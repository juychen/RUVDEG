#!/usr/bin/env bash
# ============================================================
# Run darw_gene.py (per-region dotplot pipeline) for all
# scviHarmony outputs. At most ${MAX_JOBS} jobs run concurrently.
#
# Usage:
#   ./run_darw_gene.sh                           # run all 8 regions
#   ./run_darw_gene.sh iCTX TH                   # run only these regions
#   ./run_darw_gene.sh --genegroup /path/to.xlsx # custom genegroup file
#   ./run_darw_gene.sh --max-genes 100           # override batch size
# ============================================================
set -euo pipefail

# ---- Config (edit here) ----
SCRIPT="/home/junyichen/code/RUVAEDEG/darw_gene.py"
INPUT_DIR="/data3/junyi/scvi_harmony"           # each region's h5ad lives here
OUT_DIR="/data3/junyi/scvi_harmony/dotplots"     # dotplot pdfs will be saved here
MAX_JOBS=4
CONDA_ENV="scvi-env"          # empty = use current python

# darw_gene.py specific defaults (can be overridden via CLI flags)
GENEGROUP="${GENEGROUP:-/data2st2/junyi/code/sn/data/All_degs_N_v0715FF.xlsx}"
MAX_GENES=50
GROUPBY="sample_status"
STANDARD_SCALE="var"
FIGSIZE="20 20"
FONT="/data2st1/junyi/arial.ttf"
LAYER="scvi_reconstructed_counts_harmony"

ALL_REGIONS=(iCTX TH STR PFC MB HY HPF AMY)
#ALL_REGIONS=(iCTX)

# ---- Optional conda env ----
if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "[ENV] conda env '$CONDA_ENV' activated"
fi

python -c "import scanpy" >/dev/null 2>&1 || {
    echo "[ERROR] 'scanpy' is not importable with the current python." >&2
    exit 1
}

mkdir -p "$OUT_DIR"

# ---- One job per region ----
run_one() {
    local region="$1"
    local input="${INPUT_DIR}/${region}_scviHarmony.h5ad"
    local region_out="${OUT_DIR}/${region}"
    local log="${region_out}.log"

    if [[ ! -f "$input" ]]; then
        echo "[SKIP] input not found: $input" >&2
        return 0
    fi

    mkdir -p "$region_out"

    echo "[RUN] $(date '+%F %T')  ${region}: ${input}"
    python "$SCRIPT" \
        --input  "$input" \
        --output "$region_out" \
        --genegroup "$GENEGROUP" \
        --max-genes "$MAX_GENES" \
        --groupby "$GROUPBY" \
        --standard-scale "$STANDARD_SCALE" \
        --layer "$LAYER" \
        --figsize $FIGSIZE \
        --font "$FONT" \
        >"$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[DONE] ${region}  ->  ${region_out}  (log: $log)"
    else
        echo "[FAIL] ${region} (exit=$rc)  log: $log" >&2
    fi
    return $rc
}
export -f run_one
export SCRIPT INPUT_DIR OUT_DIR GENEGROUP MAX_GENES GROUPBY STANDARD_SCALE FIGSIZE FONT

# ---- Region selection ----
if [[ $# -gt 0 ]]; then
    REGIONS=("$@")
else
    REGIONS=("${ALL_REGIONS[@]}")
fi

echo "[CONFIG] script=$SCRIPT  max_jobs=$MAX_JOBS  input_dir=$INPUT_DIR  out_dir=$OUT_DIR"
printf '%s\n' "${REGIONS[@]}" \
    | xargs -P "$MAX_JOBS" -I{} bash -c 'run_one "$1"' _ {}

echo "[ALL] finished"
