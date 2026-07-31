#!/usr/bin/env bash
# ============================================================
# Run scviHarmony.py DEG pipeline on all 8 brain-region datasets.
# At most ${MAX_JOBS} jobs run concurrently.
#
# Usage:
#   ./run_scvi_harmony.sh                          # run all 8 regions
#   ./run_scvi_harmony.sh iCTX TH                  # run only these regions
# ============================================================
set -euo pipefail

# ---- Config (edit here) ----
SCRIPT="/home/junyichen/code/RUVAEDEG/scviHarmony.py"
INPUT_DIR="/data7/mark/STG/dataset/snRNA/merge_SCH_new/six_datasets_4v3_500_1000gene"
OUT_DIR="/data3/junyi/scvi_harmony"
MAX_JOBS=4
CONDA_ENV="scvi-env"          # empty = use current python
NCLUST="celltype.L2"
HARMONY_BATCH="company"
LAMB=0.3
MAX_ITER_HARMONY=20

ALL_REGIONS=(iCTX TH STR PFC MB HY HPF AMY)
#ALL_REGIONS=(iCTX)

# ---- Optional conda env ----
# Hardcoded source line removed: activation is handled here so the script
# works with any conda install and can be disabled by setting CONDA_ENV="".
if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "[ENV] conda env '$CONDA_ENV' activated"
fi

python -c "import scvi" >/dev/null 2>&1 || {
    echo "[ERROR] 'scvi' is not importable with the current python." >&2
    exit 1
}

mkdir -p "$OUT_DIR"

# ---- One job per region ----
run_one() {
    local region="$1"
    local input="${INPUT_DIR}/${region}_downsampled_ratio.h5ad"
    local outprefix="${OUT_DIR}/${region}_scviHarmony.h5ad"
    local log="${OUT_DIR}/${region}_scviHarmony.log"

    if [[ ! -f "$input" ]]; then
        echo "[SKIP] input not found: $input" >&2
        return 0
    fi

    echo "[RUN] $(date '+%F %T')  ${region}: ${input}"
    python "$SCRIPT" \
        -i "$input" \
        -o "$outprefix" \
        --nclust "$NCLUST" \
        --harmony-batch "$HARMONY_BATCH" \
        --lamb "$LAMB" \
        --max-iter-harmony "$MAX_ITER_HARMONY" \
        >"$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[DONE] ${region}  ->  ${outprefix}.h5ad  (log: $log)"
    else
        echo "[FAIL] ${region} (exit=$rc)  log: $log" >&2
    fi
    return $rc
}
export -f run_one
export SCRIPT INPUT_DIR OUT_DIR NCLUST HARMONY_BATCH LAMB MAX_ITER_HARMONY

# ---- Main ----
if [[ $# -gt 0 ]]; then
    REGIONS=("$@")
else
    REGIONS=("${ALL_REGIONS[@]}")
fi

echo "[CONFIG] script=$SCRIPT  max_jobs=$MAX_JOBS  out_dir=$OUT_DIR"
printf '%s\n' "${REGIONS[@]}" \
    | xargs -P "$MAX_JOBS" -I{} bash -c 'run_one "$1"' _ {}

echo "[ALL] finished"
