#!/usr/bin/env bash
# ============================================================
# Run darw_gene.py to produce 8-region concatenated heatmaps +
# dotplots for all scviHarmony outputs.
#
# Default behavior: scan ${INPUT_DIR} for *scviHarmony.h5ad,
# concatenate into one AnnData, plot heatmap + dotplot showing
# all 8 regions in a single figure (per gene set).
#
# Usage:
#   ./run_darw_gene.sh                            # 8-region concat heatmap (default)
#   MODE=per_region ./run_darw_gene.sh             # old per-region xargs mode
#   MODE=per_region ./run_darw_gene.sh iCTX TH    # per-region mode, subset
#   MODE=per_region ./run_darw_gene.sh --input-dir /path/to/dir
#                                                 # per-region mode, custom input dir
# ============================================================
set -euo pipefail

# ---- Mode selection ----
# "concat"     = scan ${INPUT_DIR}, concatenate 8 regions, plot 1 big heatmap (default)
# "per_region" = old behavior: per-region xargs (separate output per region)
MODE="${MODE:-concat}"

# ---- Config (edit here) ----
SCRIPT="/home/junyichen/code/RUVAEDEG/darw_gene.py"
INPUT_DIR="/data3/junyi/scvi_harmony"   # each region's *scviHarmony.h5ad lives here
OUT_DIR="/data3/junyi/scvi_harmony/dotplots"
MAX_JOBS=4
CONDA_ENV="scvi-env"          # empty = use current python

# darw_gene.py specific defaults (can be overridden via CLI flags)
GENEGROUP="${GENEGROUP:-/data2st2/junyi/code/sn/data/All_degs_N_v0715FF.xlsx}"
MAX_GENES=50
GROUPBY="sample_status"
STANDARD_SCALE="var"
FIGSIZE="20 70"
FONT="/data2st1/junyi/arial.ttf"
LAYER="scvi_reconstructed_counts_harmony"
GLOB_PATTERN="*scviHarmony.h5ad"

# Color strategy: "percentile" (ipynb-like, p5(>0)/p95 across regions) or "fixed"
COLOR_STRATEGY="percentile"
# Draw a true heatmap (sc.pl.matrixplot) in addition to dotplot, when HEATMAP=1
HEATMAP=1
HEATMAP_CMAP="viridis"

ALL_REGIONS=(iCTX TH STR PFC MB HY HPF AMY)

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

# ---- Build common python args (shared by both modes) ----
build_python_args() {
    local -a args=(
        --genegroup "$GENEGROUP"
        --max-genes "$MAX_GENES"
        --groupby "$GROUPBY"
        --standard-scale "$STANDARD_SCALE"
        --layer "$LAYER"
        --color-strategy "$COLOR_STRATEGY"
        --figsize $FIGSIZE
        --font "$FONT"
    )
    if [[ "$HEATMAP" == "1" ]]; then
        args+=(--heatmap --heatmap-cmap "$HEATMAP_CMAP")
    fi
    printf '%s\n' "${args[@]}"
}

# ============================================================
# MODE=concat: scan INPUT_DIR, concatenate 8 regions, one heatmap
# ============================================================
if [[ "$MODE" == "concat" ]]; then
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "[ERROR] INPUT_DIR is not a directory: $INPUT_DIR" >&2
        exit 1
    fi
    mkdir -p "$OUT_DIR"
    log="${OUT_DIR}/concat.log"
    echo "[RUN-CONCAT] $(date '+%F %T')"
    echo "  input_dir   = $INPUT_DIR"
    echo "  glob        = $GLOB_PATTERN"
    echo "  out_dir     = $OUT_DIR"
    echo "  layer       = $LAYER"
    echo "  color       = $COLOR_STRATEGY"
    echo "  heatmap     = $HEATMAP (cmap=$HEATMAP_CMAP)"

    # shellcheck disable=SC2046
    python "$SCRIPT" \
        --input  "$INPUT_DIR" \
        --output "$OUT_DIR" \
        --glob "$GLOB_PATTERN" \
        $(build_python_args) \
        >"$log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[DONE-CONCAT]  ->  $OUT_DIR  (log: $log)"
    else
        echo "[FAIL-CONCAT] (exit=$rc)  log: $log" >&2
    fi
    exit $rc
fi

# ============================================================
# MODE=per_region: old behavior, per-region xargs (with subdirs)
# ============================================================
echo "[MODE=per_region]"

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
    local extra_args=()
    if [[ "$HEATMAP" == "1" ]]; then
        extra_args+=(--heatmap --heatmap-cmap "$HEATMAP_CMAP")
    fi
    python "$SCRIPT" \
        --input  "$input" \
        --output "$region_out" \
        --genegroup "$GENEGROUP" \
        --max-genes "$MAX_GENES" \
        --groupby "$GROUPBY" \
        --standard-scale "$STANDARD_SCALE" \
        --layer "$LAYER" \
        --color-strategy "$COLOR_STRATEGY" \
        --figsize $FIGSIZE \
        --font "$FONT" \
        "${extra_args[@]}" \
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
export SCRIPT INPUT_DIR OUT_DIR GENEGROUP MAX_GENES GROUPBY STANDARD_SCALE FIGSIZE FONT LAYER

# ---- Sub-dispatch for per_region: support --input-dir ----
if [[ ${1:-} == "--input-dir" ]]; then
    INPUT_DIR_OPT="${2:?missing value for --input-dir}"
    shift 2
    EXTRA_ARGS=("$@")
    if [[ ! -d "$INPUT_DIR_OPT" ]]; then
        echo "[ERROR] --input-dir path is not a directory: $INPUT_DIR_OPT" >&2
        exit 1
    fi
    INPUT_DIR="$INPUT_DIR_OPT"  # override
    echo "[CONFIG] per_region  input_dir=$INPUT_DIR  out_dir=$OUT_DIR"
    # fall through to xargs below
fi

# ---- Region selection ----
if [[ $# -gt 0 && ${1:-} != "--input-dir" ]]; then
    REGIONS=("$@")
else
    REGIONS=("${ALL_REGIONS[@]}")
fi

echo "[CONFIG] script=$SCRIPT  max_jobs=$MAX_JOBS  input_dir=$INPUT_DIR  out_dir=$OUT_DIR"
printf '%s\n' "${REGIONS[@]}" \
    | xargs -P "$MAX_JOBS" -I{} bash -c 'run_one "$1"' _ {}

echo "[ALL] finished"
