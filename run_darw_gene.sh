#!/usr/bin/env bash
# ============================================================
# Run draw_neggene.py: per-region dotplot pipeline.
# Scans ${INPUT_DIR} for *.h5ad, plots per-region dotplots
# (vertically stacked, shared color) into ${OUT_DIR}.
#
# Usage:
#   ./run_darw_gene.sh                            # default config
#   INPUT_DIR=/path/to/dir ./run_darw_gene.sh      # custom input folder
#   OUT_DIR=/path/to/out ./run_darw_gene.sh        # custom output folder
#   LAYER=scvi_reconstructed_counts_harmony ./run_darw_gene.sh
# ============================================================
set -euo pipefail

# ---- Config (edit here) ----
SCRIPT="/home/junyichen/code/RUVAEDEG/draw_neggene.py"
INPUT_DIR="${INPUT_DIR:-/data3/junyi/scvi_harmony}"          # folder containing *.h5ad
OUT_DIR="${OUT_DIR:-/data3/junyi/scvi_harmony/dotplots}"     # dotplot pdfs saved here
CONDA_ENV="scvi-env"          # empty = use current python

# draw_neggene.py specific defaults (can be overridden via env vars)
GENEGROUP="${GENEGROUP:-/data2st2/junyi/code/sn/data/All_degs_N_v0715FF.xlsx}"
MAX_GENES="${MAX_GENES:-50}"
GROUPBY="${GROUPBY:-sample_status}"
STANDARD_SCALE="${STANDARD_SCALE:-var}"
FIGSIZE="${FIGSIZE:-20 70}"
FONT="${FONT:-/data2st1/junyi/arial.ttf}"
LAYER="${LAYER:-scvi_reconstructed_counts_harmony}"  # layer to plot (default: scvi_reconstructed_counts_harmony)
VMAX="${VMAX:-1}"              # empty = auto global p95
VMIN="${VMIN:-0}"

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

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[ERROR] INPUT_DIR is not a directory: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
log="${OUT_DIR}/draw_neggene.log"

echo "[RUN] $(date '+%F %T')"
echo "  input_dir   = $INPUT_DIR"
echo "  out_dir     = $OUT_DIR"
echo "  layer       = $LAYER"
echo "  standard    = $STANDARD_SCALE"
echo "  vmin/vmax   = $VMIN / ${VMAX:-auto-p95}"
echo "  genegroup   = $GENEGROUP"

# Build args
args=(
    --input  "$INPUT_DIR"
    --output "$OUT_DIR"
    --genegroup "$GENEGROUP"
    --max-genes "$MAX_GENES"
    --groupby "$GROUPBY"
    --standard-scale "$STANDARD_SCALE"
    --layer "$LAYER"
    --figsize $FIGSIZE
    --font "$FONT"
    --vmin "$VMIN"
)
if [[ -n "$VMAX" ]]; then
    args+=(--vmax "$VMAX")
fi

python "$SCRIPT" "${args[@]}" >"$log" 2>&1
rc=$?
if [[ $rc -eq 0 ]]; then
    echo "[DONE]  ->  $OUT_DIR  (log: $log)"
else
    echo "[FAIL] (exit=$rc)  log: $log" >&2
fi
exit $rc
