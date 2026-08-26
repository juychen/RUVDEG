#!/usr/bin/env bash
# Run the distance benchmark on Harmony-corrected scVI h5ad files.
set -euo pipefail

SCRIPT="${SCRIPT:-/home/junyichen/code/RUVAEDEG/benchmarkbyauc_DM.py}"
INPUT_DIRS=(${INPUT_DIRS:-/data3/junyi/scvi_harmony})
NORMALIZED_LAYER="${NORMALIZED_LAYER:-scvi_nrom_counts_harmony}"
COUNT_LAYER="${COUNT_LAYER:-scvi_reconstructed_counts_harmony}"
CELLTYPE_KEY="${CELLTYPE_KEY:-celltype.L2}"
BATCH_KEY="${BATCH_KEY:-company}"
GENE_LIST="${GENE_LIST:-}"
OUT_ROOT="${OUT_ROOT:-/data3/junyi/benchmark_DM_results}"
MAX_JOBS="${MAX_JOBS:-4}"
CONDA_ENV="${CONDA_ENV:-scvi-env}"
PATTERN="${PATTERN:-*_scvi*.h5ad}"
MODEL="${MODEL:-CON_M}"           # empty = no Model filter

if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

mkdir -p "$OUT_ROOT"

declare -a H5AD_FILES=()
if [[ $# -gt 0 ]]; then
    H5AD_FILES=("$@")
else
    for d in "${INPUT_DIRS[@]}"; do
        [[ -d "$d" ]] || continue
        while IFS= read -r -d '' f; do H5AD_FILES+=("$f"); done \
            < <(find "$d" -maxdepth 1 -name "$PATTERN" -type f -print0 2>/dev/null)
    done
fi
[[ ${#H5AD_FILES[@]} -gt 0 ]] || { echo "[ERROR] no h5ad files found" >&2; exit 1; }

echo "[CONFIG] files=${#H5AD_FILES[@]} max_jobs=$MAX_JOBS celltype=$CELLTYPE_KEY batch=$BATCH_KEY model=${MODEL:-(skip)}"

run_one() {
    local h5ad_path="$1"
    local h out_dir out_prefix log
    h="$(basename "$h5ad_path" .h5ad)"
    out_dir="$OUT_ROOT/$h"
    out_prefix="$out_dir/$h"
    log="$out_dir/$h.log"
    mkdir -p "$out_dir"

    local args=(
        python "$SCRIPT"
        --h5ad "$h5ad_path"
        --normalized-layer "$NORMALIZED_LAYER"
        --count-layer "$COUNT_LAYER"
        --celltype-key "$CELLTYPE_KEY"
        --batch-key "$BATCH_KEY"
        --out-prefix "$out_prefix"
    )
    [[ -n "$GENE_LIST" ]] && args+=(--gene-list "$GENE_LIST")
    [[ -n "$MODEL" ]] && args+=(--model "$MODEL")

    echo "[RUN] $(date '+%F %T') $h"
    "${args[@]}" >"$log" 2>&1 \
        && echo "[DONE] $h (log: $log)" \
        || { echo "[FAIL] $h (exit=$?) log: $log" >&2; return 1; }
}
export -f run_one
export SCRIPT NORMALIZED_LAYER COUNT_LAYER CELLTYPE_KEY BATCH_KEY GENE_LIST OUT_ROOT MODEL

printf '%s\n' "${H5AD_FILES[@]}" | xargs -P "$MAX_JOBS" -n 1 bash -c 'run_one "$0"'
echo "[ALL] finished"
