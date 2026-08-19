#!/usr/bin/env bash
# Run benchmarkbyauc.py on every *_batchscvi.h5ad under INPUT_DIRS.
# Override anything via env vars, e.g.:
#   INPUT_DIRS="/data3/junyi" NORMALIZED_LAYER=foo COUNT_LAYER=bar \
#   TOP_N=10 MODEL=CON_M MAX_JOBS=4 OUT_ROOT=/tmp/bench \
#       ./run_benchmark_correct.sh /abs/path/file.h5ad
set -euo pipefail

SCRIPT="${SCRIPT:-/home/junyichen/code/RUVAEDEG/benchmarkbyauc.py}"
INPUT_DIRS=(${INPUT_DIRS:-/data3/junyi/scvi_harmony})
NORMALIZED_LAYER="${NORMALIZED_LAYER:-scvi_nrom_counts_harmony}"
COUNT_LAYER="${COUNT_LAYER:-scvi_reconstructed_counts_harmony}"
TOP_N="${TOP_N:-10}"
MODEL="${MODEL:-}"                  # empty = skip --model
OUT_ROOT="${OUT_ROOT:-/data3/junyi/benchmark_results}"
MAX_JOBS="${MAX_JOBS:-4}"
CONDA_ENV="${CONDA_ENV:-scvi-env}"
PATTERN="${PATTERN:-*_scvi*.h5ad}"

if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

mkdir -p "$OUT_ROOT"

# Build list of h5ad files: CLI args if given, else scan INPUT_DIRS for PATTERN
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

echo "[CONFIG] files=${#H5AD_FILES[@]}  max_jobs=$MAX_JOBS  model=${MODEL:-(skip)}"

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
        --top-n "$TOP_N"
        --out-prefix "$out_prefix"
    )
    [[ -n "$MODEL" ]] && args+=(--model "$MODEL")

    echo "[RUN] $(date '+%F %T')  $h"
    "${args[@]}" >"$log" 2>&1 \
        && echo "[DONE] $h  (log: $log)" \
        || { echo "[FAIL] $h (exit=$?)  log: $log" >&2; return 1; }
}
export -f run_one
export SCRIPT NORMALIZED_LAYER COUNT_LAYER TOP_N MODEL OUT_ROOT

printf '%s\n' "${H5AD_FILES[@]}" | xargs -P "$MAX_JOBS" -n 1 bash -c 'run_one "$0"'
echo "[ALL] finished"