#!/usr/bin/env bash
# ============================================================
# Benchmark the two public standard-scVI outputs that are skipped by
# run_benchmark_pubdata_transformed.sh.
#
# Inputs:
#   /data8/junyi/pubdata/transformed/GSE133549_scVI/GSE133549_scVI.h5ad
#   /data8/junyi/pubdata/transformed/GSE118767_scVI/GSE118767_scVI.h5ad
#
# The old all-method driver intentionally skips directories named *_scVI,
# assuming they contain only a model checkpoint. These two directories also
# contain h5ad files, so this driver benchmarks them explicitly.
#
# Usage:
#   ./run_benchmark_pubdata_scvi.sh
#   ./run_benchmark_pubdata_scvi.sh GSE118767
#
# Override paths/options, for example:
#   MAX_JOBS=1 OUT_ROOT=/tmp/benchmark ./run_benchmark_pubdata_scvi.sh
# ============================================================
set -euo pipefail

BENCH_SCRIPT="${BENCH_SCRIPT:-/home/junyichen/code/RUVAEDEG/benchmarkbyauc_DM.py}"
TRANSFORMED_ROOT="${TRANSFORMED_ROOT:-/data8/junyi/pubdata/transformed}"
OUT_ROOT="${OUT_ROOT:-/data8/junyi/benchmark_results_pubdata}"
NORMALIZED_LAYER="${NORMALIZED_LAYER:-scvi_nrom_counts}"
COUNT_LAYER="${COUNT_LAYER:-counts}"
GENE_LIST="${GENE_LIST:-}"
MAX_JOBS="${MAX_JOBS:-2}"
CONDA_ENV="${CONDA_ENV:-scvi-env}"

if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "[ENV] conda env '$CONDA_ENV' activated"
fi

python -c "import scanpy" >/dev/null 2>&1 || {
    echo "[ERROR] scanpy is not importable with the current python." >&2
    exit 1
}

# Resolve the environment's Python for the worker processes.
PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python)"
fi

mkdir -p "$OUT_ROOT"

# Each row: dataset, h5ad, celltype obs key, batch obs key.
DATASETS=(
    "GSE133549|${TRANSFORMED_ROOT}/GSE133549_scVI/GSE133549_scVI.h5ad|nnet2|source_file"
    "GSE118767|${TRANSFORMED_ROOT}/GSE118767_scVI/GSE118767_scVI.h5ad|meta_cell_line_demuxlet|protocol"
)

run_one() {
    local dataset="$1"
    local h5ad="$2"
    local celltype_key="$3"
    local batch_key="$4"
    local out_dir="${OUT_ROOT}/${dataset}_scVI"
    local out_prefix="${out_dir}/${dataset}_scVI"
    local log="${out_dir}/${dataset}_scVI.log"

    mkdir -p "$out_dir"

    if [[ ! -f "$h5ad" ]]; then
        echo "[FAIL] $dataset input not found: $h5ad" >&2
        return 1
    fi

    echo "[RUN] $(date '+%F %T')  $dataset"
    echo "      input=$h5ad"
    echo "      normalized=$NORMALIZED_LAYER  count=$COUNT_LAYER"
    echo "      celltype=$celltype_key  batch=$batch_key"

    local -a cmd=(
        "$PYTHON_BIN" "$BENCH_SCRIPT"
        --h5ad "$h5ad"
        --normalized-layer "$NORMALIZED_LAYER"
        --count-layer "$COUNT_LAYER"
        --celltype-key "$celltype_key"
        --batch-key "$batch_key"
        --out-prefix "$out_prefix"
    )
    [[ -n "$GENE_LIST" ]] && cmd+=(--gene-list "$GENE_LIST")

    "${cmd[@]}" >"$log" 2>&1 \
        && echo "[DONE] $dataset  results=$out_dir  log=$log" \
        || { local rc=$?; echo "[FAIL] $dataset (exit=$rc)  log=$log" >&2; return "$rc"; }
}
export -f run_one
export PYTHON_BIN BENCH_SCRIPT OUT_ROOT NORMALIZED_LAYER COUNT_LAYER GENE_LIST

if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    REQUESTED=(GSE133549 GSE118767)
fi

printf '%s\n' "${DATASETS[@]}" \
    | while IFS='|' read -r dataset h5ad celltype_key batch_key; do
        for requested in "${REQUESTED[@]}"; do
            if [[ "$dataset" == "$requested" ]]; then
                printf '%s\t%s\t%s\t%s\n' \
                    "$dataset" "$h5ad" "$celltype_key" "$batch_key"
                break
            fi
        done
    done \
    | xargs -P "$MAX_JOBS" -n 4 bash -c 'run_one "$1" "$2" "$3" "$4"' _

echo "[ALL] finished"
