#!/usr/bin/env bash
# ============================================================
# Run scVI.py with batch correction on the two
# public benchmark datasets (GSE118767 / GSE133549).
# At most ${MAX_JOBS} jobs run concurrently.
#
# Mirrors run_scvi_harmony_public.sh but passes --no-batch so that
# scVI is trained without registering any batch_key (pure baseline).
#
# Usage:
#   ./run_scvi_nobatch_public.sh                       # run both datasets
#   ./run_scvi_nobatch_public.sh GSE118767             # run one only
#   ./run_scvi_nobatch_public.sh GSE118767 GSE133549   # run selected
# ============================================================
set -euo pipefail

# ---- Config (edit here) ----
SCRIPT="/home/junyichen/code/RUVAEDEG/scVI.py"
INPUT_DIRS=(/data8/junyi/pubdata/publicdata)
OUT_ROOT="/data8/junyi/pubdata/transformed"
MAX_JOBS=4
CONDA_ENV="scvi-env"          # empty = use current python

# scVI hyperparameters (kept identical to the harmonised version for fairness)
N_LATENT=32
N_LAYERS=2

# Datasets to process (basename of <INPUT_DIR>/<name>.h5ad).
ALL_REGIONS=(GSE133549 GSE118767)

# Per-dataset cell-type column name (must be same length as ALL_REGIONS).
CELLTYPE_KEYS=(nnet2 meta_cell_line_demuxlet)

# Per-dataset pairing column (obs col used to build pairs; default 'company').
PAIR_BATCH_KEYS=(source_file protocol)

# Per-dataset condition column. 'none' = no condition key.
CONDITION_KEYS=(none none)

# ---- Optional conda env ----
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

# ---- One job per input directory and region ----
run_one() {
    local input_dir="$1"
    local region="$2"
    local dataset="${region}"                         # use region name as dataset id
    local celltype_key="$3"
    local pair_batch_key="$4"
    local condition_key="$5"
    local out_dir="${OUT_ROOT}/${dataset}_scVI"
    local input="${input_dir}/${region}.h5ad"
    local outprefix="${out_dir}/${region}_scVI.h5ad"
    local log="${out_dir}/${region}_scVI.log"

    mkdir -p "$out_dir"

    if [[ ! -f "$input" ]]; then
        echo "[SKIP] input not found: $input" >&2
        return 0
    fi

    echo "[RUN] $(date '+%F %T')  ${region}: ${input}"
    python "$SCRIPT" \
        -i "$input" \
        -o "$outprefix" \
        --batch-key "$pair_batch_key" \
        --n-latent "$N_LATENT" \
        --n-layers "$N_LAYERS" \
        --no-compare \
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
export SCRIPT OUT_ROOT N_LATENT N_LAYERS

# ---- Main ----
if [[ $# -gt 0 ]]; then
    REGIONS=("$@")
else
    REGIONS=("${ALL_REGIONS[@]}")
fi

echo "[CONFIG] script=$SCRIPT  max_jobs=$MAX_JOBS  input_dirs=${#INPUT_DIRS[@]}  regions=${REGIONS[*]}"
if [[ ${#ALL_REGIONS[@]} -ne ${#CELLTYPE_KEYS[@]} ]] \
    || [[ ${#ALL_REGIONS[@]} -ne ${#PAIR_BATCH_KEYS[@]} ]] \
    || [[ ${#ALL_REGIONS[@]} -ne ${#CONDITION_KEYS[@]} ]]; then
    echo "[ERROR] ALL_REGIONS / CELLTYPE_KEYS / PAIR_BATCH_KEYS / " \
         "CONDITION_KEYS must all have the same length" >&2
    exit 1
fi

for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$input_dir" ]]; then
        echo "[WARN] input directory not found: $input_dir" >&2
        continue
    fi
    for region in "${REGIONS[@]}"; do
        for idx in "${!ALL_REGIONS[@]}"; do
            if [[ "${ALL_REGIONS[$idx]}" == "$region" ]]; then
                printf '%s\t%s\t%s\t%s\t%s\n' \
                    "$input_dir" "$region" \
                    "${CELLTYPE_KEYS[$idx]}" \
                    "${PAIR_BATCH_KEYS[$idx]}" \
                    "${CONDITION_KEYS[$idx]}"
                break
            fi
        done
    done
done | xargs -P "$MAX_JOBS" -n 5 bash -c 'run_one "$1" "$2" "$3" "$4" "$5"' _

echo "[ALL] finished"
