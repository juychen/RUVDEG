#!/usr/bin/env bash
# ============================================================
# Run batchScVI.py DEG pipeline on public datasets.
# Each dataset can use its own cell-type column name (and its own
# pair-batch / condition columns), so you can mix datasets that were
# annotated with different cell-type vocabularies.
#
# Usage:
#   ./run_scvibatch_public.sh                          # run all datasets
#   ./run_scvibatch_public.sh GSE118767                # run only these
# ============================================================
set -euo pipefail

# ---- Config (edit here) ----
SCRIPT="/home/junyichen/code/RUVAEDEG/batchScVI.py"
INPUT_DIRS=(/data1st1/junyi/correctdata/publicdata)
OUT_ROOT="/data1st1/junyi/correctdata/transformed"
MAX_JOBS=4
CONDA_ENV="scvi-env"          # empty = use current python

# Datasets to process (basename of <INPUT_DIR>/<name>.h5ad).
ALL_REGIONS=(GSE133549 GSE118767)

# Per-dataset cell-type column name (must be same length as ALL_REGIONS).
#   GSE118767: no cell-type column yet -> pick the one you annotated
#   GSE133549: cell-line demix -> e.g. meta_cell_line / meta_cell_line_demuxlet
CELLTYPE_KEYS=(nnet2 meta_cell_line)

# Per-dataset pairing column (obs col used to build pairs; default 'company').
# Public data has no 'company' -> use source_file / protocol / sample_id.
PAIR_BATCH_KEYS=(source_file protocol)

# Per-dataset condition column. 'none' = no condition key (do NOT use "" —
# empty fields get swallowed by printf|xargs and shift the arguments).
# batchScVI.py falls back to training on ALL cells when there is no
# condition key and no 'status' column, so public data can use 'none'.
CONDITION_KEYS=(none none)

# Optional per-dataset external scVI model dir (empty = skip external decode).
# Must be the trained scVI *.model/ directory (passed to scvi.model.SCVI.load).
EXTERNAL_MODEL_DIRS=(
                     /data1st1/junyi/correctdata/transformed/GSE133549_scVInobatch/GSE133549_scVInobatch.model
                     /data1st1/junyi/correctdata/transformed/GSE118767_scVInobatch/GSE118767_scVInobatch.model
                     )

# ---- Model hyper-params (shared across datasets) ----
N_LATENT=32
N_LAYERS=2
PAIR_WEIGHT=1.0
LR=1e-3
MAX_EPOCHS=100

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

# ---- One job per dataset ----
run_one() {
    local input_dir="$1"
    local region="$2"
    local celltype_key="$3"
    local pair_batch_key="$4"
    local condition_key="$5"
    local external_model_dir="$6"

    # 'none' placeholder -> no --condition-key flag
    [[ "$condition_key" == "none" ]] && condition_key=""

    local out_dir="${OUT_ROOT}/${region}_batchscvi"
    local input="${input_dir}/${region}.h5ad"
    local outprefix="${out_dir}/${region}_batchscvi.h5ad"
    local log="${out_dir}/${region}_batchscvi.log"

    mkdir -p "$out_dir"

    if [[ ! -f "$input" ]]; then
        echo "[SKIP] input not found: $input" >&2
        return 0
    fi
    if [[ -n "$external_model_dir" && ! -d "$external_model_dir" ]]; then
        echo "[SKIP] external model not found: $external_model_dir" >&2
        return 0
    fi

    echo "[RUN] $(date '+%F %T')  ${region}: ${input}"
    echo "      celltype=$celltype_key  pair_batch=$pair_batch_key  "
    echo "      condition=${condition_key:-<none>}  ext_model=${external_model_dir:-<none>}"

    local -a cmd=(python "$SCRIPT"
        -i "$input"
        -o "$outprefix"
        --labels-key "$celltype_key"
        --pair-batch-key "$pair_batch_key"
        --n-latent "$N_LATENT"
        --n-layers "$N_LAYERS"
        --pair-weight "$PAIR_WEIGHT"
        --lr "$LR"
        --max-epochs "$MAX_EPOCHS"
        --no-compare
    )
    if [[ -n "$condition_key" ]]; then
        cmd+=(--condition-key "$condition_key")
    fi
    if [[ -n "$external_model_dir" ]]; then
        cmd+=(--external-model-dir "$external_model_dir")
    fi

    "${cmd[@]}" >"$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[DONE] ${region}  ->  ${outprefix}.h5ad  (log: $log)"
    else
        echo "[FAIL] ${region} (exit=$rc)  log: $log" >&2
    fi
    return $rc
}
export -f run_one
export SCRIPT OUT_ROOT N_LATENT N_LAYERS PAIR_WEIGHT LR MAX_EPOCHS

# ---- Main ----
if [[ $# -gt 0 ]]; then
    REGIONS=("$@")
else
    REGIONS=("${ALL_REGIONS[@]}")
fi

echo "[CONFIG] script=$SCRIPT  max_jobs=$MAX_JOBS  input_dirs=${#INPUT_DIRS[@]}"
if [[ ${#ALL_REGIONS[@]} -ne ${#CELLTYPE_KEYS[@]} ]] \
    || [[ ${#ALL_REGIONS[@]} -ne ${#PAIR_BATCH_KEYS[@]} ]] \
    || [[ ${#ALL_REGIONS[@]} -ne ${#CONDITION_KEYS[@]} ]] \
    || [[ ${#ALL_REGIONS[@]} -ne ${#EXTERNAL_MODEL_DIRS[@]} ]]; then
    echo "[ERROR] ALL_REGIONS / CELLTYPE_KEYS / PAIR_BATCH_KEYS / " \
         "CONDITION_KEYS / EXTERNAL_MODEL_DIRS must all have the same length" >&2
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
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$input_dir" "$region" \
                    "${CELLTYPE_KEYS[$idx]}" \
                    "${PAIR_BATCH_KEYS[$idx]}" \
                    "${CONDITION_KEYS[$idx]}" \
                    "${EXTERNAL_MODEL_DIRS[$idx]}"
                break
            fi
        done
    done
done | xargs -P "$MAX_JOBS" -n 6 bash -c 'run_one "$1" "$2" "$3" "$4" "$5" "$6"' _

echo "[ALL] finished"