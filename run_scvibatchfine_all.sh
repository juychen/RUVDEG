#!/usr/bin/env bash
# ============================================================
# Run scVI.py DEG pipeline on all 8 brain-region datasets.
# At most ${MAX_JOBS} jobs run concurrently.
#
# Usage:
#   ./run_scvi_all.sh                          # run all 8 regions
#   ./run_scvi_all.sh iCTX TH                  # run only these regions
# ============================================================
set -euo pipefail

# ---- Config (edit here) ----
SCRIPT="/home/junyichen/code/RUVAEDEG/batchScVI.py"
INPUT_DIRS=(
    "/data7/mark/STG/dataset/snRNA/merge_SCH_new/six_datasets_4v3_500_1000gene"
)
OUT_ROOT="/data3/junyi"
PRETRAINED_ROOT="/data3/junyi/scvi_nobatch"
CONDITION_KEY="status"
OUTPUT_DIR_SUFFIX="batchfinelr"
FINETUNE_LR="1e-4"
FINETUNE_EPOCHS=20
MAX_JOBS=4
CONDA_ENV="scvi-env"          # empty = use current python

ALL_REGIONS=(iCTX TH STR PFC MB HY HPF AMY)
#ALL_REGIONS=(iCTX)

# ---- Optional conda env ----
# Hardcoded source line removed: activation is handled here so the script
# works with any conda install and can be disabled by setting CONDA_ENV"".
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
    local dataset
    dataset="$(basename "$input_dir")"
    dataset="${dataset%_new}"
    local out_dir="${OUT_ROOT}/${dataset}_${OUTPUT_DIR_SUFFIX}"
    local input="${input_dir}/${region}_downsampled_ratio.h5ad"
    local outprefix="${out_dir}/${region}_batchscvi.h5ad"
    local pretrained_model="${PRETRAINED_ROOT}/${region}_scVImodel.model"
    local log="${out_dir}/${region}_scvi.log"

    mkdir -p "$out_dir"

    if [[ ! -f "$input" ]]; then
        echo "[SKIP] input not found: $input" >&2
        return 0
    fi
    if [[ ! -d "$pretrained_model" ]]; then
        echo "[SKIP] pretrained model not found: $pretrained_model" >&2
        return 0
    fi

    echo "[RUN] $(date '+%F %T')  ${region}: ${input}"
    python "$SCRIPT" \
        -i "$input" \
        -o "$outprefix" \
        --condition-key "$CONDITION_KEY" \
        --finetune-model-dir "$pretrained_model" \
        --external-model-dir "$pretrained_model" \
        --lr "$FINETUNE_LR" \
        --max-epochs "$FINETUNE_EPOCHS" \
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
export SCRIPT OUT_ROOT PRETRAINED_ROOT CONDITION_KEY OUTPUT_DIR_SUFFIX \
    FINETUNE_LR FINETUNE_EPOCHS

# ---- Main ----
if [[ $# -gt 0 ]]; then
    REGIONS=("$@")
else
    REGIONS=("${ALL_REGIONS[@]}")
fi

echo "[CONFIG] script=$SCRIPT  max_jobs=$MAX_JOBS  input_dirs=${#INPUT_DIRS[@]}"
for input_dir in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$input_dir" ]]; then
        echo "[WARN] input directory not found: $input_dir" >&2
        continue
    fi
    for region in "${REGIONS[@]}"; do
        printf '%s\t%s\n' "$input_dir" "$region"
    done
done | xargs -P "$MAX_JOBS" -n 2 bash -c 'run_one "$1" "$2"' _

echo "[ALL] finished"
