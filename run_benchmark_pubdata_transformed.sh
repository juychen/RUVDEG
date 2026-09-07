#!/usr/bin/env bash
# ============================================================
# Benchmark scVI / batchscVI / batchscvifix / scVInobatch / scviharmony
# outputs under /data8/junyi/pubdata/transformed/<GSE>_<method>/.
#
# Distance-based benchmark: calls benchmarkbyauc_DM.py which computes
# per-celltype × batch-pair distances (mean_diff, cohens_d, var_ratio,
# hellinger) on log1p raw vs scVI-corrected layers.
#
# Differences from run_benchmark_scvi_*.sh:
#   * Iterates the 10 method directories
#   * Auto-detects method (scVI / batchscvi / batchscvifix / scVInobatch
#     / scviharmony) and dataset (GSE133549 / GSE118767) from the
#     directory name
#   * Picks the method-appropriate normalized/count layer pair
#   * Prefers *_full.h5ad (with auto-fallback to *.h5ad when the
#     full file is missing the required normalized layer)
#   * Per-dataset --celltype-key and --batch-key (DM needs both; AUC
#     benchmark only needs --company which we still inject for safety)
#   * Runs prep_pubdata_obs.py first to inject status/company/Model
#     columns (status/company are not used by DM but harmless; Model is
#     only consulted when --model is passed)
#
# Override anything via env vars, e.g.:
#   INPUT_DIRS="/data8/junyi/pubdata/transformed/GSE118767_scviharmony" \
#   MAX_JOBS=1 OUT_ROOT=/tmp/bench \
#       ./run_benchmark_pubdata_transformed.sh
# ============================================================
set -euo pipefail

BENCH_SCRIPT="${BENCH_SCRIPT:-/home/junyichen/code/RUVAEDEG/benchmarkbyauc_DM.py}"
PREP_SCRIPT="${PREP_SCRIPT:-/home/junyichen/code/RUVAEDEG/prep_pubdata_obs.py}"

# 10 method dirs (configurable; default = all 10).
INPUT_DIRS=(${INPUT_DIRS:-\
    /data8/junyi/pubdata/transformed/GSE118767_scVI \
    /data8/junyi/pubdata/transformed/GSE133549_scVI \
    /data8/junyi/pubdata/transformed/GSE118767_batchscvi \
    /data8/junyi/pubdata/transformed/GSE133549_batchscvi \
    /data8/junyi/pubdata/transformed/GSE118767_batchscvifix \
    /data8/junyi/pubdata/transformed/GSE133549_batchscvifix \
    /data8/junyi/pubdata/transformed/GSE118767_scVInobatch \
    /data8/junyi/pubdata/transformed/GSE133549_scVInobatch \
    /data8/junyi/pubdata/transformed/GSE118767_scviharmony \
    /data8/junyi/pubdata/transformed/GSE133549_scviharmony \
})

TOP_N="${TOP_N:-10}"
MODEL="${MODEL:-}"                  # empty = skip --model
OUT_ROOT="${OUT_ROOT:-/data8/junyi/benchmark_results_pubdata}"
MAX_JOBS="${MAX_JOBS:-2}"
CONDA_ENV="${CONDA_ENV:-scvi-env}"

if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "[ENV] conda env '$CONDA_ENV' activated"
fi

python -c "import scanpy, scvi" >/dev/null 2>&1 || {
    echo "[ERROR] 'scanpy'/'scvi' not importable with the current python." >&2
    exit 1
}

# Resolve the python that actually has scanpy (the conda-env python), so
# the helpers below work even from inside subshells where PATH inheritance
# is unreliable.
CONDA_PY="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "$CONDA_PY" ]]; then
    # Fall back to whatever `python` resolves to inside the activated env.
    CONDA_PY="$(command -v python)"
fi
echo "[INFO] python for helpers: $CONDA_PY"

mkdir -p "$OUT_ROOT"

# ----------------------------- per-method config -----------------------------
# Keys: METHOD -> "normalized_layer|count_layer|primary_glob|fallback_glob|short_tag"
declare -A METHOD_LAYER=(
    [scVI]="scvi_nrom_counts|counts|*_scVI.h5ad|*_scVI_full.h5ad|scVI"
    [batchscvi]="scvi_nrom_counts|counts|*_batchscvi_full.h5ad|*_batchscvi.h5ad|batchscvi"
    [batchscvifix]="scvi_nrom_counts|counts|*_batchscvifix_full.h5ad|*_batchscvifix.h5ad|batchscvifix"
    [scVInobatch]="scvi_nrom_counts|counts|*_scVInobatch_full.h5ad|*_scVInobatch.h5ad|scVInobatch"
    [scviharmony]="scvi_nrom_counts_harmony|counts|*subset_scviHarmony.h5ad|*_scviHarmony.h5ad|scviharmony"
)

# ----------------------------- per-(dataset, method) obs keys -----------------------------
# Keys: DATASET -> METHOD -> "celltype_key|batch_key"
#
# NOTE: cell-type / batch columns differ by (dataset × method) because the
# public-data harmony / scVI re-runs use different subsets / different
# ground-truth columns. The most reliable pair (after per-dataset
# prioritisation) is:
#   * GSE133549 scVI / batchscvi / batchscvifix / scVInobatch: nnet2 (cell type) + source_file (batch)
#   * GSE133549 scviharmony:             meta_cell_line_demuxlet + meta_batch
#   * GSE118767 *:                       meta_cell_line_demuxlet + protocol
# Override per-call with --celltype-key / --batch-key CLI args.
declare -A DATASET_METHOD_KEYS=(
    [GSE133549__scVI]="nnet2|source_file"
    [GSE133549__batchscvi]="nnet2|source_file"
    [GSE133549__batchscvifix]="nnet2|source_file"
    [GSE133549__scVInobatch]="nnet2|source_file"
    [GSE133549__scviharmony]="nnet2|source_file"
    [GSE118767__scVI]="meta_cell_line_demuxlet|protocol"
    [GSE118767__batchscvi]="meta_cell_line_demuxlet|protocol"
    [GSE118767__batchscvifix]="meta_cell_line_demuxlet|protocol"
    [GSE118767__scVInobatch]="meta_cell_line_demuxlet|protocol"
    [GSE118767__scviharmony]="meta_cell_line_demuxlet|protocol"
)

# Fallback priority list when the (dataset, method)-specific column is missing
# in the chosen h5ad. The first column that exists in adata.obs wins.
GSE133549_CT_FALLBACKS=("nnet2" "ident" "meta_cell_line_demuxlet" "meta_demuxlet_cls")
GSE133549_BATCH_FALLBACKS=("source_file" "meta_batch" "sample_id")
GSE118767_CT_FALLBACKS=("meta_cell_line_demuxlet" "meta_demuxlet_cls" "meta_cell_line")
GSE118767_BATCH_FALLBACKS=("protocol" "meta_batch" "source_file")

# ----------------------------- helpers -----------------------------

# Extract method suffix from a directory basename like "GSE118767_batchscvi"
# -> "batchscvi". Echoes nothing if no method suffix matches.
extract_method() {
    local base="$1"
    case "$base" in
        GSE*_scVI|scVI) echo scVI ;;
        GSE*_batchscvi) echo batchscvi ;;
        GSE*_batchscvifix) echo batchscvifix ;;
        GSE*_scVInobatch) echo scVInobatch ;;
        GSE*_scviharmony) echo scviharmony ;;
        *) return 1 ;;
    esac
}

# Extract GSE id from a directory basename like "GSE118767_batchscvi"
# -> "GSE118767".
extract_dataset() {
    local base="$1"
    case "$base" in
        GSE*) echo "${base%%_*}" ;;
        *) return 1 ;;
    esac
}

# Pick an h5ad in $dir that:
#   1. Matches $primary_glob (preferred)
#   2. Else matches $fallback_glob
# Returns the first match found, or empty if none.
pick_h5ad() {
    local dir="$1" primary="$2" fallback="$3"
    local f
    f=$(find "$dir" -maxdepth 1 -name "$primary" -type f 2>/dev/null | head -n 1 || true)
    [[ -n "$f" ]] && { echo "$f"; return 0; }
    f=$(find "$dir" -maxdepth 1 -name "$fallback" -type f 2>/dev/null | head -n 1 || true)
    [[ -n "$f" ]] && { echo "$f"; return 0; }
    return 1
}

# Validate that $h5ad contains both required layers; auto-fallback to the
# non-full sibling when the primary file is missing the normalized layer.
# Echoes the (possibly replaced) h5ad path.

# Resolve the actual obs column for a given dataset/method/key-type
# (celltype vs batch), preferring the (dataset, method)-specific column and
# then trying the fallback list for that dataset. Echoes the column name.
resolve_obs_key() {
    local h5ad="$1" dataset="$2" method="$3" key_type="$4"
    local cfg="${DATASET_METHOD_KEYS[${dataset}__${method}]:-}"
    local pref=""
    if [[ -n "$cfg" ]]; then
        IFS='|' read -r pref_ct pref_batch <<<"$cfg"
        if [[ "$key_type" == "celltype" ]]; then pref="$pref_ct"; else pref="$pref_batch"; fi
    fi
    local -a fallbacks
    if [[ "$dataset" == "GSE118767" ]]; then
        if [[ "$key_type" == "celltype" ]]; then
            fallbacks=("${GSE118767_CT_FALLBACKS[@]}")
        else
            fallbacks=("${GSE118767_BATCH_FALLBACKS[@]}")
        fi
    else
        if [[ "$key_type" == "celltype" ]]; then
            fallbacks=("${GSE133549_CT_FALLBACKS[@]}")
        else
            fallbacks=("${GSE133549_BATCH_FALLBACKS[@]}")
        fi
    fi
    local cols candidate
    cols=$("$CONDA_PY" - "$h5ad" <<'PY' 2>/dev/null
import sys
import scanpy as sc
a = sc.read_h5ad(sys.argv[1], backed="r")
print(",".join(a.obs.columns))
PY
)
    for candidate in "$pref" "${fallbacks[@]}"; do
        [[ -z "$candidate" ]] && continue
        if [[ ",$cols," == *",$candidate,"* ]]; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

# Validate that $h5ad contains both required layers; auto-fallback to the
# non-full sibling when the primary file is missing the normalized layer.
# Echoes the (possibly replaced) h5ad path.
validate_layers() {
    local h5ad="$1" norm="$2" count="$3" fallback_glob="$4"
    local layers
    layers=$("$CONDA_PY" - "$h5ad" <<'PY' 2>/dev/null
import sys
import scanpy as sc
a = sc.read_h5ad(sys.argv[1], backed="r")
print(",".join(str(k) for k in a.layers.keys()))
PY
)
    if [[ ",$layers," == *",$norm,"* ]] && [[ ",$layers," == *",$count,"* ]]; then
        echo "$h5ad"; return 0
    fi
    local dir sibling
    dir=$(dirname "$h5ad")
    sibling=$(find "$dir" -maxdepth 1 -name "$fallback_glob" -type f 2>/dev/null | head -n 1 || true)
    if [[ -n "$sibling" && -s "$sibling" ]]; then
        local sib_layers
        sib_layers=$("$CONDA_PY" - "$sibling" <<'PY' 2>/dev/null
import sys
import scanpy as sc
a = sc.read_h5ad(sys.argv[1], backed="r")
print(",".join(str(k) for k in a.layers.keys()))
PY
)
        if [[ ",$sib_layers," == *",$norm,"* ]] && [[ ",$sib_layers," == *",$count,"* ]]; then
            echo "[WARN] $(basename "$h5ad") missing required layer ($norm/$count); falling back to $(basename "$sibling")" >&2
            echo "$sibling"; return 0
        fi
    fi
    echo "[ERROR] neither $(basename "$h5ad") nor sibling has layers ($norm, $count)" >&2
    return 1
}

# ----------------------------- per-dir worker -----------------------------

run_one() {
    local input_dir="$1" out_tag="$2" h5ad_path="$3" norm_layer="$4" count_layer="$5" celltype_key="$6" batch_key="$7"

    local out_dir="$OUT_ROOT/$out_tag"
    local prepped="${h5ad_path}.prepped.h5ad"
    local out_prefix="$out_dir/$out_tag"
    local log="$out_dir/$out_tag.log"

    mkdir -p "$out_dir"

    echo "[PREP] $(date '+%F %T')  $out_tag  <-  $(basename "$h5ad_path")"
    python "$PREP_SCRIPT" \
        --h5ad "$h5ad_path" \
        --out "$prepped" \
        >"$out_dir/prep.log" 2>&1 \
        || { echo "[FAIL] prep  $out_tag (exit=$?)  log: $out_dir/prep.log" >&2; return 1; }

    local args=(
        python "$BENCH_SCRIPT"
        --h5ad "$prepped"
        --normalized-layer "$norm_layer"
        --count-layer "$count_layer"
        --celltype-key "$celltype_key"
        --batch-key "$batch_key"
        --out-prefix "$out_prefix"
    )
    [[ -n "$MODEL" ]] && args+=(--model "$MODEL")

    echo "[RUN]  $(date '+%F %T')  $out_tag  layers=($norm_layer,$count_layer)  celltype=$celltype_key  batch=$batch_key"
    "${args[@]}" >"$log" 2>&1 \
        && echo "[DONE] $out_tag  log: $log" \
        || { echo "[FAIL] $out_tag (exit=$?)  log: $log" >&2; return 1; }
}
export -f run_one
export OUT_ROOT PREP_SCRIPT BENCH_SCRIPT TOP_N MODEL CONDA_PY

# ----------------------------- dispatch -----------------------------

echo "[CONFIG] input_dirs=${#INPUT_DIRS[@]}  max_jobs=$MAX_JOBS  out_root=$OUT_ROOT"

for d in "${INPUT_DIRS[@]}"; do
    if [[ ! -d "$d" ]]; then
        echo "[WARN] input directory not found: $d" >&2
        continue
    fi
    base=$(basename "$d")
    method=$(extract_method "$base" 2>/dev/null) || {
        echo "[SKIP] unrecognized directory name: $d" >&2
        continue
    }
    dataset=$(extract_dataset "$base" 2>/dev/null) || {
        echo "[SKIP] no GSE id in directory name: $d" >&2
        continue
    }
    cfg="${METHOD_LAYER[$method]:-}"
    if [[ -z "$cfg" ]]; then
        echo "[SKIP] no config for method=$method (dir=$d)" >&2
        continue
    fi
    IFS='|' read -r norm_layer count_layer primary fallback tag <<<"$cfg"
    out_tag="${dataset}_${tag}"

    keys_cfg="${DATASET_METHOD_KEYS[${dataset}__${method}]:-}"
    if [[ -z "$keys_cfg" ]]; then
        echo "[SKIP] no obs-key config for dataset=$dataset method=$method (dir=$d)" >&2
        continue
    fi
    IFS='|' read -r default_ct default_batch <<<"$keys_cfg"

    h5ad=$(pick_h5ad "$d" "$primary" "$fallback") || {
        echo "[SKIP] no h5ad matching $primary or $fallback in $d" >&2
        continue
    }

    h5ad=$(validate_layers "$h5ad" "$norm_layer" "$count_layer" "$fallback") || {
        echo "[SKIP] $d: no h5ad with required layers ($norm_layer, $count_layer)" >&2
        continue
    }

    # Resolve the actual obs columns present in the chosen h5ad
    celltype_key=$(resolve_obs_key "$h5ad" "$dataset" "$method" "celltype") || {
        echo "[SKIP] $d: no usable cell-type column (tried $default_ct, fallbacks)" >&2
        continue
    }
    batch_key=$(resolve_obs_key "$h5ad" "$dataset" "$method" "batch") || {
        echo "[SKIP] $d: no usable batch column (tried $default_batch, fallbacks)" >&2
        continue
    }
    if [[ "$celltype_key" != "$default_ct" || "$batch_key" != "$default_batch" ]]; then
        echo "[INFO] $d: resolved obs keys celltype=$celltype_key batch=$batch_key" >&2
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$d" "$out_tag" "$h5ad" "$norm_layer" "$count_layer" "$celltype_key" "$batch_key"
done | xargs -P "$MAX_JOBS" -n 7 bash -c 'run_one "$1" "$2" "$3" "$4" "$5" "$6" "$7"' _

echo "[ALL] finished"