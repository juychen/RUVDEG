"""
Benchmark xlsx-defined group-DEG AUC for any scVI-style h5ad.

Input contract:
- adata.obs must contain "status" (case-control, e.g. "CON") and "company",
  plus "Model" (e.g. "CON_M", "CURES_F") to split by sex/condition.
- adata.layers must contain:
    * NORMALIZED_LAYER (raw expression, used for the HK sanity check)
    * COUNT_LAYER (used to zero-out dropout entries before analysis)

Outputs (next to --out-prefix):
    <prefix>_group_auc_long.csv                       -- group DEG long-form
    <prefix>_group_auc_summary.csv                    -- group DEG per (Model, group)
    <prefix>_group_auc_wide_raw_<Model>.csv           -- gene × DML1 raw_AUC
    <prefix>_group_auc_wide_corr_<Model>.csv          -- gene × DML1 corr_AUC
    <prefix>_group_auc_wide_delta_<Model>.csv         -- gene × DML1 delta

Run:
    python benchmarkbyauc_groupgene.py \
        --h5ad /path/to/data.h5ad \
        --normalized-layer batchpair_D1_normlized \
        --count-layer batchpair_D1_counts \
        --out-prefix ./run \
        --group-deg-xlsx /data2st2/junyi/code/sn/data/All_degs_N_v0715FF.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import roc_auc_score

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = delayed = None


# ----------------------------- defaults / constants -----------------------------
# Default location of the curated group DEG workbook. Loaded lazily inside main().
DEFAULT_GROUP_DEG_XLSX = (
    "/data2st2/junyi/code/sn/data/All_degs_N_v0715FF.xlsx"
)

# ----------------------------- helpers -----------------------------
def to_sparse_int(arr, dtype=np.float32):
    """Convert a sparse / dense array to a sparse float32 matrix (int-friendly)."""
    if issparse(arr):
        return csr_matrix(arr).astype(dtype)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if dtype == np.int32:
        return csr_matrix(np.rint(arr).astype(dtype))
    return csr_matrix(arr.astype(dtype))


# ----------------------------- core pipeline -----------------------------
def prepare_layers(adata, normalized_layer, count_layer, force_rescale=True):
    """In-place: scale, zero-dropout, log1p, store corrected layer."""
    layer = adata.layers[normalized_layer]
    if force_rescale:
        row_sums = np.asarray(layer.sum(axis=1)).ravel()
        if np.allclose(row_sums, 1.0):
            print(f"[INFO] {normalized_layer} sums ≈ 1, multiplying by 1e4")
            adata.layers[normalized_layer] = layer * 1e4

    # Zero-out dropout entries (where count layer == 0)
    count = adata.layers[count_layer]
    if issparse(count):
        zero_mask = (count == 0).toarray()
    else:
        zero_mask = (count == 0)
    if zero_mask.any():
        adata.layers[normalized_layer][zero_mask] = 0
        print(f"[INFO] Zeroed {int(zero_mask.sum())} dropout entries")

    # Re-sparsify to float32 to save memory
    val = adata.layers[normalized_layer]
    if not issparse(val) or val.dtype != np.float32:
        val = to_sparse_int(val, dtype=np.float32)
        if issparse(val):
            val.eliminate_zeros()
        adata.layers[normalized_layer] = val

    # raw expression: log1p(adata.X) -> log1px（不动 adata.X）
    raw_x = adata.X.copy()
    log1px = np.log1p(raw_x.toarray()) if issparse(raw_x) else np.log1p(raw_x)
    adata.layers["log1px"] = log1px

    # scVI-corrected: 换 X 到 normalized layer 再 log1p -> log1pscvi
    adata.X = adata.layers[normalized_layer]
    scvi_x = adata.X
    log1pscvi = np.log1p(scvi_x.toarray()) if issparse(scvi_x) else np.log1p(scvi_x)
    adata.layers["log1pscvi"] = log1pscvi

    # 还原 X 为 log1px（后续 rank_genes_groups 会基于 adata.X）
    adata.X = adata.layers["log1px"]



def _normalize_gene_name(name) -> str:
    """Mouse convention: 'ACTB' / 'actb' -> 'Actb'."""
    if not isinstance(name, str):
        name = str(name)
    s = name.strip()
    if not s:
        return s
    return s[0].upper() + s[1:].lower()


def load_group_degs(
    xlsx_path,
    gene_col: str = "Gene",
    group_col: str = "DML1",
    drop_groups=("other",),
):
    """Read the curated group-DEG workbook and return {group: [gene, ...]}.

    Drops any rows whose group label matches ``drop_groups``. Gene symbols
    are normalized to mouse Title-case so they can be matched against
    ``adata.var_names``.
    """
    df = pd.read_excel(xlsx_path)
    if gene_col not in df.columns or group_col not in df.columns:
        raise KeyError(
            f"xlsx missing required columns: {gene_col!r} or {group_col!r}. "
            f"Available: {df.columns.tolist()}"
        )
    df = df[~df[group_col].astype(str).isin(drop_groups)].copy()
    out = {}
    for g, sub in df.groupby(group_col, sort=False):
        genes = [
            _normalize_gene_name(x)
            for x in sub[gene_col].dropna().astype(str).unique().tolist()
        ]
        out[str(g)] = genes
    return out


def _group_auc_rows(
    model_value: str,
    group: str,
    used: list,
    keep: np.ndarray,
    X_raw: np.ndarray,
    X_corr: np.ndarray,
    y: np.ndarray,
    n_case: int,
    n_ctrl: int,
    ctrl: str,
    case: str,
    var_index: dict,
) -> list:
    """Worker: per-gene AUC rows for one (Model, DML1) task.

    Body of the innermost loop in :func:`compute_group_degs_auc`, extracted
    so it can be dispatched via :mod:`joblib` when ``n_jobs > 1``. Returns
    a list of dicts with the same schema as the in-process loop.
    """
    rows = []
    for g in used:
        col = var_index[g]
        raw_x = X_raw[keep, col]
        cor_x = X_corr[keep, col]
        if np.unique(raw_x).size < 2 or np.unique(cor_x).size < 2:
            continue
        raw_auc = abs(roc_auc_score(y, raw_x) - 0.5) + 0.5
        cor_auc = abs(roc_auc_score(y, cor_x) - 0.5) + 0.5
        rows.append({
            "Model": model_value,
            "group": group,
            "case_label": case,
            "control_label": ctrl,
            "n_case": n_case,
            "n_control": n_ctrl,
            "gene": g,
            "raw_AUC": raw_auc,
            "corr_AUC": cor_auc,
            "delta": cor_auc - raw_auc,
        })
    return rows


def compute_group_degs_auc(
    adata,
    group_to_genes,
    control_label: str = "CON",
    n_jobs: int = 1,
):
    """One-vs-rest AUC for every (Model, DML1) on {CON_<sex>, group_<sex>} cells.

    For each Model value M whose last 2 chars are ``_M`` / ``_F`` we pick
    ``case_label  = f"{group}{suffix}"`` and
    ``control      = f"CON{suffix}"``; cells with other statuses are excluded.

    Parameters
    ----------
    n_jobs : int, default 1
        Parallel workers for the per-(Model, DML1) AUC loop. ``1`` runs the
        original sequential loop (no joblib import cost, identical output).
        ``>1`` dispatches each surviving (Model, DML1) task through
        :func:`joblib.Parallel` with the ``loky`` backend. Requires the
        ``joblib`` package. Note that each worker receives copies of
        ``adata.layers['log1px']`` and ``['log1pscvi']``; on large datasets
        prefer ``n_jobs`` that fits in available RAM.
    """
    X_raw = adata.layers["log1px"]
    X_corr = adata.layers["log1pscvi"]
    if issparse(X_raw):
        X_raw = X_raw.toarray()
    if issparse(X_corr):
        X_corr = X_corr.toarray()
    X_raw = np.asarray(X_raw)
    X_corr = np.asarray(X_corr)
    var_index = {g: i for i, g in enumerate(adata.var_names)}

    # Build the task list: one entry per (Model, DML1) that survives the
    # early-exit filters. The actual per-gene work runs in the worker.
    models = adata.obs["Model"].astype(str).unique()
    statuses = adata.obs["status"].astype(str).to_numpy()
    unique_statuses = set(statuses)
    obs_status = adata.obs["status"].astype(str)
    tasks = []
    for model_value in sorted(models):
        suffix = model_value[-2:] if model_value[-2:] in {"_M", "_F"} else ""
        if not suffix:
            continue
        ctrl = f"{control_label}{suffix}"
        if ctrl not in unique_statuses:
            continue
        for group, genes in group_to_genes.items():
            case = f"{group}{suffix}"
            if case not in unique_statuses:
                continue
            keep = obs_status.isin({ctrl, case}).to_numpy()
            if keep.sum() < 30:
                continue
            y = (statuses[keep] == case).astype(int)
            if y.sum() < 5 or y.sum() > (len(y) - 5):
                continue
            n_case = int(y.sum())
            n_ctrl = int(len(y) - n_case)
            used = [g for g in genes if g in var_index]
            if not used:
                continue
            tasks.append(
                (model_value, group, used, keep, X_raw, X_corr, y,
                 n_case, n_ctrl, ctrl, case, var_index)
            )

    # Dispatch. n_jobs == 1 (default) keeps the original in-process loop
    # bit-for-bit, avoiding any joblib import or pickling cost.
    if n_jobs <= 1 or len(tasks) <= 1:
        long_rows: list = []
        for t in tasks:
            long_rows.extend(_group_auc_rows(*t))
    else:
        if Parallel is None or delayed is None:
            raise RuntimeError(
                "joblib is required when --n-jobs > 1; install with "
                "`pip install joblib`."
            )
        chunks = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_group_auc_rows)(*t) for t in tasks
        )
        long_rows = [row for chunk in chunks for row in chunk]

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame()
    summary_df = (
        long_df.groupby(["Model", "group"], sort=False)
        .agg(
            n_genes=("gene", "count"),
            n_case=("n_case", "first"),
            n_control=("n_control", "first"),
            raw_mean=("raw_AUC", "mean"),
            corr_mean=("corr_AUC", "mean"),
            delta_mean=("delta", "mean"),
            frac_raw_gt_0_6=("raw_AUC", lambda s: (s > 0.6).mean()),
            frac_corr_gt_0_6=("corr_AUC", lambda s: (s > 0.6).mean()),
        )
        .round(4)
        .reset_index()
    )
    return long_df, summary_df


def pivot_wide(long_df, value_col: str) -> pd.DataFrame:
    """long_df -> (index=gene, columns=DML1) pivot on ``value_col``."""
    if long_df.empty:
        return pd.DataFrame()
    wide = long_df.pivot_table(
        index="gene", columns="group", values=value_col, aggfunc="first"
    )
    return wide.reindex(columns=sorted(wide.columns))


# ----------------------------- entry point -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True, type=Path,
                   help="Path to input h5ad file")
    p.add_argument("--normalized-layer", required=True,
                   help="Name of layer with raw expression "
                        "(e.g. batchpair_D1_normlized)")
    p.add_argument("--count-layer", required=True,
                   help="Name of layer with raw counts (used to mask dropout)")
    p.add_argument("--out-prefix", type=Path, default=Path("./benchmark"),
                   help="Prefix for output CSV files")
    p.add_argument("--no-rescale", action="store_true",
                   help="Skip auto-rescaling to per-cell sum = 1e4")
    p.add_argument(
        "--group-deg-xlsx",
        type=Path,
        default=Path(DEFAULT_GROUP_DEG_XLSX),
        help="Path to All_degs_N_v0715FF.xlsx (group DEG benchmark).",
    )
    p.add_argument("--gene-col", type=str, default="Gene",
                   help="Column in --group-deg-xlsx that holds the gene symbol.")
    p.add_argument("--group-col", type=str, default="DML1",
                   help="Column in --group-deg-xlsx that holds the group label.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel workers for the group-DEG AUC loop "
                        "(default 1). Uses joblib (loky backend).")
    return p.parse_args()


def main():
    args = parse_args()
    out_prefix: Path = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"[INFO] adata shape: {adata.shape}")
    print(f"[INFO] status counts:\n{adata.obs['status'].value_counts()}")
    print(f"[INFO] normalized layer = {args.normalized_layer!r}, "
          f"count layer = {args.count_layer!r}")
    
    # Step 1: prepare layers (rescale, mask dropout, log1p)
    prepare_layers(
        adata,
        normalized_layer=args.normalized_layer,
        count_layer=args.count_layer,
        force_rescale=not args.no_rescale,
    )

    if not args.group_deg_xlsx.exists():
        raise SystemExit(f"[ERROR] --group-deg-xlsx not found: {args.group_deg_xlsx}")

    group_to_genes = load_group_degs(
        args.group_deg_xlsx,
        gene_col=args.gene_col,
        group_col=args.group_col,
        drop_groups=("other",),
    )
    print(
        f"[INFO] loaded {sum(len(v) for v in group_to_genes.values())} "
        f"gene-group rows across {len(group_to_genes)} groups"
    )
    long_df, summary_df = compute_group_degs_auc(
        adata, group_to_genes, control_label="CON", n_jobs=args.n_jobs,
    )
    long_path = out_prefix.with_name(out_prefix.name + "_group_auc_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"[INFO] wrote {long_path} ({len(long_df)} rows)")
    sum_path = out_prefix.with_name(out_prefix.name + "_group_auc_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    print(f"[INFO] wrote {sum_path}")

    for model_value in sorted(long_df["Model"].unique()):
        sub = long_df[long_df["Model"] == model_value]
        for col, metric in [
            ("raw_AUC", "raw"),
            ("corr_AUC", "corr"),
            ("delta", "delta"),
        ]:
            wide = pivot_wide(sub[["gene", "group", col]], col)
            if wide.empty:
                continue
            wide_path = out_prefix.with_name(
                out_prefix.name
                + f"_group_auc_wide_{metric}_{model_value}.csv"
            )
            wide.to_csv(wide_path, index_label="gene")
            print(
                f"[INFO] wrote {wide_path}  shape={wide.shape} "
                f"(genes x diseases)"
            )

    # Per-Model × source aggregation across ALL group DEGs.
            # "source" ∈ {raw, corr} — i.e. the computational model under
            # comparison (raw gene expression vs scVI-corrected). Aggregating
            # across every (gene × group) pair lets the user ask:
            #   "which source retains the most group-DEG differences?"
    per_model_rows = []
    for model_value in sorted(long_df["Model"].unique()):
        sub = long_df[long_df["Model"] == model_value]
        if sub.empty:
            continue
        n_pairs = int(len(sub))
        n_genes = int(sub["gene"].nunique())
        n_groups = int(sub["group"].nunique())
        for source, col in [("raw", "raw_AUC"), ("corr", "corr_AUC")]:
            values = sub[col].to_numpy()
            per_model_rows.append({
                "Model": model_value,
                "source": source,
                "n_pairs": n_pairs,
                "n_genes": n_genes,
                "n_groups": n_groups,
                "mean_AUC": float(np.mean(values)),
                "median_AUC": float(np.median(values)),
                "std_AUC": float(np.std(values)),
                "frac_gt_0_6": float((values > 0.6).mean()),
            })
    per_model_df = pd.DataFrame(per_model_rows)
    per_model_path = out_prefix.with_name(
        out_prefix.name + "_group_auc_per_model.csv"
    )
    per_model_df.to_csv(per_model_path, index=False)
    print(f"[INFO] wrote {per_model_path}")

            # Pivot the per-Model summary so raw / corr sit side-by-side
            # and the corr − raw delta is easy to read.
    if not per_model_df.empty:
        piv = per_model_df.pivot_table(
            index="Model",
            columns="source",
            values="mean_AUC",
        )
        if {"raw", "corr"}.issubset(piv.columns):
            piv["mean_AUC_delta_corr_minus_raw"] = piv["corr"] - piv["raw"]
        print(
            "\n=== Per-Model group-DEG mean AUC "
            "(higher = more discriminative) ==="
        )
        print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
