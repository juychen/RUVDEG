"""
Benchmark celltype-by-batch distances for any scVI-style h5ad.

Input contract:
- adata.obs must contain "status" (case-control, e.g. "CON") and "company"
- adata.layers must contain:
    * NORMALIZED_LAYER (raw expression, used for the HK sanity check)
    * COUNT_LAYER (used to zero-out dropout entries before analysis)

The script will:
1. Re-scale the normalized layer to per-cell sum = 1e4 if needed
2. Zero-out cells where COUNT_LAYER == 0 (dropout masking)
3. Create log1p raw and corrected expression layers
4. Compute per-gene distances between batches within each cell type
5. Average each metric across genes for every celltype × batch pair

Run:
    python benchmarkbyauc.py \
        --h5ad /path/to/data.h5ad \
        --normalized-layer batchpair_D1_normlized \
        --count-layer batchpair_D1_counts \
        --celltype-key celltype.L2 \
        --batch-key company \
        --out-prefix ./run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from itertools import combinations



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

    count = adata.layers[count_layer]
    if issparse(count):
        zero_mask = (count == 0).toarray()
    else:
        zero_mask = count == 0
    if zero_mask.any():
        adata.layers[normalized_layer][zero_mask] = 0
        print(f"[INFO] Zeroed {int(zero_mask.sum())} dropout entries")

    val = adata.layers[normalized_layer]
    if not issparse(val) or val.dtype != np.float32:
        val = to_sparse_int(val, dtype=np.float32)
        val.eliminate_zeros()
        adata.layers[normalized_layer] = val

    raw_x = adata.X.copy()
    raw_x = raw_x.toarray() if issparse(raw_x) else raw_x
    adata.layers["log1px"] = np.log1p(raw_x)

    corrected_x = adata.layers[normalized_layer]
    corrected_x = corrected_x.toarray() if issparse(corrected_x) else corrected_x
    adata.layers["log1pscvi"] = np.log1p(corrected_x)


def _safe_cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled_sd = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                        / (len(a) + len(b) - 2))
    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float((a.mean() - b.mean()) / pooled_sd)


def _safe_var_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    vb = b.var(ddof=1)
    if vb == 0 or not np.isfinite(vb):
        return np.nan
    return float(a.var(ddof=1) / vb)


def _safe_bray_curtis(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    denom = np.abs(a).sum() + np.abs(b).sum()
    if denom == 0:
        return 0.0
    return float(np.abs(a - b).sum() / denom)


def _safe_hellinger(a, b, bins=50):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges); pb, _ = np.histogram(b, bins=edges)
    pa = pa.astype(float); pb = pb.astype(float)
    pa = pa / pa.sum() if pa.sum() > 0 else pa
    pb = pb / pb.sum() if pb.sum() > 0 else pb
    return float(np.linalg.norm(np.sqrt(pa) - np.sqrt(pb)) / np.sqrt(2))


def _safe_mean_abs_cross_diff(a, b):
    """Mean absolute difference over the Cartesian product of two cell sets.

    ``mean_diff`` is now ``mean_{i in a, j in b} |a_i - b_j|`` (exact cell
    pair differences) instead of ``|mean(a) - mean(b)|``.  Naive O(nA*nB)
    pairwise evaluation is replaced by a sort + prefix-sum identity:

        sum_j |a_i - b_j| =
            (# b <= a_i) * a_i - sum(b <= a_i)
            + (sum(b > a_i) - #(b > a_i) * a_i)

    which gives the exact same result in O((nA+nB) log(nA+nB)) time.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    a = np.sort(a)
    b = np.sort(b)
    # Prefix sums of b allow O(1) lookups of sum(b <= threshold).
    b_cum = np.concatenate([[0.0], np.cumsum(b)])
    total_abs = 0.0
    for a_i in a:
        k = int(np.searchsorted(b, a_i, side="right"))
        le_sum = b_cum[k]            # sum of b_j <= a_i
        gt_sum = b_cum[-1] - le_sum  # sum of b_j >  a_i
        le_cnt = k
        gt_cnt = b.size - k
        total_abs += (le_cnt * a_i - le_sum) + (gt_sum - gt_cnt * a_i)
    return float(total_abs / (a.size * b.size))


def _one_pair_metrics(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return {
        "mean_diff":   _safe_mean_abs_cross_diff(a, b),
        "cohens_d":    _safe_cohens_d(a, b),
        "var_ratio":   _safe_var_ratio(a, b),
        "hellinger":   _safe_hellinger(a, b),
    }


def _to_dense(arr):
    return np.asarray(arr.toarray() if issparse(arr) else arr)


def compute_celltype_batch_distance(
    adata,
    celltype_key: str = "celltype.L2",
    batch_key: str   = "company",
    layer_raw: str   = "log1px",
    layer_corr: str  = "log1pscvi",
    gene_filter: list[str] | None = None,
    min_cells_per_batch: int = 5,
):
    """Per (celltype, gene, batch_A, batch_B) distance on log1p layers.

    Output: long-form DataFrame whose rows are
        (celltype, gene, batch_A, batch_B, n_A, n_B)
        + 5 metrics × 2 layers (raw / corr).
    """
    # X_raw = _to_dense(adata.layers[layer_raw])
    # X_corr = _to_dense(adata.layers[layer_corr])
    var_index = {g: i for i, g in enumerate(adata.var_names)}
    genes = (
        [g for g in gene_filter if g in var_index]
        if gene_filter is not None else list(adata.var_names)
    )
    print(f"[INFO] distance: {len(genes)} genes × "
          f"{adata.obs[celltype_key].nunique()} celltypes")

    rows = []
    obs = adata.obs
    for ct, sub in obs.groupby(celltype_key, observed=True):
        if pd.isna(ct):
            continue
        idx = sub.index.to_numpy()
        ct_batch = sub[batch_key].astype("object").to_numpy()
        batches = (
            pd.Series(ct_batch).dropna().value_counts()
            .loc[lambda s: s >= min_cells_per_batch].index.tolist()
        )
        if len(batches) < 2:
            print(f"[WARN] celltype={ct!r}: <2 usable batches, skip")
            continue
        sub_X_raw  = adata[idx].layers[layer_raw]
        sub_X_corr = adata[idx].layers[layer_corr]
        for gene in genes:
            col = var_index[gene]
            raw_col  = sub_X_raw[:, col]
            corr_col = sub_X_corr[:, col]
            for bA, bB in combinations(batches, 2):
                mA = (ct_batch == bA); mB = (ct_batch == bB)
                rec = {
                    "celltype": ct, "gene": gene,
                    "batch_A": bA, "batch_B": bB,
                    "n_A": int(mA.sum()), "n_B": int(mB.sum()),
                }
                for k, v in _one_pair_metrics(raw_col[mA],  raw_col[mB]).items():
                    rec[f"raw_{k}"]  = v
                for k, v in _one_pair_metrics(corr_col[mA], corr_col[mB]).items():
                    rec[f"corr_{k}"] = v
                rows.append(rec)
    return pd.DataFrame(rows)


def summarize_distance_by_celltype_batch(long_df: pd.DataFrame):
    """Collapse to (celltype, batch_pair) mean across genes for each metric × layer."""
    metric_cols = [
        c for c in long_df.columns
        if c.startswith(("raw_", "corr_"))
        and c.endswith(("mean_diff", "cohens_d", "var_ratio",
                        "bray_curtis", "hellinger"))
    ]
    long = long_df.melt(
        id_vars=["celltype", "batch_A", "batch_B"],
        value_vars=metric_cols,
        var_name="layer_metric",
        value_name="distance",
    )
    long[["layer", "metric"]] = long["layer_metric"].str.split(
        "_", n=1, expand=True)
    summary = (
        long.groupby(["celltype", "batch_A", "batch_B", "layer", "metric"],
                     observed=True, as_index=False)
        .agg(n_genes=("distance", "size"),
             mean_d=("distance", "mean"),
             median_d=("distance", "median"))
    )
    # raw vs corr delta per (celltype, batch_pair, metric)
    pivot = summary.pivot_table(
        index=["celltype", "batch_A", "batch_B", "metric"],
        columns="layer", values="mean_d").reset_index()
    if "raw" in pivot.columns and "corr" in pivot.columns:
        pivot["delta_corr_minus_raw"] = pivot["corr"] - pivot["raw"]
    return summary, pivot


# ----------------------------- UMAP plotting -----------------------------
def _pick_umap_rep(adata, preferred: str | None = None) -> str:
    """Pick a representation key for UMAP, mirroring umap.ipynb convention.

    Priority: explicit ``preferred`` (if present in adata.obsm), else
    X_harmony > X_scVI > X_scVI_* > X_pca > first available key.
    """
    if preferred and preferred in adata.obsm:
        return preferred
    for key in ["X_harmony", "X_scVI", "X_pca"]:
        if key in adata.obsm:
            return key
    scvi_like = [k for k in adata.obsm if k.startswith("X_scVI")]
    if scvi_like:
        return scvi_like[0]
    raise KeyError(
        f"No suitable obsm key for UMAP. Tried {[preferred] if preferred else []}"
        f"+ ['X_harmony', 'X_scVI', 'X_pca']. "
        f"Available: {list(adata.obsm.keys())}")


def draw_umap(
    adata,
    out_path: Path,
    celltype_key: str = "celltype.L2",
    batch_key: str = "company",
    rep: str | None = None,
    extra_color_keys: list[str] | None = None,
):
    """Compute neighbours + UMAP on a latent rep and save a PNG.

    Mirrors the loop in ``umap.ipynb``: neighbors → umap → pl.umap with
    ``show=False`` and ``save=...``. Colours are celltype, batch, and any
    extra obs columns that exist (e.g. ``status``).
    """
    rep = _pick_umap_rep(adata, preferred=rep)
    print(f"[INFO] UMAP: using rep={rep!r}, shape={adata.obsm[rep].shape}")

    sc.pp.neighbors(adata, use_rep=rep)
    sc.tl.umap(adata)

    color_keys: list[str] = []
    if celltype_key in adata.obs.columns:
        color_keys.append(celltype_key)
    if batch_key in adata.obs.columns:
        color_keys.append(batch_key)
    if "status" in adata.obs.columns:
        color_keys.append("status")
    for k in extra_color_keys or []:
        if k in adata.obs.columns and k not in color_keys:
            color_keys.append(k)

    if not color_keys:
        raise KeyError(
            "UMAP: no usable obs columns found for colouring. "
            f"Requested celltype_key={celltype_key!r}, batch_key={batch_key!r}. "
            f"Available obs: {adata.obs.columns.tolist()}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_name = out_path.stem  # scanpy appends `_umap.png` automatically
    print(f"[INFO] UMAP: plotting {color_keys} → {out_path}")
    sc.pl.umap(adata, color=color_keys, save=save_name, show=False)
    # scanpy saves to ./figures/<save_name>_umap.png by default; relocate it
    # so the caller gets exactly the path they asked for.
    default_saved = Path("figures") / f"{save_name}_umap.png"
    if default_saved.exists() and default_saved.resolve() != out_path.resolve():
        out_path.write_bytes(default_saved.read_bytes())
        default_saved.unlink()
    print(f"[INFO] UMAP saved to {out_path}")

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
    p.add_argument("--celltype-key", type=str, default="celltype.L2",
                   help="obs column for cell type used by "
                        "the celltype×batch distance benchmark")
    p.add_argument("--batch-key", type=str, default="company",
                   help="obs column for batch used by "
                        "the celltype×batch distance benchmark")
    p.add_argument("--gene-list", type=str, default=None,
                    help="Path to a txt file with one gene per line")
    p.add_argument("--model", type=str, default=None,
                   help="Subset adata to cells with this value in "
                        "adata.obs['Model'] before the distance benchmark "
                        "(e.g. 'CON_M'). Empty = no filter.")
    p.add_argument("--umap-out", type=Path, default=None,
                   help="PNG path for the latent UMAP. Defaults to "
                        "<out-prefix>_umap.png. "
                        "UMAP is always drawn at the end (mirrors umap.ipynb); "
                        "rep is auto-picked from X_harmony > X_scVI > X_pca "
                        "unless --umap-rep is given.")
    p.add_argument("--umap-rep", type=str, default=None,
                   help="Explicit obsm key for UMAP (e.g. 'X_scVI', 'X_harmony'). "
                        "Overrides auto-detection.")
    p.add_argument("--umap-color", action="append", default=None,
                   help="Extra obs column(s) to colour the UMAP by "
                        "(e.g. --umap-color status). May be passed multiple times.")
    return p.parse_args()


def main():
    args = parse_args()
    out_prefix: Path = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"[INFO] adata shape: {adata.shape}")
    print(f"[INFO] normalized layer = {args.normalized_layer!r}, "
          f"count layer = {args.count_layer!r}")
    
    # Optional: subset adata to a specific Model (e.g. CON_M) before
    # preparing layers and computing distances.
    if args.model:
        if "Model" not in adata.obs.columns:
            raise KeyError(
                f"--model={args.model!r} requested but adata.obs has no "
                "'Model' column. Available columns: "
                f"{adata.obs.columns.tolist()}")
        before = adata.n_obs
        adata = adata[adata.obs["Model"] == args.model].copy()
        print(f"[INFO] --model {args.model!r}: "
              f"{before} -> {adata.n_obs} cells")

    # Step 1: prepare layers (rescale, mask dropout, log1p)
    prepare_layers(
        adata,
        normalized_layer=args.normalized_layer,
        count_layer=args.count_layer,
        force_rescale=not args.no_rescale,
    )
    gene_filter = None
    if args.gene_list:
        gene_filter = [
            g.strip() for g in Path(args.gene_list).read_text().splitlines()
            if g.strip() and not g.startswith("#")
        ]
        print(f"[INFO] distance: restricting to {len(gene_filter)} genes")

    dist_long = compute_celltype_batch_distance(
        adata,
        celltype_key=args.celltype_key,
        batch_key=args.batch_key,
        gene_filter=gene_filter,
    )
    long_path = out_prefix.with_name(
        out_prefix.name + "_ct_batch_distance_long.csv")
    dist_long.to_csv(long_path, index=False)
    print(f"[INFO] wrote {long_path}  ({dist_long.shape})")

    ct_summary, ct_pivot = summarize_distance_by_celltype_batch(dist_long)
    sum_path = out_prefix.with_name(
        out_prefix.name + "_ct_batch_distance_summary.csv")
    ct_summary.to_csv(sum_path, index=False)
    print(f"[INFO] wrote {sum_path}")

    pivot_path = out_prefix.with_name(
        out_prefix.name + "_ct_batch_distance_by_layer.csv")
    ct_pivot.to_csv(pivot_path, index=False)
    print(f"[INFO] wrote {pivot_path}")

    print("\n=== celltype × batch_pair × metric: raw vs corr (mean_d) ===")
    print(ct_pivot.round(4).to_string(index=False))

    # Step 2 (always): latent UMAP at benchmark end, mirroring umap.ipynb.
    #   neighbors → umap → pl.umap coloured by celltype + batch (+ status).
    umap_out = (
        args.umap_out
        if args.umap_out is not None
        else out_prefix.with_name(out_prefix.name + "_umap.png")
    )
    try:
        _pick_umap_rep(adata, preferred=args.umap_rep)
    except KeyError as exc:
        print(f"[WARN] no latent representation for UMAP, skip drawing: {exc}")
    else:
        draw_umap(
            adata,
            out_path=umap_out,
            celltype_key=args.celltype_key,
            batch_key=args.batch_key,
            rep=args.umap_rep,
            extra_color_keys=args.umap_color,
        )


if __name__ == "__main__":
    main()
