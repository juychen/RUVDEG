"""
Benchmark HK-gene / company-DEG AUC for any scVI-style h5ad.

Input contract:
- adata.obs must contain "status" (case-control, e.g. "CON") and "company"
- adata.layers must contain:
    * NORMALIZED_LAYER (raw expression, used for the HK sanity check)
    * COUNT_LAYER (used to zero-out dropout entries before analysis)

The script will:
1. Re-scale the normalized layer to per-cell sum = 1e4 if needed
2. Zero-out cells where COUNT_LAYER == 0 (dropout masking)
3. log1p() the raw expression -> "log1px"
4. log1p() the normalized layer -> "log1pscvi"
5. Compute HK-gene × company one-vs-rest AUC (raw vs corrected) -> hkg_auc_df
6. Compute top-N company-DEG one-vs-rest AUC -> top10_auc_df

Run:
    python benchmarkbyauc.py \
        --h5ad /path/to/data.h5ad \
        --normalized-layer batchpair_D1_normlized \
        --count-layer batchpair_D1_counts \
        --top-n 10 \
        --out-prefix ./run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import roc_auc_score


# ----------------------------- defaults / constants -----------------------------
HK_PRIORITY = [
    # 结构/骨架 (mouse + human symbols; human panels use uppercase)
    "Actb", "ACTB", "Tuba1a", "TUBA1A", "Tuba1b", "TUBA1B",
    "Ubc", "UBC", "Uba52", "UBA52",
    # 蛋白酶体/折叠
    "Psmd6", "PSMD6", "Psmd7", "PSMD7", "Psma5", "PSMA5",
    "Hsp90aa1", "HSP90AA1", "Hsp90ab1", "HSP90AB1",
    "Ywhaz", "YWHAZ",
    # 线粒体
    "Sdha", "SDHA", "Cyc1", "CYC1",
    "Cox4i1", "COX4I1", "Cox5b", "COX5B",
    "Ndufb8", "NDUFB8",
    "Atp5f1b", "ATP5F1B",
    # 翻译/转录
    "Eef1a1", "EEF1A1",
    "Rplp0", "RPLP0", "Rpl19", "RPL19", "Rps18", "RPS18",
    "Polr2a", "POLR2A",
    "Tbp", "TBP",
    # 代谢
    "Ppia", "PPIA", "Pgk1", "PGK1",
]


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


def per_company_aucs(x_col, y_label, comps):
    """For one gene's expression vector, return {company: |AUC-0.5|+0.5}."""
    out = {}
    for comp in comps:
        y = (y_label == comp).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            out[comp] = np.nan
            continue
        try:
            a = roc_auc_score(y, x_col)
        except ValueError:
            out[comp] = np.nan
            continue
        out[comp] = abs(a - 0.5) + 0.5
    return out


def top_n_genes_per_company_from_df(df, group_col, gene_col, lfc_col, n):
    """Return {company: top-N rows by |logFC|} from a DEG dataframe."""
    out = {}
    for company, sub in df.groupby(group_col):
        ranked = sub.assign(_abs_logfc=sub[lfc_col].abs())
        ranked = ranked.sort_values("_abs_logfc", ascending=False)
        out[company] = ranked.head(n).copy()
    return out


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



def build_company_degs(adata,model = "CON_M"):
    """Run rank_genes_groups for `company` inside the CON subset; return raw
    and scVI-corrected DEG dataframes."""
    ad_con = adata[adata.obs["Model"] == model].copy()
    ad_con.X = ad_con.layers["log1px"]
    sc.tl.rank_genes_groups(ad_con, groupby="company", method="wilcoxon", pct=True)
    raw_dfs = []
    for comp in ad_con.obs["company"].unique():
        df_c = sc.get.rank_genes_groups_df(ad_con, group=comp)
        df_c["company"] = comp
        raw_dfs.append(df_c)
    df_raw = pd.concat(raw_dfs, ignore_index=True)

    ad_con.X = ad_con.layers["log1pscvi"]
    sc.tl.rank_genes_groups(ad_con, groupby="company", method="wilcoxon", pct=True)
    corr_dfs = []
    for comp in ad_con.obs["company"].unique():
        df_c = sc.get.rank_genes_groups_df(ad_con, group=comp)
        df_c["company"] = comp
        corr_dfs.append(df_c)
    df_corr = pd.concat(corr_dfs, ignore_index=True)

    return df_raw, df_corr


def compute_hkg_auc(ad_con):
    """HK-gene × company one-vs-rest AUC (raw vs corrected) -> DataFrame."""
    con_company = ad_con.obs["company"].values
    companies = sorted(set(con_company))

    X_raw = ad_con.layers["log1px"]
    X_corr = ad_con.layers["log1pscvi"]
    if issparse(X_raw):
        X_raw = X_raw.toarray()
    if issparse(X_corr):
        X_corr = X_corr.toarray()
    X_raw = np.asarray(X_raw)
    X_corr = np.asarray(X_corr)

    hkg_in = [g for g in HK_PRIORITY if g in ad_con.var_names]
    # Also look up HK genes via an alternate var column (e.g. 'gene_symbol')
    # when var_names are Ensembl IDs.
    if not hkg_in and "gene_symbol" in getattr(ad_con, "var", pd.DataFrame()).columns:
        sym_to_idx = {s: i for i, s in enumerate(ad_con.var["gene_symbol"].astype(str).values)}
        hkg_in = [g for g in HK_PRIORITY if g in sym_to_idx]
        if hkg_in:
            print(f"[INFO] HK matched via adata.var['gene_symbol'] (Ensembl var_names)")
    print(f"[INFO] HK priority has {len(HK_PRIORITY)} genes, "
          f"{len(hkg_in)} present in adata")
    print(f"[INFO] companies: {companies}")
    print(f"[INFO] company distribution: "
          f"{dict(zip(*np.unique(con_company, return_counts=True)))}")

    rows = []
    var_name_to_col = {str(n): i for i, n in enumerate(ad_con.var_names)}
    sym_to_col = None
    if "gene_symbol" in getattr(ad_con, "var", pd.DataFrame()).columns:
        sym_to_col = {str(s): i for i, s in enumerate(ad_con.var["gene_symbol"].astype(str).values)}
    for g in hkg_in:
        if g in var_name_to_col:
            col = var_name_to_col[g]
        elif sym_to_col is not None and g in sym_to_col:
            col = sym_to_col[g]
        else:
            # gene not findable in this AnnData; skip silently
            continue
        raw_a = per_company_aucs(X_raw[:, col], con_company, companies)
        cor_a = per_company_aucs(X_corr[:, col], con_company, companies)
        for comp in companies:
            rows.append({
                "gene": g,
                "company": comp,
                "raw_AUC": raw_a.get(comp, np.nan),
                "corr_AUC": cor_a.get(comp, np.nan),
                "delta": cor_a.get(comp, np.nan) - raw_a.get(comp, np.nan),
            })
    hkg_auc_df = pd.DataFrame(rows)

    best_per_gene = (
        hkg_auc_df.groupby("gene")[["raw_AUC", "corr_AUC"]]
        .max()
        .rename(columns={"raw_AUC": "best_raw", "corr_AUC": "best_corr"})
        .reset_index()
    )
    best_per_gene["delta_best"] = (
        best_per_gene["best_corr"] - best_per_gene["best_raw"]
    )
    hkg_auc_df = hkg_auc_df.merge(best_per_gene, on="gene", how="left")
    return hkg_auc_df, best_per_gene, companies


def compute_top10_auc(df_raw_filtered, ad_con, top_n):
    """Top-N DEG per company × one-vs-rest AUC -> DataFrame."""
    top10 = top_n_genes_per_company_from_df(
        df_raw_filtered,
        group_col="company",
        gene_col="names",
        lfc_col="logfoldchanges",
        n=top_n,
    )

    con_company = ad_con.obs["company"].values
    X_raw = ad_con.layers["log1px"]
    X_corr = ad_con.layers["log1pscvi"]
    if issparse(X_raw):
        X_raw = X_raw.toarray()
    if issparse(X_corr):
        X_corr = X_corr.toarray()
    X_raw = np.asarray(X_raw)
    X_corr = np.asarray(X_corr)

    var_index = {g: i for i, g in enumerate(ad_con.var_names)}
    sym_index = None
    if "gene_symbol" in getattr(ad_con, "var", pd.DataFrame()).columns:
        sym_index = {str(s): i for i, s in enumerate(ad_con.var["gene_symbol"].astype(str).values)}
    rows = []
    for company, comp_df in top10.items():
        y_company = (con_company == company).astype(int)
        for _, gene_row in comp_df.iterrows():
            gene = str(gene_row["names"])
            if gene in var_index:
                col = var_index[gene]
            elif sym_index is not None and gene in sym_index:
                col = sym_index[gene]
            else:
                continue
            raw_auc = abs(roc_auc_score(y_company, X_raw[:, col]) - 0.5) + 0.5
            corr_auc = abs(roc_auc_score(y_company, X_corr[:, col]) - 0.5) + 0.5
            rows.append({
                "company": company,
                "gene": gene,
                "raw_logFC": float(gene_row["logfoldchanges"]),
                "raw_AUC": raw_auc,
                "corr_AUC": corr_auc,
                "delta": corr_auc - raw_auc,
            })
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["company", "raw_logFC"],
            key=lambda s: s.abs() if s.name == "raw_logFC" else s,
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


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
    p.add_argument("--top-n", type=int, default=10,
                   help="Top-N company DEGs to evaluate (default 10)")
    p.add_argument("--out-prefix", type=Path, default=Path("./benchmark"),
                   help="Prefix for output CSV files")
    p.add_argument("--no-rescale", action="store_true",
                   help="Skip auto-rescaling to per-cell sum = 1e4")
    p.add_argument("--model", type=str, default="CON_M",
                   help="Model label used to subset adata before picking "
                        "the CON subset. Default 'CON_M'. Pass 'CON_F', 'All', "
                        "or any value present in adata.obs['Model']. Pass an "
                        "empty string '' to skip Model-based filtering.")
    return p.parse_args()


def main():
    args = parse_args()
    out_prefix: Path = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"[INFO] adata shape: {adata.shape}")
    print(f"[INFO] status counts:\n{adata.obs['status'].value_counts()}")
    print(f"[INFO] company counts:\n{adata.obs['company'].value_counts()}")
    print(f"[INFO] normalized layer = {args.normalized_layer!r}, "
          f"count layer = {args.count_layer!r}")
    
    # Step 1: prepare layers (rescale, mask dropout, log1p)
    prepare_layers(
        adata,
        normalized_layer=args.normalized_layer,
        count_layer=args.count_layer,
        force_rescale=not args.no_rescale,
    )

    # Step 2: HK-gene one-vs-rest AUC (CON subset of the chosen Model)
    ad_con = adata[adata.obs["Model"] == args.model].copy()
    if len(ad_con) < 50:
        raise SystemExit(
            f"[ERROR] CON subset has only {len(ad_con)} cells - too few "
            "for reliable AUC. Check --model and adata.obs['status'] values."
        )
    hkg_auc_df, best_per_gene, _ = compute_hkg_auc(ad_con)
    hkg_path = out_prefix.with_name(out_prefix.name + "_hkg_auc.csv")
    hkg_auc_df.to_csv(hkg_path, index=False)
    print(f"[INFO] wrote {hkg_path}")

    # Step 3: rank_genes_groups on raw + scVI layers, build company DEGs
    df_raw_all, _ = build_company_degs(adata, model=args.model)
    df_raw_filtered = df_raw_all[df_raw_all["pvals_adj"] < 0.05].copy()

    # Step 4: Top-N company-DEG one-vs-rest AUC
    top10_auc_df = compute_top10_auc(df_raw_filtered, ad_con, args.top_n)
    top_path = out_prefix.with_name(
        out_prefix.name + "_top" + str(args.top_n) + "_auc.csv"
    )
    top10_auc_df.to_csv(top_path, index=False)
    print(f"[INFO] wrote {top_path}")

    # Summary prints
    print("\n=== HK AUC summary (mean across companies, per gene) ===")
    print(best_per_gene.to_string(index=False))

    print(f"\n=== Top-{args.top_n} company-DEG AUC summary ===")
    summary_top = (
        top10_auc_df.groupby("company")
        .agg(
            n=("gene", "count"),
            raw_mean=("raw_AUC", "mean"),
            corr_mean=("corr_AUC", "mean"),
            delta_mean=("delta", "mean"),
        )
        .round(4)
        .reset_index()
    )
    print(summary_top.to_string(index=False))


if __name__ == "__main__":
    main()
