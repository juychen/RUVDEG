"""
Benchmark HK-gene / company-DEG / group-DEG AUC for any scVI-style h5ad.

Input contract:
- adata.obs must contain "status" (case-control, e.g. "CON") and "company",
  plus "Model" (e.g. "CON_M", "CURES_F") to split by sex/condition.
- adata.layers must contain:
    * NORMALIZED_LAYER (raw expression, used for the HK sanity check)
    * COUNT_LAYER (used to zero-out dropout entries before analysis)

Outputs (next to --out-prefix):
    <prefix>_hkg_auc.csv                              -- HK genes × company AUC
    <prefix>_top<N>_auc.csv                           -- top-N company DEG AUC
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
        --top-n 10 \
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


# ----------------------------- defaults / constants -----------------------------
HK_PRIORITY = [
    # 结构/骨架
    "Actb", "Tuba1a", "Ubc", "Uba52",
    # 蛋白酶体/折叠
    "Psmd6", "Psmd7", "Psma5",
    "Hsp90aa1", "Hsp90ab1",
    "Ywhaz",
    # 线粒体
    "Sdha", "Cyc1",
    "Cox4i1", "Cox5b",
    "Ndufb8",
    "Atp5f1b",
    # 翻译/转录
    "Eef1a1",
    "Rplp0", "Rpl19", "Rps18",
    "Polr2a",
    "Tbp",
    # 代谢
    "Ppia", "Pgk1",
]

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
    print(f"[INFO] HK priority has {len(HK_PRIORITY)} genes, "
          f"{len(hkg_in)} present in adata")
    print(f"[INFO] companies: {companies}")
    print(f"[INFO] company distribution: "
          f"{dict(zip(*np.unique(con_company, return_counts=True)))}")

    rows = []
    for g in hkg_in:
        col = list(ad_con.var_names).index(g)
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
    rows = []
    for company, comp_df in top10.items():
        y_company = (con_company == company).astype(int)
        for _, gene_row in comp_df.iterrows():
            gene = str(gene_row["names"])
            if gene not in var_index:
                continue
            col = var_index[gene]
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


def compute_group_degs_auc(adata, group_to_genes, control_label: str = "CON"):
    """One-vs-rest AUC for every (Model, DML1) on {CON_<sex>, group_<sex>} cells.

    For each Model value M whose last 2 chars are ``_M`` / ``_F`` we pick
    ``case_label  = f"{group}{suffix}"`` and
    ``control      = f"CON{suffix}"``; cells with other statuses are excluded.
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

    long_rows = []
    models = adata.obs["Model"].astype(str).unique()
    statuses = adata.obs["status"].astype(str).to_numpy()
    unique_statuses = set(statuses)
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
            keep = adata.obs["status"].astype(str).isin({ctrl, case}).to_numpy()
            if keep.sum() < 30:
                continue
            y = (statuses[keep] == case).astype(int)
            if y.sum() < 5 or y.sum() > (len(y) - 5):
                continue
            n_case = int(y.sum())
            n_ctrl = int(len(y) - n_case)
            used = [g for g in genes if g in var_index]
            for g in used:
                col = var_index[g]
                raw_x = X_raw[keep, col]
                cor_x = X_corr[keep, col]
                if np.unique(raw_x).size < 2 or np.unique(cor_x).size < 2:
                    continue
                raw_auc = abs(roc_auc_score(y, raw_x) - 0.5) + 0.5
                cor_auc = abs(roc_auc_score(y, cor_x) - 0.5) + 0.5
                long_rows.append({
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
    p.add_argument(
        "--group-deg-xlsx",
        type=Path,
        default=Path(DEFAULT_GROUP_DEG_XLSX),
        help="Path to All_degs_N_v0715FF.xlsx (group DEG benchmark).",
    )
    p.add_argument(
        "--skip-group-degs",
        action="store_true",
        help="Skip the group-DEG benchmark entirely.",
    )
    p.add_argument("--gene-col", type=str, default="Gene",
                   help="Column in --group-deg-xlsx that holds the gene symbol.")
    p.add_argument("--group-col", type=str, default="DML1",
                   help="Column in --group-deg-xlsx that holds the group label.")
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

    # Step 5: group DEG benchmark (one-vs-rest AUC per Model × DML1).
    if not args.skip_group_degs:
        if not args.group_deg_xlsx.exists():
            print(f"[WARN] --group-deg-xlsx not found: {args.group_deg_xlsx}")
        else:
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
                adata, group_to_genes, control_label="CON"
            )
            long_path = out_prefix.with_name(
                out_prefix.name + "_group_auc_long.csv"
            )
            long_df.to_csv(long_path, index=False)
            print(f"[INFO] wrote {long_path} ({len(long_df)} rows)")
            sum_path = out_prefix.with_name(
                out_prefix.name + "_group_auc_summary.csv"
            )
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

            # Step 6: per-Model × source aggregation across ALL group DEGs.
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
                    piv["mean_AUC_delta_corr_minus_raw"] = (
                        piv["corr"] - piv["raw"]
                    )
                print(
                    "\n=== Per-Model group-DEG mean AUC "
                    "(higher = more discriminative) ==="
                )
                print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
