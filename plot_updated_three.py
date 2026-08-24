"""Regenerate 3 plots from updated batchscvi_full + batchscvi_con data:

1. delta_best bar plot (per region × model, lower = better)
2. mean ROC bar plot (per region × model, lower = better)
3. ROC curve: empirical CDF of per-(gene, company) AUC for each model
   (left-hugging curve = better residual batch correction)

Data sources:
  - /data3/junyi/benchmark_results/*_batchscvi_full/*_batchscvi_full_hkg_auc.csv
  - /data3/junyi/benchmark_results/*_batchscvi_con/condtion_batch_hkg_auc.csv
  - /data3/junyi/benchmark_results/*_scVImodel/*_scVImodel_hkg_auc.csv
  - /data3/junyi/benchmark_results/*_scviHarmony/*_scviHarmony_hkg_auc.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATTERNS = [
    "/data3/junyi/benchmark_results/*_batchscvi_full/*_batchscvi_full_hkg_auc.csv",
    "/data3/junyi/benchmark_results/*_batchscvi_con/condtion_batch_hkg_auc.csv",
    "/data3/junyi/benchmark_results/*_scVImodel/*_scVImodel_hkg_auc.csv",
    "/data3/junyi/benchmark_results/*_scviHarmony/*_scviHarmony_hkg_auc.csv",
]
OUT_DIR  = Path("/home/junyichen/code/RUVAEDEG")

PALETTE = {
    "raw":             "#52514e",  # charcoal for raw
    "scvi":            "#2a78d6",  # blue
    "scviHarmony":     "#1baf7a",  # aqua
    "batchscvi_full":  "#7a5e9e",  # violet
    "batchscvi_con":   "#d62728",  # red
}

MODEL_ORDER = ["scVImodel", "scviHarmony", "batchscvi_full", "batchscvi_con"]
LABEL = {
    "raw":              "raw",
    "scVImodel":        "scVI",
    "scviHarmony":      "scVI+Harmony",
    "batchscvi_full":   "paired-batch scVI (D1)",
    "batchscvi_con":    "paired-batch scVI + cond",
}


BAR_BG = "#fcfcfb"
INK    = "#0b0b0b"
SUB    = "#52514e"
RULE   = "#c3c2b7"
GRID   = "#e1e0d9"


def collect():
    """Return a DataFrame with per-region × model metrics."""
    rows = []
    for pattern in CSV_PATTERNS:
        for path in sorted(__import__("glob").glob(pattern)):
            base = Path(path).stem.replace("_hkg_auc", "")
            parent = Path(path).parent.name
            if parent.endswith("_batchscvi_full"):
                model = "batchscvi_full"
                region = parent[:-len("_batchscvi_full")]
            elif parent.endswith("_batchscvi_con"):
                model = "batchscvi_con"
                region = parent[:-len("_batchscvi_con")]
            elif parent.endswith("_scviHarmony"):
                model = "scviHarmony"
                region = parent[:-len("_scviHarmony")]
            elif parent.endswith("_scVImodel"):
                model = "scVImodel"
                region = parent[:-len("_scVImodel")]
            else:
                continue
            df = pd.read_csv(path)
            per_gene = df.drop_duplicates("gene")[["gene", "best_raw", "best_corr", "delta_best"]]
            rows.append({
                "region": region,
                "model": model,
                "mean_raw": per_gene["best_raw"].mean(),
                "mean_corr": per_gene["best_corr"].mean(),
                "mean_delta_best": per_gene["delta_best"].mean(),
                "best_raw": per_gene["best_raw"].values,
                "best_corr": per_gene["best_corr"].values,
                "all_raw_auc": df["raw_AUC"].values,
                "all_corr_auc": df["corr_AUC"].values,
            })
    return pd.DataFrame(rows)


# ---------- plot 1: delta_best bars ----------
def plot_delta(df, out_png):
    regions = sorted(df["region"].unique())
    n_reg = len(regions)
    bar_w = 0.26
    x = np.arange(n_reg)

    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=160)
    fig.patch.set_facecolor(BAR_BG); ax.set_facecolor(BAR_BG)

    ax.axhline(0, color="#383835", lw=1.0, zorder=1)
    ax.axhspan(df["mean_delta_best"].min() - 0.02, 0,
               color="#1baf7a", alpha=0.05, zorder=0)

    for i, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("region").reindex(regions)
        vals = sub["mean_delta_best"].values
        offsets = x + (i - (len(MODEL_ORDER) - 1) / 2) * bar_w
        # map raw internal model name to PALETTE key
        palette_key = {"scVImodel": "scvi",
                       "scviHarmony": "scviHarmony",
                       "batchscvi_full": "batchscvi_full",
                       "batchscvi_con":  "batchscvi_con"}[model]
        bars = ax.bar(offsets, vals, width=bar_w,
                      color=PALETTE[palette_key],
                      edgecolor=INK, linewidth=0.5,
                      label=LABEL[model], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2,
                    v + (0.003 if v >= 0 else -0.005),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5, color=INK)

    ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=10, color=INK)
    ax.set_xlabel("Brain region", fontsize=11, color=INK, labelpad=8)
    ax.set_ylabel(
        "HKG mean Δ_best (best_corr − best_raw)\n"
        "← lower = better; HKG more batch-invariant after correction",
        fontsize=10, color=INK, labelpad=8,
    )
    ax.tick_params(axis="y", colors=SUB, labelsize=9)
    ax.tick_params(axis="x", colors=SUB)
    ymin, ymax = df["mean_delta_best"].min(), df["mean_delta_best"].max()
    ax.set_ylim(ymin - 0.018, ymax + 0.018)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5,
                    handlelength=1.6, handletextpad=0.5, labelcolor=INK)
    for h in leg.legend_handles:
        h.set_edgecolor(INK); h.set_linewidth(0.5)

    ax.set_title(
        "HK-gene AUC drop after batch correction, per region × model\n"
        "(23 HK genes, CON_M subset; bar below 0 ⇒ correction made HKG more batch-invariant)",
        fontsize=11.5, color=INK, pad=14, loc="left",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------- plot 2: mean ROC bars ----------
def plot_roc_bars(df, out_png):
    regions = sorted(df["region"].unique())
    n_reg = len(regions); n_mod = len(MODEL_ORDER) + 1  # raw + each model
    bar_w = 0.18
    x = np.arange(n_reg)

    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=160)
    fig.patch.set_facecolor(BAR_BG); ax.set_facecolor(BAR_BG)
    ax.axhline(0.5, color=SUB, lw=1.0, ls=(0, (4, 4)), zorder=2)
    ax.text(n_reg - 0.45, 0.503, "ideal = 0.5",
            ha="right", va="bottom", fontsize=8.5, color=SUB)

    # raw (use pooled mean across models)
    raw_per_region = df.groupby("region")["mean_raw"].mean().reindex(regions)
    others = [
        ("scvi",            "scVImodel"),
        ("scviHarmony",     "scviHarmony"),
        ("batchscvi_full",  "batchscvi_full"),
        ("batchscvi_con",   "batchscvi_con"),
    ]
    series = [("raw", raw_per_region.values)] + [
        (lab, df[df["model"] == m].set_index("region")
            .reindex(regions)["mean_corr"].values)
        for lab, m in others
    ]

    for i, (lab, vals) in enumerate(series):
        offsets = x + (i - (n_mod - 1) / 2) * bar_w
        label_name = LABEL[lab] if lab == "raw" else LABEL[{"scvi": "scVImodel", "scviHarmony": "scviHarmony", "batchscvi_full": "batchscvi_full", "batchscvi_con": "batchscvi_con"}[lab]]
        bars = ax.bar(offsets, vals, width=bar_w * 0.92,
                      color=PALETTE[lab], edgecolor=INK, linewidth=0.5,
                      label=label_name,
                      zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)

    ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=10.5, color=INK)
    ax.set_xlabel("Brain region", fontsize=11, color=INK, labelpad=8)
    ax.set_ylabel("HKG mean ROC (best across companies)\nlower = better",
                  fontsize=10, color=INK, labelpad=8)
    ax.tick_params(axis="y", colors=SUB, labelsize=9)
    ax.tick_params(axis="x", colors=SUB)
    ax.set_ylim(0.46, max(0.78, np.concatenate([v for _, v in series]).max() + 0.04))

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)

    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
                    ncol=n_mod, frameon=False, fontsize=9.5,
                    handlelength=1.6, handletextpad=0.5, columnspacing=1.6,
                    labelcolor=INK)
    for h in leg.legend_handles:
        h.set_edgecolor(INK); h.set_linewidth(0.5)

    ax.set_title(
        "HK-gene ROC by brain region × correction model\n"
        "(23 HK genes per region; lower = better batch correction)",
        fontsize=12, color=INK, pad=56, loc="left",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------- plot 3: ROC curve (empirical CDF of per-gene AUC) ----------
def plot_roc_curve(df, out_png):
    """Plot empirical CDF of per-(gene, company) AUC for each model.

    Left-hugging curve = better (more genes have AUC ≈ 0.5).
    Diagonal-ish curve = bad (genes uniformly distributed between 0.5 and 1).
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=160)
    fig.patch.set_facecolor(BAR_BG); ax.set_facecolor(BAR_BG)

    # pool all (gene, company) corr_AUC across all regions
    series = []
    raw_vals = []
    for _, row in df.drop_duplicates("region").iterrows():
        pass  # placeholder
    # raw: per-(gene, company) raw_AUC is the same across all 3 model folders
    raw_per_region = []
    for r in df["region"].unique():
        sub = df[df["region"] == r].iloc[0]
        raw_per_region.append(sub["all_raw_auc"])
    raw_vals = np.concatenate(raw_per_region)

    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        per_region = []
        for r in sorted(sub["region"].unique()):
            rs = sub[sub["region"] == r].iloc[0]
            per_region.append(rs["all_corr_auc"])
        corr_vals = np.concatenate(per_region)
        series.append((model, corr_vals))

    # Plot raw first (background reference)
    sorted_raw = np.sort(raw_vals)
    n = len(sorted_raw)
    ecdf = np.arange(1, n + 1) / n
    ax.plot(sorted_raw, ecdf, color=PALETTE["raw"], lw=1.6, ls="--",
            label=LABEL["raw"], alpha=0.7, zorder=3)

    # Plot each model
    for model, vals in series:
        sorted_vals = np.sort(vals)
        n = len(sorted_vals)
        ecdf = np.arange(1, n + 1) / n
        palette_key = {"scVImodel": "scvi",
                       "scviHarmony": "scviHarmony",
                       "batchscvi_full": "batchscvi_full",
                       "batchscvi_con":  "batchscvi_con"}[model]
        ax.plot(sorted_vals, ecdf, color=PALETTE[palette_key], lw=2.2,
                label=LABEL[model], zorder=4)

    # Ideal baseline: all AUCs = 0.5 → vertical line at 0.5
    ax.axvline(0.5, color=SUB, lw=1.0, ls=(0, (4, 4)), zorder=2)
    ax.text(0.503, 0.02, "ideal = 0.5", ha="left", va="bottom",
            fontsize=9, color=SUB, transform=ax.get_yaxis_transform())

    # AUC of AUC: compute fraction-of-AUCs within tolerance of 0.5 (since
    # AUCs are symmetrized to [0.5, 1.0], use a small slack like 0.55)
    def frac_near_05(vals, tol=0.05):
        return float((vals <= 0.5 + tol).sum()) / len(vals)
    summary_lines = []
    summary_lines.append(f"fraction of (gene,company) AUCs ≤ 0.55  (closer to 0.5 = better)")
    summary_lines.append(f"raw:               {frac_near_05(raw_vals)*100:5.1f}%")
    for model, vals in series:
        summary_lines.append(f"{LABEL[model]:<17} {frac_near_05(vals)*100:5.1f}%")

    ax.set_xlabel("HKG AUC (one-vs-rest, per gene × company)",
                  fontsize=10.5, color=INK, labelpad=8)
    ax.set_ylabel("Empirical CDF (fraction of pairs with AUC ≤ x)",
                  fontsize=10.5, color=INK, labelpad=8)
    ax.set_xlim(0.45, 0.80)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="both", colors=SUB, labelsize=9)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5,
                    handlelength=1.8, handletextpad=0.5, labelcolor=INK)
    for h in leg.legend_handles:
        if hasattr(h, "set_edgecolor"):
            h.set_edgecolor(INK); h.set_linewidth(0.5)

    ax.set_title(
        "ROC of HK gene residuals — cumulative distribution of per-(gene,company) AUC\n"
        "Pooled across 8 regions × 23 HK genes × 2-3 companies (left-hugging curve = better)",
        fontsize=11.5, color=INK, pad=14, loc="left",
    )

    # inset summary text
    txt = "\n".join(summary_lines)
    ax.text(0.98, 0.30, txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=INK,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BAR_BG,
                      edgecolor=RULE, linewidth=0.6))

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    df = collect()
    print(f"loaded {len(df)} region × model rows")
    print()

    # overall summary
    print("=== per-region summary (raw / scVI / scVI+Harmony / batchscvi / batchscvi+cond) ===")
    regions = sorted(df["region"].unique())
    print(f"{'region':<6} {'raw':>6} {'scVI':>6} {'scVI+Harm':>10} {'batchscvi':>10} {'batchscvi+cond':>14} {'Δ_bsc':>8} {'Δ_bsc+cond':>12}")
    for r in regions:
        sub = df[df["region"] == r]
        raw = sub["mean_raw"].mean()
        scvi = sub[sub["model"]=="scVImodel"]["mean_corr"].iat[0]
        harm = sub[sub["model"]=="scviHarmony"]["mean_corr"].iat[0]
        bsc = sub[sub["model"]=="batchscvi_full"]["mean_corr"].iat[0]
        bsc_con = sub[sub["model"]=="batchscvi_con"]["mean_corr"].iat[0]
        delta = bsc - raw
        delta_con = bsc_con - raw
        print(f"{r:<6} {raw:>6.3f} {scvi:>6.3f} {harm:>10.3f} {bsc:>10.3f} {bsc_con:>14.3f} {delta:>+8.3f} {delta_con:>+12.3f}")
    print()
    print("=== mean across regions ===")
    grp = df.groupby("model").agg(
        mean_raw=("mean_raw", "mean"),
        mean_corr=("mean_corr", "mean"),
        mean_delta=("mean_delta_best", "mean"),
    ).round(4)
    print(grp.to_string())

    p1 = OUT_DIR / "hkg_delta_best_barplot.png"
    p2 = OUT_DIR / "hkg_mean_roc_by_region.png"
    p3 = OUT_DIR / "hkg_roc_curve.png"

    plot_delta(df, p1)
    plot_roc_bars(df, p2)
    plot_roc_curve(df, p3)
    print()
    print(f"→ wrote {p1}")
    print(f"→ wrote {p2}")
    print(f"→ wrote {p3}")


if __name__ == "__main__":
    main()