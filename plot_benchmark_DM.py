#!/usr/bin/env python3
"""
Plot per-celltype batch-distance metrics comparing scVI, scVI-Harmony, scVI-batch.

For cohens_d, hellinger, mean_diff: bar = delta_corr_minus_raw
    (more negative = better; correction shrinks the batch distance)
For var_ratio: bar = |corr-1| - |raw-1|
    (more negative = better; correction moves variance ratio closer to 1)

Error bars are SEM (standard error of the mean) over the ~6 batch pairs
contributing to each celltype. With few pairs (n=4..6 for some celltypes)
SEM tracks closely with SD/sqrt(n).

One figure per brain region, four subplots (one per metric).
X axis: cell type. Three bars per cell type (one per method).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/data3/junyi/benchmark_DM_results")
OUT_DIR = Path("/home/junyichen/code/RUVAEDEG/benchmark_DM_results_plots")
OUT_DIR.mkdir(exist_ok=True, parents=True)

BRAIN_REGIONS = ["AMY", "HPF", "HY", "MB", "PFC", "STR", "TH", "iCTX"]
METHODS = [
    ("scVImodel", "scVI"),
    ("scviHarmony", "scVI-Harmony"),
    ("batchscvi_full", "scVI-batch"),
]

# Categorical palette — fixed order across all plots.
PALETTE = {
    "scVI":          "#1f77b4",
    "scVI-Harmony":  "#ff7f0e",
    "scVI-batch":    "#2ca02c",
}

DISTANCE_METRICS = ["cohens_d", "hellinger", "mean_diff"]
VAR_METRIC = "var_ratio"


def _agg_per_celltype(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Aggregate across batch pairs to one mean + SEM per celltype.

    For distance metrics (cohens_d/hellinger/mean_diff):
        improvement = delta_corr_minus_raw
    For var_ratio:
        improvement = |corr-1| - |raw-1|

    Returns DataFrame with columns: celltype, mean, sem, n
    """
    sub = df[df["metric"] == metric].copy()
    if metric in DISTANCE_METRICS:
        sub["improvement"] = sub["delta_corr_minus_raw"]
    else:  # var_ratio
        sub["improvement"] = (sub["corr"] - 1).abs() - (sub["raw"] - 1).abs()

    grouped = sub.groupby("celltype")["improvement"]
    out = pd.DataFrame({
        "mean": grouped.mean(),
        "sem":  grouped.std(ddof=1) / np.sqrt(grouped.count()),
        "n":    grouped.count(),
    }).reset_index()
    return out


def plot_region(region: str) -> Path:
    """Build a 2x2 figure (4 metrics) for one brain region."""
    panel_data = {}  # method_label -> {metric: DataFrame[celltype, mean, sem, n]}
    for sub_dir, label in METHODS:
        csv_path = RESULTS_DIR / f"{region}_{sub_dir}" / f"{region}_{sub_dir}_ct_batch_distance_by_layer.csv"
        df = pd.read_csv(csv_path)
        per_metric = {m: _agg_per_celltype(df, m) for m in [VAR_METRIC] + DISTANCE_METRICS}
        panel_data[label] = per_metric

    # Union of celltypes across methods (preserve sorted order for readability)
    all_celltypes = sorted(set().union(*[set(d["cohens_d"]["celltype"]) for d in panel_data.values()]))

    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
    metric_order = [("cohens_d", "Cohen's d  (corr − raw)"),
                    ("hellinger", "Hellinger  (corr − raw)"),
                    ("mean_diff", "Mean diff  (corr − raw)"),
                    (VAR_METRIC, "Var ratio  (|corr−1| − |raw−1|)")]

    n_ct = len(all_celltypes)
    n_methods = len(METHODS)
    bar_w = 0.8 / n_methods
    x = np.arange(n_ct)

    for ax, (metric, title) in zip(axes.flat, metric_order):
        for j, (_, label) in enumerate(METHODS):
            df_metric = panel_data[label][metric].set_index("celltype")
            means = np.array([df_metric.loc[ct, "mean"] if ct in df_metric.index else np.nan
                              for ct in all_celltypes])
            sems = np.array([df_metric.loc[ct, "sem"] if ct in df_metric.index else np.nan
                             for ct in all_celltypes])
            ax.bar(x + (j - 1) * bar_w, means,
                   width=bar_w * 0.92,
                   color=PALETTE[label],
                   edgecolor="white", linewidth=0.4,
                   yerr=sems,
                   error_kw=dict(ecolor="#333", elinewidth=0.9, capsize=3, capthick=0.9),
                   label=label)
        ax.axhline(0, color="#444", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("improvement (↓ better)")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(all_celltypes, rotation=70, ha="right", fontsize=8)
        ax.set_xlabel("cell type")

    fig.suptitle(f"{region} — per-celltype batch-effect correction  "
                 f"(mean ± SEM over batch pairs)",
                 fontsize=15, y=0.995)
    fig.tight_layout()
    out_path = OUT_DIR / f"{region}_per_celltype_bars.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    saved = []
    for region in BRAIN_REGIONS:
        path = plot_region(region)
        saved.append(path)
        print(f"  wrote {path}")
    print(f"\n{len(saved)} figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()