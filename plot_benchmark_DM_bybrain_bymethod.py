"""
Bar chart of batch-distance metrics across cell types, grouped by method,
faceted by brain region and metric.

Data source: /data3/junyi/benchmark_DM_results/<REGION>_<METHOD>/<...>_ct_batch_distance_by_layer.csv
- For cohens_d / hellinger / mean_diff: lower (more negative) corr = better
- For var_ratio: closer to 1 is better

Each cell type × batch-pair has one row per metric. We plot the corr value of
each metric per cell type, grouping the 3 methods as side-by-side bars.
Errorbar is the SEM across batch pairs (mean ± SEM, n=batch_pairs).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/data3/junyi/benchmark_DM_results")
OUT_DIR = Path("/home/junyichen/code/RUVAEDEG/figures_benchmark_DM_bybrain_bymethod")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# brain region order and method order (fixed, since categorical hues are
# assigned in fixed order — never cycled)
REGIONS = ["AMY", "HPF", "HY", "iCTX", "MB", "PFC", "STR", "TH"]
METHODS = ["scVImodel", "scviHarmony", "batchscvi_full"]
METHOD_LABELS = {"scVImodel": "scVI", "scviHarmony": "scVI+Harmony",
                 "batchscvi_full": "scVI+batch"}
METHOD_COLORS = {"scVImodel": "#2a78d6",
                 "scviHarmony": "#eb6834",
                 "batchscvi_full": "#1baf7a"}
METRICS = ["cohens_d", "hellinger", "mean_diff", "var_ratio"]

# categorical palette light-mode (from skill reference, slots 1/2/3)
PALETTE_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
METHOD_COLORS = dict(zip(METHODS, PALETTE_CATEGORICAL))


def region_to_method_to_path(region: str) -> dict[str, Path]:
    """Return method -> by_layer CSV for one region."""
    out: dict[str, Path] = {}
    for method in METHODS:
        folder = RESULTS_DIR / f"{region}_{method}"
        candidates = list(folder.glob(f"{region}_{method}_ct_batch_distance_by_layer.csv"))
        if not candidates:
            raise FileNotFoundError(f"no by_layer CSV in {folder}")
        out[method] = candidates[0]
    return out


def load_region(region: str) -> pd.DataFrame:
    """Concatenate all 3 methods for a region into one DataFrame."""
    paths = region_to_method_to_path(region)
    frames = []
    for method, p in paths.items():
        df = pd.read_csv(p)
        df["method"] = method
        frames.append(df[["celltype", "method", "metric", "corr"]])
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean / SEM / count of corr per (method, celltype) for one metric."""
    sub = df[df["metric"] == metric]
    g = (sub.groupby(["method", "celltype"])["corr"]
            .agg(mean="mean", std="sem", n="count")
            .reset_index())
    return g


def plot_metric(metric: str, regions_data: dict[str, pd.DataFrame]) -> plt.Figure:
    """One figure per metric, 8 brain-region subplots."""
    # 4 rows × 2 cols gives 8 panels
    fig, axes = plt.subplots(4, 2, figsize=(20, 22),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    # y-axis label / title copy
    metric_titles = {
        "cohens_d":  "Cohen's d  (lower = better)",
        "hellinger": "Hellinger distance  (lower = better)",
        "mean_diff": "Mean difference  (lower = better)",
        "var_ratio": "Variance ratio  (closer to 1 = better)",
    }
    fig.suptitle(f"Batch-distance per cell type — {metric_titles[metric]}",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, region in enumerate(REGIONS):
        ax = axes_flat[idx]
        agg = aggregate(regions_data[region], metric)

        # build celltype order: union of celltypes seen across methods, sorted
        celltypes = sorted(agg["celltype"].unique())
        n_ct = len(celltypes)
        x = np.arange(n_ct)
        bar_w = 0.8 / len(METHODS)

        for m_idx, method in enumerate(METHODS):
            sub = agg[agg["method"] == method].set_index("celltype")
            # align to celltype order
            means = sub["mean"].reindex(celltypes).values
            sems = sub["std"].reindex(celltypes).values
            offsets = (m_idx - (len(METHODS) - 1) / 2) * bar_w
            ax.bar(x + offsets, means, bar_w,
                   yerr=sems, capsize=2,
                   color=METHOD_COLORS[method],
                   edgecolor="white", linewidth=0.5,
                   label=METHOD_LABELS[method],
                   error_kw=dict(ecolor="#52514e", lw=0.8))

        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(celltypes, rotation=80, fontsize=6,
                           ha="right")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric == "var_ratio":
            # reference line at 1 — the ideal target
            ax.axhline(1.0, color="#898781", linestyle=":", linewidth=1,
                       alpha=0.7)

        if idx == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8)

    # any unused axes — none, but safe
    return fig


def main() -> None:
    print("Loading per-region data ...")
    regions_data = {r: load_region(r) for r in REGIONS}

    for metric in METRICS:
        print(f"Plotting {metric} ...")
        fig = plot_metric(metric, regions_data)
        out = OUT_DIR / f"benchmark_DM_{metric}_bybrain_bymethod.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")

    print(f"Done. Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
