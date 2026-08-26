"""
One-bar-per-method summary across brain regions.

Each subplot is one metric. Within a subplot:
  - x-axis: brain regions (8)
  - 3 grouped bars per region (one per method)
  - bar height = mean across cell types and batch pairs of corr
  - errorbar = SEM across (celltype × batch-pair) samples

For cohens_d / hellinger / mean_diff: lower is better.
For var_ratio: closer to 1 is better — so we plot |corr − 1|.

Output: figures_benchmark_DM_summary/benchmark_DM_<metric>_summary.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/data3/junyi/benchmark_DM_results")
OUT_DIR = Path("/home/junyichen/code/RUVAEDEG/figures_benchmark_DM_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["AMY", "HPF", "HY", "iCTX", "MB", "PFC", "STR", "TH"]
METHODS = ["scVImodel", "scviHarmony", "batchscvi_full"]
METHOD_LABELS = {"scVImodel": "scVI", "scviHarmony": "scVI+Harmony",
                 "batchscvi_full": "scVI+batch"}
METHOD_COLORS = {"scVImodel": "#2a78d6",
                 "scviHarmony": "#eb6834",
                 "batchscvi_full": "#1baf7a"}
METRICS = ["cohens_d", "hellinger", "mean_diff", "var_ratio"]

METRIC_TITLES = {
    "cohens_d":  "Cohen's d  (|corr| — lower is better)",
    "hellinger": "Hellinger  (|corr| — lower is better)",
    "mean_diff": "Mean diff  (|corr| — lower is better)",
    "var_ratio": "Var ratio  (|corr−1| — closer to 0 is better)",
}


def region_to_method_to_path(region: str) -> dict[str, Path]:
    out = {}
    for method in METHODS:
        folder = RESULTS_DIR / f"{region}_{method}"
        candidates = list(folder.glob(f"{region}_{method}_ct_batch_distance_by_layer.csv"))
        if not candidates:
            raise FileNotFoundError(f"no by_layer CSV in {folder}")
        out[method] = candidates[0]
    return out


def load_region(region: str) -> pd.DataFrame:
    paths = region_to_method_to_path(region)
    frames = []
    for method, p in paths.items():
        df = pd.read_csv(p)
        df["method"] = method
        frames.append(df[["celltype", "method", "metric", "corr"]])
    return pd.concat(frames, ignore_index=True)


def value_for(df: pd.DataFrame, metric: str) -> pd.Series:
    """Per-row metric value, where 0 = best for var_ratio."""
    sub = df[df["metric"] == metric].copy()
    if metric == "var_ratio":
        return (sub["corr"] - 1.0).abs().rename("value")
    return sub["corr"].abs().rename("value")


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    v = value_for(df, metric)
    sub = df[df["metric"] == metric].copy()
    sub["value"] = v.values
    g = (sub.groupby("method")["value"]
            .agg(mean="mean", sem="sem")
            .reindex(METHODS)
            .reset_index())
    return g


def plot_metric(metric: str, regions_data: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9),
                             constrained_layout=True, sharey=False)
    axes_flat = axes.flatten()
    fig.suptitle(f"Cross-region summary — {METRIC_TITLES[metric]}",
                 fontsize=16, fontweight="bold", y=1.04)

    n_reg = len(REGIONS)
    x = np.arange(n_reg)
    bar_w = 0.8 / len(METHODS)

    for r_idx, region in enumerate(REGIONS):
        ax = axes_flat[r_idx]
        agg = aggregate(regions_data[region], metric)

        for m_idx, method in enumerate(METHODS):
            row = agg[agg["method"] == method].iloc[0]
            offset = (m_idx - (len(METHODS) - 1) / 2) * bar_w
            ax.bar(x[r_idx] + offset, row["mean"], bar_w,
                   yerr=row["sem"], capsize=3,
                   color=METHOD_COLORS[method],
                   edgecolor="white", linewidth=0.5,
                   error_kw=dict(ecolor="#52514e", lw=0.8))

        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.set_xticks([x[r_idx]])
        ax.set_xticklabels([""], rotation=0)
        ax.set_ylabel("value" if r_idx % 4 == 0 else "")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # log y so large scVI+batch spikes don't flatten everything else
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(1e-4, 1e-3))

    # one shared legend on top-left subplot
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m])
               for m in METHODS]
    labels = [METHOD_LABELS[m] for m in METHODS]
    axes_flat[0].legend(handles, labels, loc="upper left",
                        frameon=False, fontsize=9)

    return fig


def main() -> None:
    print("Loading per-region data ...")
    regions_data = {r: load_region(r) for r in REGIONS}

    for metric in METRICS:
        print(f"Plotting {metric} ...")
        fig = plot_metric(metric, regions_data)
        out = OUT_DIR / f"benchmark_DM_{metric}_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")

    print(f"Done. Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()