"""Summary-of-summary plot: every brain region × every method (incl. raw) on one chart per metric.

Layout:
- 2x2 subplot grid, one subplot per metric (cohens_d / hellinger / mean_diff / var_ratio).
- Within each subplot: x-axis = 8 brain regions; 4 grouped bars per region
  (raw + scVI + scVI+Harmony + scVI+batch).
- Each bar height = mean across (celltype × batch-pair) samples.
- Errorbar = SEM across (celltype × batch-pair) samples.
- cohens_d / hellinger / mean_diff: plot |corr| (lower is better).
- var_ratio: plot |corr − 1| (closer to 1 is better).

Output: figures_benchmark_DM_summary/benchmark_DM_<metric>_summary_of_summary.png
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
METHOD_LABELS = {
    "raw": "raw",
    "scVImodel": "scVI",
    "scviHarmony": "scVI+Harmony",
    "batchscvi_full": "scVI+batch",
}
METHOD_COLORS = {
    "raw": "#9aa0a6",        # grey for raw
    "scVImodel": "#2a78d6",
    "scviHarmony": "#eb6834",
    "batchscvi_full": "#1baf7a",
}
METHOD_ORDER = ["raw", "scVImodel", "scviHarmony", "batchscvi_full"]
METRICS = ["cohens_d", "hellinger", "mean_diff", "var_ratio"]

METRIC_TITLES = {
    "cohens_d": "Cohen's d  (|corr| — lower better)",
    "hellinger": "Hellinger  (|corr| — lower better)",
    "mean_diff": "Mean diff  (|corr| — lower better)",
    "var_ratio": "Var ratio  (|corr − 1| — closer to 0 better)",
}


def _csv_for(region: str, method: str) -> Path:
    folder = RESULTS_DIR / f"{region}_{method}"
    candidates = list(folder.glob(f"{region}_{method}_ct_batch_distance_by_layer.csv"))
    if not candidates:
        raise FileNotFoundError(f"no CSV found under {folder}")
    return candidates[0]


def _load_region_method(region: str, method: str) -> pd.DataFrame:
    """Load the per-(celltype × batch-pair) rows for one region+method, return
    columns celltype, batch_A, batch_B, metric, value, kind."""
    df = pd.read_csv(_csv_for(region, method))
    return df[["celltype", "batch_A", "batch_B", "metric", "corr"]].copy()


def _load_raw(region: str) -> pd.DataFrame:
    """Raw values are method-independent; load from any method and tag kind='raw'."""
    df = _load_region_method(region, "scVImodel")
    df = df.rename(columns={"corr": "raw_val"})
    return df


def _value_for(sub: pd.DataFrame, metric: str) -> pd.Series:
    """Transform corr → scalar value (per-metric, abs or |corr-1|)."""
    if metric == "var_ratio":
        return (sub["corr"] - 1.0).abs()
    return sub["corr"].abs()


def _value_for_raw(sub_raw: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "var_ratio":
        return (sub_raw["raw_val"] - 1.0).abs()
    return sub_raw["raw_val"].abs()


def aggregate_region(region_data: dict, raw_data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For one region, build a per-method table: method, mean, sem, n."""
    rows = []
    # raw (always available from raw_data)
    sub_r = raw_data[raw_data["metric"] == metric].copy()
    vals_r = _value_for_raw(sub_r, metric)
    rows.append({
        "method": "raw",
        "mean": vals_r.mean(),
        "sem": vals_r.sem(),
        "n": len(vals_r),
    })
    # methods
    for method in METHODS:
        sub = region_data[method][region_data[method]["metric"] == metric].copy()
        vals = _value_for(sub, metric)
        rows.append({
            "method": method,
            "mean": vals.mean(),
            "sem": vals.sem(),
            "n": len(vals),
        })
    return pd.DataFrame(rows)


def plot_metric(metric: str, all_data: dict, all_raw: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    n_reg = len(REGIONS)
    x = np.arange(n_reg)
    n_meth = len(METHOD_ORDER)
    bar_w = 0.8 / n_meth

    for r_idx, region in enumerate(REGIONS):
        agg = aggregate_region(all_data[region], all_raw[region], metric)
        for m_idx, method in enumerate(METHOD_ORDER):
            row = agg[agg["method"] == method].iloc[0]
            offset = (m_idx - (n_meth - 1) / 2) * bar_w
            ax.bar(
                x[r_idx] + offset, row["mean"], bar_w,
                yerr=row["sem"], capsize=3,
                color=METHOD_COLORS[method],
                edgecolor="white", linewidth=0.5,
                error_kw=dict(ecolor="#52514e", lw=0.8),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, fontsize=11)
    ax.set_xlabel("Brain region", fontsize=11)
    ax.set_ylabel(METRIC_TITLES[metric].split("  ")[1] if "  " in METRIC_TITLES[metric] else "value",
                   fontsize=11)
    ax.set_title(f"Summary of summary — {METRIC_TITLES[metric]}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in METHOD_ORDER]
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    ax.legend(handles, labels, loc="upper right", fontsize=10, frameon=False, ncol=4)

    return fig


def main() -> None:
    # load all data once
    all_data: dict[str, dict[str, pd.DataFrame]] = {}
    all_raw: dict[str, pd.DataFrame] = {}
    for region in REGIONS:
        all_data[region] = {m: _load_region_method(region, m) for m in METHODS}
        all_raw[region] = _load_raw(region)

    for metric in METRICS:
        fig = plot_metric(metric, all_data, all_raw)
        out = OUT_DIR / f"benchmark_DM_{metric}_summary_of_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()