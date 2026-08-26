"""Per-region × per-celltype plot: every celltype × every method (incl. raw) on one chart per metric.

Layout (per metric PNG):
- 2x4 subplot grid, one subplot per brain region (8 regions).
- Within each subplot: x-axis = celltypes of that region (alphabetical), 4 grouped bars
  per celltype (raw + scVI + scVI+Harmony + scVI+batch).
- Bar height = mean across batch-pairs for that celltype (i.e. mean of the
  per-(celltype × batch-pair) values for that celltype).
- Errorbar = SEM across batch-pairs for that celltype.
- cohens_d / hellinger / mean_diff: plot |corr| (lower is better).
- var_ratio: plot |corr − 1| (closer to 1 is better).

Celltype labels for AMY have a redundant "AMY " prefix — stripped for compactness.

Output: figures_benchmark_DM_summary/benchmark_DM_<metric>_by_region_celltype.png
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
    "raw": "#9aa0a6",
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

YLABEL = {
    "cohens_d": "|Cohen's d|",
    "hellinger": "Hellinger",
    "mean_diff": "|mean diff|",
    "var_ratio": "|var ratio − 1|",
}


def _csv_for(region: str, method: str) -> Path:
    folder = RESULTS_DIR / f"{region}_{method}"
    candidates = list(folder.glob(f"{region}_{method}_ct_batch_distance_by_layer.csv"))
    if not candidates:
        raise FileNotFoundError(f"no CSV found under {folder}")
    return candidates[0]


def _load_region_method(region: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(_csv_for(region, method))
    return df[["celltype", "batch_A", "batch_B", "metric", "corr"]].copy()


def _load_raw(region: str) -> pd.DataFrame:
    df = _load_region_method(region, "scVImodel").rename(columns={"corr": "raw_val"})
    return df


def _value_for(sub: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "var_ratio":
        return (sub["corr"] - 1.0).abs()
    return sub["corr"].abs()


def _value_for_raw(sub_raw: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "var_ratio":
        return (sub_raw["raw_val"] - 1.0).abs()
    return sub_raw["raw_val"].abs()


def _short_ct(region: str, ct: str) -> str:
    """Strip redundant region prefix (only AMY labels start with 'AMY ')."""
    prefix = f"{region} "
    if ct.startswith(prefix):
        return ct[len(prefix):]
    return ct


def aggregate_per_celltype(region_data: dict, raw_data: pd.DataFrame, metric: str
                            ) -> tuple[list[str], pd.DataFrame]:
    """For one region, build a per-celltype table covering raw + 3 methods.

    Returns:
      cts: ordered list of celltype labels (short, alphabetical)
      agg: DataFrame with columns celltype, method, mean, sem, n
    """
    # collect all celltypes from raw (== from any method, same universe)
    sub_r = raw_data[raw_data["metric"] == metric]
    cts = sorted(sub_r["celltype"].unique())
    rows = []
    for ct in cts:
        # raw
        v = _value_for_raw(sub_r[sub_r["celltype"] == ct], metric)
        rows.append({"celltype": ct, "method": "raw",
                     "mean": v.mean(), "sem": v.sem(), "n": len(v)})
        # methods
        for method in METHODS:
            sub = region_data[method]
            sub_m = sub[(sub["metric"] == metric) & (sub["celltype"] == ct)]
            v = _value_for(sub_m, metric)
            rows.append({"celltype": ct, "method": method,
                         "mean": v.mean(), "sem": v.sem(), "n": len(v)})
    return cts, pd.DataFrame(rows)


def plot_metric(metric: str, all_data: dict, all_raw: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(28, 14), constrained_layout=True)
    axes_flat = axes.flatten()
    fig.suptitle(f"Per-celltype summary by brain region — {METRIC_TITLES[metric]}",
                 fontsize=18, fontweight="bold")

    n_meth = len(METHOD_ORDER)
    bar_w = 0.8 / n_meth

    for r_idx, region in enumerate(REGIONS):
        ax = axes_flat[r_idx]
        cts, agg = aggregate_per_celltype(all_data[region], all_raw[region], metric)
        n_ct = len(cts)
        x = np.arange(n_ct)

        for m_idx, method in enumerate(METHOD_ORDER):
            sub = agg[agg["method"] == method]
            # align by celltype order
            means = [sub[sub["celltype"] == ct]["mean"].iloc[0] for ct in cts]
            sems = [sub[sub["celltype"] == ct]["sem"].iloc[0] for ct in cts]
            offset = (m_idx - (n_meth - 1) / 2) * bar_w
            ax.bar(
                x + offset, means, bar_w,
                yerr=sems, capsize=2,
                color=METHOD_COLORS[method],
                edgecolor="white", linewidth=0.4,
                error_kw=dict(ecolor="#52514e", lw=0.6),
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_short_ct(region, ct) for ct in cts],
                            rotation=75, ha="right", fontsize=7)
        ax.set_title(f"{region}  (n_celltypes={n_ct})", fontsize=12, fontweight="bold")
        if r_idx % 4 == 0:
            ax.set_ylabel(YLABEL[metric], fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # one shared legend on the figure
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in METHOD_ORDER]
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    fig.legend(handles, labels, loc="upper right", fontsize=12,
               frameon=False, ncol=4, bbox_to_anchor=(1.0, 1.01))

    return fig


def main() -> None:
    all_data: dict[str, dict[str, pd.DataFrame]] = {}
    all_raw: dict[str, pd.DataFrame] = {}
    for region in REGIONS:
        all_data[region] = {m: _load_region_method(region, m) for m in METHODS}
        all_raw[region] = _load_raw(region)

    for metric in METRICS:
        fig = plot_metric(metric, all_data, all_raw)
        out = OUT_DIR / f"benchmark_DM_{metric}_by_region_celltype.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()