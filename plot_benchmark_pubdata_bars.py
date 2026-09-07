#!/usr/bin/env python3
"""
Plot per-celltype batch-distance for the publicdata benchmark.

Sources (one subdirectory per dataset x model):
    /data8/junyi/benchmark_results_pubdata/<DATASET>_<MODEL>/<DATASET>_<MODEL>_ct_batch_distance_summary.csv

Columns: celltype, batch_A, batch_B, layer, metric, n_genes, mean_d, median_d
- layer in {raw, corr}: distance before/after batch correction
- metric in {cohens_d, hellinger, mean_diff, var_ratio}: 4 distance metrics

Plots produced (saved under figures_benchmark_pubdata_summary_bars/):
1. Per-dataset grouped bars (cell types on x-axis, 3 models grouped per cell type)
   - panel per metric (4 panels per figure)
   - uses corr layer (post-correction; smaller = better)
2. Average-distance summary (dataset x model): one bar per (dataset, model) plus
   a global mean row, computed as mean of per-celltype corr distances averaged
   across all 4 metrics.

Error bars = SEM over batch pairs (n_batch_pairs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/data8/junyi/benchmark_results_pubdata")
OUT_DIR = Path("/home/junyichen/code/RUVAEDEG/figures_benchmark_pubdata_summary_bars")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (subdir key, display label) — display order is fixed across all plots.
METHODS: list[tuple[str, str]] = [
    ("scVI", "scVI"),
    ("scVInobatch", "scVI no-batch"),
    ("scviharmony", "scVI-Harmony"),
    ("batchscvi", "scVI-batch"),
    ("batchscvifix", "scVI-batch-fix"),
]

DATASETS: list[tuple[str, str]] = [
    ("GSE118767", "GSE118767"),
    ("GSE133549", "GSE133549"),
]

# Categorical palette — fixed order, identity (never cycled).
PALETTE: dict[str, str] = {
    "scVI": "#7f7f7f",              # gray  (baseline)
    "scVI no-batch": "#1f77b4",     # blue
    "scVI-Harmony": "#ff7f0e",      # orange
    "scVI-batch": "#2ca02c",        # green
    "scVI-batch-fix": "#9467bd",    # purple
}

DISTANCE_METRICS: list[tuple[str, str]] = [
    ("cohens_d", "Cohen's d"),
    ("hellinger", "Hellinger distance"),
    ("mean_diff", "Mean difference"),
    ("var_ratio", "Variance ratio"),
]


def load_summary(dataset: str, model_key: str) -> pd.DataFrame | None:
    """Return the summary CSV for one (dataset, model) or None if missing."""
    sub = RESULTS_DIR / f"{dataset}_{model_key}"
    csv_path = sub / f"{dataset}_{model_key}_ct_batch_distance_summary.csv"
    if not csv_path.is_file():
        print(f"[WARN] missing: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def _sem(x: pd.Series) -> float:
    """Standard error of the mean, NaN-safe."""
    x = x.dropna()
    n = len(x)
    if n < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(n))


def aggregate_per_celltype(df: pd.DataFrame, layer: str = "corr") -> pd.DataFrame:
    """
    Aggregate per (celltype, metric): mean and SEM of mean_d across batch pairs.

    Returns long-form DataFrame: celltype, metric, mean, sem, n_pairs
    """
    sub = df[df["layer"] == layer].copy()
    rows: list[dict] = []
    for (ct, metric), grp in sub.groupby(["celltype", "metric"], sort=True):
        rows.append(
            {
                "celltype": ct,
                "metric": metric,
                "mean": float(grp["mean_d"].mean()),
                "sem": _sem(grp["mean_d"]),
                "n_pairs": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_overall(df: pd.DataFrame, layer: str = "corr") -> dict[str, float]:
    """Return dict metric -> (mean over all celltypes x batch pairs).

    Also stores `__avg3__` — the mean of the three scale-comparable metrics
    (Cohen's d, Hellinger, mean diff). var_ratio is excluded from this scalar
    because its values span 1e1-1e6 while the other three span 1e-2-1e0.
    """
    sub = df[df["layer"] == layer]
    out: dict[str, float] = {}
    for metric, _ in DISTANCE_METRICS:
        v = sub.loc[sub["metric"] == metric, "mean_d"]
        out[metric] = float(v.mean()) if len(v) else float("nan")
    scale_comparable = [out["cohens_d"], out["hellinger"], out["mean_diff"]]
    out["__avg3__"] = float(np.nanmean(scale_comparable))
    return out


def plot_dataset_bars(
    dataset: str,
    panel_data: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """One figure per dataset; 4 metric panels; 3 models grouped per cell type."""
    # Union of cell types across available models, sorted alphabetically for legibility.
    all_celltypes = sorted(
        set().union(
            *(set(df["celltype"]) for df in panel_data.values() if df is not None)
        )
    )
    n_ct = len(all_celltypes)
    available_models = [(k, lbl) for k, lbl in METHODS if panel_data.get(k) is not None]
    n_methods = len(available_models)

    fig, axes = plt.subplots(2, 2, figsize=(max(14, n_ct * 1.4), 10), sharex=True)
    bar_w = 0.8 / max(n_methods, 1)
    x = np.arange(n_ct)

    for ax, (metric_key, metric_title) in zip(axes.flat, DISTANCE_METRICS):
        for j, (model_key, label) in enumerate(available_models):
            df_metric = panel_data[model_key]
            df_metric = df_metric[df_metric["metric"] == metric_key].set_index(
                "celltype"
            )
            means = np.array(
                [df_metric.loc[ct, "mean"] if ct in df_metric.index else np.nan
                 for ct in all_celltypes],
                dtype=float,
            )
            sems = np.array(
                [df_metric.loc[ct, "sem"] if ct in df_metric.index else np.nan
                 for ct in all_celltypes],
                dtype=float,
            )
            ax.bar(
                x + (j - (n_methods - 1) / 2) * bar_w,
                means,
                width=bar_w * 0.92,
                color=PALETTE[label],
                edgecolor="white",
                linewidth=0.4,
                yerr=sems,
                error_kw=dict(ecolor="#333", elinewidth=0.9, capsize=3, capthick=0.9),
                label=label,
            )
        # var_ratio values span 1e0-1e6 across cell types; log-scale so small bars
        # remain visible alongside the megakaryocytes/FCGR3A outliers.
        if metric_key == "var_ratio":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(metric_title, fontsize=12)
        ax.set_ylabel("distance (corr layer)")
        ax.grid(axis="y", linewidth=0.6, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9, ncol=n_methods, frameon=False)

    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(all_celltypes, rotation=45, ha="right", fontsize=9)
    axes[-1, 0].set_xlabel("cell type")
    axes[-1, 1].set_xticks(x)
    axes[-1, 1].set_xticklabels(all_celltypes, rotation=45, ha="right", fontsize=9)
    axes[-1, 1].set_xlabel("cell type")

    fig.suptitle(
        f"{dataset} — per-celltype batch distance (mean ± SEM over batch pairs, corr layer)",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_summary_bars(
    summary: dict[tuple[str, str], dict[str, float]],
    out_path: Path,
) -> None:
    """
    Bar chart of mean distance averaged over all cell types and batch pairs.

    X axis: datasets (GSE118767, GSE133549, Overall).
    Y axis: distance (corr layer). 5 panels:
      - 4 metrics (one per panel)
      - 1 panel: average of the 3 scale-comparable metrics (excludes var_ratio).
    3 bars per dataset (one per model).
    """
    x_labels = [d[1] for d in DATASETS] + ["Overall"]
    n_groups = len(x_labels)
    available_models = [
        (k, lbl) for k, lbl in METHODS
        if any(summary.get((d[0], k)) is not None for d in DATASETS)
    ]
    n_methods = len(available_models)
    bar_w = 0.8 / max(n_methods, 1)
    x = np.arange(n_groups)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharex=True)
    panel_specs: list[tuple[str, str, bool]] = [
        ("cohens_d", "Cohen's d", False),
        ("hellinger", "Hellinger distance", False),
        ("mean_diff", "Mean difference", False),
        ("var_ratio", "Variance ratio (log)", True),
        ("__avg3__", "Mean of 3 metrics (no var_ratio)", False),
    ]

    for ax, (metric_key, metric_title, log_y) in zip(axes, panel_specs):
        for j, (model_key, label) in enumerate(available_models):
            vals: list[float] = []
            for ds_key, _ in DATASETS:
                m = summary.get((ds_key, model_key))
                vals.append(m[metric_key] if (m and metric_key in m) else np.nan)
            ds_vals = [v for v in vals if not np.isnan(v) and v > 0]
            vals.append(float(np.mean(ds_vals)) if ds_vals else np.nan)

            ax.bar(
                x + (j - (n_methods - 1) / 2) * bar_w,
                vals,
                width=bar_w * 0.92,
                color=PALETTE[label],
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_title(metric_title, fontsize=11)
        ax.set_ylabel("distance (corr layer)")
        ax.grid(axis="y", linewidth=0.6, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9, frameon=False)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)

    fig.suptitle(
        "Publicdata benchmark — mean batch distance per (dataset, model) "
        "(averaged across cell types, corr layer)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def write_summary_table(
    summary: dict[tuple[str, str], dict[str, float]],
    out_csv: Path,
) -> None:
    """Write a tidy CSV of the summary table (rows = dataset x model)."""
    rows: list[dict] = []
    for ds_key, _ in DATASETS:
        for model_key, label in METHODS:
            m = summary.get((ds_key, model_key))
            if m is None:
                continue
            rows.append(
                {
                    "dataset": ds_key,
                    "model": label,
                    "cohens_d": m["cohens_d"],
                    "hellinger": m["hellinger"],
                    "mean_diff": m["mean_diff"],
                    "var_ratio": m["var_ratio"],
                    "mean_of_3_metrics": m["__avg3__"],
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")


def main() -> None:
    # Load & aggregate everything once.
    panel_data_per_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    summary: dict[tuple[str, str], dict[str, float]] = {}

    for ds_key, ds_label in DATASETS:
        panel_data_per_dataset[ds_key] = {}
        for model_key, _ in METHODS:
            df = load_summary(ds_key, model_key)
            if df is None:
                continue
            panel_data_per_dataset[ds_key][model_key] = aggregate_per_celltype(df)
            summary[(ds_key, model_key)] = aggregate_overall(df)

    # Per-dataset per-celltype bar charts (one figure per dataset).
    for ds_key, ds_label in DATASETS:
        out_path = OUT_DIR / f"{ds_label}_per_celltype_bars.png"
        plot_dataset_bars(ds_label, panel_data_per_dataset[ds_key], out_path)

    # Overall summary bar chart (dataset x model).
    plot_summary_bars(summary, OUT_DIR / "summary_bars.png")

    # Tidy summary CSV for downstream reuse.
    write_summary_table(summary, OUT_DIR / "summary_table.csv")


if __name__ == "__main__":
    main()
