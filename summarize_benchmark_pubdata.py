#!/usr/bin/env python
"""
summarize_benchmark_pubdata.py

Summarize the per-(dataset, method) benchmark results under
/data8/junyi/benchmark_results_pubdata/ and produce bar plots comparing
the performance of each method (smaller delta = better batch correction).

For each (dataset, method) directory, three plot inputs are aggregated:

1. *_hkg_auc.csv         -> mean delta_best = corr_best - raw_best per dataset
                              (housekeeping-gene one-vs-rest AUC across companies;
                              values closer to 0 / negative = better correction
                              for genes that should NOT discriminate between
                              cell types)
2. *_top10_auc.csv       -> mean delta = corr_AUC - raw_AUC across companies
                              (cell-type marker AUC; same-direction interpretation)
3. *_ct_batch_distance_*.csv (when present)
                           -> mean delta_corr_minus_raw per metric per dataset
                              (per-celltype x batch-pair distances on log1p
                              expression; smaller |corr - raw| = correction worked)

Output:
  /home/junyichen/code/RUVAEDEG/figures_benchmark_pubdata_summary/
      hk_auc_delta_per_dataset.png
      top10_auc_delta_per_dataset.png
      dm_distance_delta_per_metric.png
      summary_table.csv
      summary_table.md

Run:
  python summarize_benchmark_pubdata.py \
      --results-root /data8/junyi/benchmark_results_pubdata \
      --out-dir /home/junyichen/code/RUVAEDEG/figures_benchmark_pubdata_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Reference palette from /tmp/.../dataviz/references/palette.md
SERIES_COLORS = {
    "blue":   "#2a78d6",
    "orange": "#eb6834",
    "aqua":   "#1baf7a",
    "yellow": "#eda100",
    "magenta":"#e87ba4",
    "green":  "#008300",
    "violet": "#4a3aa7",
    "red":    "#e34948",
}
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRID_HAIR     = "#e1e0d9"
SURFACE       = "#fcfcfb"

# Categorical hue order for fixed assignment (by method)
METHOD_COLOR_SLOTS = {
    "batchscvi":   "blue",
    "scVInobatch": "orange",
    "scviharmony": "aqua",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("/data8/junyi/benchmark_results_pubdata"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/junyichen/code/RUVAEDEG/figures_benchmark_pubdata_summary"),
    )
    return p.parse_args()


def split_name(dir_name: str) -> tuple[str, str]:
    """'GSE118767_batchscvi' -> ('GSE118767', 'batchscvi')."""
    base = dir_name
    if base.startswith("GSE"):
        ds, _, method = base.partition("_")
        return ds, method
    return base, ""


def collect_hk_auc(root: Path) -> pd.DataFrame:
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        ds, method = split_name(sub.name)
        fp = sub / f"{sub.name}_hkg_auc.csv"
        if not fp.is_file():
            continue
        df = pd.read_csv(fp)
        if "delta_best" in df.columns and len(df):
            rows.append({
                "dataset": ds,
                "method": method,
                "n_genes": int(df["gene"].nunique()),
                "n_companies": int(df["company"].nunique()),
                "mean_delta_best": float(df["delta_best"].mean()),
                "median_delta_best": float(df["delta_best"].median()),
                "mean_raw_AUC": float(df["raw_AUC"].mean()),
                "mean_corr_AUC": float(df["corr_AUC"].mean()),
            })
    return pd.DataFrame(rows)


def collect_top10_auc(root: Path) -> pd.DataFrame:
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        ds, method = split_name(sub.name)
        fp = sub / f"{sub.name}_top10_auc.csv"
        if not fp.is_file():
            continue
        df = pd.read_csv(fp)
        if "delta" in df.columns and len(df):
            rows.append({
                "dataset": ds,
                "method": method,
                "n_companies": int(df["company"].nunique()),
                "mean_delta": float(df["delta"].mean()),
                "median_delta": float(df["delta"].median()),
                "mean_raw_AUC": float(df["raw_AUC"].mean()),
                "mean_corr_AUC": float(df["corr_AUC"].mean()),
            })
    return pd.DataFrame(rows)


def collect_dm_distance(root: Path) -> pd.DataFrame:
    """Mean delta_corr_minus_raw per (dataset, method, metric)."""
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        ds, method = split_name(sub.name)
        fp = sub / f"{sub.name}_ct_batch_distance_by_layer.csv"
        if not fp.is_file():
            continue
        df = pd.read_csv(fp)
        if "delta_corr_minus_raw" not in df.columns or len(df) == 0:
            continue
        for metric, mdf in df.groupby("metric"):
            rows.append({
                "dataset": ds,
                "method": method,
                "metric": metric,
                "n_pairs": int(len(mdf)),
                "mean_delta": float(mdf["delta_corr_minus_raw"].mean()),
                "median_delta": float(mdf["delta_corr_minus_raw"].median()),
                "mean_raw": float(mdf["raw"].abs().mean()),
                "mean_corr": float(mdf["corr"].abs().mean()),
            })
    return pd.DataFrame(rows)


def _bar_with_axes(ax, x, y, color, ylabel, title, *, label_y=True, max_label=None):
    """Draw a single horizontal bar plot onto ax (clean style, no top/right spines).

    `max_label` (optional): the maximum absolute label value to use for xlim
    and label offset. When bars have very different magnitudes within the same
    plot (e.g., DM var_ratio 2.7 vs cohens_d 0.12), pass the absolute max of the
    smallest metric so labels fit.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID_HAIR, linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Zero line for reference
    ax.axvline(0, color=INK_MUTED, linewidth=0.9, zorder=1)

    bars = ax.barh(
        x, y,
        color=color, edgecolor="none", height=0.7, zorder=2,
    )
    ax.set_yticks(range(len(x)))
    ax.set_yticklabels(x, color=INK_SECONDARY)

    ax.set_xlabel(ylabel, color=INK_SECONDARY, fontsize=9)

    # Value labels at the bar end - use a generous xlim so labels don't
    # collide with the y-tick labels on the left.
    if label_y:
        xmax_data = max(abs(v) for v in y) if len(y) else 1.0
        xmax = max(xmax_data, max_label or 0.0) * 1.4
        ax.set_xlim(-xmax, xmax)
        for bar, v in zip(bars, y):
            offset = 0.04 * xmax
            ha = "left" if v >= 0 else "right"
            tx = v + (offset if v >= 0 else -offset)
            ax.text(tx, bar.get_y() + bar.get_height() / 2,
                    f"{v:+.3f}", va="center", ha=ha,
                    fontsize=8.5, color=INK_PRIMARY)

    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=8)


def plot_hk_auc_delta(df: pd.DataFrame, out: Path):
    if df.empty:
        print("[skip] hk_auc_delta: no input")
        return
    df = df.copy()
    df["label"] = df["dataset"] + " · " + df["method"]
    df = df.sort_values("mean_delta_best", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, max(3.5, 0.45 * len(df) + 1.5)))
    colors = [SERIES_COLORS[METHOD_COLOR_SLOTS.get(m, "blue")] for m in df["method"]]
    _bar_with_axes(
        ax,
        x=df["label"].tolist(),
        y=df["mean_delta_best"].tolist(),
        color=colors,
        ylabel="HK-gene one-vs-rest AUC delta (corr − raw)",
        title=(
            f"Housekeeping-gene AUC delta — {len(df)} runs. "
            "Smaller = better batch correction."
        ),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"[wrote] {out}")


def plot_top10_auc_delta(df: pd.DataFrame, out: Path):
    if df.empty:
        print("[skip] top10_auc_delta: no input")
        return
    df = df.copy()
    df["label"] = df["dataset"] + " · " + df["method"]
    df = df.sort_values("mean_delta", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, max(3.5, 0.45 * len(df) + 1.5)))
    colors = [SERIES_COLORS[METHOD_COLOR_SLOTS.get(m, "blue")] for m in df["method"]]
    _bar_with_axes(
        ax,
        x=df["label"].tolist(),
        y=df["mean_delta"].tolist(),
        color=colors,
        ylabel="Top-10 company-DEG AUC delta (corr − raw)",
        title=(
            f"Top-10 cell-type DEG AUC delta — {len(df)} runs. "
            "Smaller = marker signature preserved."
        ),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"[wrote] {out}")


def plot_dm_distance_delta(df: pd.DataFrame, out: Path):
    """Small multiples: 1 subplot per metric; within each, bars per (dataset, method)."""
    if df.empty:
        print("[skip] dm_distance_delta: no input")
        return
    df = df.copy()
    df["label"] = df["dataset"] + " · " + df["method"]

    metrics = sorted(df["metric"].unique())
    n = len(metrics)
    fig, axes = plt.subplots(
        1, n, figsize=(4.5 * n, max(3.5, 0.55 * df["label"].nunique() + 1.2)),
        sharey=False,
    )
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric].sort_values(
            "mean_delta", ascending=True).reset_index(drop=True)
        colors = [
            SERIES_COLORS[METHOD_COLOR_SLOTS.get(m, "blue")] for m in sub["method"]
        ]
        _bar_with_axes(
            ax,
            x=sub["label"].tolist(),
            y=sub["mean_delta"].tolist(),
            color=colors,
            ylabel=f"{metric} delta (corr − raw)",
            title=f"{metric} — mean per celltype × batch-pair",
        )
    fig.suptitle(
        "Celltype × batch-pair distance delta (DM benchmark). "
        "Smaller magnitude = better batch correction.",
        color=INK_PRIMARY, fontsize=12, x=0.01, ha="left", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def write_summary_table(hk: pd.DataFrame, top10: pd.DataFrame, dm: pd.DataFrame, out_csv: Path, out_md: Path):
    # Build a per-(label) table where each metric/dataset is one column.
    # Each input gets prefixed so columns from different inputs don't collide.

    def _with_label(d, prefix):
        if d.empty:
            return d
        d = d.copy()
        d["label"] = d["dataset"] + " · " + d["method"]
        drop_cols = [c for c in ("dataset", "method") if c in d.columns]
        d = d.drop(columns=drop_cols)
        return d.add_prefix(prefix).rename(columns={f"{prefix}label": "label"})

    parts = [
        _with_label(hk,    "hk_"),
        _with_label(top10, "top10_"),
    ]

    # Wide DM table: rows = label, columns = metric
    if not dm.empty:
        dm_wide = dm.pivot_table(
            index=["dataset", "method"],
            columns="metric",
            values="mean_delta",
            aggfunc="first",
        ).reset_index()
        dm_wide["label"] = dm_wide["dataset"] + " · " + dm_wide["method"]
        dm_wide = dm_wide.drop(columns=["dataset", "method"])
        dm_wide = dm_wide.add_prefix("dm_").rename(columns={"dm_label": "label"})
        parts.append(dm_wide)

    # Drop empty parts
    parts = [p for p in parts if not p.empty]
    if not parts:
        print("[skip] summary_table: no input")
        return

    # All parts now share a "label" column (without prefix). Concatenate horizontally.
    summary = parts[0]
    for p in parts[1:]:
        summary = summary.merge(p, on="label", how="outer")

    summary = summary.sort_values("label").reset_index(drop=True)
    summary.to_csv(out_csv, index=False)
    print(f"[wrote] {out_csv}")

    md_lines = ["| " + " | ".join(summary.columns) + " |"]
    md_lines.append("|" + "|".join(["---"] * len(summary.columns)) + "|")
    for _, row in summary.iterrows():
        md_lines.append("| " + " | ".join(
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
        ) + " |")
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"[wrote] {out_md}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.results_root
    if not root.is_dir():
        raise SystemExit(f"[ERROR] results root not found: {root}")

    print(f"[scan] {root}")
    hk = collect_hk_auc(root)
    top10 = collect_top10_auc(root)
    dm = collect_dm_distance(root)
    print(f"[scan] hk_auc rows={len(hk)}  top10 rows={len(top10)}  dm rows={len(dm)}")

    plot_hk_auc_delta(
        hk, args.out_dir / "hk_auc_delta_per_dataset.png",
    )
    plot_top10_auc_delta(
        top10, args.out_dir / "top10_auc_delta_per_dataset.png",
    )
    plot_dm_distance_delta(
        dm, args.out_dir / "dm_distance_delta_per_metric.png",
    )
    write_summary_table(
        hk, top10, dm,
        args.out_dir / "summary_table.csv",
        args.out_dir / "summary_table.md",
    )


if __name__ == "__main__":
    main()