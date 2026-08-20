"""Per-region grouped bar plot of HKG mean ROC across 4 correction models.

For each brain region, 4 bars side-by-side:
    raw · scVI · scVI+Harmony · paired-batch scVI (D1 counts)

Y-axis: HKG mean ROC across the 23 HK genes (lower = better).
Horizontal dashed line at 0.5 = ideal (no company signal).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_GLOB = "/data3/junyi/benchmark_results/*/*_hkg_auc.csv"
OUT_PNG  = Path("/home/junyichen/code/RUVAEDEG/hkg_mean_roc_by_region.png")
OUT_CSV  = Path("/home/junyichen/code/RUVAEDEG/hkg_mean_roc_by_region.csv")

PALETTE = {
    "raw":         "#2a78d6",  # blue
    "scvi":        "#1baf7a",  # aqua
    "scviHarmony": "#eb6834",  # orange
    "svibatch":    "#7a5e9e",  # violet
}
ORDER = ["raw", "scvi", "scviHarmony", "svibatch"]
LABELS = {
    "raw":         "raw",
    "scvi":        "scVI",
    "scviHarmony": "scVI+Harmony",
    "svibatch":    "paired-batch scVI (D1)",
}


def collect_per_region():
    """For each region, return {model: mean_roc_value}.

    raw is identical across all 3 model folders (best_raw), so use the
    best_raw mean from any of the three (they're equal); here we average
    across all 24 entries to mirror the pooled raw bar in the summary view.
    """
    rows = []
    for path in sorted(__import__("glob").glob(CSV_GLOB)):
        base = Path(path).stem.replace("_hkg_auc", "")
        if base.endswith("_batchscvi_full"):
            model, region = "batchscvi_full", base[: -len("_batchscvi_full")]
        elif base.endswith("_scviHarmony"):
            model, region = "scviHarmony", base[: -len("_scviHarmony")]
        elif base.endswith("_scVImodel"):
            model, region = "scVImodel", base[: -len("_scVImodel")]
        else:
            continue
        df = pd.read_csv(path)
        per_gene = df.drop_duplicates("gene")[["gene", "best_raw", "best_corr"]]
        rows.append({
            "region": region,
            "model": model,
            "raw_value": per_gene["best_raw"].mean(),
            "corr_value": per_gene["best_corr"].mean(),
        })
    df = pd.DataFrame(rows)
    regions = sorted(df["region"].unique())

    # Build per-region dict keyed by 4-model label
    out = {}
    for r in regions:
        sub = df[df["region"] == r]
        out[r] = {
            "raw":         sub["raw_value"].mean(),     # all 3 equal, mean = value
            "scvi":        sub[sub["model"] == "scVImodel"]["corr_value"].iat[0],
            "scviHarmony": sub[sub["model"] == "scviHarmony"]["corr_value"].iat[0],
            "svibatch":    sub[sub["model"] == "batchscvi_full"]["corr_value"].iat[0],
        }
    return regions, out


def plot(regions: list, per_region: dict, out_png: Path):
    n_reg = len(regions)
    n_mod = len(ORDER)
    bar_w = 0.20
    group_w = bar_w * n_mod
    x = np.arange(n_reg)

    fig, ax = plt.subplots(figsize=(13.0, 6.4), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # ideal baseline at 0.5
    ax.axhline(0.5, color="#52514e", lw=1.0, ls=(0, (4, 4)), zorder=2)
    ax.text(n_reg - 0.45, 0.503, "ideal = 0.5",
            ha="right", va="bottom", fontsize=8.5, color="#52514e")

    for i, model in enumerate(ORDER):
        vals = np.array([per_region[r][model] for r in regions])
        offsets = x + (i - (n_mod - 1) / 2) * bar_w
        bars = ax.bar(
            offsets, vals, width=bar_w * 0.92,
            color=PALETTE[model],
            edgecolor="#0b0b0b", linewidth=0.5,
            label=LABELS[model], zorder=3,
        )
        # direct label above each bar
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.005,
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#0b0b0b",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=10.5, color="#0b0b0b")
    ax.set_xlabel("Brain region", fontsize=11, color="#0b0b0b", labelpad=8)
    ax.set_ylabel(
        "HKG mean ROC (best across companies)\n"
        "lower = HK genes carry less company signal",
        fontsize=10, color="#0b0b0b", labelpad=8,
    )
    ax.tick_params(axis="y", colors="#52514e", labelsize=9)
    ax.tick_params(axis="x", colors="#52514e")

    # y-limits with headroom for labels
    all_vals = np.array([per_region[r][m] for r in regions for m in ORDER])
    ax.set_ylim(0.46, max(0.78, all_vals.max() + 0.04))

    # recessive chrome
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # legend (top, horizontal) — pushed above the title's two-line block
    leg = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
        ncol=4, frameon=False, fontsize=9.5,
        handlelength=1.6, handletextpad=0.5, columnspacing=1.6,
        labelcolor="#0b0b0b",
    )
    for h in leg.legend_handles:
        h.set_edgecolor("#0b0b0b")
        h.set_linewidth(0.5)

    ax.set_title(
        "HK-gene ROC by brain region × correction model\n"
        "(23 HK genes per region; lower = better batch correction)",
        fontsize=12, color="#0b0b0b", pad=56, loc="left",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    regions, per_region = collect_per_region()
    print(f"regions: {regions}")
    print()
    print(f"{'region':<6}", end="")
    for m in ORDER:
        print(f" {LABELS[m]:>22}", end="")
    print()
    for r in regions:
        print(f"{r:<6}", end="")
        for m in ORDER:
            print(f" {per_region[r][m]:>22.4f}", end="")
        print()

    # save csv
    rows = []
    for r in regions:
        for m in ORDER:
            rows.append({"region": r, "model": m, "label": LABELS[m],
                         "mean_roc": per_region[r][m]})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n→ wrote summary: {OUT_CSV}")

    plot(regions, per_region, OUT_PNG)
    print(f"→ wrote plot   : {OUT_PNG}")


if __name__ == "__main__":
    main()