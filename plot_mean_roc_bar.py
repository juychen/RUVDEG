"""Barplot: HKG mean ROC (lower = better batch correction).

4 models compared:
  raw        — pre-correction (best_raw across 8 regions × 23 HK genes)
  scvi       — scVImodel   (best_corr across 8 regions × 23 HK genes)
  scviHarmony— scviHarmony (best_corr across 8 regions × 23 HK genes)
  svibatch   — batchscvi_full (best_corr across 8 regions × 23 HK genes)

Ideal = 0.5 (HKG carries no company signal). Lower bar = better.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_GLOB = "/data3/junyi/benchmark_results/*/*_hkg_auc.csv"
OUT_PNG  = Path("/home/junyichen/code/RUVAEDEG/hkg_mean_roc_barplot.png")
OUT_CSV  = Path("/home/junyichen/code/RUVAEDEG/hkg_mean_roc_summary.csv")

# dataviz palette (light)
PALETTE = {
    "raw":         "#2a78d6",  # blue
    "scvi":        "#1baf7a",  # aqua
    "scviHarmony": "#eb6834",  # orange
    "svibatch":    "#7a5e9e",  # violet
}
ORDER = ["raw", "scvi", "scviHarmony", "svibatch"]

LABELS = {
    "raw":         "raw\n(no correction)",
    "scvi":        "scVI",
    "scviHarmony": "scVI + Harmony",
    "svibatch":    "paired-batch scVI\n(D1 counts)",
}


def collect():
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
            "region": region, "model": model,
            "mean_raw": per_gene["best_raw"].mean(),
            "mean_corr": per_gene["best_corr"].mean(),
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame):
    """Return {model_label: (mean, std)} for the 4-model bar plot."""
    out = {}
    # raw: same across all 3 model folders; pool all 24 entries
    out["raw"] = (df["mean_raw"].mean(), df["mean_raw"].std())
    for model, label in [("scVImodel", "scvi"),
                         ("scviHarmony", "scviHarmony"),
                         ("batchscvi_full", "svibatch")]:
        sub = df[df["model"] == model]["mean_corr"]
        out[label] = (sub.mean(), sub.std())
    return out


def plot(summary: dict, out_png: Path):
    labels = [LABELS[k] for k in ORDER]
    means  = [summary[k][0] for k in ORDER]
    stds   = [summary[k][1] for k in ORDER]
    colors = [PALETTE[k] for k in ORDER]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    x = np.arange(len(ORDER))
    bars = ax.bar(
        x, means, width=0.62,
        color=colors,
        edgecolor="#0b0b0b", linewidth=0.6,
        yerr=stds,
        zorder=3,
    )

    # ideal baseline at 0.5
    ax.axhline(0.5, color="#52514e", lw=1.0, ls=(0, (4, 4)), zorder=2)
    ax.text(len(ORDER) - 0.5, 0.503,
            "ideal = 0.5 (no company signal)",
            ha="right", va="bottom", fontsize=9, color="#52514e")

    # direct labels on each bar (relief for sub-3:1 aqua slot)
    for b, m, s in zip(bars, means, stds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            m + s + 0.005,
            f"{m:.3f} ± {s:.3f}",
            ha="center", va="bottom",
            fontsize=9.5, color="#0b0b0b",
        )

    # ranking arrow + caption
    best_idx = int(np.argmin(means))
    worst_idx = int(np.argmax(means))
    arrow_y = 0.74
    ax.annotate(
        "", xy=(best_idx, arrow_y + 0.005), xytext=(worst_idx, arrow_y + 0.005),
        arrowprops=dict(arrowstyle="->", color="#0ca30c", lw=1.6),
    )
    ax.text(
        (best_idx + worst_idx) / 2, arrow_y + 0.012,
        "← lower mean ROC = better batch correction",
        ha="center", va="bottom", fontsize=9.5, color="#0ca30c", fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="#0b0b0b")
    ax.set_xlabel("Correction model", fontsize=11, color="#0b0b0b", labelpad=8)
    ax.set_ylabel(
        "HKG mean ROC (best across companies)\n"
        "lower = HK genes carry less company signal after correction",
        fontsize=10, color="#0b0b0b", labelpad=8,
    )
    ax.tick_params(axis="y", colors="#52514e", labelsize=9)
    ax.tick_params(axis="x", colors="#52514e")

    # y-limits with headroom for error bars + labels
    ymin = min(m - s for m, s in zip(means, stds))
    ymax = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(max(0.46, ymin - 0.02), ymax + 0.05)

    # recessive chrome
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "HK-gene ROC across batch-correction models\n"
        "(mean ± SD over 23 HK genes × 8 brain regions; lower = better)",
        fontsize=12, color="#0b0b0b", pad=14, loc="left",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    df = collect()
    summary = aggregate(df)
    print("=== 4-model mean ROC (best across companies) ===")
    print(f"{'model':<14} {'mean':>7} {'std':>7}")
    for k in ORDER:
        m, s = summary[k]
        print(f"{k:<14} {m:7.4f} {s:7.4f}")

    # save summary csv
    out = pd.DataFrame([
        {"model": k, "label": LABELS[k].replace("\n", " "),
         "mean_roc": summary[k][0], "std_roc": summary[k][1]}
        for k in ORDER
    ])
    out.to_csv(OUT_CSV, index=False)
    print(f"\n→ wrote summary: {OUT_CSV}")

    plot(summary, OUT_PNG)
    print(f"→ wrote plot   : {OUT_PNG}")


if __name__ == "__main__":
    main()