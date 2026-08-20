"""Grouped bar plot: HKG delta_best mean per (region × model).

delta_best = best_corr - best_raw, computed per HK gene in
    benchmarkbyauc.py.  Lower mean = better (HKG stays closer to 0.5
    after batch correction).

Models present in /data3/junyi/benchmark_results:
    - scVImodel        (scVI raw -> log1p corrected)
    - scviHarmony      (scVI -> Harmony)
    - batchscvi_full   (paired-batch scVI, full genes)
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_GLOB = "/data3/junyi/benchmark_results/*/*_hkg_auc.csv"
OUT_PNG  = Path("/home/junyichen/code/RUVAEDEG/hkg_delta_best_barplot.png")
OUT_CSV  = Path("/home/junyichen/code/RUVAEDEG/hkg_delta_summary.csv")

# dataviz palette (light) — categorical slots 1,2,3 from palette.md
PALETTE = {
    "scVImodel":      "#2a78d6",  # blue
    "scviHarmony":    "#1baf7a",  # aqua
    "batchscvi_full": "#eb6834",  # orange
}
MODEL_ORDER = ["scVImodel", "scviHarmony", "batchscvi_full"]


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
        per_gene = df.drop_duplicates("gene")[["gene", "best_raw", "best_corr", "delta_best"]]
        rows.append({
            "region": region,
            "model":  model,
            "mean_delta_best": per_gene["delta_best"].mean(),
            "mean_best_raw":   per_gene["best_raw"].mean(),
            "mean_best_corr":  per_gene["best_corr"].mean(),
            "n_genes":         len(per_gene),
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out_png: Path):
    regions = sorted(df["region"].unique())
    n_reg = len(regions)
    n_mod = len(MODEL_ORDER)
    bar_w = 0.26
    x = np.arange(n_reg)

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # horizontal baseline at y=0 — separates "improved" (good) from "worsened" (bad)
    ax.axhline(0, color="#383835", lw=1.0, zorder=1)

    # shaded "good" band (negative delta_best = HKG got more stable after correction)
    ymin, ymax = df["mean_delta_best"].min(), df["mean_delta_best"].max()
    pad = max(0.01, (ymax - ymin) * 0.10)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axhspan(ymin - pad, 0, color="#1baf7a", alpha=0.05, zorder=0)

    for i, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("region").reindex(regions)
        vals = sub["mean_delta_best"].values
        offsets = x + (i - (n_mod - 1) / 2) * bar_w
        bars = ax.bar(
            offsets, vals, width=bar_w,
            color=PALETTE[model],
            edgecolor="#0b0b0b", linewidth=0.6,
            label=model, zorder=3,
        )
        # direct labels on every bar (relief for sub-3:1 aqua slot)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + (0.003 if v >= 0 else -0.006),
                f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color="#0b0b0b",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=10, color="#0b0b0b")
    ax.set_xlabel("Brain region", fontsize=11, color="#0b0b0b", labelpad=8)
    ax.set_ylabel(
        "HKG mean ΔAUC (best_corr − best_raw)\n← lower is better · HKG more stable after correction",
        fontsize=10.5, color="#0b0b0b", labelpad=8,
    )
    ax.tick_params(axis="y", colors="#52514e", labelsize=9)
    ax.tick_params(axis="x", colors="#52514e")

    # recessive spines
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")

    # light horizontal gridlines only
    ax.yaxis.grid(True, color="#e1e0d9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "HK-gene AUC drop after batch correction, per region × model\n"
        "(23 HK genes, CON_M subset; bar below 0 ⇒ correction made HKG more batch-invariant)",
        fontsize=11.5, color="#0b0b0b", pad=12, loc="left",
    )

    leg = ax.legend(
        loc="upper right", frameon=False, fontsize=9.5,
        handlelength=1.6, handletextpad=0.6, borderaxespad=0.6,
        labelcolor="#0b0b0b",
    )
    # legend swatch outline
    for h in leg.legend_handles:
        h.set_edgecolor("#0b0b0b")
        h.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    df = collect()
    df.to_csv(OUT_CSV, index=False)
    print(df.round(4).to_string(index=False))
    print()
    print("=== mean Δ_best across regions (lower = better) ===")
    agg = df.groupby("model")["mean_delta_best"].agg(["mean", "std", "count"]).round(4)
    print(agg.to_string())
    plot(df, OUT_PNG)
    print(f"\n→ wrote {OUT_PNG}")
    print(f"→ wrote {OUT_CSV}")


if __name__ == "__main__":
    main()