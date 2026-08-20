"""Compact per-region montage of the 4 status/recon/batchLR panels.

Cropping rules (auto-detected per image):
- panel 1 (raw):     keep full Y-axis labels; drop legend (right side)
- panel 2 (scVI):    drop Y-axis labels; drop legend
- panel 3 (harmony): drop Y-axis labels; drop legend
- panel 4 (bLR):     drop Y-axis labels; KEEP legend

Output:
    /data3/junyi/image/montage/{region}_raw_scvi_harmony_batchlr_compact.png
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OUT_DIR = Path("/data3/junyi/image/montage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["AMY", "HPF", "HY", "MB", "PFC", "STR", "TH", "iCTX"]

PANELS = [
    # (label, src_template, crop_left_skip, keep_legend)
    ("raw",            "/data3/junyi/six_datasets_4v3_500_1000gene_batchfinelr/{region}_batchscvi_full.raw_status.png",                   False, False),
    ("scVI",           "/data3/junyi/scvi/{region}_scVImodel.hkg_recon_status.png",                                                      True,  False),
    ("scVI + Harmony", "/data3/junyi/scvi_harmony/{region}_scviHarmony.hkg_recon_harmony_status.png",                                    True,  False),
    ("batchLR (D1)",   "/data3/junyi/six_datasets_4v3_500_1000gene_batchfinelr/{region}_batchscvi_full.hkg_batchpair_D1_counts.png",    True,  True),
]

BG          = (252, 252, 251)
TITLE_INK   = (11, 11, 11)
SUB_INK     = (82, 81, 78)
LABEL_BG    = (240, 239, 236)
LABEL_INK   = (11, 11, 11)
ACCENTS = [
    (42, 120, 214),   # blue   - raw
    (27, 175, 122),   # aqua   - scVI
    (235, 104, 52),   # orange - harmony
    (133, 100, 169),  # violet - batchLR
]


def detect_blocks(arr: np.ndarray, axis: int) -> list[tuple[int, int]]:
    """Return list of (start, end) of contiguous non-empty slices along `axis`."""
    if axis == 0:
        ink = ((arr < 240).any(axis=2)).sum(axis=1)
    else:
        ink = ((arr < 240).any(axis=2)).sum(axis=0)
    blocks = []
    i = 0
    n = len(ink)
    while i < n:
        if ink[i] > 0:
            j = i
            while j < n and ink[j] > 0:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def find_plot_borders(img: np.ndarray) -> tuple[int, int]:
    """Return (left_border_x, right_border_x) — the plot's vertical edges.

    The plot is bounded by tall vertical lines with very high ink density
    (the axis lines). We find the first and last column with ink > 500
    to anchor the plot region. Anything to the left is Y-axis labels;
    anything to the right is the legend block.
    """
    H, W = img.shape[:2]
    ink = ((img < 240).any(axis=2)).sum(axis=0)
    thick_lines = [x for x in range(W) if ink[x] > 500]
    if not thick_lines:
        return 0, W
    return thick_lines[0], thick_lines[-1]


def auto_crop(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) crop coords for a dotplot panel."""
    H, W = img.shape[:2]
    plot_left, plot_right = find_plot_borders(img)

    # Default crop = plot area + a small margin to include the border lines
    left = max(0, plot_left - 8)
    right = min(W, plot_right + 8)

    # Top crop = drop whitespace above first ink row
    ink_rows = ((img < 240).any(axis=2)).sum(axis=1)
    top = 0
    for y in range(H):
        if ink_rows[y] > 0:
            top = y
            break

    return left, top, right, H


def get_font(size: int):
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def make_montage(region: str) -> Path:
    panels = []
    for label, tpl, drop_yaxis, keep_legend in PANELS:
        path = Path(tpl.format(region=region))
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        left, top, right, bottom = auto_crop(arr)

        # If we want to KEEP the Y-axis labels, expand left to image edge
        if not drop_yaxis:
            left = 0
        # If we want to KEEP the legend, expand right to image edge
        if keep_legend:
            right = arr.shape[1]

        cropped = img.crop((left, top, right, bottom))
        panels.append((label, cropped))

    # All panels now share the same height (auto_crop keeps bottom = H, top differs
    # only if some panel has different layout — none here).
    h = max(p.size[1] for _, p in panels)
    # Pad shorter panels to align top edges
    aligned = []
    for label, p in panels:
        if p.size[1] < h:
            new_p = Image.new("RGB", (p.size[0], h), BG)
            new_p.paste(p, (0, 0))
            aligned.append((label, new_p))
        else:
            aligned.append((label, p))

    gap = 8
    margin_x = 16
    label_h = 36
    title_h = 64

    panel_widths = [p.size[0] for _, p in aligned]
    canvas_w = margin_x * 2 + sum(panel_widths) + gap * (len(aligned) - 1)
    canvas_h = title_h + label_h + h + 16

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    f_title = get_font(28)
    f_sub = get_font(15)
    title = f"{region}  —  batch correction comparison"
    sub = "raw  ·  scVI  ·  scVI + Harmony  ·  paired-batch scVI on D1 counts"
    bbox = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((canvas_w - (bbox[2] - bbox[0])) / 2, 18), title, font=f_title, fill=TITLE_INK)
    bbox2 = draw.textbbox((0, 0), sub, font=f_sub)
    draw.text(((canvas_w - (bbox2[2] - bbox2[0])) / 2, 50), sub, font=f_sub, fill=SUB_INK)

    f_label = get_font(18)

    y_img = title_h + label_h
    x = margin_x
    for i, (label, p) in enumerate(aligned):
        # accent stripe + label card
        draw.rectangle([x, title_h, x + p.size[0], title_h + 4], fill=ACCENTS[i])
        draw.rectangle([x, title_h + 10, x + p.size[0], title_h + label_h - 4], fill=LABEL_BG)
        bb = draw.textbbox((0, 0), label, font=f_label)
        lw = bb[2] - bb[0]
        draw.text(
            (x + (p.size[0] - lw) / 2, title_h + 10 + (label_h - 14 - (bb[3] - bb[1])) / 2 - 4),
            label, font=f_label, fill=LABEL_INK,
        )
        canvas.paste(p, (x, y_img))
        x += p.size[0] + gap

    out_path = OUT_DIR / f"{region}_raw_scvi_harmony_batchlr_compact.png"
    canvas.save(out_path, "PNG", optimize=True)
    return out_path, panel_widths


def main():
    for region in REGIONS:
        path, widths = make_montage(region)
        sz_mb = path.stat().st_size / 1e6
        img = Image.open(path)
        print(
            f"  {region:<5} → {path.name}  "
            f"({img.size[0]}x{img.size[1]}, {sz_mb:.1f} MB, "
            f"panel widths={widths})"
        )
    print(f"\n=== wrote 8 compact montages → {OUT_DIR} ===")


if __name__ == "__main__":
    main()