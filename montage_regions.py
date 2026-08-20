"""Concatenate the 4 status / recon / batchLR panels per brain region.

For each region, lay 4 panels horizontally:
   raw  →  scVI (recon)  →  scVI+Harmony (recon)  →  batchLR (D1 counts)

Source paths (verified byte-identical for raw_status across all 3 folders):
    raw:          six_datasets_4v3_500_1000gene_batchfinelr/{r}_batchscvi_full.raw_status.png
    scVI recon:   scvi/{r}_scVImodel.hkg_recon_status.png
    harm recon:   scvi_harmony/{r}_scviHarmony.hkg_recon_harmony_status.png
    batchLR:      six_datasets_4v3_500_1000gene_batchfinelr/{r}_batchscvi_full.hkg_batchpair_D1_counts.png

Output:
    /data3/junyi/image/montage/{region}_raw_scvi_harmony_batchlr.png
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("/data3/junyi/image/montage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["AMY", "HPF", "HY", "MB", "PFC", "STR", "TH", "iCTX"]

PANELS = [
    # (label, src_path_template with {region})
    ("raw",            "/data3/junyi/six_datasets_4v3_500_1000gene_batchfinelr/{region}_batchscvi_full.raw_status.png"),
    ("scVI",           "/data3/junyi/scvi/{region}_scVImodel.hkg_recon_status.png"),
    ("scVI + Harmony", "/data3/junyi/scvi_harmony/{region}_scviHarmony.hkg_recon_harmony_status.png"),
    ("batchLR (D1)",   "/data3/junyi/six_datasets_4v3_500_1000gene_batchfinelr/{region}_batchscvi_full.hkg_batchpair_D1_counts.png"),
]

# dataviz palette (light)
BG          = (252, 252, 251)
TITLE_INK   = (11, 11, 11)
SUB_INK     = (82, 81, 78)
LABEL_BG    = (240, 239, 236)
LABEL_INK   = (11, 11, 11)

# Per-panel accent stripe color (subtle bar above the panel name)
ACCENTS = [
    (42, 120, 214),   # blue   - raw
    (27, 175, 122),   # aqua   - scVI
    (235, 104, 52),   # orange - harmony
    (133, 100, 169),  # violet - batchLR
]


def get_font(size: int):
    """Try DejaVuSans (always present in Linux); fallback to default bitmap."""
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def make_montage(region: str) -> Path:
    paths = [Path(p.format(region=region)) for _, p in PANELS]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    imgs = [Image.open(p).convert("RGB") for p in paths]
    w, h = imgs[0].size

    gap = 24
    label_h = 56
    title_h = 100
    margin_x = 32

    canvas_w = margin_x * 2 + w * 4 + gap * 3
    canvas_h = title_h + label_h + h + margin_x

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    # Title
    f_title = get_font(34)
    f_sub = get_font(18)
    title = f"{region}  —  batch correction comparison"
    sub = ("raw  ·  scVI  ·  scVI + Harmony  ·  paired-batch scVI on D1 counts  "
           "(status / HKG distribution, all panels share the same scale)")
    bbox = draw.textbbox((0, 0), title, font=f_title)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) / 2, 24), title, font=f_title, fill=TITLE_INK)
    bbox2 = draw.textbbox((0, 0), sub, font=f_sub)
    sw = bbox2[2] - bbox2[0]
    draw.text(((canvas_w - sw) / 2, 70), sub, font=f_sub, fill=SUB_INK)

    f_label = get_font(20)
    f_panel = get_font(15)

    y_img = title_h + label_h
    x = margin_x
    for i, (img, (label, _path)) in enumerate(zip(imgs, PANELS)):
        # accent bar (4px tall) above the label
        bar_top = title_h
        bar_bot = title_h + 6
        draw.rectangle([x, bar_top, x + w, bar_bot], fill=ACCENTS[i])

        # label background card
        lbl_top = title_h + 14
        lbl_bot = title_h + label_h - 6
        draw.rectangle([x, lbl_top, x + w, lbl_bot], fill=LABEL_BG)
        # centered label text
        bb = draw.textbbox((0, 0), label, font=f_label)
        lw = bb[2] - bb[0]
        lh = bb[3] - bb[1]
        draw.text((x + (w - lw) / 2, lbl_top + (label_h - 14 - lh) / 2 - 4),
                  label, font=f_label, fill=LABEL_INK)

        # paste image
        canvas.paste(img, (x, y_img))

        x += w + gap

    out_path = OUT_DIR / f"{region}_raw_scvi_harmony_batchlr.png"
    canvas.save(out_path, "PNG", optimize=True)
    return out_path


def main():
    written = []
    for r in REGIONS:
        p = make_montage(r)
        size_mb = p.stat().st_size / 1e6
        print(f"  {r:<5} → {p.name}  ({size_mb:.1f} MB, {Image.open(p).size[0]}×{Image.open(p).size[1]})")
        written.append(p)
    print()
    print(f"=== wrote {len(written)} montages → {OUT_DIR} ===")


if __name__ == "__main__":
    main()