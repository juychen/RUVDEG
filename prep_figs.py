"""Prepare figures for the minimal PPTX: trim whitespace, strip redundant titles."""
from PIL import Image
import numpy as np
import os

OUT = "slide_figs"
os.makedirs(OUT, exist_ok=True)


def load_gray(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    # "ink" = any pixel meaningfully darker/more saturated than white
    ink = (255 - a.min(axis=2)) > 12
    return im, ink


def autotrim(im, ink, pad=6):
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0:
        return im
    t, b = rows[0], rows[-1]
    l, r = cols[0], cols[-1]
    W, H = im.size
    return im.crop((max(0, l - pad), max(0, t - pad),
                    min(W, r + pad + 1), min(H, b + pad + 1)))


def strip_title(im, ink, search_frac=0.45, min_gap=12):
    """Drop a title band: the top ink band followed by a tall blank gap."""
    H = ink.shape[0]
    rowink = ink.any(axis=1)
    limit = int(H * search_frac)
    i = 0
    while i < limit and not rowink[i]:
        i += 1
    # end of first ink band
    while i < limit and rowink[i]:
        i += 1
    gap_start = i
    while i < limit and not rowink[i]:
        i += 1
    if i - gap_start >= min_gap and i < limit:
        return im.crop((0, gap_start, im.size[0], im.size[1])), True
    return im, False


def prep(src, dst, do_strip_title=False, crop=None):
    im, ink = load_gray(src)
    if crop:
        im = im.crop(crop)
        a = np.asarray(im.convert("RGB")).astype(np.int16)
        ink = (255 - a.min(axis=2)) > 12
    if do_strip_title:
        im, hit = strip_title(im, ink)
        a = np.asarray(im.convert("RGB")).astype(np.int16)
        ink = (255 - a.min(axis=2)) > 12
    im = autotrim(im, ink)
    im.save(os.path.join(OUT, dst))
    print(f"{dst:32s} {im.size}  aspect={im.size[0]/im.size[1]:.2f}")


six = "ppt_figs/fig2_results_6panel.png"
W, H = Image.open(six).size

# Slide 3: bio latent z + UV latent w (bottom-left two panels)
prep(six, "latents.png", crop=(0, H // 2, 1088, H))
# Slide 4: volcano
prep(six, "volcano.png", crop=(0, 0, W // 3, H // 2), do_strip_title=True)

# Slide 2: training loss
prep("ppt_figs/fig1_training_loss.png", "loss.png")

# Slide 4: DEG dotplots before/after (titles go in slide text)
prep("ppt_figs/fig3_deg_before_uv.png", "deg_before.png", do_strip_title=True)
prep("ppt_figs/fig4_deg_after_uv.png", "deg_after.png", do_strip_title=True)

# Slide 5: HKG before/after
prep("ppt_figs/fig5_hkg_before_uv.png", "hkg_before.png", do_strip_title=True)
prep("ppt_figs/fig6_hkg_after_uv.png", "hkg_after.png", do_strip_title=True)
