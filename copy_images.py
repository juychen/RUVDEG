"""Copy all image files from three source dirs into /data3/junyi/image,
preserving the relative path under each source folder name.

Layout after copy:
    /data3/junyi/image/<src_folder>/<original_relative_path>

Examples:
    /data3/junyi/image/six_datasets_4v3_500_1000gene_batchfinelr/AMY_batchscvi_full.count_diff.png
    /data3/junyi/image/scvi_harmony/dotplots/MB/abc.png
    /data3/junyi/image/scvi/dotplots/xyz.pdf
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

TARGET_ROOT = Path("/data3/junyi/image")
SOURCES = [
    Path("/data3/junyi/six_datasets_4v3_500_1000gene_batchfinelr"),
    Path("/data3/junyi/scvi_harmony"),
    Path("/data3/junyi/scvi"),
]
EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tif", ".tiff"}


def iter_images(source: Path):
    """Yield (src_path, rel_to_source) for every image under `source`."""
    for root, _dirs, files in os.walk(source):
        for f in files:
            if Path(f).suffix.lower() in EXTS:
                p = Path(root) / f
                yield p, p.relative_to(source)


def copy_one(src: Path, dst: Path) -> tuple[str, str]:
    """Copy src→dst, creating parents. If dst exists, rename with suffix."""
    if not dst.exists():
        shutil.copy2(src, dst)
        return "copied", ""
    # collision — append numeric suffix before extension
    stem, suf = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}__dup{i}{suf}")
        if not cand.exists():
            shutil.copy2(src, cand)
            return "copied-dup", cand.name
        i += 1


def main():
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    grand_total = 0
    grand_bytes = 0
    grand_dup = 0

    for source in SOURCES:
        if not source.is_dir():
            print(f"[SKIP] missing source: {source}", file=sys.stderr)
            continue

        dest_root = TARGET_ROOT / source.name
        dest_root.mkdir(parents=True, exist_ok=True)

        copied = 0
        dup = 0
        bytes_total = 0
        for src, rel in iter_images(source):
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            status, dup_name = copy_one(src, dst)
            copied += 1
            bytes_total += src.stat().st_size
            if status == "copied-dup":
                dup += 1
        grand_total += copied
        grand_bytes += bytes_total
        grand_dup += dup
        print(
            f"[{source.name}] {copied} files  "
            f"({bytes_total/1e6:.1f} MB){' · ' + str(dup) + ' dup' if dup else ''}"
        )

    print()
    print(
        f"=== TOTAL: {grand_total} files · "
        f"{grand_bytes/1e9:.2f} GB · "
        f"{grand_dup} duplicates renamed ==="
    )
    print(f"→ {TARGET_ROOT}")


if __name__ == "__main__":
    main()