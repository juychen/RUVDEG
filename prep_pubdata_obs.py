#!/usr/bin/env python
"""
prep_pubdata_obs.py

Add the obs columns (status, company, Model) that benchmarkbyauc.py requires to
h5ad files produced by scVI-style training pipelines (the new "format-B"
outputs under /data8/junyi/pubdata/transformed/), where these columns are
absent.

Per-dataset `company` (cell-type) key:
  GSE133549  -> nnet2
  GSE118767  -> meta_cell_line_demuxlet

`status` is set to a synthetic "CON" (these are training outputs, not
case-control designs) and `Model` to "CON_M" so the default
`args.model="CON_M"` filter in benchmarkbyauc.py keeps every cell.

Usage:
  python prep_pubdata_obs.py --h5ad PATH [--out PATH] [--celltype-key KEY]

Default --out is <PATH>.prepped.h5ad (writes next to the input; never
overwrites the training-output h5ad).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scanpy as sc


# Per-dataset cell-type key, indexed by the GSE id extracted from the
# directory/file basename. Public datasets in /data8/junyi/pubdata/.
DEFAULT_CELLTYPE_KEYS: dict[str, str] = {
    "GSE133549": "nnet2",
    "GSE118767": "meta_cell_line_demuxlet",
}

# Fallbacks tried in order if the per-dataset cell-type key is missing.
CELLTYPE_FALLBACKS: tuple[str, ...] = (
    "meta_cell_line_demuxlet",
    "meta_cell_line",
    "sample_id",
    "source_file",
    "protocol",
)


def detect_dataset_id(path: Path) -> str | None:
    """Extract the GSE id from a path or its basename (e.g. 'GSE118767')."""
    import re

    text = str(path)
    m = re.search(r"GSE\d+", text)
    return m.group(0) if m else None


def pick_company_key(adata, dataset_id: str | None, override: str | None) -> str:
    """Return the obs column name to use as `company`."""
    if override is not None:
        if override not in adata.obs:
            raise KeyError(
                f"--celltype-key {override!r} not in adata.obs "
                f"(available: {list(adata.obs.columns)})"
            )
        return override

    if dataset_id and dataset_id in DEFAULT_CELLTYPE_KEYS:
        candidate = DEFAULT_CELLTYPE_KEYS[dataset_id]
        if candidate in adata.obs:
            return candidate

    for k in CELLTYPE_FALLBACKS:
        if k in adata.obs:
            return k

    raise KeyError(
        f"Could not find any usable cell-type column for company. "
        f"Tried {[dataset_id and DEFAULT_CELLTYPE_KEYS.get(dataset_id), *CELLTYPE_FALLBACKS]}; "
        f"adata.obs has: {list(adata.obs.columns)}. "
        f"Pass --celltype-key to override."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad",
        required=True,
        type=Path,
        help="Path to the training-output h5ad file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output h5ad path (default: <input>.prepped.h5ad)",
    )
    parser.add_argument(
        "--celltype-key",
        type=str,
        default=None,
        help="Override the per-dataset cell-type column used as `company`",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="CON",
        help="Value to assign to obs['status'] (default: 'CON')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="CON_M",
        help="Value to assign to obs['Model'] (default: 'CON_M')",
    )
    args = parser.parse_args()

    h5ad_path: Path = args.h5ad.resolve()
    out_path: Path = (args.out or h5ad_path.with_suffix(h5ad_path.suffix + ".prepped.h5ad")).resolve()

    if not h5ad_path.is_file():
        print(f"[ERROR] input not found: {h5ad_path}", file=sys.stderr)
        sys.exit(1)

    if out_path == h5ad_path:
        print(
            f"[ERROR] --out must differ from --h5ad to avoid clobbering the "
            f"training output",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] reading {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[INFO] shape: {adata.shape}")

    # status (case-control): synthetic, since these are training outputs.
    if "status" not in adata.obs:
        adata.obs["status"] = args.status
        print(f"[INFO] added obs['status'] = {args.status!r}")
    else:
        print(f"[INFO] obs['status'] already present ({dict(adata.obs['status'].value_counts())})")

    # Model (filter key for benchmarkbyauc.py): keep all cells by default.
    if "Model" not in adata.obs:
        adata.obs["Model"] = args.model
        print(f"[INFO] added obs['Model'] = {args.model!r}")
    else:
        print(f"[INFO] obs['Model'] already present ({dict(adata.obs['Model'].value_counts())})")

    # company (cell-type key): per-dataset, with fallbacks.
    dataset_id = detect_dataset_id(h5ad_path)
    if "company" in adata.obs:
        print(
            f"[INFO] obs['company'] already present "
            f"({dict(adata.obs['company'].value_counts())})"
        )
    else:
        company_key = pick_company_key(adata, dataset_id, args.celltype_key)
        adata.obs["company"] = adata.obs[company_key].astype(str).values
        print(
            f"[INFO] added obs['company'] from {company_key!r} "
            f"({dict(adata.obs['company'].value_counts())})"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] writing {out_path}")
    adata.write_h5ad(out_path, compression="gzip")
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()