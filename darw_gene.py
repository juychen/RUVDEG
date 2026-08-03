#!/usr/bin/env python3
"""
按脑区绘制不同基因集的 dotplot（纵向拼接，统一色标），并支持基因数超过
max_genes 时自动分批输出。

用法示例:
    python darw_gene.py \
        --input  /data2st1/junyi/final.h5ad \
        --output /data2st2/junyi/code/sn/figures \
        --genegroup /data2st2/junyi/code/sn/data/All_degs_N_v0715FF.xlsx \
        --max-genes 50
"""
import os
import argparse
import glob
import warnings
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib as mpl

from collections import OrderedDict
from scipy.cluster.hierarchy import linkage as sp_linkage


# ---------- 函数：复用 sub_dict 画出不同基因集的 dotplot（纵向拼接，统一色标）---
def _shared_percentile_vminmax(sub_dict, genes, layer, p_lo=5, p_hi=95):
    """Compute shared vmin/vmax across all regions (mirroring ipynb 60_SummerizeHigenes).

    vmin = percentile_p(>0)   -- only positive entries
    vmax = percentile_p(all)  -- all entries
    Returns (vmin, vmax), or (None, None) if not computable.
    """
    arrs = []
    for reg, sub in sub_dict.items():
        if layer is None:
            X = sub.X
        elif layer in sub.layers:
            X = sub.layers[layer]
        else:
            continue
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        # Use only the genes of interest
        g_mask = np.isin(np.asarray(sub.var_names), np.asarray(genes))
        if g_mask.sum() == 0:
            continue
        X_sub = X[:, g_mask]
        arrs.append(X_sub.ravel())
    if not arrs:
        return None, None
    a = np.concatenate(arrs)
    pos = a[a > 0]
    vmin = float(np.percentile(pos, p_lo)) if pos.size else 0.0
    vmax = float(np.percentile(a, p_hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def dotplot_by_region(sub_dict, genes,
                      adata_full=None,
                      groupby='sample_status',
                      standard_scale='var',
                      figsize=(20, 70),
                      hspace=-0.1,
                      out_path='/data2st2/junyi/output/dotplot_by_region.pdf',
                      show=False,
                      max_genes=50,
                      layer=None,
                      vmin=None, vmax=None,
                      color_strategy='fixed'):
    """
    利用 sub_dict（已按脑切分好的子集 dict）绘制各脑区 dotplot，纵向拼接。
    参数:
        sub_dict : dict, {region: ad_subset} 由上方完整 cell 计算产生
        genes    : list, 基因名列表
        adata_full : 若提供，用于检查基因是否存在于 var_names
        groupby  : dotplot 的 x 轴分组列
        standard_scale : 'var' 或 'group' 或 None
        height_per_region, fig_width, hspace : 布局参数
        out_path : pdf 保存路径
        show     : 是否 plt.show()
        max_genes : int, 单个图中最多基因数。有效基因数超过该值时自动分批绘制，
                    每批仍用相同 figsize，输出文件名追加 _part01/_part02 等后缀。
        layer    : str 或 None，dotplot 可视化使用的 layer 名（None=adata.X）。
        vmin, vmax : 显式覆盖色标范围；仅在 standard_scale=None 时生效
        color_strategy : 'fixed' (默认, vmin/vmax 由调用方决定) /
                         'percentile' (跨 region 算 p5(>0)/p95, 和 ipynb 一致)
    返回:
        分批时返回 (list_of_figs, None)；否则返回 (fig, axes)
    """
    # 检查基因存在
    if adata_full is not None:
        genes_plot = [g for g in genes if g in adata_full.var_names]
    else:
        first_key = list(sub_dict.keys())[0]
        genes_plot = [g for g in genes if g in sub_dict[first_key].var_names]
    if len(genes_plot) < 2:
        raise ValueError(f"有效基因 {len(genes_plot)} 个，至少需 2 个")

    # ---- 有效基因数超过 max_genes 时：按 max_genes 个一批，递归分批绘制 ----
    if len(genes_plot) > max_genes:
        n_batches = int(np.ceil(len(genes_plot) / max_genes))
        print(f"有效基因 {len(genes_plot)} 个 > {max_genes}，分 {n_batches} 批绘制")
        figs = []
        for b in range(n_batches):
            batch_genes = genes_plot[b * max_genes:(b + 1) * max_genes]
            stem, ext = os.path.splitext(out_path)
            batch_out = f"{stem}_part{b + 1:02d}{ext}"
            fig, _ = dotplot_by_region(
                sub_dict, batch_genes, adata_full=adata_full,
                groupby=groupby, standard_scale=standard_scale,
                figsize=figsize, hspace=hspace,
                out_path=batch_out, show=show, max_genes=max_genes,
                layer=layer, vmin=vmin, vmax=vmax,
                color_strategy=color_strategy)
            figs.append(fig)
        return figs, None

    # ---- Resolve shared vmin/vmax ----
    if color_strategy == "percentile":
        vmin_calc, vmax_calc = _shared_percentile_vminmax(sub_dict, genes_plot, layer)
        if vmin is None:
            vmin = vmin_calc
        if vmax is None:
            vmax = vmax_calc
        print(f"shared color (percentile, p5(>0)/p95): vmin={vmin:.3f}, vmax={vmax:.3f}")
    # else: keep user-provided vmin/vmax (or hard 0/1 by default below)

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 1

    regions_sorted = sorted(sub_dict.keys())
    n = len(regions_sorted)

    fig, axes = plt.subplots(n, 1, figsize=figsize,
                             squeeze=False)

    for i, reg in enumerate(regions_sorted):
        ax = axes[i][0]
        sub = sub_dict[reg]
        stusmap = {
            "CON": "CTRL",
            "RES": "CURES",
            "SUS": "CUSUS",
            "CSDS":"CSSUS",
            "CSRES": "CSRES"

        }
        sub.obs['sample_status'] = sub.obs['sample'].astype(str) + '_' + sub.obs['status'].astype(str).map(stusmap)
        sub.obs['sample_status'] = sub.obs['sample_status'].astype('category')
        sc.tl.dendrogram(sub, groupby=groupby)
        sc.pl.dotplot(sub, var_names=genes_plot, groupby=groupby,
                      layer=layer,
                      standard_scale=standard_scale, dendrogram=True,
                      vmin=vmin, vmax=vmax, show=False, ax=ax)

    # 仅保留最后一个子图的 x 轴和图例
    last_row_y0 = axes[-1][0].get_position().bounds[1]
    main_bbox = axes[0][0].get_position().bounds
    main_x1 = main_bbox[0] + main_bbox[2]
    for a in fig.axes:
        bbox = a.get_position().bounds
        if bbox[1] > last_row_y0 + 0.01:
            a.set_xticks([])
            a.set_xticklabels([])
            a.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            a.xaxis.set_visible(False)
            a.spines['bottom'].set_visible(False)
            if bbox[0] > main_x1 - 0.01:
                a.set_visible(False)

    plt.subplots_adjust(hspace=hspace)
    plt.savefig(out_path,bbox_inches="tight")
    if show==True:
        plt.show()
    print(f"已保存 -> {out_path}")
    return fig, axes


def heatmap_by_region(sub_dict, genes,
                      adata_full=None,
                      groupby='sample_status',
                      figsize=(20, 70),
                      hspace=-0.05,
                      out_path='/data2st2/junyi/output/heatmap_by_region.pdf',
                      show=False,
                      max_genes=50,
                      layer=None,
                      cmap='viridis',
                      swap_axes=True,
                      show_dendrogram=True,
                      standard_scale=None,
                      vmin=None, vmax=None,
                      color_strategy='percentile'):
    """8 region 拼一起的真热图 (sc.pl.matrixplot)，色标按 percentile (和 ipynb 一致)。"""
    if adata_full is not None:
        genes_plot = [g for g in genes if g in adata_full.var_names]
    else:
        first_key = list(sub_dict.keys())[0]
        genes_plot = [g for g in genes if g in sub_dict[first_key].var_names]
    if len(genes_plot) < 2:
        raise ValueError(f"有效基因 {len(genes_plot)} 个，至少需 2 个")

    # ---- 批次 ----
    if len(genes_plot) > max_genes:
        n_batches = int(np.ceil(len(genes_plot) / max_genes))
        print(f"[heatmap] 有效基因 {len(genes_plot)} 个 > {max_genes}，分 {n_batches} 批绘制")
        figs = []
        for b in range(n_batches):
            batch_genes = genes_plot[b * max_genes:(b + 1) * max_genes]
            stem, ext = os.path.splitext(out_path)
            batch_out = f"{stem}_part{b + 1:02d}{ext}"
            fig = heatmap_by_region(
                sub_dict, batch_genes, adata_full=adata_full,
                groupby=groupby, figsize=figsize, hspace=hspace,
                out_path=batch_out, show=show, max_genes=max_genes,
                layer=layer, cmap=cmap, swap_axes=swap_axes,
                show_dendrogram=show_dendrogram,
                standard_scale=standard_scale, vmin=vmin, vmax=vmax,
                color_strategy=color_strategy)
            figs.append(fig)
        return figs

    # ---- Compute shared vmin/vmax if percentile ----
    if color_strategy == "percentile":
        vmin_calc, vmax_calc = _shared_percentile_vminmax(sub_dict, genes_plot, layer)
        if vmin is None:
            vmin = vmin_calc
        if vmax is None:
            vmax = vmax_calc
        print(f"[heatmap] shared color (percentile, p5(>0)/p95): vmin={vmin:.3f}, vmax={vmax:.3f}")

    # ---- Build sample_status on each subset ----
    stusmap = {
        "CON": "CTRL", "RES": "CURES", "SUS": "CUSUS",
        "CSDS": "CSSUS", "CSRES": "CSRES",
    }
    for reg, sub in sub_dict.items():
        sub.obs['sample_status'] = (sub.obs['sample'].astype(str) + '_' +
                                     sub.obs['status'].astype(str).map(stusmap))
        sub.obs['sample_status'] = sub.obs['sample_status'].astype('category')

    # ---- Concat all subsets into one AnnData (preserves region label) ----
    from anndata import concat as ad_concat
    pieces = []
    for reg in sorted(sub_dict.keys()):
        s = sub_dict[reg].copy()
        s.obs['__region__'] = reg
        pieces.append(s)
    ad_all = ad_concat(pieces, axis=0, join="outer", merge="same", label="region_chunk")

    # ---- Plot heatmap: region on rows, sample_status on columns ----
    sc.settings.set_figure_params(dpi=110, figsize=figsize)
    fig, axes = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax = axes[0][0]
    ax.clear()

    plot_kwargs = dict(
        var_names=genes_plot,
        groupby=['__region__', 'sample_status'],
        layer=layer,
        cmap=cmap,
        standard_scale=standard_scale,
        swap_axes=swap_axes,
        show=False,
        return_fig=True,
    )
    if not show_dendrogram:
        plot_kwargs['dendrogram'] = False
    if vmin is not None and standard_scale is None:
        plot_kwargs['vmin'] = vmin
    if vmax is not None and standard_scale is None:
        plot_kwargs['vmax'] = vmax

    fig = sc.pl.matrixplot(ad_all, **plot_kwargs)

    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"已保存 -> {out_path}")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="按脑区绘制基因集 dotplot，支持自动分批。")
    parser.add_argument("--input", required=True,
                        help="输入 h5ad 文件路径，或包含多个 h5ad 的目录（目录下所有 "
                             "*.h5ad 会被自动扫描、concatenate，并按文件名前缀识别 region）")
    parser.add_argument("--output", required=True,
                        help="输出目录，dotplot 图片将保存到该目录")
    parser.add_argument("--genegroup", default="/data2st2/junyi/code/sn/data/All_degs_N_v0715FF.xlsx",
                        help="可选：基因分组 Excel 文件（含 DML2 列与基因列）")
    parser.add_argument("--max-genes", type=int, default=50,
                        help="单个 dotplot 最多基因数，超过则自动分批（默认 50）")
    parser.add_argument("--groupby", default="sample_status",
                        help="dotplot 的 x 轴分组列（默认 sample_status）")
    parser.add_argument("--standard-scale", default="var",
                        help="standard_scale 参数：var / group / None（默认 var）")
    parser.add_argument("--figsize", nargs=2, type=float, default=[20, 70],
                        help="图片尺寸 (宽 高)，默认 20 70")
    parser.add_argument("--font", default="/data2st1/junyi/arial.ttf",
                        help="字体文件路径")
    parser.add_argument("--layer", default="scvi_reconstructed_counts_harmony",
                        help="dotplot 可视化使用的 layer 名，例如 counts / "
                             "scvi_reconstructed_counts / scvi_reconstructed_counts_harmony / "
                             "count_diff。不传则使用 adata.X（默认行为）。")
    parser.add_argument("--glob", default="*.h5ad",
                        help="当 --input 是目录时使用的 glob 模式（默认 *.h5ad）")
    parser.add_argument("--region-from", choices=["obs", "filename"], default="obs",
                        help="region 标签来源：obs=adata.obs['region']（默认）；"
                             "filename=从文件名第一个 '_' 之前提取")
    parser.add_argument("--color-strategy", choices=["fixed", "percentile"], default="percentile",
                        help="dotplot 共享色标策略：fixed=用 vmin/vmax（默认 0-1）；"
                             "percentile=跨 region 按 p5(>0)/p95 自动算（和 ipynb 一致）")
    parser.add_argument("--heatmap", action="store_true",
                        help="除了 dotplot，再额外画一张 8 region 拼一起的真热图 (sc.pl.matrixplot)")
    parser.add_argument("--heatmap-cmap", default="viridis",
                        help="热图 colormap（默认 viridis）")
    parser.add_argument("--vmin", type=float, default=None,
                        help="显式 vmin（仅在 standard-scale=None 且 --color-strategy=fixed 时生效）")
    parser.add_argument("--vmax", type=float, default=None,
                        help="显式 vmax（仅在 standard-scale=None 且 --color-strategy=fixed 时生效）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 线程与警告设置
    default_n_threads = 8
    os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
    os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
    os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

    warnings.filterwarnings("ignore")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    # 字体与 matplotlib 全局设置
    if os.path.exists(args.font):
        fm.fontManager.addfont(args.font)
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['font.sans-serif'] = ["Arial"]

    # 输出目录
    os.makedirs(args.output, exist_ok=True)

    # ---------- 读取输入 adata（支持单文件或目录批量拼接）----------
    if os.path.isdir(args.input):
        # Directory mode: scan *.h5ad, concatenate
        pattern = os.path.join(args.input, args.glob)
        h5ad_files = sorted(glob.glob(pattern))
        if not h5ad_files:
            raise FileNotFoundError(
                f"在目录 {args.input!r} 下没有找到匹配 {args.glob!r} 的 h5ad 文件"
            )
        print(f"[input dir] {args.input}: 找到 {len(h5ad_files)} 个 h5ad")
        for f in h5ad_files:
            print(f"  - {os.path.basename(f)}")

        adatas = []
        for f in h5ad_files:
            a = sc.read(f)
            adatas.append(a)
            print(f"    {os.path.basename(f)}: shape={a.shape}")

        # Concatenate (join='outer' to keep union of genes; obs columns are unioned)
        ad_all = sc.concat(
            adatas, axis=0, join="outer", merge="same",
            label="region_from_file", keys=[os.path.basename(f) for f in h5ad_files],
            index_unique=None,
        )
        print(f"拼接后: {ad_all.shape}")

        # If obs['region'] missing, fill from filename prefix
        if "region" not in ad_all.obs.columns and args.region_from == "obs":
            args.region_from = "filename"

        if args.region_from == "filename" or "region" not in ad_all.obs.columns:
            def _region_from_filename(fname):
                base = os.path.basename(fname)
                # 形如 iCTX_scviHarmony.h5ad -> iCTX
                stem = os.path.splitext(base)[0]
                return stem.split("_")[0]
            regions = [_region_from_filename(f) for f in h5ad_files]
            # 为每个 cell 标 region（用 label 索引 + keys 长度推断）
            region_per_cell = np.empty(ad_all.n_obs, dtype=object)
            cursor = 0
            for a, reg in zip(adatas, regions):
                n = a.n_obs
                region_per_cell[cursor:cursor + n] = reg
                cursor += n
            ad_all.obs["region"] = pd.Categorical(region_per_cell)
            print(f"[region] from filename: {sorted(ad_all.obs['region'].unique().tolist())}")
    else:
        ad_all = sc.read(args.input)
        print(f"已读取 adata: {args.input}, shape={ad_all.shape}")

    # ---------- 校验可视化 layer ----------
    if args.layer is None:
        print(f"[layer] 使用 adata.X（默认）")
    else:
        if args.layer not in ad_all.layers:
            raise ValueError(
                f"--layer={args.layer!r} 不在 adata.layers 中。"
                f" 可选: {list(ad_all.layers.keys())}"
            )
        print(f"[layer] 使用 adata.layers[{args.layer!r}]")

    # ---------- 内参基因 ----------
    hkgene = ['Rpl13a', 'Rplp0', 'Rps18', 'Rps27a', 'Rps23', 'Rps29', 'Rpl32',
              'Eef1a1', 'Eef2', 'Ppia', 'Hsp90ab1', 'Psma1', 'Cyc1', 'Ubb', 'Psmb2']
    hkgene_present = [g for g in hkgene if g in ad_all.var_names]
    if 'hkgene_expression' not in ad_all.obsm_keys():
        ad_all.obsm['hkgene_expression'] = ad_all[:, hkgene_present].X.toarray()

    # ---------- 按脑区切分 ----------
    region_data = OrderedDict()
    sub_dict = {}
    for reg in sorted(ad_all.obs['region'].unique()):
        print(f"Processing region: {reg}")
        sub = ad_all[ad_all.obs['region'] == reg]
        sub_dict[reg] = sub

    # ---------- 基因集 ----------
    rps_genes = [gene for gene in ad_all.var_names if gene.startswith('Rps')]
    mt_genes = [gene for gene in ad_all.var_names if gene.startswith('mt-')]

    # ---------- 对各基因集依次调用 dotplot_by_region ----------
    gene_sets = [
        ('hkgene', hkgene_present),
        ('rps',    rps_genes),
        ('mt',     mt_genes)
    ]
    for name, genes in gene_sets:
        print(f"\n===== {name} =====")
        dotplot_by_region(
            sub_dict, genes, adata_full=ad_all,
            groupby=args.groupby,
            standard_scale=args.standard_scale,
            figsize=tuple(args.figsize),
            max_genes=args.max_genes,
            layer=args.layer,
            color_strategy=args.color_strategy,
            vmin=args.vmin, vmax=args.vmax,
            out_path=os.path.join(args.output, f'dotplot_{name}_per_region.pdf'))
        # 真热图（可选，--heatmap）
        if args.heatmap:
            heatmap_by_region(
                sub_dict, genes, adata_full=ad_all,
                groupby=args.groupby,
                figsize=tuple(args.figsize),
                max_genes=args.max_genes,
                layer=args.layer,
                cmap=args.heatmap_cmap,
                standard_scale=None,  # 真热图用 raw 表达，色标按 percentile
                color_strategy="percentile",
                out_path=os.path.join(args.output, f'heatmap_{name}_per_region.pdf'))

    # ---------- 按 DML2 基因分组绘制（可选） ----------
    if args.genegroup is not None:
        df_genegroup = pd.read_excel(args.genegroup)

        dml2_col = next((c for c in df_genegroup.columns if c.lower() == 'dml2'), None)
        if dml2_col is None:
            raise ValueError("df_genegroup 中找不到 DML2 列")

        gene_col_candidates = [
            c for c in df_genegroup.columns
            if c.lower() in {'gene', 'gene_symbol', 'genesymbol', 'symbol', 'gene_name', 'genename'}
        ]
        if len(gene_col_candidates) > 0:
            gene_col = gene_col_candidates[0]
        else:
            non_dml2_cols = [c for c in df_genegroup.columns if c != dml2_col]
            if len(non_dml2_cols) == 0:
                raise ValueError("df_genegroup 中没有可用的基因列")
            gene_col = non_dml2_cols[0]
            print(f"[Warning] 未识别标准基因列名，临时使用: {gene_col}")

        if 'sample_group' not in ad_all.obs.columns:
            ad_all.obs['sample_group'] = ad_all.obs['sample'].astype(str) + '_' + ad_all.obs['status'].astype(str)
        ad_all.obs['sample_group'] = ad_all.obs['sample_group'].astype('category')

        df_tmp = df_genegroup[[dml2_col, gene_col]].copy()
        df_tmp = df_tmp.dropna(subset=[dml2_col, gene_col])
        df_tmp[dml2_col] = df_tmp[dml2_col].astype(str).str.strip()
        df_tmp[gene_col] = df_tmp[gene_col].astype(str).str.strip()

        var_set = set(ad_all.var_names)
        min_genes = 3

        group_summary = []
        for dml2_name, sub_df in df_tmp.groupby(dml2_col):
            raw_genes = pd.unique(sub_df[gene_col]).tolist()
            genes_present = [g for g in raw_genes if g in var_set]

            if len(genes_present) < min_genes:
                print(f"[Skip] {dml2_name}: 匹配到基因数 {len(genes_present)} < {min_genes}")
                continue

            safe_name = ''.join(ch if (ch.isalnum() or ch in ['_', '-']) else '_' for ch in dml2_name)

            # 按脑区分开画、纵向拼接、统一色标
            dotplot_by_region(
                sub_dict, genes_present, adata_full=ad_all,
                groupby=args.groupby,
                standard_scale=args.standard_scale,
                figsize=tuple(args.figsize),
                max_genes=args.max_genes,
                layer=args.layer,
                color_strategy=args.color_strategy,
                vmin=args.vmin, vmax=args.vmax,
                show=False,
                out_path=os.path.join(args.output, f'dotplot_DML2_{safe_name}_per_region.pdf'))
            if args.heatmap:
                heatmap_by_region(
                    sub_dict, genes_present, adata_full=ad_all,
                    groupby=args.groupby,
                    figsize=tuple(args.figsize),
                    max_genes=args.max_genes,
                    layer=args.layer,
                    cmap=args.heatmap_cmap,
                    standard_scale=None,
                    color_strategy="percentile",
                    out_path=os.path.join(args.output, f'heatmap_DML2_{safe_name}_per_region.pdf'))

            group_summary.append({
                'DML2': dml2_name,
                'n_genes_raw': len(raw_genes),
                'n_genes_present': len(genes_present),
            })

        df_dml2_summary = pd.DataFrame(group_summary).sort_values('n_genes_present', ascending=False)
        print(f"完成 DML2 基因集处理: {len(df_dml2_summary)} 个分组")
        print(df_dml2_summary.to_string())


if __name__ == '__main__':
    main()


