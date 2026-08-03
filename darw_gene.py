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
def dotplot_by_region(sub_dict, genes,
                      adata_full=None,
                      groupby='sample_status',
                      standard_scale='var',
                      figsize=(20, 70),
                      hspace=-0.1,
                      out_path='/data2st2/junyi/output/dotplot_by_region.pdf',
                      show=False,
                      max_genes=50,
                      layer=None):
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
                layer=layer)
            figs.append(fig)
        return figs, None

    regions_sorted = sorted(sub_dict.keys())
    n = len(regions_sorted)

    vmin_g, vmax_g = 0, 1
    print(f"全局 vmin={vmin_g:.4f}, vmax={vmax_g:.4f}")

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
                      vmin=vmin_g, vmax=vmax_g, show=False, ax=ax)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="按脑区绘制基因集 dotplot，支持自动分批。")
    parser.add_argument("--input", required=True,
                        help="输入 h5ad 文件路径（adata）")
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
    parser.add_argument("--layer", default=None,
                        help="dotplot 可视化使用的 layer 名，例如 counts / "
                             "scvi_reconstructed_counts / scvi_reconstructed_counts_harmony / "
                             "count_diff。不传则使用 adata.X（默认行为）。")
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

    # ---------- 读取输入 adata ----------
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
            out_path=os.path.join(args.output, f'dotplot_{name}_per_region.pdf'))

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
                show=False,
                out_path=os.path.join(args.output, f'dotplot_DML2_{safe_name}_per_region.pdf'))

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


