# %%
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
from rich import print
from scipy.stats import false_discovery_control
from scipy.sparse import csr_matrix


def to_sparse_int(arr):
    """把 numpy / float / 稀疏数组转换成 (n_cells, n_genes) 的 csr int 矩阵。

    - 输入若为稀疏矩阵：保持格式，只把 dtype 转成 int32。
    - 输入若为稠密 ndarray：rint 截断后构造 csr（counts 必须是离散整数）。
    - 用于把 decoder 抽样得到的 counts 写入 adata.layers["..."]，避免 h5ad
      把 float 数组按 ~8 字节/元素存储（counts 通常很稀疏 + 数值小）。
    """
    import numpy as np
    from scipy.sparse import csr_matrix, issparse
    if issparse(arr):
        return csr_matrix(arr).astype(np.int32)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # 防御性：误传 1D 时强制 reshape 成单行矩阵
        arr = arr.reshape(1, -1)
    return csr_matrix(np.rint(arr).astype(np.int32))

# ===== 命令行参数（可通过 argv 覆盖；在 Jupyter 中运行时自动使用默认值） =====
parser = argparse.ArgumentParser(description="SCVI DEG pipeline (RUVDEG mirror)")
parser.add_argument(
    "--input", "-i",
    default="/data7/mark/STG/dataset/snRNA/merge_SCH_new/six_datasets_4v3_500_1000gene/TH_downsampled_ratio.h5ad",
    help="输入 h5ad 文件路径",
)
parser.add_argument(
    "--outprefix", "-o",
    default="/home/junyichen/code/RUVAEDEG/scVImodel_output.h5ad",
    help="Output prefix (base path). All outputs are written as <prefix-base>.<suffix>.",
)
parser.add_argument(
    "--transform-batch",
    default="beirui",
    help="get_normalized_expression / posterior_predictive_sample 的 transform_batch 标签",
)
parser.add_argument(
    "--no-batch",
    action="store_true",
    help="不使用 batch key（batch_key=None，无 batch 校正）。此时 --transform-batch 会被忽略。",
)
parser.add_argument(
    "--no-cont-cov",
    action="store_true",
    help="不使用连续协变量（CONT_COVS=None），不向 SCVI 传入任何 covariate。",
)
parser.add_argument(
    "--n-latent", type=int, default=32,
    help="latent 维度（默认 32）",
)
parser.add_argument(
    "--n-layers", type=int, default=2,
    help="编码器/解码器层数（默认 2）",
)
parser.add_argument(
    "--no-compare", action="store_true",
    help="跳过末尾的 HKG dotplot（仍会训练、抽样并写 h5ad）",
)
args, _ = parser.parse_known_args()

INPUT_H5AD = os.path.abspath(args.input)
OUTPREFIX = os.path.abspath(args.outprefix)
N_LATENT = args.n_latent
N_LAYERS = args.n_layers
USE_BATCH = not args.no_batch
USE_CONT_COVS = not args.no_cont_cov
SHOW_COMPARE = not args.no_compare

# 使用 batch key 时 transform_batch 有效；不使用 batch 时必须为 None
TRANSFORM_BATCH = args.transform_batch if USE_BATCH else None

# Strip the extension so all outputs share one base path: <base>.<suffix>
OUTBASE = os.path.splitext(OUTPREFIX)[0]
OUTDIR = os.path.dirname(OUTBASE) or "."
os.makedirs(OUTDIR, exist_ok=True)

print(f"input h5ad : {INPUT_H5AD}")
print(f"out prefix : {OUTPREFIX}")
print(f"out base   : {OUTBASE}")
print(f"transform  : {TRANSFORM_BATCH}  |  n_latent={N_LATENT}  n_layers={N_LAYERS}"
      f"  |  use_batch={USE_BATCH}  use_cont_cov={USE_CONT_COVS}")

# ===== 数据读取 =====
adata_subset = sc.read_h5ad(INPUT_H5AD)

# transform_batch 必须是当前数据中实际存在的 company；缺失时使用细胞数最多的 company。
if USE_BATCH:
    if "company" not in adata_subset.obs.columns:
        raise KeyError("使用 batch 校正时，adata.obs 必须包含 'company' 列")

    company_counts = adata_subset.obs["company"].dropna().astype(str).value_counts()
    if company_counts.empty:
        raise ValueError("adata.obs['company'] 没有有效值，无法选择 transform_batch")

    if TRANSFORM_BATCH not in company_counts.index:
        requested_transform_batch = TRANSFORM_BATCH
        TRANSFORM_BATCH = company_counts.index[0]
        print(
            f"⚠ transform_batch={requested_transform_batch!r} 不在当前数据中，"
            f"改用细胞数最多的 company={TRANSFORM_BATCH!r} "
            f"(n={company_counts.iloc[0]})"
        )

print(f"resolved transform_batch: {TRANSFORM_BATCH!r}")

# 关键列 value counts —— 与 RUVDEG 一致的元信息
for col in ["status", "company", "celltype.L2", "sex", "sample", "region"]:
    if col in adata_subset.obs.columns:
        vc = adata_subset.obs[col].value_counts()
        print(f"\n{col} (n_unique={vc.size}):")
        print(vc.head(10))



# %%
# === 5. n_genes_on 协变量（mirror RUVDEG cell 4） ===
# 用 raw counts 计算每个细胞 >0 的基因数，标准化后作为连续 nuisance covariate。
# 与 RUVDEG 完全一致：mean/std 由本次数据估计，z-score 后写入 adata.obs["n_genes_on"]。
# 仅在 --no-cont-cov 未开启（USE_CONT_COVS=True）时才需要计算。
if USE_CONT_COVS:
    n_genes_on_raw = (adata_subset.layers["counts"] > 0).sum(axis=1).astype(np.float32)
    n_genes_on_mean = float(n_genes_on_raw.mean())
    n_genes_on_std  = float(n_genes_on_raw.std())
    if n_genes_on_std < 1e-8:
        raise ValueError("n_genes_on 没有足够变异，无法作为连续协变量")

    adata_subset.obs["n_genes_on"] = (
        (n_genes_on_raw - n_genes_on_mean) / n_genes_on_std
    ).astype(np.float32)

    print(f"raw mean / std    : {n_genes_on_mean:.1f} / {n_genes_on_std:.1f}")
    print(f"standardized range: [{adata_subset.obs['n_genes_on'].min():.3f}, "
          f"{adata_subset.obs['n_genes_on'].max():.3f}]")
else:
    print("⚠ --no-cont-cov: 跳过 n_genes_on 计算，不传入任何连续协变量")


# %%
adata_subset.X = adata_subset.layers["counts"].copy()

# %%
# === 6. SCVI setup_anndata（mirror RUVDEG covariate 设计） ===
#
# RUVDEG → SCVI 映射：
#   batch (技术)         -> batch_key="company"
#   n_genes_on (连续)    -> continuous_covariate_keys=["n_genes_on"]
#   group (生物学 status) -> NOT 注册：SCVI 无监督，biology 体现在 latent z；
#                              保留在 adata.obs["status"] 供下游 DEG 使用。
# 不传 labels_key：本数据只含单个 L2 细胞类型，且 labels_key 语义上是细胞类型标签、
# 不是 biology-of-interest，作 covariate-as-label 会语义错误。

# --no-batch 时 BATCH_KEY=None：scvi 不注册 batch 字段，模型完全不使用 batch
# --no-cont-cov 时 CONT_COVS=None：scvi 不注册任何连续协变量
BATCH_KEY = "company" if USE_BATCH else None
CONT_COVS = ["n_genes_on"] if USE_CONT_COVS else None

scvi.model.SCVI.setup_anndata(
    adata_subset,
    layer=None,                              # raw counts 在 adata.X
    batch_key=BATCH_KEY,                     # technical nuisance (RUVDEG `batch`)；--no-batch 时为 None
    labels_key=None,                         # no cell-type label here
    categorical_covariate_keys=None,         # no extra categorical nuisance
    continuous_covariate_keys=CONT_COVS,     # technical continuous nuisance (RUVDEG `n_genes_on`)
)
print(f"✓ scvi.model.SCVI.setup_anndata complete")
print(f"  manager uuid: {adata_subset.uns['_scvi_manager_uuid'][:8]}…")


# %%
# === 6b. Auto-select the GPU with the most free memory ===
if torch.cuda.is_available():
    gpu_memory = []
    for gpu_idx in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu_idx):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu_memory.append((gpu_idx, free_bytes, total_bytes))

    BEST_GPU, BEST_GPU_FREE, BEST_GPU_TOTAL = max(gpu_memory, key=lambda x: x[1])
    TRAIN_ACCELERATOR = "gpu"
    TRAIN_DEVICES = [BEST_GPU]
    print(
        f"✓ select gpu:{BEST_GPU}  free={BEST_GPU_FREE / 1024**3:.2f} GiB / "
        f"total={BEST_GPU_TOTAL / 1024**3:.2f} GiB"
    )
else:
    BEST_GPU = None
    TRAIN_ACCELERATOR = "cpu"
    TRAIN_DEVICES = 1
    print("CUDA not available, fallback to CPU")


# %%
model = scvi.model.SCVI(adata_subset, n_layers=N_LAYERS, n_latent=N_LATENT, gene_likelihood="zinb")
model.train(accelerator=TRAIN_ACCELERATOR, devices=TRAIN_DEVICES)


# %%
SCVI_LATENT_KEY = "X_scVI"
adata_subset.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

# %%
# === 8. DEG 比较（CON 为共同对照组） ===
# status 共 5 类：CON / CURES / CUSUS / CSRES / CSSUS
# 命名约定：C=control stress / S=CSDS stress，末尾 R/S = Resilient / Susceptible
#
# 比较列表：
#   RES_vs_CON    : CURES vs CON        (对照+抗压 vs 对照)
#   SUS_vs_CON    : CUSUS vs CON        (对照+易感 vs 对照)
#   CSDS_vs_CON   : (CSRES ∪ CSSUS) vs CON
#   CSRES_vs_CON  : CSRES vs CON        (CSDS+抗压 vs 对照)

# status = adata_subset.obs["status"].astype(str)
# print("status counts:")
# print(status.value_counts())

# adata_de = adata_subset.copy()
# adata_de.obs["status_3grp"] = np.where(
#     status.isin(["CSRES", "CSSUS"]), "CSDS",
#     np.where(status == "CON", "CON", status),
# ).astype(str)
# print("\nstatus_3grp counts:")
# print(adata_de.obs["status_3grp"].value_counts())

# comparisons = [
#     ("RES_vs_CON",    "status",      ["CURES"], "CON"),
#     ("SUS_vs_CON",    "status",      ["CUSUS"], "CON"),
#     ("CSDS_vs_CON",   "status_3grp", ["CSDS"],  "CON"),
#     ("CSRES_vs_CON",  "status",      ["CSRES"], "CON"),
# ]

# deg_results = {}
# for label, gb, g1, g2 in comparisons:
#     print(f"\n>>> {label}:  {g1} vs {g2}  (groupby={gb})")

#     # 检查 g1 / g2 组是否有细胞：某些样本可能缺少某个 status（如 ICTX 没有 CURES），
#     # 若任一组细胞数为 0 则跳过该比较
#     vc = adata_de.obs[gb].astype(str).value_counts()
#     g1_groups = [g1] if isinstance(g1, str) else list(g1)
#     missing = [g for g in g1_groups + [g2] if int(vc.get(g, 0)) == 0]
#     if missing:
#         print(f"  ⚠ skip {label}: 以下组没有细胞 -> {missing}  (细胞数: {dict(vc)})")
#         continue

#     deg = model.differential_expression(
#         adata=adata_de,
#         groupby=gb,
#         group1=g1,
#         group2=g2,
#         mode="change",
#         delta=0.1,
#         batch_correction=USE_BATCH,
#         silent=False,
#     )
#     deg = deg.copy()
#     deg.index.name = "gene"
#     deg = deg.reset_index()
#     deg["comparison"] = label
#     deg["fdr"] = false_discovery_control(deg["proba_de"].values)
#     deg_results[label] = deg

# deg_all = pd.concat(deg_results.values(), ignore_index=True)
# deg_csv = OUTBASE + ".deg_4comparisons.csv"
# deg_all.to_csv(deg_csv, index=False)
# print(f"✓ saved: {deg_csv}")

# summary = (
#     deg_all.groupby("comparison")
#     .agg(
#         n_genes=("gene", "size"),
#         n_de_fdr05=("fdr", lambda s: int((s < 0.05).sum())),
#         n_up_lfc1=(
#             "lfc_mean",
#             lambda s: int(((deg_all.loc[s.index, "fdr"] < 0.05) & (s > 1)).sum()),
#         ),
#         n_down_lfc1=(
#             "lfc_mean",
#             lambda s: int(((deg_all.loc[s.index, "fdr"] < 0.05) & (s < -1)).sum()),
#         ),
#     )
# )
# print("\n=== DEG 摘要 ===")
# print(summary)

# # %%
# deg_all[deg_all.lfc_median.abs() > 0.1].group1.value_counts()

# %%
# 未使用 batch key（--no-batch）时没有 batch registry，跳过；
# 未使用连续协变量（--no-cont-cov）时没有 extra_continuous_covs registry，跳过。
batch_idx = model.adata_manager.get_from_registry("batch") if USE_BATCH else None
cont_cov = (
    model.adata_manager.get_from_registry("extra_continuous_covs") if USE_CONT_COVS else None
)
if cont_cov is not None:
    cont_cov = torch.as_tensor(
        pd.DataFrame(cont_cov).to_numpy(dtype=np.float32),
        device=next(model.module.parameters()).device,
    )


# %%
if USE_BATCH:
    adata_subset.obs['batch_idx'] = batch_idx

# %%

if USE_BATCH:
    batch_map = dict(zip(adata_subset.obs['company'].values, adata_subset.obs['batch_idx'].values))

# %%
if USE_BATCH:
    batch_map

# %%
X_norm = model.get_normalized_expression(
    lib_size=1e4, transform_batch=TRANSFORM_BATCH
)
adata_subset.layers["scvi_nrom_counts"] = X_norm

# %%
# === SCVI reconstruct counts（真·后验预测抽样） ===

# 1) 拿到 batch / cov 输入
# 2) 从后验预测分布 p(x_hat | x) 抽样 n_samples 次
n_samples = 1   # 每细胞 1 个 reconstruction sample
reconstructed = model.posterior_predictive_sample(
    adata=adata_subset,
    n_samples=n_samples,
    batch_size=512,
    transform_batch=TRANSFORM_BATCH,
    silent=False,
)
# 返回 shape: (n_samples, n_cells, n_genes) → squeeze 到 (n_cells, n_genes)
reconstructed = np.asarray(reconstructed.todense())

print(f"reconstructed shape: {reconstructed.shape}  dtype={reconstructed.dtype}")
print(f"range: [{reconstructed.min()}, {reconstructed.max()}]  median={np.median(reconstructed):.2f}")
print(f"fraction non-zero: {(reconstructed > 0).mean():.3f}")

# 3) 写回 adata —— 稀疏 int 矩阵，节省 h5ad 存储
adata_subset.layers["scvi_reconstructed_counts"] = to_sparse_int(reconstructed)
adata_subset.write_h5ad(OUTBASE + ".h5ad", compression="gzip")
print(f"✓ saved: {OUTBASE}.h5ad")

if not SHOW_COMPARE:
    print("⚠ --no-compare: skip HKG dotplot")
    raise SystemExit(0)


# %%
hkg_priority = [
    "Aars", "Sars",
    "Polr2a", "Polr2f",
    "Psmd6", "Psmd7", "Psma5",
    "Rer1", "Ipo8", "Pop4", "Pes1",
    "Oaz1",
    "Rpl13a", "Rpl27", "Rps13", "Rps20",
    "Hprt1", "Gusb", "Ppia", "Ywhaz"
]

candidate_negative_control_genes = [
    # 在大规模跨组织分析中相对稳定
    "Oaz1",
    "Rps13",
    "Rps20",
    "Rpl27",

    # 小鼠脑中报告相对稳定的基础基因
    "Aars",
    "Polr2f",
    "Psmd6",
    "Psmd7",
    "Psma5",

    # RNA加工、核质运输与基础细胞维护
    "Ipo8",
    "Pop4",
    "Pes1",
    "Rer1",

    # 可作为补充候选，但必须在本数据中验证
    "Rpl13a",
    "Cyc1",
    "Sdha",
    "Ubc"
]

hkg_priority = candidate_negative_control_genes + hkg_priority

# %%
# ========== 9d. HKG dotplot: scvi_reconstructed_counts × status ==========

# 1) 准备 HKG 列表（保持 hkg_priority 顺序、去重、保留本数据中存在的）
seen = set()
hkg_genes = []
for g in hkg_priority:
    if g in adata_subset.var_names and g not in seen:
        seen.add(g)
        hkg_genes.append(g)
print(f"HKG available: {len(hkg_genes)} / {len(set(hkg_priority))}")

# 2) status 排序为 5 类固定顺序
#    用 set_categories 而非 reorder_categories：某些数据（如 ICTX）缺少某个 status
#    （例如没有 CURES），reorder_categories 要求新旧类别一致会报错；
#    set_categories 允许缺失类别，只强制固定顺序。
adata_subset.obs["status"] = adata_subset.obs["status"].astype("category")
adata_subset.obs["status"] = adata_subset.obs["status"].cat.set_categories(
    ["CON", "CURES", "CUSUS", "CSRES", "CSSUS"]
)

# 3) 共享颜色尺度：把 reconstructed_counts 一起看，避免每 panel 自适应
recon = adata_subset.layers["scvi_reconstructed_counts"]
recon_arr = recon.toarray() if hasattr(recon, "toarray") else np.asarray(recon)
vmin = float(np.percentile(recon_arr[recon_arr > 0], 5))
vmax = float(np.percentile(recon_arr, 95))
print(f"shared color: vmin={vmin:.2f}, vmax={vmax:.2f}")

# 4) 画 dotplot — color = mean expr, size = % cells expressing
sc.settings.figdir = OUTDIR
fig = sc.pl.dotplot(
    adata_subset,
    var_names=hkg_genes,
    groupby="sample",
    layer="scvi_reconstructed_counts",
    cmap="Reds",
    standard_scale=None,     # 用真实 counts 量级，跨组可比
    swap_axes=False,
    dendrogram=False,
    return_fig=True,
    show=False,
    save="scvi_hkg_recon_status.png",
)
fig_out = OUTBASE + ".hkg_recon_status.png"
fig.savefig(fig_out, bbox_inches="tight", dpi=200)
print(f"✓ saved: {fig_out}")

sc.settings.figdir = OUTDIR
fig = sc.pl.dotplot(
    adata_subset,
    var_names=hkg_genes,
    groupby="sample",
    layer="counts",
    cmap="Reds",
    standard_scale=None,     # 用真实 counts 量级，跨组可比
    swap_axes=False,
    dendrogram=False,
    return_fig=True,
    show=False,
    save="raw_status.png",
)
fig_out = OUTBASE + ".raw_status.png"
fig.savefig(fig_out, bbox_inches="tight", dpi=200)
print(f"✓ saved: {fig_out}")

adata_subset.layers["count_diff"] = to_sparse_int(
    adata_subset.layers["scvi_reconstructed_counts"] - adata_subset.layers["counts"]
)
fig = sc.pl.dotplot(
    adata_subset,
    var_names=hkg_genes,
    groupby="sample",
    layer="count_diff",
    cmap="bwr",
    expression_cutoff=-1000,
    standard_scale=None,     # 用真实 counts 量级，跨组可比
    swap_axes=False,
    dendrogram=False,
    return_fig=True,
    show=False,
    save="raw_status.png",
)
fig_out = OUTBASE + ".count_diff.png"
fig.savefig(fig_out, bbox_inches="tight", dpi=200)
print(f"✓ saved: {fig_out}")

