# %%
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import scvi
import seaborn as sns
import torch
from rich import print

# ===== 命令行参数（可通过 argv 覆盖；在 Jupyter 中运行时自动使用默认值） =====
parser = argparse.ArgumentParser(description="SCVI + Harmony DEG pipeline (RUVDEG mirror)")
parser.add_argument(
    "--input", "-i",
    default="/data7/mark/STG/dataset/snRNA/merge_SCH_new/six_datasets_4v3_500_1000gene/TH_downsampled_ratio.h5ad",
    help="输入 h5ad 文件路径",
)
parser.add_argument(
    "--outprefix", "-o",
    default="/home/junyichen/code/RUVAEDEG/scviHarmony_output.h5ad",
    help="Output prefix (base path). All outputs are written as <prefix-base>.<suffix>.",
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
# ===== Harmony 特有参数 =====
parser.add_argument(
    "--harmony-batch",
    default="company",
    help="run_harmony 的 vars_use：用于 batch 校正的 obs 列名（默认 company）",
)
parser.add_argument(
    "--nclust", type=str, default="celltype.L2",
    help="Harmony 内部软聚类数：传整数（如 3）直接指定；传 obs 列名（如 celltype.L2）取该列 unique 类别数作为 nclust。默认 celltype.L2。",
)
parser.add_argument(
    "--lamb", type=float, default=0.3,
    help="Harmony ridge 回归惩罚 lambda（默认 0.3）",
)
parser.add_argument(
    "--max-iter-harmony", type=int, default=20,
    help="Harmony 最大迭代次数（默认 20）",
)
args, _ = parser.parse_known_args()

INPUT_H5AD = os.path.abspath(args.input)
OUTPREFIX = os.path.abspath(args.outprefix)
N_LATENT = args.n_latent
N_LAYERS = args.n_layers
# 本 pipeline 的 SCVI 不用 batch key：batch 校正交给 Harmony（run_harmony 的 vars_use）
# 因此 batch_key 恒为 None，transform_batch 恒为 None。
USE_BATCH = False
USE_CONT_COVS = not args.no_cont_cov

HARMONY_BATCH = args.harmony_batch
NCLUST_RAW = args.nclust
LAMB = args.lamb
MAX_ITER_HARMONY = args.max_iter_harmony

TRANSFORM_BATCH = None

# Strip the extension so all outputs share one base path: <base>.<suffix>
OUTBASE = os.path.splitext(OUTPREFIX)[0]
OUTDIR = os.path.dirname(OUTBASE) or "."
os.makedirs(OUTDIR, exist_ok=True)

print(f"input h5ad : {INPUT_H5AD}")
print(f"out prefix : {OUTPREFIX}")
print(f"out base   : {OUTBASE}")
print(f"n_latent={N_LATENT}  n_layers={N_LAYERS}  use_cont_cov={USE_CONT_COVS}")
print(f"scvi       : batch_key=None (batch 校正由 Harmony 完成)")
print(f"harmony    : vars_use={HARMONY_BATCH}  nclust={NCLUST_RAW}  lamb={LAMB}  max_iter={MAX_ITER_HARMONY}")

# ===== 数据读取 =====
adata_subset = sc.read_h5ad(INPUT_H5AD)

# 关键列 value counts —— 与 RUVDEG 一致的元信息
for col in ["status", "company", "celltype.L2", "sex", "sample", "region"]:
    if col in adata_subset.obs.columns:
        vc = adata_subset.obs[col].value_counts()
        print(f"\n{col} (n_unique={vc.size}):")
        print(vc.head(10))

# %%
# === 3. 数据结构探索（SCVI setup 前置） ===
print(f"shape: {adata_subset.shape}")
print(f"X dtype : {adata_subset.X.dtype}   X range: [{adata_subset.X.min()}, {adata_subset.X.max()}]")
print(f"layers  : {[k for k in adata_subset.layers.keys() if k is not None]}")

# 关键列 value counts —— 与 RUVDEG 一致的元信息
for col in ["status", "company", "celltype.L2", "sex", "sample", "region"]:
    if col in adata_subset.obs.columns:
        vc = adata_subset.obs[col].value_counts()
        print(f"\n{col} (n_unique={vc.size}):")
        print(vc.head(10))

# 解析 nclust（type=str 所以纯数字也是字符串）：
#   - 纯数字字符串 -> int（如 --nclust 3）
#   - obs 列名     -> 取该列 unique 类别数（如默认 celltype.L2）
#   - 其他         -> 报错提示
if isinstance(NCLUST_RAW, str) and NCLUST_RAW.isdigit():
    NCLUST = int(NCLUST_RAW)
elif isinstance(NCLUST_RAW, str) and NCLUST_RAW in adata_subset.obs.columns:
    NCLUST = int(adata_subset.obs[NCLUST_RAW].nunique())
    print(f"  nclust from obs['{NCLUST_RAW}'] = {NCLUST}")
else:
    raise ValueError(f"--nclust 必须是整数或 obs 列名，收到: {NCLUST_RAW!r}")

print(f"nclust: {NCLUST}")

# %%
adata_subset.obs.status.value_counts()

# %%
adata_subset.layers["counts"]

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
# 本 pipeline：batch 校正由 Harmony 完成，SCVI 不注册 batch key。
#   batch (技术)         -> None（SCVI 不用；run_harmony vars_use 用 HARMONY_BATCH）
#   n_genes_on (连续)    -> continuous_covariate_keys=["n_genes_on"]（--no-cont-cov 时 None）
#   group (生物学 status) -> NOT 注册：SCVI 无监督，biology 体现在 latent z；
#                              保留在 adata.obs["status"] 供下游 DEG 使用。
# 不传 labels_key：本数据只含单个 L2 细胞类型，且 labels_key 语义上是细胞类型标签、
# 不是 biology-of-interest，作 covariate-as-label 会语义错误。

# SCVI 永远不用 batch（batch_key=None）；--no-cont-cov 时 CONT_COVS=None
BATCH_KEY = None
CONT_COVS = ["n_genes_on"] if USE_CONT_COVS else None

scvi.model.SCVI.setup_anndata(
    adata_subset,
    layer=None,                              # raw counts 在 adata.X
    batch_key=BATCH_KEY,                     # SCVI 不用 batch（Harmony 负责校正）
    labels_key=None,                         # no cell-type label here
    categorical_covariate_keys=None,         # no extra categorical nuisance
    continuous_covariate_keys=CONT_COVS,     # technical continuous nuisance (RUVDEG `n_genes_on`)；--no-cont-cov 时为 None
)
print(f"✓ scvi.model.SCVI.setup_anndata complete")
print(f"  manager uuid: {adata_subset.uns['_scvi_manager_uuid'][:8]}…")


# %%
# === 6b. 自动选择最空闲的 GPU ===

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
adata_subset.obsm['X_pcaBACK'] = adata_subset.obsm['X_pca'].copy()


# %%
adata_subset.obsm['X_pca'] = adata_subset.obsm[SCVI_LATENT_KEY].copy()

# %%
adata_subset.obsm['X_pca'].shape

# %%
adata_subset.obs.columns

# %%
from harmonypy import run_harmony
import pandas as pd

# %%
ho = run_harmony(adata_subset.obsm['X_pca'],
                 meta_data=pd.DataFrame(adata_subset.obs[HARMONY_BATCH]),
                 vars_use=HARMONY_BATCH,
                 random_state=0,
                 nclust=NCLUST,
                 lamb=LAMB,
                 max_iter_harmony=MAX_ITER_HARMONY)
# sce.pp.harmony_integrate(adata_concat, 'batch',
#                          **{'nclust':args.nclust,'lamb':args.lamb,'epsilon_harmony':0,'epsilon_cluster':0,'max_iter_harmony':10})
## lamb 11-12,lamb:0.3 is good
adata_subset.obsm['X_harmony'] = ho.Z_corr
# adata_concat.obs['umap_0'] = adata_concat.obsm['X_umap'][:, 0]
# adata_concat.obs['umap_1'] = adata_concat.obsm['X_umap'][:, 1]
# 
adata_subset.write_h5ad(OUTBASE + ".h5ad", compression="gzip")
import torch
import numpy as np


def decode_from_z(model, adata, z, library_size=None, device="cpu"):
    """用自定义 latent z 直接跑 decoder（generative），返回 ZINB 分布。

    Parameters
    ----------
    model : scvi.model.SCVI（训练后）
    adata : AnnData（与模型 setup 结构一致）
    z : (n_cells, n_latent) 数组 —— 自定义 latent，例如：
        - adata_subset.obsm['X_harmony']（harmony 校正后的 z）
        - adata_subset.obsm['X_scVI']（latent mean，即 q(z|x) 的均值，无采样噪声）
    library_size : (n_cells,) 或 None
        None 时用观测 library size = log(rowSums(counts))，即固定测序深度。
        想模拟"真实后验"的 library 可改用 model.get_latent_representation() 同源估计，
        或用 encoder 的 ql 均值（见下方注释掉的替代方案）。

    Returns
    -------
    px : scvi.distributions 分布对象
        px.sample() 得到 (n_cells, n_genes) counts 抽样
        px.mean    得到 (n_cells, n_genes) 期望表达（rate）
    """
    module = model.module
    module.eval()  # 关闭 dropout/BN 训练态，保证确定性 decode
    module.to(device)  # 移到目标设备（默认 CPU：释放 GPU 显存，避免全量采样 OOM）

    z = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device)

    # batch_index：SCVI 未注册 batch（本 pipeline 恒为 None），恒用全 0
    batch_index = torch.zeros((adata.n_obs, 1), dtype=torch.long, device=device)

    # library：观测 library size（log 尺度），decoder 内部 exp(library)*px_scale
    if library_size is None:
        counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
        lib = np.asarray(counts.sum(axis=1)).ravel()
        library = torch.log(
            torch.tensor(lib, dtype=torch.float32, device=device)
        ).unsqueeze(1)
    else:
        library = torch.log(
            torch.tensor(np.asarray(library_size, dtype=np.float32), device=device)
        ).unsqueeze(1)

    # cont_covs：USE_CONT_COVS 时从 registry 取连续协变量编码，否则 None
    if USE_CONT_COVS:
        cont_cov = model.adata_manager.get_from_registry("extra_continuous_covs")
        cont_covs = torch.as_tensor(
            pd.DataFrame(cont_cov).to_numpy(dtype=np.float32), device=device
        )
    else:
        cont_covs = None

    with torch.inference_mode():
        gen_out = module.generative(
            z=z,
            library=library,
            batch_index=batch_index,
            cont_covs=cont_covs,
        )
    return gen_out["px"]  # px: ZINB 分布


def decode_sample_from_z(model, adata, z, library_size=None, batch_size=2048, device="cpu"):
    """分块 decode + 抽样 counts，避免一次性物化全量 (n_cells, n_genes) 导致 GPU/内存 OOM。

    等价于 decode_from_z(model, adata, z).sample()，但按 batch_size 分块进行，
    每块只持有 (batch_size, n_genes) 的中间张量，峰值内存大幅降低。

    Parameters
    ----------
    model : scvi.model.SCVI（训练后）
    adata : AnnData（与模型 setup 结构一致）
    z : (n_cells, n_latent) 数组 —— 自定义 latent
    library_size : (n_cells,) 或 None（同 decode_from_z）
    batch_size : int，每块细胞数（默认 2048）
    device : str，decode 所在设备。默认 "cpu"：
        模型在 GPU 上训练，decode 时把 module 移到 CPU 可立即释放显存，
        也避免全量采样再次撑爆 GPU（默认最稳；显存充足可传 device="cuda" 加速）。

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes) —— counts 抽样（等价于 posterior predictive sample）
    """
    module = model.module
    module.eval()
    module.to(device)

    z = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device)
    n = adata.n_obs

    batch_index = torch.zeros((n, 1), dtype=torch.long, device=device)

    if library_size is None:
        counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
        lib = np.asarray(counts.sum(axis=1)).ravel()
        library = torch.log(
            torch.tensor(lib, dtype=torch.float32, device=device)
        ).unsqueeze(1)
    else:
        library = torch.log(
            torch.tensor(np.asarray(library_size, dtype=np.float32), device=device)
        ).unsqueeze(1)

    if USE_CONT_COVS:
        cont_cov = model.adata_manager.get_from_registry("extra_continuous_covs")
        cont_covs = torch.as_tensor(
            pd.DataFrame(cont_cov).to_numpy(dtype=np.float32), device=device
        )
    else:
        cont_covs = None

    samples = []
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            px = module.generative(
                z=z[i:j],
                library=library[i:j],
                batch_index=batch_index[i:j],
                cont_covs=cont_covs[i:j] if cont_covs is not None else None,
            )["px"]
            samples.append(px.sample().cpu())

    return torch.cat(samples, dim=0).numpy()


def normalized_expression_from_z(model, adata, z, lib_size=1):
    """用自定义 latent z 直接算归一化表达（等价于"指定 z 的 get_normalized_expression"）。

    语义与 scvi 官方 get_normalized_expression 完全一致：
    返回 px_scale * lib_size，即每个基因在解码分布中的相对表达（softmax 权重）
    乘以 library_size 尺度。默认 lib_size=1 -> 每细胞行和为 1（相对比例）。

    与官方版本的差别仅在于：不经过 encoder 采样 z，而是用传入的自定义 z。

    Parameters
    ----------
    model : scvi.model.SCVI（训练后）
    adata : AnnData（与模型 setup 结构一致）
    z : (n_cells, n_latent) 数组 —— 自定义 latent
    lib_size : float 或 (n_cells,)
        同官方 library_size 参数：乘以 px_scale 的尺度因子。
        默认 1（返回每细胞行和=1 的相对表达比例）；
        传 1e4 则得到每 1 万 counts 的标准化表达。

    Returns
    -------
    np.ndarray of shape (n_cells, n_genes) —— 归一化表达（rate 的比例）
    """
    px = decode_from_z(model, adata, z)
    scale = px.scale  # (n_cells, n_genes) softmax 权重
    if np.isscalar(lib_size) or (isinstance(lib_size, (int, float))):
        scale = scale * lib_size
    else:
        scale = scale * torch.tensor(np.asarray(lib_size, dtype=np.float32)).unsqueeze(1)
    return scale.cpu().numpy()

# %%
# === 演示：用指定 z 直接 decode / 归一化表达 ===

# 0) 训练结束后释放 PyTorch 缓存的显存（训练占用了 GPU 5 大量显存）
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 1) 用 harmony 校正后的 z（或 X_scVI latent mean）
z_custom = adata_subset.obsm["X_harmony"]  # (n_cells, n_latent)

# 2) decode：返回 ZINB 分布（默认移到 CPU 执行，释放 GPU 显存）
px = decode_from_z(model, adata_subset, z_custom)

# 3) 从分布抽样 counts —— 分块抽样，避免全量 (n_cells, n_genes) 在 GPU 上 OOM
recon_custom = decode_sample_from_z(model, adata_subset, z_custom, batch_size=512)
print(f"reconstructed shape: {recon_custom.shape}")
print(f"range: [{recon_custom.min()}, {recon_custom.max()}]  median={np.median(recon_custom):.2f}")
print(f"fraction non-zero: {(recon_custom > 0).mean():.3f}")

# 4) 写回 adata（注意不要覆盖默认的 scvi_reconstructed_counts）
adata_subset.layers["scvi_reconstructed_counts_harmony"] = recon_custom

# 5) 若只要期望表达（rate，不带抽样噪声）：
# px_mean = px.mean.numpy()
# adata_subset.layers["scvi_rate_harmony"] = px_mean

# 6) 等价于 get_normalized_expression 的自定义 z 版本：
#    lib_size=1   -> 相对表达（每细胞行和=1）
#    lib_size=1e4 -> 每 1 万 counts 的标准化表达（同官方常用调用）
X_norm_harmony = normalized_expression_from_z(
    model, adata_subset, z_custom, lib_size=1e4
)
print(f"normalized shape: {X_norm_harmony.shape}")
print(f"row sum (lib_size=1e4): {X_norm_harmony.sum(axis=1).mean():.0f}")

adata_subset.layers["scvi_nrom_counts_harmony"] = X_norm_harmony

# %%
recon_custom

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
import scanpy as sc
import numpy as np

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
recon = adata_subset.layers["scvi_reconstructed_counts_harmony"]
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
    layer="scvi_reconstructed_counts_harmony",
    cmap="Reds",
    standard_scale=None,     # 用真实 counts 量级，跨组可比
    swap_axes=False,
    dendrogram=False,
    return_fig=True,
    show=False,
    save="scvi_hkg_recon_harmony_status.png",
)
fig_out = OUTBASE + ".hkg_recon_harmony_status.png"
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

adata_subset.layers["count_diff"] = adata_subset.layers["scvi_reconstructed_counts_harmony"] - adata_subset.layers["counts"]
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

adata_subset.write_h5ad(OUTBASE + ".h5ad", compression="gzip")
print(f"✓ saved: {OUTBASE}.h5ad")
