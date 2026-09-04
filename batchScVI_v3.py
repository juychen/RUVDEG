# %%
"""batchScVI_v3.py — two-stage pipeline with TRANSFORM_BATCH pairwise alignment.

Stage 1 (mirror of scVI.py --no-batch):
    Train a plain ``scvi.model.SCVI`` WITHOUT any batch indicator on the
    same adata (CON cells by default). The model is saved to
    ``<outbase>_scvi_nobatch.model``. Because it is trained on the exact
    same adata as stage 2, no external-model loading / gene-space alignment
    is ever needed.

Stage 2 (pairwise, --transform-batch):
    Pick a TRANSFORM_BATCH category as the alignment target. For every
    other batch ``b`` a SEPARATE stage-2 ``SCVIWithBatchPairLoss`` model is
    trained on the pair subset {TRANSFORM_BATCH, b} (initialised from
    stage-1). Each pair model is saved to
    ``<outbase>__pair_<b>.model``.

Outputs (all written under the same ``--outprefix``):
    <outbase>_scvi_nobatch.model/   stage-1 plain scVI checkpoint
    <outbase>__pair_<b>.model/      one stage-2 pair model per other batch
    <outbase>.h5ad                  training-subset results:
                                    TRANSFORM_BATCH keeps its stage-1
                                    embedding, other batches carry their
                                    pair-model embedding; all layers are
                                    decoded by the stage-1 decoder
    <outbase>_full.h5ad             full-data results (same assembly):
                                    TRANSFORM_BATCH rows use stage-1
                                    embedding + stage-1 norm decode
                                    expression, other batches use their
                                    pair-model embedding decoded by the
                                    frozen stage-1 decoder
    <outbase>_full.hkg_*.png        HKG dotplots

Without ``--transform-batch`` the script falls back to the v2 behaviour
(one stage-2 model trained on all batches jointly).
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
from rich import print
from scipy.sparse import csr_matrix, issparse
from scvi.module._constants import MODULE_KEYS


def sanitize_batch_name(name):
    """Make a batch label safe to embed in file paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def save_pair_loss_history(model, out_prefix, label):
    """打印并保存 stage-2 的 pair loss（仅 batch_pair_mse / batch_pair_n）。

    Parameters
    ----------
    model
        已训练完的 SCVIWithBatchPairLoss 模型（其 history_ 含
        batch_pair_mse_train 等键）。
    out_prefix
        输出路径前缀，实际写 ``<out_prefix>.pair_loss.csv``。
    label
        日志标签（如 out base 或 pair 名）。
    """
    history = model.history_
    pair_keys = [k for k in history.keys() if "batch_pair" in k]
    if not pair_keys:
        print(f"⚠ {label}: 未记录 pair loss，history keys = {list(history.keys())}")
        return
    frame = pd.DataFrame(
        {
            k: pd.Series(np.asarray(history[k]).ravel()).astype(
                float if "mse" in k else int
            )
            for k in pair_keys
        }
    )
    frame.index.name = "epoch"
    csv_path = out_prefix + ".pair_loss.csv"
    frame.to_csv(csv_path)
    print(f"✓ {label} pair loss saved: {csv_path}")
    mse_train = frame["batch_pair_mse_train"]
    print(
        f"  {label} batch_pair_mse_train: "
        f"final={mse_train.iloc[-1]:.6g}  min={mse_train.min():.6g}  "
        f"mean={mse_train.mean():.6g}  (n_epochs={len(mse_train)})"
    )
    if "batch_pair_mse_validation" in frame:
        mse_val = frame["batch_pair_mse_validation"]
        print(
            f"  {label} batch_pair_mse_validation: "
            f"final={mse_val.iloc[-1]:.6g}  min={mse_val.min():.6g}"
        )
    return frame


def to_sparse_int(arr, dtype=np.int32):
    """Convert an array to CSR while preserving floats when requested."""
    if not issparse(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    result = csr_matrix(arr)
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        result.data = np.rint(result.data).astype(dtype)
    else:
        result = result.astype(dtype)
    result.eliminate_zeros()
    return result


def decode_with_fixed_latent(model, adata, z, batch_size=512, lib_size=1e4):
    """Decode fixed latent values with a trained scVI decoder."""
    z = np.asarray(z)
    model._validate_anndata(adata)
    model.module.eval()
    counts_list = []
    normalized_list = []

    with torch.no_grad():
        loader = model._make_data_loader(
            adata=adata,
            indices=np.arange(adata.n_obs),
            batch_size=batch_size,
            shuffle=False,
        )
        cell_start = 0
        for tensors in loader:
            inference_input = model.module._get_inference_input(tensors)
            inference_outputs = model.module.inference(**inference_input)
            generative_inputs = model.module._get_generative_input(
                tensors, inference_outputs
            )

            cell_stop = cell_start + tensors["X"].shape[0]
            generative_inputs[MODULE_KEYS.Z_KEY] = torch.as_tensor(
                z[cell_start:cell_stop],
                dtype=inference_outputs[MODULE_KEYS.Z_KEY].dtype,
                device=inference_outputs[MODULE_KEYS.Z_KEY].device,
            )

            counts_output = model.module.generative(**generative_inputs)
            counts_list.append(
                counts_output[MODULE_KEYS.PX_KEY].mu.detach().cpu()
            )

            generative_inputs[MODULE_KEYS.LIBRARY_KEY] = torch.full_like(
                generative_inputs[MODULE_KEYS.LIBRARY_KEY],
                float(np.log(lib_size)),
            )
            normalized_output = model.module.generative(**generative_inputs)
            normalized_list.append(
                normalized_output[MODULE_KEYS.PX_KEY].mu.detach().cpu()
            )
            cell_start = cell_stop

    return (
        torch.cat(counts_list, dim=0).numpy(),
        torch.cat(normalized_list, dim=0).numpy().astype(np.float32),
    )


# ===== 命令行参数 =====
parser = argparse.ArgumentParser(
    description="Two-stage pipeline: plain scVI (nobatch) -> SCVIWithBatchPairLoss"
)
parser.add_argument(
    "--input", "-i",
    default="/data7/mark/STG/dataset/snRNA/merge_SCH_new/six_datasets_4v3_500_1000gene/TH_downsampled_ratio.h5ad",
    help="输入 h5ad 文件路径",
)
parser.add_argument(
    "--outprefix", "-o",
    default="/home/junyichen/code/RUVAEDEG/batchScVImodel_output.h5ad",
    help="Output prefix (base path). 所有输出写成 <prefix-base>.<suffix>。",
)
parser.add_argument(
    "--pair-batch-key",
    default="company",
    help="用于配对的 obs 列（默认 'company'），会被复制成 _pair_batch 后注册成 cat cov",
)
parser.add_argument(
    "--labels-key",
    default="celltype.L2",
    help="labels_key，注册到 pair model（限制 pair-MSE 在同一 cell type 内）",
)
parser.add_argument(
    "--condition-key",
    default=None,
    help="可选 condition 列；指定后按 celltype.L2 + condition 分组 pair，且使用全部细胞训练",
)
parser.add_argument(
    "--no-cont-cov",
    action="store_true",
    help="不传入连续协变量 (n_genes_on)",
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
    "--pair-weight", type=float, default=1.0,
    help="pair-MSE 项在总 loss 中的权重（默认 1.0；0 等价于退化为普通 scVI）",
)
parser.add_argument(
    "--lr", type=float, default=1e-3,
    help="training learning rate passed to plan_kwargs (default: 1e-3)",
)
parser.add_argument(
    "--max-epochs", type=int, default=None,
    help="maximum training epochs; None uses scVI automatic epoch selection",
)
parser.add_argument(
    "--align-on",
    default="z",
    choices=["mu", "z"],
    help="pair-MSE 对齐的量：'mu' (decoder rate, 默认) 或 'z' (latent)",
)
parser.add_argument(
    "--no-compare",
    action="store_true",
    help="跳过末尾的 HKG dotplot（仍会训练、抽样并写 h5ad）",
)
parser.add_argument(
    "--skip-stage1",
    action="store_true",
    help="跳过 stage-1 plain scVI 训练（若 <outbase>_scvi_nobatch.model 已存在）",
)
parser.add_argument(
    "--transform-batch",
    default=None,
    help="TRANSFORM_BATCH 目标 batch：指定后 stage-2 改为逐对训练"
    "（TRANSFORM_BATCH 与其他 batch 两两配对，各训一个 stage-2 模型），"
    "最终其他 batch 的 embedding 对齐到 TRANSFORM_BATCH；"
    "不指定则保持 v2 单一 stage-2 模型行为",
)
args, _ = parser.parse_known_args()

INPUT_H5AD = os.path.abspath(args.input)
OUTPREFIX = os.path.abspath(args.outprefix)
N_LATENT = args.n_latent
N_LAYERS = args.n_layers
USE_CONT_COVS = not args.no_cont_cov
SHOW_COMPARE = not args.no_compare
PAIR_WEIGHT = float(args.pair_weight)
LEARNING_RATE = float(args.lr)
MAX_EPOCHS = args.max_epochs
ALIGN_ON = args.align_on
PAIR_BATCH_KEY = args.pair_batch_key
LABELS_KEY = args.labels_key
CONDITION_KEY = args.condition_key
SKIP_STAGE1 = args.skip_stage1
TRANSFORM_BATCH = args.transform_batch
PAIRWISE_MODE = TRANSFORM_BATCH is not None

OUTBASE = os.path.splitext(OUTPREFIX)[0]
OUTDIR = os.path.dirname(OUTBASE) or "."
MODEL_DIR = OUTBASE + ".model"
STAGE1_MODEL_DIR = OUTBASE + "_scvi_nobatch.model"
os.makedirs(OUTDIR, exist_ok=True)

print(f"input h5ad      : {INPUT_H5AD}")
print(f"out prefix      : {OUTPREFIX}")
print(f"out base        : {OUTBASE}")
print(f"stage-1 model   : {STAGE1_MODEL_DIR}")
if PAIRWISE_MODE:
    print(f"transform batch : {TRANSFORM_BATCH}")
else:
    print(f"stage-2 model   : {MODEL_DIR}")
print(f"n_latent={N_LATENT}  n_layers={N_LAYERS}  use_cont_cov={USE_CONT_COVS}")
print(f"pair_weight={PAIR_WEIGHT}  align_on={ALIGN_ON}  "
    f"pair_batch_key={PAIR_BATCH_KEY!r}  labels_key={LABELS_KEY!r}  "
    f"condition_key={CONDITION_KEY!r}  lr={LEARNING_RATE:g}  "
    f"max_epochs={MAX_EPOCHS}")

# %%
# ===== 1. 数据读取 =====
adata_all = sc.read_h5ad(INPUT_H5AD)

if CONDITION_KEY is not None:
    if CONDITION_KEY not in adata_all.obs.columns:
        raise KeyError(
            f"condition key {CONDITION_KEY!r} 不在 adata.obs 中，"
            f"现有列: {adata_all.obs.columns.tolist()}"
        )
    if LABELS_KEY not in adata_all.obs.columns:
        raise KeyError(
            f"labels key {LABELS_KEY!r} 不在 adata.obs 中，"
            f"现有列: {adata_all.obs.columns.tolist()}"
        )

    pair_group_key = "_pair_group"
    adata_all.obs[pair_group_key] = (
        adata_all.obs[LABELS_KEY].astype(str)
        + "__"
        + adata_all.obs[CONDITION_KEY].astype(str)
    ).astype("category")
    LABELS_KEY = pair_group_key
    print(
        f"condition-aware pairing enabled: {CONDITION_KEY!r} + "
        f"{args.labels_key!r} -> {LABELS_KEY!r} "
        f"({adata_all.obs[LABELS_KEY].nunique()} groups)"
    )

# %%
# ===== 2. n_genes_on：在全量 adata_all 上算一次，训练 / eval 共享同一列 =====
if USE_CONT_COVS:
    if "counts" not in adata_all.layers:
        if adata_all.X is None:
            raise KeyError(
                "layers['counts'] 缺失且 X 为空，无法自动生成 counts layer"
            )
        print("⚠ layers['counts'] 缺失: 将 X.copy() 写入 counts layer")
        adata_all.layers["counts"] = adata_all.X.copy()
    n_genes_on_raw = (adata_all.layers["counts"] > 0).sum(axis=1).astype(np.float32)
    n_genes_on_mean = float(n_genes_on_raw.mean())
    n_genes_on_std = float(n_genes_on_raw.std())
    if n_genes_on_std < 1e-8:
        raise ValueError("n_genes_on 没有足够变异，无法作为连续协变量")

    n_genes_on_z = (
        (n_genes_on_raw - n_genes_on_mean) / n_genes_on_std
    ).astype(np.float32)
    adata_all.obs["n_genes_on"] = n_genes_on_z
    print(f"n_genes_on (全量): raw μ/σ = {n_genes_on_mean:.1f} / {n_genes_on_std:.1f}")
    print(
        f"standardized range (all cells): "
        f"[{adata_all.obs['n_genes_on'].min():.3f}, "
        f"{adata_all.obs['n_genes_on'].max():.3f}]"
    )
else:
    print("⚠ --no-cont-cov: 跳过 n_genes_on 计算")

# 训练子集：默认只用 CON（指定 condition-key 后用全量细胞）
if CONDITION_KEY is None and "status" in adata_all.obs.columns:
    TRAIN_STATUSES = ["CON"]
    adata_subset = adata_all[adata_all.obs.status.isin(TRAIN_STATUSES)].copy()
    print(f"training subset (status in {TRAIN_STATUSES}): {adata_subset.shape}")
else:
    if CONDITION_KEY is None:
        print(
            "⚠ no --condition-key and 'status' not in obs: "
            "falling back to training on ALL cells"
        )
    else:
        print(
            f"condition-aware training: all cells, "
            f"condition_key={CONDITION_KEY!r}"
        )
    adata_subset = adata_all.copy()
    print(f"training subset (all cells): {adata_subset.shape}")

# adata_eval 用全部数据 —— 下游 encode / decode / HKG dotplot 用这个。
adata_eval = adata_all.copy()
print(f"eval / decode subset (all status)         : {adata_eval.shape}")

if PAIR_BATCH_KEY not in adata_all.obs.columns:
    raise KeyError(
        f"pair-batch key {PAIR_BATCH_KEY!r} 不在 adata.obs 中，"
        f"现有列: {adata_all.obs.columns.tolist()}"
    )
if USE_CONT_COVS and "counts" not in adata_all.layers:
    raise KeyError(
        "未提供 n_genes_on 且 layers['counts'] 缺失，无法自动计算连续协变量"
    )

print(f"\nadata_all   shape: {adata_all.shape}")
for col in ["status", "company", "celltype.L2", "sex", "sample", "region"]:
    if col in adata_all.obs.columns:
        vc = adata_all.obs[col].value_counts()
        print(f"\n{col} (n_unique={vc.size}):")
        print(vc.head(10))

# %%
# ===== 3. raw counts 写入 X =====
adata_subset.X = adata_subset.layers["counts"].copy()

# %%
# ===== 4. Drop NaN + 强制 dtype（BEFORE scVI registration） =====
mask = adata_subset.obs[PAIR_BATCH_KEY].notna()
adata_subset = adata_subset[mask].copy()
for _cat_key in {PAIR_BATCH_KEY, LABELS_KEY}:
    adata_subset.obs[_cat_key] = adata_subset.obs[_cat_key].astype("category")
if USE_CONT_COVS:
    adata_subset.obs["n_genes_on"] = adata_subset.obs["n_genes_on"].astype(np.float32)
print(f"after dropna shape : {adata_subset.shape}")
print(f"{PAIR_BATCH_KEY} categories : "
      f"{adata_subset.obs[PAIR_BATCH_KEY].cat.categories.tolist()}")

# %%
# ===== 5. Auto-select GPU =====
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

CONT_COVS = ["n_genes_on"] if USE_CONT_COVS else None

# %%
# ============================================================
# ===== Stage 1: plain scVI WITHOUT batch indicator ==========
# ============================================================
# mirror of scVI.py --no-batch：batch_key=None, labels_key=None,
# 只注册 X + n_genes_on。在同一个 adata_subset 上训练，因此与 stage-2
# 完全同源 —— 不需要任何 external model 加载 / 基因对齐。
# 用独立副本注册，避免与 stage-2 的 pair model manager 互相覆盖 uuid
adata_stage1 = adata_subset.copy()
scvi.model.SCVI.setup_anndata(
    adata_stage1,
    layer=None,
    batch_key=None,
    labels_key=None,
    categorical_covariate_keys=None,
    continuous_covariate_keys=CONT_COVS,
)
print(f"✓ stage-1 setup_anndata complete  "
      f"uuid: {adata_stage1.uns['_scvi_manager_uuid'][:8]}…")

if SKIP_STAGE1 and os.path.isdir(STAGE1_MODEL_DIR):
    print(f"== stage-1 skipped (--skip-stage1): {STAGE1_MODEL_DIR} ==")
    stage1_model = scvi.model.SCVI.load(STAGE1_MODEL_DIR, adata=adata_stage1)
else:
    print(f"\n== stage-1: plain scVI (nobatch) on {adata_subset.shape} ==")
    stage1_model = scvi.model.SCVI(
        adata_stage1, n_layers=N_LAYERS, n_latent=N_LATENT,
        gene_likelihood="zinb",
    )
    stage1_model.train(
        max_epochs=MAX_EPOCHS,
        accelerator=TRAIN_ACCELERATOR,
        devices=TRAIN_DEVICES,
        plan_kwargs={"lr": LEARNING_RATE},
    )
    stage1_model.save(STAGE1_MODEL_DIR, overwrite=True, save_anndata=False)
    print(f"✓ stage-1 model saved: {STAGE1_MODEL_DIR}")

# %%
# ============================================================
# ===== Stage 2: SCVIWithBatchPairLoss (pair-MSE) ============
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_scvi_batch_pair import SCVIWithBatchPairLoss  # noqa: E402

# ---- stage-1 latent on the training subset（stage-1 与 stage-2 同一 adata）----
z_s1_subset = np.asarray(stage1_model.get_latent_representation())

if not PAIRWISE_MODE:
    # ================= v2 fallback：单一 stage-2 模型，全 batch 联合训练 =================
    print(f"\n== stage-2 (joint): SCVIWithBatchPairLoss on {adata_subset.shape} ==")
    SCVIWithBatchPairLoss.setup_anndata(
        adata_subset,
        pair_batch_obs_key=PAIR_BATCH_KEY,
        batch_key=None,
        layer=None,
        labels_key=LABELS_KEY,
        categorical_covariate_keys=None,
        continuous_covariate_keys=CONT_COVS,
    )
    print(f"✓ stage-2 setup_anndata complete")
    print(f"  manager uuid: {adata_subset.uns['_scvi_manager_uuid'][:8]}…")

    model = SCVIWithBatchPairLoss(
        adata_subset,
        n_layers=N_LAYERS,
        n_latent=N_LATENT,
        gene_likelihood="zinb",
        pair_weight=PAIR_WEIGHT,
        align_on=ALIGN_ON,
        deeply_inject_covariates=False,  # decoder 不看 pair-batch
    )
    compatible_state = {
        name: value
        for name, value in stage1_model.module.state_dict().items()
        if name in model.module.state_dict()
        and model.module.state_dict()[name].shape == value.shape
    }
    model.module.load_state_dict(compatible_state, strict=False)
    print(
        f"✓ initialized stage-2 from stage-1: "
        f"{len(compatible_state)} compatible parameter tensors"
    )
    print(f"module class    : {type(model.module).__name__}")
    print(f"pair_weight     : {model.module.pair_weight}")
    print(f"align_on        : {model.module.align_on}")
    print(f"n_input genes   : {model.module.n_input}")
    model.train(
        max_epochs=MAX_EPOCHS,
        accelerator=TRAIN_ACCELERATOR,
        devices=TRAIN_DEVICES,
        plan_kwargs={"lr": LEARNING_RATE},
    )
    save_pair_loss_history(model, OUTBASE, os.path.basename(OUTBASE))
    model.save(MODEL_DIR, overwrite=True, save_anndata=False)
    print(f"✓ stage-2 model saved: {MODEL_DIR}")
    # z_s1_subset 用作对比基线
    adata_subset.obsm["X_scVI_nobatch"] = z_s1_subset
else:
    # ============ stage-1 plain scVI 的 baseline（不重新注册 manager）============
    adata_subset.obsm["X_scVI_nobatch"] = z_s1_subset

# ---- pairwise stage-2：TRANSFORM_BATCH 与其他 batch 两两配对 ----
pair_models = {}                                   # b -> (model, model_dir, cells)
pair_models["__joint__"] = (
    (model, MODEL_DIR, None) if not PAIRWISE_MODE else (None, None, None)
)
if PAIRWISE_MODE:
    batch_categories = adata_subset.obs[PAIR_BATCH_KEY].cat.categories.tolist()
    print(f"\n== stage-2 (pairwise): TRANSFORM_BATCH={TRANSFORM_BATCH!r} ==")
    print(f"  _pair_batch categories: {batch_categories}")
    if TRANSFORM_BATCH not in batch_categories:
        raise KeyError(
            f"TRANSFORM_BATCH {TRANSFORM_BATCH!r} 不在 {PAIR_BATCH_KEY!r} 类别中: "
            f"{batch_categories}"
        )
    others = [b for b in batch_categories if b != TRANSFORM_BATCH]
    if not others:
        raise ValueError(
            f"{PAIR_BATCH_KEY!r} 只有 TRANSFORM_BATCH {TRANSFORM_BATCH!r} 一个类别，"
            f"没有其他 batch 可配对"
        )

    # TRANSFORM_BATCH 的 stage-1 编码 —— 其余 batch 对齐到这里
    is_transform = adata_subset.obs[PAIR_BATCH_KEY] == TRANSFORM_BATCH
    print(f"  TRANSFORM_BATCH cells: {int(is_transform.sum())}")

    for b in others:
        b_safe = sanitize_batch_name(b)
        pair_model_dir = OUTBASE + f"__pair_{b_safe}.model"
        print(f"\n---- pair model: {TRANSFORM_BATCH!r} <-> {b!r} -> {pair_model_dir} ----")
        is_pair = is_transform | (adata_subset.obs[PAIR_BATCH_KEY] == b)
        adata_cells = adata_subset[is_pair].copy()
        # 每个配对模型注册独立 manager —— scVI 要求每模型一个 uuid
        SCVIWithBatchPairLoss.setup_anndata(
            adata_cells,
            pair_batch_obs_key=PAIR_BATCH_KEY,
            batch_key=None,
            layer=None,
            labels_key=LABELS_KEY,
            categorical_covariate_keys=None,
            continuous_covariate_keys=CONT_COVS,
        )
        print(f"  pair cells: {adata_cells.shape}")
        # pair model
        pair_model = SCVIWithBatchPairLoss(
            adata_cells,
            n_layers=N_LAYERS,
            n_latent=N_LATENT,
            gene_likelihood="zinb",
            pair_weight=PAIR_WEIGHT,
            align_on=ALIGN_ON,
            deeply_inject_covariates=False,  # decoder 不看 pair-batch
        )
        # 初始权重来自 stage-1
        compatible_state = {
            name: value
            for name, value in stage1_model.module.state_dict().items()
            if name in pair_model.module.state_dict()
            and pair_model.module.state_dict()[name].shape == value.shape
        }
        pair_model.module.load_state_dict(compatible_state, strict=False)
        print(
            f"  ✓ initialized from stage-1: "
            f"{len(compatible_state)} compatible parameter tensors"
        )
        print(f"  module class  : {type(pair_model.module).__name__}")
        print(f"  n_input genes : {pair_model.module.n_input}")
        pair_model.train(
            max_epochs=MAX_EPOCHS,
            accelerator=TRAIN_ACCELERATOR,
            devices=TRAIN_DEVICES,
            plan_kwargs={"lr": LEARNING_RATE},
        )
        save_pair_loss_history(
            pair_model,
            f"{OUTBASE}__pair_{b_safe}",
            f"pair {TRANSFORM_BATCH!r}-{b!r}",
        )
        pair_model.save(pair_model_dir, overwrite=True, save_anndata=False)
        print(f"  ✓ pair model saved: {pair_model_dir}")
        pair_models[b] = (pair_model, pair_model_dir, adata_cells)

    model = pair_models[others[0]][0]                  # 占位：下游用 pair model
    print(f"\n✓ stage-2 pairwise models: {len(pair_models) - 1}")

# %%
# ===== 8. latent 表示 =====
SCVI_LATENT_KEY = "X_scVI"
if not PAIRWISE_MODE:
    adata_subset.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
else:
    # pairwise 拼装：TRANSFORM_BATCH 用 stage-1 z，其他 batch 用各自 pair model z
    subset_batch = adata_subset.obs[PAIR_BATCH_KEY].astype(str).to_numpy()
    z_pair_subset = z_s1_subset.copy()
    for b, (pair_model, _, adata_cells) in pair_models.items():
        if b.startswith("__"):
            continue
        is_b = subset_batch == b
        if not bool(is_b.any()):
            continue
        z_b = pair_model.get_latent_representation(batch_size=512)
        # adata_cells 保持 adata_subset 原始行序，用 batch 掩码选出 b 的行
        cells_b_mask = (adata_cells.obs[PAIR_BATCH_KEY] == b).to_numpy()
        z_pair_subset[is_b] = z_b[cells_b_mask]
    adata_subset.obsm[SCVI_LATENT_KEY] = z_pair_subset
    print(f"  subset z assembly: TRANSFORM_BATCH={int((subset_batch == TRANSFORM_BATCH).sum())} "
          f"(stage-1 z), other={int((subset_batch != TRANSFORM_BATCH).sum())} (pair model z)")

# %%
# ===== 9. 归一化表达 + 后验预测抽样 =====
if not PAIRWISE_MODE:
    X_norm = model.get_normalized_expression(
        lib_size=1e4,
        transform_batch=None,
    )
    adata_subset.layers["scvi_nrom_counts"] = X_norm
    n_samples = 1
    reconstructed = model.posterior_predictive_sample(
        adata=adata_subset,
        n_samples=n_samples,
        batch_size=512,
        transform_batch=None,
        silent=False,
    )
    reconstructed = np.asarray(reconstructed.todense())
    print(f"reconstructed shape: {reconstructed.shape}  dtype={reconstructed.dtype}")
    print(f"range: [{reconstructed.min()}, {reconstructed.max()}]  "
          f"median={np.median(reconstructed):.2f}")
    print(f"fraction non-zero: {(reconstructed > 0).mean():.3f}")
    adata_subset.layers["scvi_reconstructed_counts"] = to_sparse_int(reconstructed)
else:
    # 全部用 stage-1 decoder 解码（TRANSFORM 行 = stage-1 z → stage-1 表达；
    # 其他行 = stage-2 z → 回到 stage-1 decoder 解码的 norm decode expression）
    subset_counts, subset_normalized = decode_with_fixed_latent(
        stage1_model, adata_stage1, z_pair_subset, batch_size=512
    )
    adata_subset.layers["scvi_reconstructed_counts"] = to_sparse_int(subset_counts)
    adata_subset.layers["scvi_nrom_counts"] = csr_matrix(subset_normalized)

# 持久化 AnnData + 模型
adata_subset.write_h5ad(OUTBASE + ".h5ad", compression="gzip")
print(f"✓ saved: {OUTBASE}.h5ad")

# %%
# ===== 10a. 切换到全部细胞：pair model 在全量 adata_eval 上 encode/decode =====
adata_eval = adata_all.copy()
print(f"\n== full-data eval switch ==")
print(f"adata_eval shape (all status): {adata_eval.shape}")

# The RAW decoder was trained on counts, so evaluation must use the same X.
adata_eval.X = adata_eval.layers["counts"].copy()

eval_mask = adata_eval.obs[PAIR_BATCH_KEY].notna()
if not bool(eval_mask.all()):
    adata_eval = adata_eval[eval_mask].copy()
    adata_eval.obs[PAIR_BATCH_KEY] = adata_eval.obs[PAIR_BATCH_KEY].astype("category")
print(f"adata_eval after NaN drop : {adata_eval.shape}")

for _cat_key in {PAIR_BATCH_KEY, LABELS_KEY}:
    adata_eval.obs[_cat_key] = adata_eval.obs[_cat_key].astype("category")
if USE_CONT_COVS:
    adata_eval.obs["n_genes_on"] = adata_eval.obs["n_genes_on"].astype(np.float32)
    print(
        f"adata_eval n_genes_on (shared, 全量 z-score): "
        f"range=[{adata_eval.obs['n_genes_on'].min():.3f}, "
        f"{adata_eval.obs['n_genes_on'].max():.3f}]"
    )

# ---- 全量 stage-1 编码：TRANSFORM_BATCH 一直用 stage-1 z ----
adata_eval_stage1 = adata_eval.copy()
scvi.model.SCVI.setup_anndata(
    adata_eval_stage1,
    layer=None,
    batch_key=None,
    labels_key=None,
    categorical_covariate_keys=None,
    continuous_covariate_keys=CONT_COVS,
)
stage1_eval_decoder = scvi.model.SCVI.load(
    STAGE1_MODEL_DIR,
    adata=adata_eval_stage1,
)
z_s1_eval = stage1_eval_decoder.get_latent_representation(batch_size=512)

OUTBASE_FULL = OUTBASE + "_full"
OUTDIR_FULL = os.path.dirname(OUTBASE_FULL) or "."
os.makedirs(OUTDIR_FULL, exist_ok=True)

if not PAIRWISE_MODE:
    # ================= v2 fallback：stage-2 model 编码全部细胞 =================
    SCVIWithBatchPairLoss.setup_anndata(
        adata_eval,
        pair_batch_obs_key=PAIR_BATCH_KEY,
        batch_key=None,
        layer=None,
        labels_key=LABELS_KEY,
        categorical_covariate_keys=None,
        continuous_covariate_keys=CONT_COVS,
    )
    print(f"  manager uuid (eval)     : {adata_eval.uns['_scvi_manager_uuid'][:8]}…")
    z_pair = model.get_latent_representation(adata=adata_eval, batch_size=512)
    adata_eval.obsm[SCVI_LATENT_KEY] = z_pair
else:
    # ============ pairwise：TRANSFORM_BATCH 用 stage-1 z，其他 batch 用 pair model z ============
    eval_batch = adata_eval.obs[PAIR_BATCH_KEY].astype(str).to_numpy()
    z_pair_eval = np.zeros((adata_eval.n_obs, N_LATENT), dtype=np.float32)
    for b, (pair_model, _, adata_cells) in pair_models.items():
        if b.startswith("__"):
            continue
        is_b = eval_batch == b
        if not bool(is_b.any()):
            print(f"⚠ eval 中没有 {b!r} 细胞，跳过其 pair model 编码")
            continue
        # pair model 只在 b 的细胞上编码（其训练集 = {TRANSFORM_BATCH, b}）
        adata_cells_b = adata_eval[is_b].copy()
        for _cat_key in {PAIR_BATCH_KEY, LABELS_KEY}:
            adata_cells_b.obs[_cat_key] = (
                adata_cells_b.obs[_cat_key].astype(str).astype("category")
            )
        if USE_CONT_COVS:
            adata_cells_b.obs["n_genes_on"] = (
                adata_cells_b.obs["n_genes_on"].astype(np.float32)
            )
        SCVIWithBatchPairLoss.setup_anndata(
            adata_cells_b,
            pair_batch_obs_key=PAIR_BATCH_KEY,
            batch_key=None,
            layer=None,
            labels_key=LABELS_KEY,
            categorical_covariate_keys=None,
            continuous_covariate_keys=CONT_COVS,
        )
        z_b = pair_model.get_latent_representation(
            adata=adata_cells_b, batch_size=512
        )
        print(f"  pair model {b!r} z: {z_b.shape}")
        z_pair_eval[is_b] = z_b
    # TRANSFORM_BATCH 用 stage-1 z
    is_transform_eval = eval_batch == TRANSFORM_BATCH
    z_pair_eval[is_transform_eval] = z_s1_eval[is_transform_eval]
    adata_eval.obsm[SCVI_LATENT_KEY] = z_pair_eval
    print(f"  eval z assembly: TRANSFORM_BATCH={int(is_transform_eval.sum())} (stage-1 z), "
          f"other={int((~is_transform_eval).sum())} (pair model z)")

# Decode the updated pair-loss Z with the original RAW scVI decoder.
adata_eval_raw = adata_eval.copy()
scvi.model.SCVI.setup_anndata(
    adata_eval_raw,
    layer=None,
    batch_key=None,
    labels_key=None,
    categorical_covariate_keys=None,
    continuous_covariate_keys=CONT_COVS,
)
raw_scvi_decoder = scvi.model.SCVI.load(
    STAGE1_MODEL_DIR,
    adata=adata_eval_raw,
)
z_pair = adata_eval.obsm[SCVI_LATENT_KEY]
raw_counts, raw_normalized = decode_with_fixed_latent(
    raw_scvi_decoder,
    adata_eval_raw,
    z_pair,
)
adata_eval.layers["scvi_reconstructed_counts"] = to_sparse_int(raw_counts)
adata_eval.layers["scvi_nrom_counts"] = csr_matrix(raw_normalized)

# %%
# ===== HKG dotplots（全部在 adata_eval 上，pair model 解码） =====
hkg_priority = [
    "Aars", "Sars",
    "Polr2a", "Polr2f",
    "Psmd6", "Psmd7", "Psma5",
    "Rer1", "Ipo8", "Pop4", "Pes1",
    "Oaz1",
    "Rpl13a", "Rpl27", "Rps13", "Rps20",
    "Hprt1", "Gusb", "Ppia", "Ywhaz",
]

candidate_negative_control_genes = [
    "Oaz1", "Rps13", "Rps20", "Rpl27",
    "Aars", "Polr2f", "Psmd6", "Psmd7", "Psma5",
    "Ipo8", "Pop4", "Pes1", "Rer1",
    "Rpl13a", "Cyc1", "Sdha", "Ubc",
]

hkg_priority = candidate_negative_control_genes + hkg_priority

seen = set()
hkg_genes = []
for g in hkg_priority:
    if g in adata_eval.var_names and g not in seen:
        seen.add(g)
        hkg_genes.append(g)
print(f"HKG available: {len(hkg_genes)} / {len(set(hkg_priority))}")

PLOT_GROUPBY = "sample" if "sample" in adata_eval.obs.columns else PAIR_BATCH_KEY
if PLOT_GROUPBY != "sample":
    print(f"⚠ 'sample' not in obs: dotplot groupby falls back to {PLOT_GROUPBY!r}")

if SHOW_COMPARE:
    if "status" in adata_eval.obs.columns:
        adata_eval.obs["status"] = adata_eval.obs["status"].astype("category")
        adata_eval.obs["status"] = adata_eval.obs["status"].cat.set_categories(
            ["CON", "CURES", "CUSUS", "CSRES", "CSSUS"]
        )
    else:
        print("⚠ 'status' not in obs: skip status category ordering")

if SHOW_COMPARE:
    recon = adata_eval.layers["scvi_reconstructed_counts"]
    recon_arr = recon.toarray() if hasattr(recon, "toarray") else np.asarray(recon)
    vmin = float(np.percentile(recon_arr[recon_arr > 0], 5))
    vmax = float(np.percentile(recon_arr, 95))
    print(f"shared color: vmin={vmin:.2f}, vmax={vmax:.2f}")

    sc.settings.figdir = OUTDIR_FULL
    fig = sc.pl.dotplot(
        adata_eval,
        var_names=hkg_genes,
        groupby=PLOT_GROUPBY,
        layer="scvi_reconstructed_counts",
        cmap="Reds",
        standard_scale=None,
        swap_axes=False,
        dendrogram=False,
        return_fig=True,
        show=False,
        save="batchscvi_full_hkg_recon_status.png",
    )
    fig_out = OUTBASE_FULL + ".hkg_recon_status.png"
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)
    print(f"✓ saved: {fig_out}")

    sc.settings.figdir = OUTDIR_FULL
    fig = sc.pl.dotplot(
        adata_eval,
        var_names=hkg_genes,
        groupby=PLOT_GROUPBY,
        layer="counts",
        cmap="Reds",
        standard_scale=None,
        swap_axes=False,
        dendrogram=False,
        return_fig=True,
        show=False,
        save="batchscvi_full_raw_status.png",
    )
    fig_out = OUTBASE_FULL + ".raw_status.png"
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)
    print(f"✓ saved: {fig_out}")

    adata_eval.layers["count_diff"] = to_sparse_int(
        adata_eval.layers["scvi_reconstructed_counts"] - adata_eval.layers["counts"]
    )
    fig = sc.pl.dotplot(
        adata_eval,
        var_names=hkg_genes,
        groupby=PLOT_GROUPBY,
        layer="count_diff",
        cmap="bwr",
        expression_cutoff=-1000,
        standard_scale=None,
        swap_axes=False,
        dendrogram=False,
        return_fig=True,
        show=False,
        save="batchscvi_full_count_diff.png",
    )
    fig_out = OUTBASE_FULL + ".count_diff.png"
    fig.savefig(fig_out, bbox_inches="tight", dpi=200)
    print(f"✓ saved: {fig_out}")
else:
    print("⚠ --no-compare: skip HKG dotplots (and posterior_predictive_sample on eval set)")

# ----- 持久化最终全量结果 -----
adata_eval.write_h5ad(OUTBASE_FULL + ".h5ad", compression="gzip")
print(f"✓ final full-data h5ad: {OUTBASE_FULL}.h5ad")
