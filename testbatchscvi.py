# %% [markdown]
# # testSCVIbatch.ipynb — cross-batch pair-MSE alignment loss
# 
# Inherits from `testRUVVAE_ZINB.ipynb` (uses the same TH subset + scVI-style
# data prep). Calls [`model_scvi_batch_pair.SCVIWithBatchPairLoss`](model_scvi_batch_pair.py)
# which adds an extra cross-batch MSE term to the standard scVI loss.
# 
# Pipeline:
# 1. Load data & prepare scVI format (counts in X, n_genes_on covariate)
# 2. `setup_anndata` + build `SCVIWithBatchPairLoss`
# 3. Forward validation: `pair_loss > 0` only when mini-batch has ≥ 2 batches
# 4. Train (100 epochs), watch `batch_pair_mse`
# 5. Evaluate alignment: iLISI / ASW vs baseline SCVI
# 6. Visualise: UMAP by batch, per-pair MSE convergence, latent `z` distribution

# %%
# ========== Load the same data subset as testRUVVAE_ZINB.ipynb ==========
import scanpy as sc
from pathlib import Path
import sys
sys.path.insert(0, "/home/junyichen/code/RUVAEDEG")

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, issparse
from torch.utils.data import Sampler
import matplotlib.pyplot as plt
import scanpy.external as sce
from sklearn.metrics import silhouette_score
from scvi.model import SCVI as _SCVI

from model_scvi_batch_pair import (
    SCVIWithBatchPairLoss,
    VAEWithBatchPairLoss,
    _cross_batch_pair_mse,
)
from scvi.module._constants import MODULE_KEYS


def to_sparse_int(arr):
    """把 numpy / float / 稀疏数组转换成 (n_cells, n_genes) 的 csr int 矩阵。

    - 输入若为稀疏矩阵：保持格式，只把 dtype 转成 int32。
    - 输入若为稠密 ndarray：rint 截断后构造 csr（counts 必须是离散整数）。
    - 用于把 decoder 抽样得到的 counts 写入 adata.layers["..."]，避免 h5ad
      把 float 数组按 ~8 字节/元素存储（counts 通常很稀疏 + 数值小）。
    """
    if issparse(arr):
        return csr_matrix(arr).astype(np.int32)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # 防御性：误传 1D 时强制 reshape 成单行矩阵
        arr = arr.reshape(1, -1)
    return csr_matrix(np.rint(arr).astype(np.int32))


adata = sc.read_h5ad(
    "/data7/mark/STG/dataset/snRNA/merge_SCH_new/six_datasets_4v3_500_1000gene/TH_downsampled_ratio.h5ad"
)

adata_subset = adata[
    adata.obs["celltype.L2"].isin(
        adata.obs["celltype.L2"].value_counts().head(3).index[1:3]
    )
]
adata_subset = adata_subset[adata_subset.obs["sex"] == "M"]

SLIDE_FIG_DIR = Path("slide_fig")
SLIDE_FIG_DIR.mkdir(parents=True, exist_ok=True)
adata_subset = adata_subset[adata_subset.obs["status"].isin(["CON"])]
adata_subset = adata_subset.copy()  # small slice for fast iteration

print(f"adata_subset shape: {adata_subset.shape}")
print(f"status counts:\n{adata_subset.obs['status'].value_counts()}")
print(f"company counts:\n{adata_subset.obs['company'].value_counts()}")

# %%
# ========== Prepare scVI-format data: raw counts in X + n_genes_on covariate ==========
adata_scvi = adata_subset.copy()

# counts → X
if "counts" in adata_scvi.layers:
    adata_scvi.X = adata_scvi.layers["counts"].copy()

X = adata_scvi.X.toarray() if hasattr(adata_scvi.X, "toarray") else adata_scvi.X.copy()
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
adata_scvi.X = csr_matrix(X.astype(np.float32))

# n_genes_on covariate (z-scored)
n_genes_on_raw = (adata_scvi.X > 0).sum(axis=1).astype(np.float32)
mu_, sd_ = float(n_genes_on_raw.mean()), float(n_genes_on_raw.std())
adata_scvi.obs["n_genes_on"] = (
    (n_genes_on_raw - mu_) / (sd_ + 1e-8)
).astype(np.float32)

print(f"adata_scvi shape: {adata_scvi.shape}")
print(f"X dtype/range: {adata_scvi.X.dtype} / [{adata_scvi.X.min()}, {adata_scvi.X.max()}]")
print(f"n_batches: {adata_scvi.obs['company'].nunique()}")
print(f"n_status: {adata_scvi.obs['status'].nunique()}")

# %%
# ========== Drop NaN company + force dtype BEFORE scVI registration ==========
# ``company`` (batch_key) must be ``category``; ``n_genes_on`` (continuous)
# must be ``float32``. Mixed dtypes are the typical cause of the
# ``RuntimeError: expand(... size=[1, 1, 255])`` raised by one_hot on a
# 4-D integer tensor coming out of the cat-covs registry.
mask = adata_scvi.obs["company"].notna()
adata_scvi = adata_scvi[mask].copy()
adata_scvi.obs["company"] = adata_scvi.obs["company"].astype("category")
adata_scvi.obs["n_genes_on"] = adata_scvi.obs["n_genes_on"].astype(np.float32)
print("after dropna shape :", adata_scvi.shape)
print("company categories :", adata_scvi.obs["company"].cat.categories.tolist())
print("company dtype      :", adata_scvi.obs["company"].dtype)
print("n_genes_on dtype   :", adata_scvi.obs["n_genes_on"].dtype)

# %%
# ========== Import the new model and run setup_anndata + instantiation ==========
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

SCVIWithBatchPairLoss.setup_anndata(
    adata_scvi,
    pair_batch_obs_key="company",  # copied to `_pair_batch` and registered as cat cov
    batch_key=None,                # SCVI does NOT apply batch-correction at all
    labels_key="celltype.L2",
    continuous_covariate_keys=["n_genes_on"],  # pair-batch run uses no continuous covariates
)

# For align_on="mu", pair loss uses mean-adjusted log1p counts and sums
# over genes, so use a smaller coefficient than the raw per-gene MSE version.
PAIR_WEIGHT = 1
ALIGN_ON    = "z"                            # "mu" (decoder rate) or "z" (latent)

model = SCVIWithBatchPairLoss(
    adata_scvi,
    n_latent=32,
    n_layers=2,
    gene_likelihood="zinb",
    pair_weight=PAIR_WEIGHT,
    align_on=ALIGN_ON,
    deeply_inject_covariates=False,  # decoder NEVER sees the pair batch — loss only
)

print(f"Module class:  {type(model.module).__name__}")
print(f"pair_weight:   {model.module.pair_weight}")
print(f"align_on:      {model.module.align_on}")
print(f"n_input genes: {model.module.n_input}")
print(f"n_batch:       {model.module.n_batch}")

# %% [markdown]
# ## Forward validation: pair loss is positive only when a mini-batch spans ≥ 2 batches

# %%
# ========== StratifiedBatchSampler: guarantee every mini-batch spans all batches ==========
class StratifiedBatchSampler(Sampler):
    """Within each mini-batch draw (roughly) the same number of cells from
    every ``batch_key`` category. Guarantees every mini-batch spans >=2
    batches, so the cross-batch pair-MSE term is never degenerate.
    """

    def __init__(self, batch_codes: np.ndarray, batch_size: int, seed: int = 0):
        self.batch_codes = np.asarray(batch_codes)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        self.indices_per_batch = {
            b: np.where(self.batch_codes == b)[0]
            for b in np.unique(self.batch_codes)
        }
        self.batch_categories = list(self.indices_per_batch.keys())
        self.n_batches = len(self.batch_categories)
        self.per_batch_quota = max(1, self.batch_size // self.n_batches)
        self.effective_batch_size = self.per_batch_quota * self.n_batches

        # Number of mini-batches per epoch = the longest batch pool divided by quota.
        max_len = max(len(v) for v in self.indices_per_batch.values())
        self.num_iter = max_len // self.per_batch_quota

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {b: rng.permutation(idx).tolist()
                 for b, idx in self.indices_per_batch.items()}
        for it in range(self.num_iter):
            chunk = []
            for ids in pools.values():
                chunk.extend(ids[it * self.per_batch_quota:(it + 1) * self.per_batch_quota])
            rng.shuffle(chunk)
            yield chunk

    def __len__(self):
        return self.num_iter


# Build a sampler on the company batches, with effective batch_size = 256.
batch_codes = adata_scvi.obs["company"].cat.codes.to_numpy()
stratified_sampler = StratifiedBatchSampler(batch_codes, batch_size=256, seed=seed)
print(f"stratified sampler: {stratified_sampler.n_batches} batches × "
      f"{stratified_sampler.per_batch_quota} cells/batch = "
      f"effective batch size {stratified_sampler.effective_batch_size}, "
      f"{stratified_sampler.num_iter} mini-batches per epoch")

# %%
# ========== Forward sanity check (shuffle=True → mini-batches mix batches) ==========
train_loader = model._make_data_loader(
    adata=adata_scvi,
    indices=np.arange(adata_scvi.n_obs),
    batch_size=256,
    shuffle=True,
)

for tensors in train_loader:
    inf, gen, loss = model.module.forward(tensors, compute_loss=True)
    print(f"px.mu shape:        {tuple(gen['px'].mu.shape)}")
    print(f"inference z shape:  {tuple(inf['z'].shape)}")
    print(f"loss (with pair):   {float(loss.loss):.4f}")
    print(f"  extra_metrics:    {loss.extra_metrics}")

    assert "batch_pair_mse" in loss.extra_metrics
    assert "batch_pair_n" in loss.extra_metrics
    assert loss.extra_metrics["batch_pair_n"] > 0, (
        f"expected cross-batch pairs, got n={loss.extra_metrics['batch_pair_n']}"
    )
    print("✓ pair_mse > 0  ✓ n_pairs > 0 (cross-batch pairs present)")
    break

# %%


# %%
# ========== Degenerate case: single-batch mini-batch must give pair_mse == 0 ==========
# (a fresh subset AnnData re-uses the same registry; its _pair_batch column has
#  one single category, so no cross-batch pairs exist.)
single_idx = adata_scvi.obs.index[adata_scvi.obs["company"] == adata_scvi.obs["company"].iloc[0]]
sub = adata_scvi[single_idx].copy()
scdl_single = model._make_data_loader(adata=sub, batch_size=64, shuffle=False)
for tensors in scdl_single:
    inf, gen, loss = model.module.forward(tensors, compute_loss=True)
    print(f"single-batch mini: pair_mse={loss.extra_metrics['batch_pair_mse']}, "
          f"n_pairs={loss.extra_metrics['batch_pair_n']}")
    assert loss.extra_metrics["batch_pair_n"] == 0
    assert loss.extra_metrics["batch_pair_mse"] == 0.0
    print("✓ single-batch mini returns 0 (degenerate case handled)")
    break

# %% [markdown]
# ## Training: watch `batch_pair_mse` drop while scVI loss stays normal

# %%
# ========== Train ==========
model.train(
    max_epochs=100,
    train_size=0.9,
    batch_size=256,
    # early_stopping=True,
    # early_stopping_patience=5,
    plan_kwargs={"lr": 1e-3},
)
print("✓ training complete")

# %%
# ========== Training curves: total loss + pair_mse ==========
history = model.history_
elbo  = history["elbo_train"].astype(float)
recon = history["reconstruction_loss_train"].astype(float)
kl    = history["kl_local_train"].astype(float)

# scVI logs LossOutput.extra_metrics with names such as
# ``batch_pair_mse_train`` and ``batch_pair_n_train``.
metric_keys = [k for k in history.keys() if "batch_pair" in k]
print(f"Logged pair metrics: {metric_keys}")

pair_mse_per_epoch = None
for k in metric_keys:
    if "batch_pair_mse" in k:
        pair_mse_per_epoch = np.asarray(history[k], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(elbo,  label="ELBO (train)")
axes[0].plot(recon, label="reconstruction_loss_train")
axes[0].plot(kl,    label="kl_local_train")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")
axes[0].legend()
axes[0].set_title("Training losses")
axes[0].grid(alpha=0.3)

if pair_mse_per_epoch is not None:
    axes[1].plot(pair_mse_per_epoch, label="batch_pair_mse", color="tab:red")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("pair MSE")
    axes[1].set_title("Cross-batch pair MSE (should drop)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, "batch_pair_mse not logged", ha="center")

plt.tight_layout()
fig.savefig(SLIDE_FIG_DIR / "training_losses_pair_mse.png", dpi=300, bbox_inches="tight")
fig.savefig(SLIDE_FIG_DIR / "training_losses_pair_mse.pdf", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Latent space & alignment evaluation

# %%
# ========== Latent embedding + UMAP by batch ==========
adata_scvi.obsm["X_scVI_pair"] = model.get_latent_representation()

sc.pp.neighbors(adata_scvi, use_rep="X_scVI_pair")
sc.tl.umap(adata_scvi)
umap_fig = sc.pl.umap(
    adata_scvi,
    color=["company", "status"],
    ncols=2,
    title=["UMAP by company (batch)", "UMAP by status (bio)"],
    show=False,
    return_fig=True,
)
umap_fig.savefig(SLIDE_FIG_DIR / "umap_pair_model.png", dpi=300, bbox_inches="tight")
umap_fig.savefig(SLIDE_FIG_DIR / "umap_pair_model.pdf", bbox_inches="tight")
plt.close(umap_fig)

# %%
# ========== iLISI + silhouette: how well are batches mixed? ==========
try:
    sce.pp.lisi_knn(
        adata_scvi,
        key=["company", "status"],
        use_rep="X_scVI_pair",
    )
    ilisi_batch = float(adata_scvi.obsm["X_lisi_company"].mean())
    ilisi_status = float(adata_scvi.obsm["X_lisi_status"].mean())
    n_batch = adata_scvi.obs["company"].nunique()
    print(f"iLISI (company):  {ilisi_batch:.3f}   (max = {n_batch}, larger = better mixed)")
    print(f"iLISI (status):   {ilisi_status:.3f}   (small = biology preserved)")
except Exception as exc:
    print(f"[warn] scanpy.external unavailable: {exc}")

asw_batch = silhouette_score(adata_scvi.obsm["X_scVI_pair"], adata_scvi.obs["company"])
status_labels = adata_scvi.obs["status"].astype(str)
if status_labels.nunique() >= 2:
    asw_bio = silhouette_score(adata_scvi.obsm["X_scVI_pair"], status_labels)
else:
    asw_bio = np.nan
    print("[warn] ASW(status) skipped: only one status category is present")
print(f"ASW (company, ↓): {asw_batch:.4f}")
print(f"ASW (status,  ↑): {asw_bio:.4f}" if np.isfinite(asw_bio)
      else "ASW (status,  ↑): NaN")

# %%
# ========== Compare to plain SCVI (no pair loss) ==========
_SCVI.setup_anndata(
    adata_scvi,
    batch_key="company",
    labels_key=None,
    continuous_covariate_keys=["n_genes_on"],
)
m_plain = _SCVI(adata_scvi, n_latent=32, n_layers=2, gene_likelihood="zinb")
m_plain.train(max_epochs=100, train_size=0.9, batch_size=256,
              early_stopping=True, early_stopping_patience=5,
              plan_kwargs={"lr": 1e-3})
# With batch_key="company", this is the native scVI embedding after
# removing company effects. transform_batch is for expression reconstruction;
# the latent representation is already batch-corrected by the SCVI model.
adata_scvi.obsm["X_scVI_batch_corrected"] = m_plain.get_latent_representation()

sc.pp.neighbors(
    adata_scvi,
    use_rep="X_scVI_batch_corrected",
    key_added="neighbors_scvi_batch_corrected",
)
sc.tl.umap(
    adata_scvi,
    neighbors_key="neighbors_scvi_batch_corrected",
    key_added="umap_scvi_batch_corrected",
)
scvi_corrected_fig = sc.pl.embedding(
    adata_scvi,
    basis="umap_scvi_batch_corrected",
    color=["company", "status"],
    ncols=2,
    title=["Native SCVI batch-corrected: company", "Native SCVI batch-corrected: status"],
    show=False,
    return_fig=True,
)
scvi_corrected_fig.savefig(
    SLIDE_FIG_DIR / "umap_scvi_batch_corrected.png",
    dpi=300,
    bbox_inches="tight",
)
scvi_corrected_fig.savefig(
    SLIDE_FIG_DIR / "umap_scvi_batch_corrected.pdf",
    bbox_inches="tight",
)
plt.close(scvi_corrected_fig)

asw_batch_plain = silhouette_score(
    adata_scvi.obsm["X_scVI_batch_corrected"], adata_scvi.obs["company"]
)
if status_labels.nunique() >= 2:
    asw_bio_plain = silhouette_score(
        adata_scvi.obsm["X_scVI_batch_corrected"], status_labels
    )
else:
    asw_bio_plain = np.nan

summary = pd.DataFrame(
    {
        "model": ["Native SCVI batch-corrected", "SCVI + pair-MSE"],
        "ASW_company(↓)": [asw_batch_plain, asw_batch],
        "ASW_status(↑)":  [asw_bio_plain,   asw_bio],
    }
)
print("\n=== Batch alignment summary ===")
print(summary.to_string(index=False))

# %% [markdown]
# ## Direct check: decoder `mu` is more uniform across batches after training

# %%
# ========== Compare baseline SCVI vs pair model: per-pair cross-batch MSE on mu ==========
def _per_cell_mu(model_obj):
    out = []
    scdl = model_obj._make_data_loader(adata=adata_scvi, batch_size=512)
    for tensors in scdl:
        inf, gen = model_obj.module.forward(tensors, compute_loss=False)
        out.append(gen["px"].mu.detach().cpu())
    return torch.cat(out, 0)

mu_plain = _per_cell_mu(m_plain)
mu_pair  = _per_cell_mu(model)

batch_idx = torch.as_tensor(
    adata_scvi.obs["company"].cat.codes.to_numpy(), dtype=torch.long
)
loss_plain, n = _cross_batch_pair_mse(mu_plain, batch_idx)
loss_pair,  _ = _cross_batch_pair_mse(mu_pair,  batch_idx)

print(f"cross-batch mu-MSE on baseline SCVI: {float(loss_plain):.4f}")
print(f"cross-batch mu-MSE on pair-trained : {float(loss_pair):.4f}")
print(f"reduction: {(1 - float(loss_pair) / float(loss_plain)) * 100:.1f}%")

# %%
# ========== Final summary ==========
print("=== testSCVIbatch summary ===")
print(f"Pair weight: {PAIR_WEIGHT}, align_on: {ALIGN_ON}")
print(f"mu-MSE baseline={float(loss_plain):.4f}  pair={float(loss_pair):.4f}")
print("✓ Inherited SCVI training pipeline + cross-batch pair-MSE term")


n_samples = 1   # 每细胞 1 个 reconstruction sample
reconstructed = model.posterior_predictive_sample(
    adata=adata_scvi,
    n_samples=n_samples,
    batch_size=512,
    silent=False,
)
# 返回 shape: (n_samples, n_cells, n_genes) → squeeze 到 (n_cells, n_genes)
reconstructed = np.asarray(reconstructed.todense())

print(f"reconstructed shape: {reconstructed.shape}  dtype={reconstructed.dtype}")
print(f"range: [{reconstructed.min()}, {reconstructed.max()}]  median={np.median(reconstructed):.2f}")
print(f"fraction non-zero: {(reconstructed > 0).mean():.3f}")

# 3) 写回 adata —— 稀疏 int 矩阵，节省 h5ad 存储
adata_scvi.layers["batchpair_counts"] = to_sparse_int(reconstructed)


# %%
# ========== Native SCVI reconstruction with company transformed to beirui ==========
# ``m_plain`` was trained with batch_key="company". Keeping each cell's
# latent biology but decoding every cell under the same company condition
# gives the native scVI batch-corrected posterior-predictive counts.
SCVI_TRANSFORM_BATCH = "beirui"
available_companies = adata_scvi.obs["company"].astype(str).unique().tolist()
if SCVI_TRANSFORM_BATCH not in available_companies:
    raise ValueError(
        f"SCVI_TRANSFORM_BATCH={SCVI_TRANSFORM_BATCH!r} is not present in "
        f"company values: {available_companies}"
    )

scvi_reconstructed = m_plain.posterior_predictive_sample(
    adata=adata_scvi,
    n_samples=1,
    batch_size=512,
    transform_batch=[SCVI_TRANSFORM_BATCH],
    silent=False,
)

if hasattr(scvi_reconstructed, "todense"):
    scvi_reconstructed = np.asarray(scvi_reconstructed.todense())
else:
    scvi_reconstructed = np.asarray(scvi_reconstructed)

# scVI may return (n_samples, n_cells, n_genes) or (n_cells, n_genes).
if scvi_reconstructed.ndim == 3 and scvi_reconstructed.shape[0] == 1:
    scvi_reconstructed = scvi_reconstructed[0]

print(f"native SCVI transformed batch: {SCVI_TRANSFORM_BATCH}")
print(f"reconstructed shape: {scvi_reconstructed.shape}  dtype={scvi_reconstructed.dtype}")
print(
    f"range: [{scvi_reconstructed.min()}, {scvi_reconstructed.max()}] "
    f"median={np.median(scvi_reconstructed):.2f}"
)
print(f"fraction non-zero: {(scvi_reconstructed > 0).mean():.3f}")

adata_scvi.layers["scvi_beirui_counts"] = to_sparse_int(scvi_reconstructed)

# Deterministic native-SCVI normalized expression at a shared library size.
# This is preferable to posterior-predictive counts for downstream plots.
SCVI_LIBRARY_SIZE = 1e4
scvi_beirui_normalized = m_plain.get_normalized_expression(
    adata=adata_scvi,
    transform_batch=[SCVI_TRANSFORM_BATCH],
    library_size=SCVI_LIBRARY_SIZE,
    batch_size=512,
    return_numpy=True,
)
scvi_beirui_normalized = np.asarray(scvi_beirui_normalized, dtype=np.float32)
adata_scvi.layers["scvi_beirui_normalized"] = csr_matrix(
    scvi_beirui_normalized
)
print(
    "native SCVI normalized layer:",
    scvi_beirui_normalized.shape,
    f"library_size={SCVI_LIBRARY_SIZE:g}",
)


# %%
# ========== Decode pair-model embeddings with the native SCVI decoder ==========
# This intentionally does NOT use transform_batch. The pair-model latent
# embedding is injected into the native SCVI decoder, while each cell keeps
# its original company index and library size from the plain SCVI model.
z_pair = model.get_latent_representation(
    adata=adata_scvi,
    batch_size=512,
)

plain_decoder_means = []
m_plain.module.eval()
with torch.no_grad():
    plain_loader = m_plain._make_data_loader(
        adata=adata_scvi,
        indices=np.arange(adata_scvi.n_obs),
        batch_size=512,
        shuffle=False,
    )
    cell_start = 0
    for tensors in plain_loader:
        plain_inference_inputs = m_plain.module._get_inference_input(tensors)
        plain_inference = m_plain.module.inference(**plain_inference_inputs)
        plain_generative_inputs = m_plain.module._get_generative_input(
            tensors,
            plain_inference,
        )

        cell_stop = cell_start + tensors["X"].shape[0]
        plain_generative_inputs[MODULE_KEYS.Z_KEY] = torch.as_tensor(
            z_pair[cell_start:cell_stop],
            dtype=plain_inference[MODULE_KEYS.Z_KEY].dtype,
            device=plain_inference[MODULE_KEYS.Z_KEY].device,
        )
        plain_decoder_outputs = m_plain.module.generative(
            **plain_generative_inputs,
        )
        plain_decoder_means.append(
            plain_decoder_outputs[MODULE_KEYS.PX_KEY].mu.detach().cpu()
        )
        cell_start = cell_stop

plain_decoder_pair_embedding = torch.cat(plain_decoder_means, dim=0).numpy()
print(
    "native SCVI decoder + pair embedding shape:",
    plain_decoder_pair_embedding.shape,
)
print(
    "native SCVI decoder + pair embedding range:",
    f"[{plain_decoder_pair_embedding.min():.2f}, "
    f"{plain_decoder_pair_embedding.max():.2f}]",
)

adata_scvi.layers["scvi_decoder_pair_embedding_counts"] = to_sparse_int(
    plain_decoder_pair_embedding
)


# ========== Pair-model normalized expression with a shared library size ==========
# Decode the pair model's own aligned latent means with its own decoder.
# Unlike posterior_predictive_sample, this is deterministic and does not use
# each cell's original library size.
pair_decoder_means = []
model.module.eval()
with torch.no_grad():
    pair_loader = model._make_data_loader(
        adata=adata_scvi,
        indices=np.arange(adata_scvi.n_obs),
        batch_size=512,
        shuffle=False,
    )
    cell_start = 0
    for tensors in pair_loader:
        inference_inputs = model.module._get_inference_input(tensors)
        inference_outputs = model.module.inference(**inference_inputs)
        generative_inputs = model.module._get_generative_input(
            tensors,
            inference_outputs,
        )

        cell_stop = cell_start + tensors["X"].shape[0]
        generative_inputs[MODULE_KEYS.Z_KEY] = torch.as_tensor(
            z_pair[cell_start:cell_stop],
            dtype=inference_outputs[MODULE_KEYS.Z_KEY].dtype,
            device=inference_outputs[MODULE_KEYS.Z_KEY].device,
        )
        generative_inputs[MODULE_KEYS.LIBRARY_KEY] = torch.full_like(
            generative_inputs[MODULE_KEYS.LIBRARY_KEY],
            np.log(SCVI_LIBRARY_SIZE),
        )
        generated = model.module.generative(**generative_inputs)
        pair_decoder_means.append(
            generated[MODULE_KEYS.PX_KEY].mu.detach().cpu()
        )
        cell_start = cell_stop

batchpair_normalized = torch.cat(pair_decoder_means, dim=0).numpy().astype(
    np.float32
)
adata_scvi.layers["batchpair_normalized"] = csr_matrix(batchpair_normalized)
print(
    "pair model normalized layer:",
    batchpair_normalized.shape,
    f"library_size={SCVI_LIBRARY_SIZE:g}",
)


hkg_priority = [
    "Aars", "Sars", "Polr2a", "Polr2f", "Psmd6", "Psmd7", "Psma5",
    "Rer1", "Ipo8", "Pop4", "Pes1", "Oaz1", "Rpl13a", "Rpl27",
    "Rps13", "Rps20", "Hprt1", "Gusb", "Ppia", "Ywhaz", "Cyc1",
    "Sdha", "Ubc",
]

# Keep only housekeeping genes available in this AnnData object and preserve
# the curated order so both methods use exactly the same y-axis genes.
hkg_genes = []
seen_genes = set()
for gene in hkg_priority:
    if gene in adata_scvi.var_names and gene not in seen_genes:
        hkg_genes.append(gene)
        seen_genes.add(gene)

print(f"HKG genes available: {len(hkg_genes)} / {len(hkg_priority)}")
if len(hkg_genes) < 2:
    print("[warn] Skip HKG dotplots: fewer than 2 HKG genes are present")
else:
    # Since this test subset contains CON cells only, company is the useful
    # comparison axis for these two corrected expression layers.
    for layer_name, file_stem, title in [
        ("batchpair_counts", "hkg_dotplot_batchpair", "Pair-MSE model"),
        ("scvi_beirui_counts", "hkg_dotplot_scvi_beirui", "Native SCVI -> beirui"),
        (
            "scvi_decoder_pair_embedding_counts",
            "hkg_dotplot_scvi_decoder_pair_embedding",
            "Native SCVI decoder <- pair embedding",
        ),
        (
            "batchpair_normalized",
            "hkg_dotplot_batchpair_normalized",
            "Pair-MSE normalized",
        ),
        (
            "scvi_beirui_normalized",
            "hkg_dotplot_scvi_beirui_normalized",
            "Native SCVI -> beirui normalized",
        ),
        ("counts", "hkg_dotplot_raw", "raw counts"),
    ]:
        fig = sc.pl.dotplot(
            adata_scvi,
            var_names=hkg_genes,
            groupby="sample",
            layer=layer_name,
            swap_axes=False,
            dendrogram=False,
            return_fig=True,
            vmax=1,
            show=False,
        )
        fig_out = SLIDE_FIG_DIR / f"{file_stem}.count.png"
        fig.savefig(fig_out, bbox_inches="tight", dpi=200)
        print(f"✓ saved: {fig_out}")

