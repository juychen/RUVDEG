"""Train a frozen RAW scVI plus a low-dimensional UV correction.

Stage 1 trains or loads a plain no-batch RAW scVI model.  Stage 2 never
updates that model.  It learns a small deterministic pathway:

    z_raw -> UV embedding u -> delta_z
    z_emend = z_raw + delta_z

The pair loss is evaluated on ``z_emend`` for same-celltype, cross-batch
pairs.  The saved UV embedding can be used as a downstream covariate, while
corrected expression layers are decoded with the frozen RAW scVI decoder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy.sparse import csr_matrix, issparse
from scvi.module._constants import MODULE_KEYS
from torch.utils.data import DataLoader, TensorDataset

from model_scvi_batch_pair import CelltypeBatchStratifiedSampler
from model_scvi_batch_pair_embedding import UVEmbeddingModel
from scvi.model._utils import get_max_epochs_heuristic


parser = argparse.ArgumentParser(
    description="Frozen RAW scVI with low-dimensional UV embedding pair correction"
)
parser.add_argument("--input", "-i", required=True, help="Input h5ad path")
parser.add_argument("--outprefix", "-o", required=True, help="Output h5ad prefix")
parser.add_argument("--pair-batch-key", default="company")
parser.add_argument("--labels-key", default="celltype.L2")
parser.add_argument("--condition-key", default=None)
parser.add_argument("--no-cont-cov", action="store_true")
parser.add_argument("--n-latent", type=int, default=32)
parser.add_argument("--n-layers", type=int, default=2)
parser.add_argument("--uv-dim", type=int, default=2)
parser.add_argument("--uv-hidden", type=int, default=32)
parser.add_argument("--uv-layers", type=int, default=1)
parser.add_argument("--delta-scale", type=float, default=1.0)
parser.add_argument("--pair-weight", type=float, default=1.0)
parser.add_argument("--reconstruction-weight", type=float, default=1.0)
parser.add_argument("--delta-penalty", type=float, default=1e-3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--max-epochs", type=int, default=None,
    help="maximum training epochs; None uses scVI automatic epoch selection "
    "(get_max_epochs_heuristic, cap=400)",
)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--min-cells-per-group", type=int, default=2)
parser.add_argument("--skip-stage1", action="store_true")
parser.add_argument("--raw-model-dir", default="")
parser.add_argument("--no-decoder-output", action="store_true")
args, _ = parser.parse_known_args()

INPUT_H5AD = os.path.abspath(args.input)
OUTPREFIX = os.path.abspath(args.outprefix)
OUTBASE = os.path.splitext(OUTPREFIX)[0]
OUTBASE_FULL = OUTBASE + "_full"
OUTDIR = os.path.dirname(OUTBASE) or "."
MODEL_DIR = OUTBASE + ".model"
STAGE1_MODEL_DIR = (
    os.path.abspath(args.raw_model_dir)
    if args.raw_model_dir.strip()
    else OUTBASE + "_scvi_nobatch.model"
)
USE_CONT_COVS = not args.no_cont_cov
PAIR_BATCH_KEY = args.pair_batch_key
LABELS_KEY = args.labels_key
CONDITION_KEY = args.condition_key
os.makedirs(OUTDIR, exist_ok=True)


def to_sparse(arr, dtype=np.float32):
    if issparse(arr):
        return arr.astype(dtype)
    return csr_matrix(np.asarray(arr, dtype=dtype))


def to_sparse_int(arr, dtype=np.int32):
    if not issparse(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    result = csr_matrix(arr)
    result.data = np.rint(result.data).astype(dtype)
    result.eliminate_zeros()
    return result


def set_counts_as_x(adata):
    if "counts" not in adata.layers:
        if adata.X is None:
            raise KeyError("layers['counts'] is missing and X is empty")
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()


def decode_with_fixed_latent(model, adata, z, batch_size=512, lib_size=1e4):
    """Decode fixed latent values with a frozen scVI model."""
    z = np.asarray(z, dtype=np.float32)
    if z.shape[0] != adata.n_obs:
        raise ValueError("z and adata must contain the same number of cells")
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


def collect_reconstruction_context(model, adata, batch_size):
    """Cache frozen RAW decoder inputs and observed counts in AnnData order."""
    model._validate_anndata(adata)
    model.module.eval()
    context = {
        "x": [],
        "library": [],
        "batch_index": [],
        "cont_covs": [],
        "cat_covs": [],
        "size_factor": [],
        "y": [],
    }
    with torch.no_grad():
        loader = model._make_data_loader(
            adata=adata,
            indices=np.arange(adata.n_obs),
            batch_size=batch_size,
            shuffle=False,
        )
        for tensors in loader:
            inference_input = model.module._get_inference_input(tensors)
            inference_outputs = model.module.inference(**inference_input)
            generative_inputs = model.module._get_generative_input(
                tensors, inference_outputs
            )
            context["x"].append(tensors["X"].detach().cpu())
            for key in context:
                if key == "x":
                    continue
                module_key = getattr(MODULE_KEYS, f"{key.upper()}_KEY")
                value = generative_inputs.get(module_key)
                if value is None:
                    context[key].append(None)
                else:
                    context[key].append(value.detach().cpu())

    output = {"x": torch.cat(context["x"], dim=0)}
    for key in context:
        if key == "x":
            continue
        values = context[key]
        if all(value is None for value in values):
            output[key] = None
        elif any(value is None for value in values):
            raise RuntimeError(f"inconsistent RAW decoder context for {key}")
        else:
            output[key] = torch.cat(values, dim=0)
    return output


def reconstruction_loss(raw_model, z_emend, context, indices, device):
    """RAW decoder NLL per gene; gradients flow only through ``z_emend``."""
    def select(value):
        if value is None:
            return None
        return value[indices].to(device)

    generative_inputs = {
        MODULE_KEYS.Z_KEY: z_emend,
        MODULE_KEYS.LIBRARY_KEY: select(context["library"]),
        MODULE_KEYS.BATCH_INDEX_KEY: select(context["batch_index"]),
        MODULE_KEYS.CONT_COVS_KEY: select(context["cont_covs"]),
        MODULE_KEYS.CAT_COVS_KEY: select(context["cat_covs"]),
        MODULE_KEYS.SIZE_FACTOR_KEY: select(context["size_factor"]),
        MODULE_KEYS.Y_KEY: select(context["y"]),
    }
    distribution = raw_model.module.generative(**generative_inputs)[
        MODULE_KEYS.PX_KEY
    ]
    x = context["x"][indices].to(device)
    return -distribution.log_prob(x).sum(dim=-1).mean() / x.shape[1]


def prepare_data():
    adata_all = sc.read_h5ad(INPUT_H5AD)
    set_counts_as_x(adata_all)

    if CONDITION_KEY is not None:
        if CONDITION_KEY not in adata_all.obs:
            raise KeyError(f"condition key not found: {CONDITION_KEY!r}")
        if LABELS_KEY not in adata_all.obs:
            raise KeyError(f"labels key not found: {LABELS_KEY!r}")
        adata_all.obs["_pair_group"] = (
            adata_all.obs[LABELS_KEY].astype(str)
            + "__"
            + adata_all.obs[CONDITION_KEY].astype(str)
        ).astype("category")
        pair_labels_key = "_pair_group"
    else:
        pair_labels_key = LABELS_KEY

    if pair_labels_key not in adata_all.obs:
        raise KeyError(f"labels key not found: {pair_labels_key!r}")
    if PAIR_BATCH_KEY not in adata_all.obs:
        raise KeyError(f"pair-batch key not found: {PAIR_BATCH_KEY!r}")

    if USE_CONT_COVS:
        n_genes_raw = np.asarray(
            (adata_all.layers["counts"] > 0).sum(axis=1)
        ).ravel().astype(np.float32)
        mean = float(n_genes_raw.mean())
        std = float(n_genes_raw.std())
        if std < 1e-8:
            raise ValueError("n_genes_on has insufficient variation")
        adata_all.obs["n_genes_on"] = ((n_genes_raw - mean) / std).astype(
            np.float32
        )

    if CONDITION_KEY is None and "status" in adata_all.obs:
        train_mask = adata_all.obs["status"].isin(["CON"])
    else:
        train_mask = np.ones(adata_all.n_obs, dtype=bool)

    train_mask &= adata_all.obs[PAIR_BATCH_KEY].notna().to_numpy()
    train_mask &= adata_all.obs[pair_labels_key].notna().to_numpy()
    adata_train = adata_all[train_mask].copy()
    set_counts_as_x(adata_train)
    if adata_train.n_obs == 0:
        raise ValueError("No cells remain for UV training")
    for key in [PAIR_BATCH_KEY, pair_labels_key]:
        adata_train.obs[key] = adata_train.obs[key].astype("category")
    if USE_CONT_COVS:
        adata_train.obs["n_genes_on"] = adata_train.obs["n_genes_on"].astype(
            np.float32
        )

    return adata_all, adata_train, pair_labels_key


def setup_raw_anndata(adata, continuous_covariates):
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=None,
        batch_key=None,
        labels_key=None,
        categorical_covariate_keys=None,
        continuous_covariate_keys=continuous_covariates,
    )


def load_or_train_raw(adata_train, continuous_covariates, accelerator, devices):
    setup_raw_anndata(adata_train, continuous_covariates)
    if args.skip_stage1:
        if not os.path.isdir(STAGE1_MODEL_DIR):
            raise FileNotFoundError(
                f"--skip-stage1 requested but model does not exist: "
                f"{STAGE1_MODEL_DIR}"
            )
        raw_model = scvi.model.SCVI.load(STAGE1_MODEL_DIR, adata=adata_train)
    else:
        raw_model = scvi.model.SCVI(
            adata_train,
            n_layers=args.n_layers,
            n_latent=args.n_latent,
            gene_likelihood="zinb",
        )
        raw_model.train(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            devices=devices,
            plan_kwargs={"lr": args.lr},
        )
        raw_model.save(STAGE1_MODEL_DIR, overwrite=True, save_anndata=False)
    raw_model.module.eval()
    for parameter in raw_model.module.parameters():
        parameter.requires_grad_(False)
    return raw_model


def add_model_outputs(adata, outputs):
    adata.obsm["X_scVI_raw"] = outputs["z_raw"]
    adata.obsm["X_scVI_uv"] = outputs["uv_embedding"]
    adata.obsm["X_scVI_delta"] = outputs["delta_z"]
    adata.obsm["X_scVI_emend"] = outputs["z_emend"]


print(f"input             : {INPUT_H5AD}")
print(f"output base       : {OUTBASE}")
print(f"RAW model         : {STAGE1_MODEL_DIR}")
print(f"uv_dim            : {args.uv_dim}")
print(f"pair_weight       : {args.pair_weight}")
print(f"reconstruction_weight: {args.reconstruction_weight}")
print(f"delta_penalty     : {args.delta_penalty}")
print(f"max_epochs        : {args.max_epochs} (None -> scVI heuristic, cap=400)")
print(f"lr                : {args.lr}")
print(f"n_latent          : {args.n_latent}")
print(f"n_layers          : {args.n_layers}")

adata_all, adata_train, pair_labels_key = prepare_data()
CONT_COVS = ["n_genes_on"] if USE_CONT_COVS else None

if torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
    devices = [0]
else:
    device = torch.device("cpu")
    accelerator = "cpu"
    devices = 1

raw_model = load_or_train_raw(adata_train, CONT_COVS, accelerator, devices)
raw_state_before = {
    name: value.detach().cpu().clone()
    for name, value in raw_model.module.state_dict().items()
}
z_train = raw_model.get_latent_representation(
    adata=adata_train, give_mean=True, batch_size=args.batch_size
).astype(np.float32)
reconstruction_context = collect_reconstruction_context(
    raw_model, adata_train, args.batch_size
)

batch_codes = adata_train.obs[PAIR_BATCH_KEY].cat.codes.to_numpy(dtype=np.int64)
cell_type_codes = adata_train.obs[pair_labels_key].cat.codes.to_numpy(dtype=np.int64)
if (batch_codes < 0).any() or (cell_type_codes < 0).any():
    raise ValueError("Training pair labels contain missing category codes")

sampler = CelltypeBatchStratifiedSampler(
    celltype_codes=cell_type_codes,
    batch_codes=batch_codes,
    batch_size=args.batch_size,
    min_cells_per_group=args.min_cells_per_group,
)
print(
    f"training cells={len(z_train)}  pair groups={sampler.n_groups}  "
    f"effective batch size={sampler.effective_batch_size}  "
    f"iterations/epoch={sampler.num_iter}"
)

uv_model = UVEmbeddingModel(
    raw_latent_dim=z_train.shape[1],
    uv_dim=args.uv_dim,
    hidden_dim=args.uv_hidden,
    uv_layers=args.uv_layers,
    delta_scale=args.delta_scale,
    pair_weight=args.pair_weight,
    reconstruction_weight=args.reconstruction_weight,
    delta_penalty=args.delta_penalty,
).to(device)
train_dataset = TensorDataset(
    torch.from_numpy(z_train),
    torch.from_numpy(batch_codes),
    torch.from_numpy(cell_type_codes),
)
train_loader = sampler
optimizer = torch.optim.Adam(uv_model.parameters(), lr=args.lr)

history = []
if args.max_epochs is None:
    max_epochs = get_max_epochs_heuristic(len(z_train))
    print(
        f"max_epochs not provided; using scVI heuristic: {max_epochs} "
        f"epochs (n_train={len(z_train)})"
    )
else:
    max_epochs = args.max_epochs
for epoch in range(max_epochs):
    uv_model.train()
    epoch_metrics = []
    for batch_indices in train_loader:
        batch_indices = np.asarray(batch_indices, dtype=np.int64)
        z_batch = train_dataset.tensors[0][batch_indices]
        batch_batch = train_dataset.tensors[1][batch_indices]
        cell_type_batch = train_dataset.tensors[2][batch_indices]
        z_batch = z_batch.to(device)
        batch_batch = batch_batch.to(device)
        cell_type_batch = cell_type_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = uv_model.training_loss(
            z_batch,
            batch_batch,
            cell_type_batch,
            reconstruction_loss=reconstruction_loss(
                raw_model,
                uv_model(z_batch)["z_emend"],
                reconstruction_context,
                batch_indices,
                device,
            ),
        )
        loss.backward()
        optimizer.step()
        epoch_metrics.append(metrics)

    if not epoch_metrics:
        raise RuntimeError("The pair-aware training loader yielded no batches")
    summary = {
        key: float(np.mean([item[key] for item in epoch_metrics]))
        for key in [
            "loss",
            "pair_loss",
            "reconstruction_loss",
            "delta_penalty",
            "delta_rms",
        ]
    }
    summary["pair_n"] = int(sum(item["pair_n"] for item in epoch_metrics))
    summary["epoch"] = epoch + 1
    history.append(summary)
    print(
        f"epoch {epoch + 1:03d}: loss={summary['loss']:.6g}  "
        f"pair={summary['pair_loss']:.6g}  "
        f"recon={summary['reconstruction_loss']:.6g}  "
        f"delta_rms={summary['delta_rms']:.6g}  "
        f"pairs={summary['pair_n']}"
    )

for name, value in raw_model.module.state_dict().items():
    if not torch.equal(value.detach().cpu(), raw_state_before[name]):
        raise AssertionError(f"RAW scVI parameter changed during UV training: {name}")
print("RAW scVI parameter snapshot unchanged")

uv_model.save(MODEL_DIR, raw_model_dir=STAGE1_MODEL_DIR)
Path(MODEL_DIR, "history.json").write_text(
    json.dumps(history, indent=2), encoding="utf-8"
)

uv_model.eval()
with torch.no_grad():
    train_outputs = {
        key: value.detach().cpu().numpy()
        for key, value in uv_model(torch.from_numpy(z_train).to(device)).items()
    }
add_model_outputs(adata_train, train_outputs)

adata_eval = adata_all.copy()
set_counts_as_x(adata_eval)
setup_raw_anndata(adata_eval, CONT_COVS)
z_full = raw_model.get_latent_representation(
    adata=adata_eval, give_mean=True, batch_size=args.batch_size
).astype(np.float32)
with torch.no_grad():
    full_outputs = {
        key: value.detach().cpu().numpy()
        for key, value in uv_model(torch.from_numpy(z_full).to(device)).items()
    }
add_model_outputs(adata_eval, full_outputs)

if not args.no_decoder_output:
    train_counts, train_normalized = decode_with_fixed_latent(
        raw_model, adata_train, train_outputs["z_emend"], args.batch_size
    )
    adata_train.layers["scvi_uv_reconstructed_counts"] = to_sparse_int(
        train_counts
    )
    adata_train.layers["scvi_uv_normalized_counts"] = to_sparse(
        train_normalized
    )

    full_counts, full_normalized = decode_with_fixed_latent(
        raw_model, adata_eval, full_outputs["z_emend"], args.batch_size
    )
    adata_eval.layers["scvi_uv_reconstructed_counts"] = to_sparse_int(full_counts)
    adata_eval.layers["scvi_uv_normalized_counts"] = to_sparse(full_normalized)

adata_train.write_h5ad(OUTBASE + ".h5ad", compression="gzip")
adata_eval.write_h5ad(OUTBASE_FULL + ".h5ad", compression="gzip")

uv_columns = [f"UV_{index + 1}" for index in range(args.uv_dim)]
uv_frame = pd.DataFrame(
    full_outputs["uv_embedding"], index=adata_eval.obs_names, columns=uv_columns
)
uv_frame.index.name = "cell"
uv_frame.to_csv(OUTBASE_FULL + ".uv_embedding.csv")

print(f"saved training output : {OUTBASE}.h5ad")
print(f"saved full output     : {OUTBASE_FULL}.h5ad")
print(f"saved UV covariate    : {OUTBASE_FULL}.uv_embedding.csv")
print(f"saved UV model        : {MODEL_DIR}")
