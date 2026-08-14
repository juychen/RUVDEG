"""SCVI model with an additional cross-batch MSE alignment loss.

Inherits from :class:`scvi.model.SCVI`. The module's ``loss`` is augmented
with a term that minimises the squared difference between cells from
*different* batches:

    pair_loss = mean over (b_i, b_j), i<j  mean over (cell in b_i, cell in b_j)
                    || mu[i] - mu[j] ||^2

where ``log1p(mu)`` is the decoder rate per cell (default), or the latent
``z`` when ``align_on="z"``. For ``mu``, squared differences are summed over
genes and averaged over cell pairs, matching scVI's per-cell reconstruction
reduction more closely while avoiding raw-count scale effects. The total loss is

    loss = scvi_loss + pair_weight * pair_loss

Notes
-----
* Pairing is done **per mini-batch**: for each mini-batch we build the
  Cartesian product of cells across distinct batch indices (i, j with
  i < j), so each unordered cross-batch pair is counted once.
* Cells from the same batch index are NOT paired (avoid within-batch
  trivial comparisons and save compute).
* If a mini-batch contains only one batch, ``pair_loss = 0``.

Usage
-----
>>> from model_scvi_batch_pair import SCVIWithBatchPairLoss
>>> SCVIWithBatchPairLoss.setup_anndata(
...     adata, batch_key=None,
...     categorical_covariate_keys=["_pair_batch"],
...     continuous_covariate_keys=["n_genes_on"],
... )
>>> model = SCVIWithBatchPairLoss(
...     adata, n_latent=32, n_layers=2, gene_likelihood="zinb",
...     pair_weight=1.0, align_on="mu",
... )
>>> model.train()

The pairing batch (``adata.obs["_pair_batch"]``) is registered as an
**extra categorical covariate**, NOT as ``batch_key``. SCVI therefore
does not apply any batch-correction effect to it; the column is only
used to compute the cross-batch pair-MSE loss.
"""

from __future__ import annotations

import torch

from scvi import REGISTRY_KEYS
from scvi.model import SCVI
from scvi.module import VAE
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import LossOutput

# Private obs column where the pairing batch is copied before registration.
# The actual tensor inside a mini-batch arrives under
# ``REGISTRY_KEYS.CAT_COVS_KEY`` ("extra_categorical_covs"), NOT under this
# name, because scVI routes categorical covariates through its registry.
PAIR_BATCH_KEY = "_pair_batch"


def _cross_batch_pair_mse(
    x: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_type_idx: torch.Tensor | None = None,
    reduce: str = "mean",
) -> tuple[torch.Tensor, int]:
    """Mean squared difference between cells from *different* batches.

    By default, pairs are taken across the full mini-batch (any two cells
    with different batch indices). If ``cell_type_idx`` is provided, pairs
    are restricted to cells of the **same** cell type -- this avoids pulling
    excitatory/inhibitory cell profiles of different batches together when
    cell-type composition is batch-confounded.

    Parameters
    ----------
    x
        Tensor of shape ``(n_obs, n_features)`` -- per-cell quantity to
        align (e.g. decoder rate ``mu`` or latent ``z``).
    batch_idx
        Long tensor of shape ``(n_obs,)`` -- batch index of each cell.
    cell_type_idx
        Optional long tensor of shape ``(n_obs,)`` -- cell-type index of
        each cell. If ``None``, behaves as the original cross-batch version.
    reduce
        ``"mean"`` or ``"sum"`` reduction over features within each pair.

    Returns
    -------
    pair_loss
        Scalar tensor (requires grad): mean over cell pairs of the summed
        squared feature differences. Returns ``0`` when the mini-batch
        contains fewer than two batches (within a cell type when filtered).
    n_pairs
        Number of unordered cell-cell pairs used (= 0 when degenerate).
    """
    if reduce not in {"mean", "sum"}:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")

    if cell_type_idx is not None:
        if cell_type_idx.shape != batch_idx.shape:
            raise ValueError(
                "cell_type_idx must have the same shape as batch_idx; "
                f"got {tuple(cell_type_idx.shape)} vs {tuple(batch_idx.shape)}"
            )

    pair_sq = []
    n_pairs = 0

    # ---- (a) Iterate over cell types when cell_type_idx is given ----
    cell_types = (
        torch.unique(cell_type_idx) if cell_type_idx is not None
        else [None]  # single "all" bucket
    )

    for ct in cell_types:
        if ct is not None:
            ct_mask = cell_type_idx == ct
            if not ct_mask.any():
                continue
            ct_batch = batch_idx[ct_mask]
            ct_x = x[ct_mask]
        else:
            ct_batch = batch_idx
            ct_x = x

        unique_batches = torch.unique(ct_batch)
        if unique_batches.numel() < 2:
            continue

        # Map cell-type-local batch labels to compact 0..k-1 indices so the
        # pair counting below stays correct inside each cell type.
        batch_to_local = {int(b): i for i, b in enumerate(unique_batches)}
        local_idx = torch.tensor(
            [batch_to_local[int(b)] for b in ct_batch.tolist()],
            dtype=torch.long,
            device=ct_batch.device,
        )
        per_local = [ct_x[local_idx == li] for li in range(len(unique_batches))]
        local_sizes = [t.shape[0] for t in per_local]

        for i in range(len(unique_batches)):
            for j in range(i + 1, len(unique_batches)):
                a = per_local[i]
                b = per_local[j]
                diff_sq = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2)
                if reduce == "mean":
                    diff_sq = diff_sq.mean(dim=-1)
                else:
                    diff_sq = diff_sq.sum(dim=-1)
                pair_sq.append(diff_sq.flatten())

        # Unordered cross-batch pair count inside this cell type
        local_n = (
            sum(local_sizes) ** 2
            - sum(int(s) * int(s) for s in local_sizes)
        ) // 2
        n_pairs += int(local_n)

    if n_pairs == 0:
        zero = x.new_zeros(())
        return zero, 0

    all_sq = torch.cat(pair_sq)
    pair_loss = all_sq.sum() / all_sq.numel()
    return pair_loss, int(n_pairs)


class VAEWithBatchPairLoss(VAE):
    """VAE module that adds a cross-batch MSE alignment loss.

    Parameters
    ----------
    pair_weight
        Coefficient of the pair-MSE term added to the scVI loss
        (default 1.0; set to 0 to disable).
    align_on
        Quantity to align across batches: ``"mu"`` (log1p decoder rate,
        default) or ``"z"`` (latent representation).
    """

    def __init__(
        self,
        *args,
        pair_weight: float = 1.0,
        align_on: str = "mu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if align_on not in {"mu", "z"}:
            raise ValueError(f"align_on must be 'mu' or 'z', got {align_on!r}")
        self.pair_weight = float(pair_weight)
        self.align_on = align_on

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | None],
        generative_outputs: dict[str, torch.Tensor | None],
        kl_weight: torch.Tensor | float = 1.0,
    ) -> LossOutput:
        """scVI loss + cross-batch pair MSE loss."""
        loss_output = super().loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )

        if self.pair_weight == 0.0:
            return loss_output

        # ---- Pick the quantity to align ----
        if self.align_on == "mu":
            # Stabilize count scale and sum over genes, matching scVI's
            # reconstruction reduction without letting high-count genes
            # dominate as strongly as raw-count squared error.
            x = torch.log1p(generative_outputs[MODULE_KEYS.PX_KEY].mu)
            reduce = "sum"
        else:  # "z"
            x = inference_outputs["z"]                     # (n_obs, n_latent)
            reduce = "sum"

        # Pair batch comes from the categorical covariate registry, NOT
        # the scVI batch_key registry. This keeps the cross-batch pair
        # loss independent of SCVI's batch-correction machinery.
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None)
        if cat_covs is None:
            cat_covs = tensors.get(PAIR_BATCH_KEY, None)  # fallback key name
        if cat_covs is None or cat_covs.numel() == 0:
            # No pair batch registered (e.g. adata had a single category).
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            batch_idx = cat_covs.long().view(-1)

        # Optional cell-type mask: restrict cross-batch pairing to within
        # the same cell type so different cell types never get pulled
        # together. Cell type comes from the labels registry
        # (REGISTRY_KEYS.LABELS_KEY, "labels" key).
        cell_type_idx = tensors.get(REGISTRY_KEYS.LABELS_KEY, None)
        if cell_type_idx is not None:
            cell_type_idx = cell_type_idx.long().view(-1)

        pair_loss, n_pairs = _cross_batch_pair_mse(
            x, batch_idx, cell_type_idx=cell_type_idx, reduce=reduce
        )

        # ---- Combine with scVI loss ----
        if n_pairs > 0:
            total_loss = loss_output.loss + self.pair_weight * pair_loss
        else:
            total_loss = loss_output.loss

        loss_output.loss = total_loss
        if loss_output.extra_metrics is None:
            loss_output.extra_metrics = {}
        loss_output.extra_metrics["batch_pair_mse"] = float(pair_loss.detach().cpu())
        loss_output.extra_metrics["batch_pair_n"] = int(n_pairs)

        return loss_output


class SCVIWithBatchPairLoss(SCVI):
    """SCVI model with cross-batch pair-MSE alignment loss.

    Inherits everything from :class:`scvi.model.SCVI`; only the module's
    ``loss`` adds a term. Pairing is done per mini-batch; cells from the
    same batch index are skipped.

    Parameters
    ----------
    pair_weight
        Coefficient of the pair-MSE term in the loss (default 1.0).
    align_on
        ``"mu"`` (decoder rate, default) or ``"z"`` (latent).
    """

    _module_cls = VAEWithBatchPairLoss

    def __init__(
        self,
        adata=None,
        registry=None,
        *args,
        pair_weight: float = 1.0,
        align_on: str = "mu",
        **kwargs,
    ):
        super().__init__(
            adata,
            registry,
            *args,
            pair_weight=pair_weight,
            align_on=align_on,
            **kwargs,
        )
        self.pair_weight = pair_weight
        self.align_on = align_on

    @classmethod
    def setup_anndata(
        cls,
        adata,
        pair_batch_obs_key: str = "company",
        batch_key: str | None = None,
        **kwargs,
    ):
        """Register the pairing batch as an extra categorical covariate.

        Parameters
        ----------
        pair_batch_obs_key
            Column in ``adata.obs`` that defines the pairing batches
            (default ``"company"``). It is **copied** into a fresh
            ``_pair_batch`` column so we never touch the user's
            original column.
        batch_key
            Pass-through to the parent ``SCVI.setup_anndata``. Default
            is ``None`` so SCVI does NOT apply batch correction -- the
            pairing batch is consumed only by the pair-MSE loss.
        **kwargs
            Forwarded to ``SCVI.setup_anndata`` (labels_key, continuous
            and categorical covariates, etc.).
        """
        # Copy the column under a private name. We never overwrite the
        # user's ``adata.obs[pair_batch_obs_key]`` so multiple instances
        # of this model can coexist.
        import numpy as np
        import pandas as pd

        adata.obs[PAIR_BATCH_KEY] = adata.obs[pair_batch_obs_key].astype("category")
        # Re-codify as integer codes (the categorical-covariate registry
        # accepts categories the same way batch_key does, without
        # routing through SCVI's batch effect / one-hot encoder).
        codes = adata.obs[PAIR_BATCH_KEY].cat.codes.to_numpy()
        adata.obs[PAIR_BATCH_KEY] = pd.Categorical.from_codes(
            codes, categories=adata.obs[PAIR_BATCH_KEY].cat.categories
        )

        cat_covs = list(kwargs.pop("categorical_covariate_keys", []) or [])
        if PAIR_BATCH_KEY not in cat_covs:
            cat_covs.append(PAIR_BATCH_KEY)

        return super().setup_anndata(
            adata,
            batch_key=batch_key,
            categorical_covariate_keys=cat_covs,
            **kwargs,
        )