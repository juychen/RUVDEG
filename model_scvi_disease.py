"""SCVI model with a linear disease-group effect on reconstruction.

This model inherits **completely** from :class:`scvi.model.SCVI`. Only the
module's ``generative`` (the reconstruction / forward) and ``loss`` are
modified:

* A **linear disease-group effect** ``W_group`` of shape ``(n_labels, n_genes)``
  is added to the log-rate of the count distribution.
* The reconstruction is split into two parts:

  - ``mu_scvi``    : the scVI reconstruction (base rate from the decoder)
  - ``mu_disease`` : the disease contribution ``exp(W_group[y])``

  and the combined mean is ``mu = mu_scvi * mu_disease``.

The disease group is the ``labels_key`` column passed to ``setup_anndata``
(e.g. ``status`` / ``condition``). Each category gets its own linear per-gene
effect; the control group's row is learned too (a baseline shift), so the
*contrast* ``W_group[disease] - W_group[control]`` is the interpretable
disease log-fold-change.

Usage
-----
>>> from model_scvi_disease import SCVIWithDiseaseEffect
>>> SCVIWithDiseaseEffect.setup_anndata(
...     adata, batch_key="batch", labels_key="status",
...     continuous_covariate_keys=["n_genes_on"],
... )
>>> model = SCVIWithDiseaseEffect(
...     adata, n_latent=32, n_layers=2, gene_likelihood="zinb",
...     group_effect_scale=0.01, group_effect_prior=0.0,
... )
>>> model.train()
>>> # combined reconstruction (scVI part * disease part)
>>> recon = model.get_reconstruction(return_mean=True)
>>> # split parts
>>> mu_scvi, mu_disease = model.get_reconstruction_parts()
>>> # per-gene disease log-fold-change (disease vs control)
>>> lfc = model.get_disease_logfc()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scvi import REGISTRY_KEYS
from scvi.distributions import (
    NegativeBinomial,
    Normal,
    Poisson,
    ZeroInflatedNegativeBinomial,
)
from scvi.model import SCVI
from scvi.module import VAE
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import LossOutput


class VAEWithDiseaseEffect(VAE):
    """VAE module with a linear disease-group effect on the reconstruction rate.

    Parameters
    ----------
    group_effect_scale
        Std of the normal init for ``W_group`` (default 0.01).
    group_effect_prior
        Weight of the L2 penalty on ``W_group`` added to the loss
        (default 0.0 = no penalty).
    """

    def __init__(
        self,
        *args,
        group_effect_scale: float = 0.01,
        group_effect_prior: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group_effect_scale = group_effect_scale
        self.group_effect_prior = group_effect_prior
        # Linear disease-group effect: (n_labels, n_genes)
        self.W_group = nn.Parameter(
            torch.randn(self.n_labels, self.n_input) * group_effect_scale
        )

    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Run the generative process with a linear disease-group effect.

        The base scVI reconstruction is computed first (``super().generative``),
        then the rate is multiplied by the disease factor
        ``exp(W_group[y])``. The returned ``px`` distribution therefore has
        mean ``mu = mu_scvi * mu_disease``, and the two parts are stored in
        the output dict under ``"mu_scvi"`` and ``"mu_disease"``.
        """
        # Base scVI reconstruction
        generative_outputs = super().generative(
            z,
            library,
            batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            size_factor=size_factor,
            y=y,
            transform_batch=transform_batch,
        )
        px = generative_outputs[MODULE_KEYS.PX_KEY]

        # ---- Linear disease-group effect ----
        if y is None:
            y_idx = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        else:
            y_idx = y.squeeze(-1).long()
        group_effect = self.W_group[y_idx]  # (n_obs, n_genes)

        # ---- Split reconstruction ----
        #   mu_scvi    : scVI reconstruction (base rate)
        #   mu_disease : disease multiplicative factor exp(W_group[y])
        mu_scvi = px.mu
        mu_disease = torch.exp(group_effect)
        mu = mu_scvi * mu_disease

        # Reconstruct the distribution with the combined mean
        if self.gene_likelihood == "zinb":
            px_new = ZeroInflatedNegativeBinomial(
                mu=mu, theta=px.theta, zi_logits=px.zi_logits, scale=px.scale
            )
        elif self.gene_likelihood == "nb":
            px_new = NegativeBinomial(mu=mu, theta=px.theta, scale=px.scale)
        elif self.gene_likelihood == "poisson":
            px_new = Poisson(rate=mu, scale=px.scale)
        elif self.gene_likelihood == "normal":
            px_new = Normal(mu, px.scale, normal_mu=px.scale)
        else:
            raise ValueError(f"Unsupported gene_likelihood: {self.gene_likelihood}")

        generative_outputs[MODULE_KEYS.PX_KEY] = px_new
        generative_outputs["mu_scvi"] = mu_scvi
        generative_outputs["mu_disease"] = mu_disease
        generative_outputs["group_effect"] = group_effect
        return generative_outputs

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | None],
        generative_outputs: dict[str, torch.Tensor | None],
        kl_weight: torch.Tensor | float = 1.0,
    ) -> LossOutput:
        """Compute the loss.

        The reconstruction loss is the standard scVI NLL evaluated on the
        combined mean ``mu = mu_scvi * mu_disease`` (computed in
        ``generative``). An optional L2 penalty on ``W_group`` is added when
        ``group_effect_prior > 0``.
        """
        loss_output = super().loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )
        if self.group_effect_prior > 0:
            penalty = self.group_effect_prior * torch.mean(self.W_group**2)
            loss_output.loss = loss_output.loss + penalty
            if loss_output.extra_metrics is None:
                loss_output.extra_metrics = {}
            loss_output.extra_metrics["group_effect_penalty"] = penalty
        return loss_output


class SCVIWithDiseaseEffect(SCVI):
    """SCVI model with a linear disease-group effect on reconstruction.

    Inherits everything from :class:`scvi.model.SCVI`; only the module's
    ``generative`` (reconstruction) and ``loss`` are modified.

    Parameters
    ----------
    group_effect_scale
        Std of the normal init for ``W_group`` (default 0.01).
    group_effect_prior
        Weight of the L2 penalty on ``W_group`` in the loss (default 0.0).
    """

    _module_cls = VAEWithDiseaseEffect

    def __init__(
        self,
        adata=None,
        registry=None,
        *args,
        group_effect_scale: float = 0.01,
        group_effect_prior: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            adata,
            registry,
            *args,
            group_effect_scale=group_effect_scale,
            group_effect_prior=group_effect_prior,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Reconstruction helpers
    # ------------------------------------------------------------------
    def _iter_generative_outputs(self, adata=None, indices=None, batch_size=None,
                                 n_samples=1):
        """Yield ``(inference_outputs, generative_outputs)`` per minibatch."""
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        for tensors in scdl:
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs={"n_samples": n_samples},
                compute_loss=False,
            )
            yield inference_outputs, generative_outputs

    def get_reconstruction_parts(
        self,
        adata=None,
        indices=None,
        batch_size=None,
        n_samples: int = 1,
        return_mean: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the two reconstruction parts.

        Returns
        -------
        (mu_scvi, mu_disease) : tuple of np.ndarray, each (n_obs, n_genes)
            - ``mu_scvi``    : scVI reconstruction (base rate)
            - ``mu_disease`` : disease multiplicative factor ``exp(W_group[y])``
        """
        mu_scvi_list, mu_disease_list = [], []
        for _, gen in self._iter_generative_outputs(
            adata=adata, indices=indices, batch_size=batch_size, n_samples=n_samples
        ):
            mu_scvi = gen["mu_scvi"]
            mu_disease = gen["mu_disease"]
            if n_samples > 1 and return_mean:
                mu_scvi = mu_scvi.mean(0)
                mu_disease = mu_disease.mean(0)
            mu_scvi_list.append(mu_scvi.detach().cpu())
            mu_disease_list.append(mu_disease.detach().cpu())
        mu_scvi = torch.cat(mu_scvi_list, dim=0).numpy()
        mu_disease = torch.cat(mu_disease_list, dim=0).numpy()
        return mu_scvi, mu_disease

    def get_reconstruction(
        self,
        adata=None,
        indices=None,
        batch_size=None,
        n_samples: int = 1,
        return_mean: bool = True,
    ) -> np.ndarray:
        """Return the combined reconstruction ``mu = mu_scvi * mu_disease``."""
        mu_scvi, mu_disease = self.get_reconstruction_parts(
            adata=adata, indices=indices, batch_size=batch_size,
            n_samples=n_samples, return_mean=return_mean,
        )
        return mu_scvi * mu_disease

    def get_disease_logfc_by_group(
        self,
        control_group: int | str = 0,
        group_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return disease effects for every label versus one control label.

        Parameters
        ----------
        control_group
            Control label index or category name. Defaults to the first label.
        group_names
            Optional label names. If omitted, they are read from the SCVI
            labels registry.

        Returns
        -------
        pandas.DataFrame
            One row per gene and one ``logFC_<group>_vs_<control>`` column per
            label. The control column is zero by definition.
        """
        W = self.module.W_group.detach().cpu().numpy()
        if W.shape[0] < 2:
            raise ValueError(
                "Need at least 2 disease groups (labels); "
                f"got n_labels={W.shape[0]}. Set labels_key in setup_anndata."
            )

        if group_names is None:
            registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
            group_names = [str(x) for x in registry.categorical_mapping]
        else:
            group_names = [str(x) for x in group_names]
        if len(group_names) != W.shape[0]:
            raise ValueError(
                f"group_names has {len(group_names)} labels, but W_group has "
                f"{W.shape[0]} rows."
            )

        if isinstance(control_group, str):
            if control_group not in group_names:
                raise ValueError(
                    f"Unknown control group {control_group!r}; "
                    f"available groups: {group_names}"
                )
            control_idx = group_names.index(control_group)
        else:
            control_idx = int(control_group)
            if not 0 <= control_idx < len(group_names):
                raise ValueError(
                    f"control_group index {control_idx} is out of range for "
                    f"{len(group_names)} groups."
                )

        control_name = group_names[control_idx]
        effects = W - W[control_idx][None, :]
        gene_names = np.asarray(self.adata.var_names).astype(str)
        result = pd.DataFrame({"gene": gene_names})
        for idx, group_name in enumerate(group_names):
            result[f"logFC_{group_name}_vs_{control_name}"] = effects[idx]
        return result

    def get_disease_logfc(
        self,
        disease_group: int | str | None = None,
        control_group: int | str = 0,
    ) -> np.ndarray:
        """Return one disease logFC vector for backward compatibility.

        For multiple disease groups, pass ``disease_group`` explicitly. Use
        :meth:`get_disease_logfc_by_group` to retrieve all contrasts at once.
        """
        effects = self.get_disease_logfc_by_group(control_group=control_group)
        effect_columns = [c for c in effects.columns if c.startswith("logFC_")]
        if disease_group is None:
            if len(effect_columns) != 2:
                raise ValueError(
                    "Multiple disease groups detected. Pass disease_group or "
                    "call get_disease_logfc_by_group()."
                )
            disease_group = effect_columns[1].split("_vs_")[0].removeprefix("logFC_")
        if isinstance(disease_group, int):
            group_name = list(self.adata_manager.get_state_registry(
                REGISTRY_KEYS.LABELS_KEY
            ).categorical_mapping)[disease_group]
        else:
            group_name = str(disease_group)
        column = next(
            (c for c in effect_columns if c.startswith(f"logFC_{group_name}_vs_")),
            None,
        )
        if column is None:
            raise ValueError(
                f"Unknown disease group {disease_group!r}; "
                f"available columns: {effect_columns}"
            )
        return effects[column].to_numpy()
