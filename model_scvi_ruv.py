"""RUV-style scVI model: split latent into z_bio / z_uv, company embedding
decoder, and a linear disease-group effect, with a ZINB likelihood.

This model inherits from :mod:`model_scvi_disease` (``VAEWithDiseaseEffect`` /
``SCVIWithDiseaseEffect``). The reconstruction is the sum of three parts:

    z_bio = encoder_bio(x)                 # biological latent (scVI z_encoder)
    z_uv  = encoder_uv(x)                  # unwanted-variation latent
    company_emb = company_embedding(company_idx)   # only enters the UV branch

    bio_part = decoder_bio(z_bio)          # biological log-rate
    uv_part  = decoder_uv(z_uv, company_emb, n_genes_on)  # technical branch
    group_effect = W_group[y]              # linear disease effect

    log_mu = library_log_size + bio_part + uv_part + group_effect
    px = ZeroInflatedNegativeBinomial(mu=exp(log_mu), theta=theta,
                                      zi_logits=zi_logits)

* ``z_bio`` is the standard scVI latent (used for UMAP / clustering / DEG).
* ``z_uv`` + ``company_embedding`` + continuous covariates absorb technical
  variation in the decoder only, so they never leak company info into the
  encoder / latent space.
* The linear disease effect ``W_group`` of shape ``(n_labels, n_genes)`` is
  kept exactly as in :mod:`model_scvi_disease`: ``exp(W_group[y])`` multiplies
  the mean, and ``W_group[disease] - W_group[control]`` is the interpretable
  disease log-fold-change.

Usage
-----
>>> from model_scvi_ruv import SCVIRUVWithDisease
>>> SCVIRUVWithDisease.setup_anndata(
...     adata, batch_key="company", labels_key="status",
...     continuous_covariate_keys=["n_genes_on"],
... )
>>> model = SCVIRUVWithDisease(
...     adata, n_latent=32, n_latent_uv=8, company_embed_dim=8,
...     gene_likelihood="zinb", group_effect_scale=0.01,
... )
>>> model.train()
>>> z_bio = model.get_z_bio()                  # latent for downstream analysis
>>> z_uv = model.get_z_uv()                    # unwanted variation latent
>>> company_emb = model.get_company_embedding()  # (n_company, embed_dim)
>>> mu_scvi, mu_disease = model.get_reconstruction_parts(return_mean=True)
>>> lfc = model.get_disease_logfc(disease_group="CURES")
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
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import LossOutput, auto_move_data
from scvi.nn import Encoder

from model_scvi_disease import SCVIWithDiseaseEffect, VAEWithDiseaseEffect


class RUVVAEWithDisease(VAEWithDiseaseEffect):
    """VAE module: stock scVI bio branch + small scVI UV branch + linear
    disease-group effect.

    Parameters
    ----------
    n_latent_uv
        Dimension of the UV latent ``z_uv`` (default 8).
    company_embed_dim
        Dimension of the company embedding (default 8). Ignored when no batch
        is registered (``n_batch == 0``).
    uv_n_hidden
        Hidden width of the UV decoder (default 32).
    uv_n_layers
        Number of hidden layers of the UV decoder (default 1).
    uv_dropout_rate
        Dropout rate of the UV encoder (default 0.1).
    group_effect_scale
        Std of the normal init for ``W_group`` (default 0.01).
    group_effect_prior
        Weight of the L2 penalty on ``W_group`` (default 0.0).
    """

    def __init__(
        self,
        *args,
        n_latent_uv: int = 8,
        company_embed_dim: int = 8,
        uv_n_hidden: int = 32,
        uv_n_layers: int = 1,
        uv_dropout_rate: float = 0.1,
        group_effect_scale: float = 0.01,
        group_effect_prior: float = 0.0,
        **kwargs,
    ):
        # n_continuous_cov is consumed by VAE.__init__; capture it here so the
        # UV decoder input dimension can be computed afterwards.
        n_continuous_cov = kwargs.pop("n_continuous_cov", 0)

        super().__init__(
            *args,
            n_continuous_cov=n_continuous_cov,
            group_effect_scale=group_effect_scale,
            group_effect_prior=group_effect_prior,
            **kwargs,
        )

        if self.dispersion not in ("gene", "gene-batch"):
            raise ValueError(
                "RUVVAEWithDisease only supports dispersion='gene' or "
                f"'gene-batch', got {self.dispersion!r}"
            )

        self.n_latent_uv = n_latent_uv
        self.company_embed_dim = company_embed_dim
        self.n_continuous_cov = n_continuous_cov
        self.uv_n_hidden = uv_n_hidden
        self.uv_n_layers = uv_n_layers
        self.uv_dropout_rate = uv_dropout_rate

        # ---- UV encoder: q(z_uv | x, company, cont_covs) ----
        # company (as batch) is injected via n_cat_list; continuous
        # covariates (n_genes_on) are concatenated to the input in
        # ``_regular_inference`` (scVI's Encoder does not forward a ``cont``
        # argument), so the input width is n_input + n_continuous_cov.
        self.encoder_uv = Encoder(
            self.n_input + n_continuous_cov,
            n_latent_uv,
            n_cat_list=[self.n_batch] if self.n_batch > 0 else None,
            n_layers=uv_n_layers,
            n_hidden=uv_n_hidden,
            dropout_rate=uv_dropout_rate,
            distribution="normal",
            return_dist=True,
        )

        # ---- Company embedding (only consumed by the UV branch) ----
        if self.n_batch > 0:
            self.company_embedding = nn.Embedding(self.n_batch, company_embed_dim)
        else:
            self.company_embedding = None

        # ---- Bio decoder: log-rate contribution from z_bio ----
        self.decoder_bio = nn.Sequential(
            nn.Linear(self.n_latent, 256), nn.GELU(),
            nn.Linear(256, self.n_input),
        )

        # ---- UV decoder: z_uv + company_emb + cont_covs ----
        uv_in_dim = (
            n_latent_uv
            + (company_embed_dim if self.n_batch > 0 else 0)
            + n_continuous_cov
        )
        self.decoder_uv = nn.Sequential(
            nn.Linear(uv_in_dim, uv_n_hidden), nn.GELU(),
            nn.Linear(uv_n_hidden, self.n_input),
        )

        # ---- Dropout logits (zero-inflation) from biology only ----
        self.decoder_dropout = nn.Sequential(
            nn.Linear(self.n_latent, 256), nn.GELU(),
            nn.Linear(256, self.n_input),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | None]:
        """Run inference and additionally encode the UV latent ``z_uv``.

        ``z_uv`` is encoded from ``log1p(x)`` with company (batch) and the
        continuous covariates (e.g. ``n_genes_on``) injected into the UV
        encoder, so it can directly absorb company / technical variation.
        """
        out = super()._regular_inference(
            x, batch_index, cont_covs=cont_covs, cat_covs=cat_covs, n_samples=n_samples
        )

        x_ = torch.log1p(x) if self.log_variational else x
        # Continuous covariates (e.g. n_genes_on) are concatenated to the
        # input, and company (batch) is passed through the cat_list.
        if cont_covs is not None:
            x_uv = torch.cat([x_, cont_covs], dim=-1)
        else:
            x_uv = x_
        uv_cat = (batch_index,) if self.n_batch > 0 else ()
        qz_uv, z_uv = self.encoder_uv(x_uv, *uv_cat)
        if n_samples > 1:
            z_uv = z_uv.unsqueeze(0).expand(n_samples, -1, -1)
        out["z_uv"] = z_uv
        out["qz_uv"] = qz_uv
        return out

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Add ``z_uv`` to the generative inputs."""
        generative_inputs = super()._get_generative_input(tensors, inference_outputs)
        generative_inputs["z_uv"] = inference_outputs["z_uv"]
        return generative_inputs

    # ------------------------------------------------------------------
    # Generative process
    # ------------------------------------------------------------------
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
        z_uv: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Run the generative process.

        ``log_mu = library_log_size + bio_part + uv_part + W_group[y]``,
        where:

        * ``bio_part = decoder_bio(z_bio)`` — the biological log-rate.
        * ``uv_part  = decoder_uv(z_uv, company_emb, cont_covs)`` — technical.
        * ``W_group[y]`` — the linear disease-group effect.

        ``px`` is a ZINB distribution with mean ``exp(log_mu)``, ``theta``
        from ``exp(px_r)`` and ``zi_logits`` from ``decoder_dropout(z_bio)``
        (biology only, so the corrected expression stays usable).
        """
        from torch.nn.functional import linear, one_hot

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        batch_idx = batch_index.squeeze(-1).long()
        multi_sample = z.dim() == 3

        # ---- Company embedding (UV branch only) ----
        company_emb = None
        if self.company_embedding is not None:
            company_emb = self.company_embedding(batch_idx)
            if multi_sample:
                company_emb = company_emb.unsqueeze(0).expand(z.size(0), -1, -1)

        # ---- Bio / UV log-rate contributions ----
        bio_part = self.decoder_bio(z)  # (n_obs | n_samples, n_obs, n_genes)

        uv_inputs = [z_uv]
        if company_emb is not None:
            uv_inputs.append(company_emb)
        if cont_covs is not None:
            if multi_sample:
                cont_covs = cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)
            uv_inputs.append(cont_covs)
        uv_part = self.decoder_uv(torch.cat(uv_inputs, dim=-1))

        # ---- Linear disease-group effect ----
        if y is None:
            y_idx = torch.zeros(
                batch_idx.shape[0], dtype=torch.long, device=batch_idx.device
            )
        else:
            y_idx = y.squeeze(-1).long()
        group_effect = self.W_group[y_idx]  # (n_obs, n_genes)
        if multi_sample:
            group_effect = group_effect.unsqueeze(0).expand(z.size(0), -1, -1)

        # ---- Rate: log_mu = log(library) + bio + uv + group ----
        log_mu = library + bio_part + uv_part + group_effect
        mu = torch.exp(log_mu) + 1e-6

        # ---- Dispersion theta (scVI convention: exp(px_r)) ----
        if self.dispersion == "gene":
            theta = torch.exp(self.px_r)  # (n_genes,)
        else:  # "gene-batch"
            theta = torch.exp(
                linear(one_hot(batch_idx, self.n_batch).float(), self.px_r)
            )  # (n_obs, n_genes)
            if multi_sample:
                theta = theta.unsqueeze(0).expand(z.size(0), -1, -1)

        # ---- Zero-inflation logits from biology only ----
        zi_logits = self.decoder_dropout(z)

        # ---- Likelihood ----
        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=mu, theta=theta, zi_logits=zi_logits, scale=None
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=mu, theta=theta, scale=None)
        elif self.gene_likelihood == "poisson":
            px = Poisson(rate=mu, scale=None)
        elif self.gene_likelihood == "normal":
            px = Normal(mu, theta, normal_mu=None)
        else:
            raise ValueError(f"Unsupported gene_likelihood: {self.gene_likelihood}")

        # ---- Priors ----
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        pz_uv = Normal(torch.zeros_like(z_uv), torch.ones_like(z_uv))
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
            "pz_uv": pz_uv,
            # Parts kept for the SCVIWithDiseaseEffect helpers:
            "mu_scvi": torch.exp(library + bio_part + uv_part),
            "mu_disease": torch.exp(group_effect),
            "group_effect": group_effect,
            "bio_part": bio_part,
            "uv_part": uv_part,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | None],
        generative_outputs: dict[str, torch.Tensor | None],
        kl_weight: torch.Tensor | float = 1.0,
    ) -> LossOutput:
        """Compute the loss.

        ``loss = E[-log p(x | z_bio, z_uv, company, W_group)]
                + kl_weight * (KL(q(z_bio)|p) + KL(q(z_uv)|p))
                + KL(q(l)|p(l))``
        plus the optional L2 penalty on ``W_group``.
        """
        from torch.distributions import kl_divergence

        x = tensors[REGISTRY_KEYS.X_KEY]
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        qz_uv = inference_outputs["qz_uv"]
        pz = generative_outputs[MODULE_KEYS.PZ_KEY]
        pz_uv = generative_outputs["pz_uv"]
        px = generative_outputs[MODULE_KEYS.PX_KEY]

        kl_divergence_z = kl_divergence(qz, pz).sum(dim=-1)
        kl_divergence_uv = kl_divergence(qz_uv, pz_uv).sum(dim=-1)

        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs[MODULE_KEYS.QL_KEY],
                generative_outputs[MODULE_KEYS.PL_KEY],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        reconst_loss = -px.log_prob(x).sum(-1)

        # KL warm-up is applied to the two latent terms (scVI convention)
        kl_local_for_warmup = kl_divergence_z + kl_divergence_uv
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        extra_metrics = {}
        if self.group_effect_prior > 0:
            penalty = self.group_effect_prior * torch.mean(self.W_group**2)
            loss = loss + penalty
            extra_metrics["group_effect_penalty"] = penalty.detach()
        extra_metrics["kl_uv"] = torch.mean(kl_divergence_uv).detach()

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={
                MODULE_KEYS.KL_Z_KEY: kl_divergence_z,
                "kl_divergence_uv": kl_divergence_uv,
                MODULE_KEYS.KL_L_KEY: kl_divergence_l,
            },
            extra_metrics=extra_metrics,
        )


class SCVIRUVWithDisease(SCVIWithDiseaseEffect):
    """SCVI model: stock scVI bio branch + small scVI UV branch + linear
    disease-group effect.

    Inherits everything from :class:`~model_scvi_disease.SCVIWithDiseaseEffect`
    (including ``get_reconstruction_parts``, ``get_reconstruction`` and
    ``get_disease_logfc*``); only the module is replaced by
    :class:`RUVVAEWithDisease`.

    Parameters
    ----------
    n_latent_uv
        Dimension of the UV latent (default 8).
    company_embed_dim
        Dimension of the company embedding (default 8).
    uv_n_hidden
        Hidden width of the UV decoder (default 32).
    uv_n_layers
        Hidden layers of the UV decoder (default 1).
    uv_dropout_rate
        Dropout rate of the UV encoder (default 0.1).
    group_effect_scale
        Std of the normal init for ``W_group`` (default 0.01).
    group_effect_prior
        Weight of the L2 penalty on ``W_group`` (default 0.0).
    """

    _module_cls = RUVVAEWithDisease

    def __init__(
        self,
        adata=None,
        registry=None,
        *args,
        n_latent_uv: int = 8,
        company_embed_dim: int = 8,
        uv_n_hidden: int = 32,
        uv_n_layers: int = 1,
        uv_dropout_rate: float = 0.1,
        group_effect_scale: float = 0.01,
        group_effect_prior: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            adata,
            registry,
            *args,
            n_latent_uv=n_latent_uv,
            company_embed_dim=company_embed_dim,
            uv_n_hidden=uv_n_hidden,
            uv_n_layers=uv_n_layers,
            uv_dropout_rate=uv_dropout_rate,
            group_effect_scale=group_effect_scale,
            group_effect_prior=group_effect_prior,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------
    def _iter_inference_outputs(self, adata=None, indices=None, batch_size=None,
                                n_samples=1):
        """Yield inference outputs per minibatch."""
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        for tensors in scdl:
            inference_outputs, _ = self.module.forward(
                tensors=tensors,
                inference_kwargs={"n_samples": n_samples},
                compute_loss=False,
            )
            yield inference_outputs

    def get_z_bio(self, adata=None, indices=None, batch_size=None,
                  n_samples: int = 1) -> np.ndarray:
        """Return the biological latent posterior means ``q(z_bio | x)``.

        Shape ``(n_obs, n_latent)``. Uses the posterior mean (no sampling
        noise), matching the semantics of ``get_latent_representation``.
        """
        z_list = []
        for inference_outputs in self._iter_inference_outputs(
            adata=adata, indices=indices, batch_size=batch_size, n_samples=n_samples
        ):
            z_list.append(inference_outputs[MODULE_KEYS.QZ_KEY].loc.detach().cpu())
        return torch.cat(z_list, dim=0).numpy()

    def get_z_uv(self, adata=None, indices=None, batch_size=None,
                 n_samples: int = 1) -> np.ndarray:
        """Return the unwanted-variation latent posterior means ``q(z_uv | x)``.

        Shape ``(n_obs, n_latent_uv)``. Uses the posterior mean (no sampling
        noise).
        """
        z_uv_list = []
        for inference_outputs in self._iter_inference_outputs(
            adata=adata, indices=indices, batch_size=batch_size, n_samples=n_samples
        ):
            z_uv_list.append(inference_outputs["qz_uv"].loc.detach().cpu())
        return torch.cat(z_uv_list, dim=0).numpy()

    def get_company_embedding(self) -> pd.DataFrame:
        """Return the learned company embeddings.

        Returns
        -------
        pandas.DataFrame of shape (n_company, company_embed_dim), indexed by
        the company names registered in ``setup_anndata(batch_key="company")``.
        """
        if self.module.company_embedding is None:
            raise ValueError(
                "No company embedding: batch_key was not registered in "
                "setup_anndata."
            )
        emb = self.module.company_embedding.weight.detach().cpu().numpy()
        registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        names = [str(x) for x in registry.categorical_mapping]
        return pd.DataFrame(emb, index=names)
