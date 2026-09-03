"""Low-dimensional UV embedding correction on top of a frozen RAW scVI latent.

The RAW scVI model is deliberately kept outside this module.  The training
script computes its deterministic latent mean once, then this module learns:

    z_raw -> UVEncoder -> u -> UVDecoder -> delta_z
    z_emend = z_raw + delta_z

Only the UV encoder and decoder parameters are trainable.  Batch and cell-type
labels are used only to construct the cross-batch pair loss on ``z_emend``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from model_scvi_batch_pair import _cross_batch_pair_mse


class UVEmbeddingModel(nn.Module):
    """Learn a low-dimensional unwanted-variation correction.

    Parameters
    ----------
    raw_latent_dim
        Dimensionality of the frozen RAW scVI latent.
    uv_dim
        Dimensionality of the UV embedding.
    hidden_dim
        Width of the UV encoder and decoder MLPs.
    uv_layers
        Number of hidden layers in each MLP.
    delta_scale
        Initial/output scale for the latent correction.  The final decoder
        layer is initialized to zero, so the initial emended latent equals
        the RAW latent.
    pair_weight
        Weight of the same-celltype, cross-batch pair loss.
    delta_penalty
        L2 penalty on ``delta_z``.
    """

    def __init__(
        self,
        raw_latent_dim: int,
        uv_dim: int = 2,
        hidden_dim: int = 32,
        uv_layers: int = 1,
        delta_scale: float = 1.0,
        pair_weight: float = 1.0,
        delta_penalty: float = 1e-3,
        reconstruction_weight: float = 1.0,
    ):
        super().__init__()
        if raw_latent_dim < 1:
            raise ValueError("raw_latent_dim must be positive")
        if uv_dim < 1:
            raise ValueError("uv_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if uv_layers < 1:
            raise ValueError("uv_layers must be positive")
        if delta_scale <= 0:
            raise ValueError("delta_scale must be positive")
        if pair_weight < 0:
            raise ValueError("pair_weight must be non-negative")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative")
        if delta_penalty < 0:
            raise ValueError("delta_penalty must be non-negative")

        self.raw_latent_dim = int(raw_latent_dim)
        self.uv_dim = int(uv_dim)
        self.hidden_dim = int(hidden_dim)
        self.uv_layers = int(uv_layers)
        self.delta_scale = float(delta_scale)
        self.pair_weight = float(pair_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.delta_penalty = float(delta_penalty)

        self.uv_encoder = self._make_mlp(
            self.raw_latent_dim, self.uv_dim, self.hidden_dim, self.uv_layers
        )
        self.uv_decoder = self._make_mlp(
            self.uv_dim, self.raw_latent_dim, self.hidden_dim, self.uv_layers
        )
        self._zero_initialize_output(self.uv_decoder)

    @staticmethod
    def _make_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _zero_initialize_output(module: nn.Sequential) -> None:
        output = module[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("MLP output layer must be nn.Linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, z_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return UV embedding, correction, and emended latent."""
        if z_raw.ndim != 2 or z_raw.shape[1] != self.raw_latent_dim:
            raise ValueError(
                "z_raw must have shape (n_cells, raw_latent_dim); "
                f"got {tuple(z_raw.shape)} with expected width "
                f"{self.raw_latent_dim}"
            )
        z_raw = z_raw.float()
        uv_embedding = self.uv_encoder(z_raw)
        delta_z = self.delta_scale * self.uv_decoder(uv_embedding)
        z_emend = z_raw + delta_z
        return {
            "z_raw": z_raw,
            "uv_embedding": uv_embedding,
            "delta_z": delta_z,
            "z_emend": z_emend,
        }

    def pair_loss(
        self,
        z_emend: torch.Tensor,
        batch_idx: torch.Tensor,
        cell_type_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Compute dimension-normalized cross-batch pair MSE."""
        return _cross_batch_pair_mse(
            z_emend,
            batch_idx.long().view(-1),
            cell_type_idx=(
                None
                if cell_type_idx is None
                else cell_type_idx.long().view(-1)
            ),
            reduce="mean",
        )

    def training_loss(
        self,
        z_raw: torch.Tensor,
        batch_idx: torch.Tensor,
        cell_type_idx: torch.Tensor | None = None,
        reconstruction_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Return pair, RAW-decoder reconstruction, and correction losses."""
        outputs = self(z_raw)
        pair_loss, n_pairs = self.pair_loss(
            outputs["z_emend"], batch_idx, cell_type_idx
        )
        delta_penalty = outputs["delta_z"].square().mean()
        if reconstruction_loss is None:
            reconstruction_loss = z_raw.new_zeros(())
        loss = (
            self.pair_weight * pair_loss
            + self.reconstruction_weight * reconstruction_loss
            + self.delta_penalty * delta_penalty
        )
        metrics: dict[str, float | int] = {
            "loss": float(loss.detach().cpu()),
            "pair_loss": float(pair_loss.detach().cpu()),
            "pair_n": int(n_pairs),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "delta_penalty": float(delta_penalty.detach().cpu()),
            "delta_rms": float(
                outputs["delta_z"].square().mean().sqrt().detach().cpu()
            ),
        }
        return loss, metrics

    def save(self, path: str | Path, raw_model_dir: str | None = None) -> None:
        """Save UV parameters and configuration to a model directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "raw_latent_dim": self.raw_latent_dim,
            "uv_dim": self.uv_dim,
            "hidden_dim": self.hidden_dim,
            "uv_layers": self.uv_layers,
            "delta_scale": self.delta_scale,
            "pair_weight": self.pair_weight,
            "reconstruction_weight": self.reconstruction_weight,
            "delta_penalty": self.delta_penalty,
            "raw_model_dir": raw_model_dir,
        }
        torch.save(
            {"config": config, "state_dict": self.state_dict()},
            path / "uv_embedding.pt",
        )
        (path / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "UVEmbeddingModel":
        """Load a saved UV embedding model."""
        checkpoint = torch.load(
            Path(path) / "uv_embedding.pt",
            map_location=map_location,
            weights_only=False,
        )
        config = dict(checkpoint["config"])
        config.pop("raw_model_dir", None)
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model
