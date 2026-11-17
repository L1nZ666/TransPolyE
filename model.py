"""TransPolyE model definition."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SpatioTemporalPositionalEncoding(nn.Module):
    """Project physical time and space coordinates into the model dimension."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even.")
        self.time_proj = nn.Linear(1, d_model // 2)
        self.space_proj = nn.Linear(1, d_model // 2)

    def forward(self, t_coords: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        """Return coordinate embeddings with shape [B, L, d_model]."""
        t_enc = self.time_proj(t_coords)
        x_enc = self.space_proj(x_coords)
        return torch.cat([t_enc, x_enc], dim=-1)


class TransPolyE(nn.Module):
    """Transformer for joint polynomial-class and parameter prediction."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        nx: int = 200,
        nt: int = 100,
        num_classes: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nx = nx
        self.nt = nt
        self.num_classes = num_classes

        self.value_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_encoding = SpatioTemporalPositionalEncoding(d_model)
        self.type_embedding = nn.Embedding(3, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes),
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self,
        u0: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
        t_coords: Dict[str, torch.Tensor],
        x_coords: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict polynomial parameters and class logits.

        Parameters
        ----------
        u0 : [B, 2, nx]
            Real/imaginary channels of the initial trace.
        u1, u2 : [B, 2, nt]
            Real/imaginary channels of the left and right boundary traces.
        t_coords, x_coords
            Physical coordinates for the three trace types.

        Returns
        -------
        params : [B, 2]
            Predicted continuous polynomial parameters.
        logits : [B, num_classes]
            Polynomial-class logits.
        """
        batch_size = u0.size(0)
        device = u0.device

        u0_tokens = self.value_proj(u0.transpose(1, 2))
        u1_tokens = self.value_proj(u1.transpose(1, 2))
        u2_tokens = self.value_proj(u2.transpose(1, 2))

        u0_tokens = (
            u0_tokens
            + self.pos_encoding(t_coords["u0"], x_coords["u0"])
            + self.type_embedding(
                torch.zeros(batch_size, self.nx, dtype=torch.long, device=device)
            )
        )
        u1_tokens = (
            u1_tokens
            + self.pos_encoding(t_coords["u1"], x_coords["u1"])
            + self.type_embedding(
                torch.ones(batch_size, self.nt, dtype=torch.long, device=device)
            )
        )
        u2_tokens = (
            u2_tokens
            + self.pos_encoding(t_coords["u2"], x_coords["u2"])
            + self.type_embedding(
                2 * torch.ones(batch_size, self.nt, dtype=torch.long, device=device)
            )
        )

        tokens = torch.cat([u0_tokens, u1_tokens, u2_tokens], dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        encoded = self.transformer(tokens)
        global_feature = encoded[:, 0]

        logits = self.classifier(global_feature)
        params = self.regressor(global_feature)
        return params, logits
