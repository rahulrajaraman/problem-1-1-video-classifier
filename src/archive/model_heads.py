"""Archived MLP / GRU heads formerly in src/model.py. Kept for train.py experiments."""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(frame_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(frame_features.dtype)
    return (frame_features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class DomainMLP(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frame_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mlp(masked_mean(frame_features, mask))


class DomainGRU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * self.directions, num_classes))

    def forward(self, frame_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.input_dropout(frame_features), lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        last_layer = hidden[-self.directions:].transpose(0, 1).reshape(frame_features.shape[0], -1)
        return self.classifier(last_layer)
