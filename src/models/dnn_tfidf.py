"""
dnn_tfidf.py — Baseline model: TF-IDF features → Feed-forward DNN → Softmax.

This serves as the non-sequential, feature-engineering baseline for comparison
with the proposed BiLSTM model.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DNNClassifier(nn.Module):
    """Fully-connected DNN for TF-IDF input.

    Each hidden layer follows the pattern:
        Linear → BatchNorm → ReLU → Dropout

    Args:
        input_dim   : TF-IDF feature dimension (vocabulary size)
        hidden_dims : list of hidden layer widths, e.g. [512, 256, 128]
        num_classes : number of output classes
        dropout     : dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x : TF-IDF feature tensor, shape (batch_size, input_dim)

        Returns:
            Raw logits, shape (batch_size, num_classes).
            Apply CrossEntropyLoss during training.
        """
        return self.net(x)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
