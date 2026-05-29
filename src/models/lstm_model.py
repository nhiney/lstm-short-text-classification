"""
lstm_model.py — Proposed model: Bidirectional LSTM with Additive Attention.

This is the core contribution of the thesis:
    "Deep Learning LSTM for Vietnamese Short Text Classification"

Architecture
────────────
    Input tokens (B, L)
        │
    ┌───▼───────────────────────────────────────────────────────────┐
    │  Embedding Layer                                              │
    │  (vocab_size × embed_dim)  +  Dropout(p)                      │
    └───────────────────────────────────────────────────────────────┘
        │  (B, L, embed_dim)
    ┌───▼───────────────────────────────────────────────────────────┐
    │  Bidirectional LSTM  — num_layers stacked                     │
    │  Forward  LSTM:  embed_dim  →  hidden_size                    │
    │  Backward LSTM:  embed_dim  →  hidden_size                    │
    │  Packed-padded-sequence for variable-length efficiency         │
    └───────────────────────────────────────────────────────────────┘
        │  (B, L, 2 * hidden_size)
    ┌───▼───────────────────────────────────────────────────────────┐
    │  Additive Attention  (Bahdanau, 2015)                         │
    │                                                               │
    │  score_t  =  v ᵀ · tanh(W · h_t)                             │
    │  α        =  softmax(score)        shape: (B, L)              │
    │  context  =  Σ_t  α_t · h_t        shape: (B, 2H)            │
    └───────────────────────────────────────────────────────────────┘
        │  (B, 2 * hidden_size)
    ┌───▼───────────────────────────────────────────────────────────┐
    │  Classifier Head                                              │
    │  Linear(2H → H) → LayerNorm → GELU → Dropout → Linear(H → C)│
    └───────────────────────────────────────────────────────────────┘
        │  (B, num_classes)  — raw logits

Notation: B = batch, L = seq_len, H = hidden_size, C = num_classes
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Bahdanau (additive) attention over a sequence of hidden states.

    Computes a scalar score for each position, then returns the
    attention-weighted context vector.

    Args:
        hidden_size : dimensionality of the input hidden states
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights.

        Args:
            hidden_states : BiLSTM output, shape (B, L, 2H)
            mask          : boolean tensor (B, L), True = real token.
                            Padding positions are set to -inf before softmax.

        Returns:
            context : weighted sum, shape (B, 2H)
            weights : attention distribution, shape (B, L)
        """
        # score_t = v · tanh(W · h_t),  shape: (B, L, 1) → (B, L)
        scores = self.v(torch.tanh(self.W(hidden_states))).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)                 # (B, L)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, 2H)
        return context, weights


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with optional Additive Attention for text classification.

    This is the proposed model in the thesis. The bidirectional encoding
    captures both left and right context for each token, while the attention
    mechanism learns to focus on emotionally salient words.

    Args:
        vocab_size    : number of unique tokens in the vocabulary
        embed_dim     : word embedding dimension (default 128)
        hidden_size   : LSTM hidden state dimension per direction (default 256)
        num_layers    : number of stacked LSTM layers (default 2)
        num_classes   : number of output emotion classes (default 7)
        dropout       : dropout probability used throughout (default 0.3)
        bidirectional : use BiLSTM when True (recommended)
        use_attention : use Bahdanau attention when True (recommended)
        pad_idx       : padding token index (default 0)
    """

    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = 128,
        hidden_size:   int   = 256,
        num_layers:    int   = 2,
        num_classes:   int   = 7,
        dropout:       float = 0.3,
        bidirectional: bool  = True,
        use_attention: bool  = True,
        pad_idx:       int   = 0,
    ) -> None:
        super().__init__()

        self.use_attention = use_attention
        self.bidirectional = bidirectional
        num_directions     = 2 if bidirectional else 1
        lstm_out_size      = hidden_size * num_directions

        # ── Embedding ────────────────────────────────────────────────────────
        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # Dropout is applied between layers (not on last layer output)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Attention ────────────────────────────────────────────────────────
        if use_attention:
            self.attention = AttentionLayer(lstm_out_size)

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute class logits from token index sequences.

        Args:
            input_ids : padded token indices, shape (B, L)

        Returns:
            Raw logits, shape (B, num_classes).
        """
        # Non-padding mask: True for real tokens
        mask    = input_ids != 0                        # (B, L)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()    # (B,)

        # Embedding with dropout
        x = self.embed_dropout(self.embedding(input_ids))   # (B, L, E)

        # Pack for efficiency (skips padding in LSTM computation)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed)

        # Unpack to (B, L, 2H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )

        # ── Context representation ───────────────────────────────────────────
        if self.use_attention:
            context, _ = self.attention(output, mask)   # (B, 2H)
        else:
            if self.bidirectional:
                # Concatenate the final forward and backward hidden states
                # hidden shape: (num_layers * 2, B, H)
                context = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (B, 2H)
            else:
                context = hidden[-1]                    # (B, H)

        return self.classifier(context)                 # (B, C)

    # ── Inference helpers ────────────────────────────────────────────────────

    def get_attention_weights(
        self, input_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Return attention weights for interpretability analysis.

        Returns:
            Attention weight tensor (B, L), or None if use_attention=False.
        """
        if not self.use_attention:
            return None

        mask    = input_ids != 0
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        x       = self.embed_dropout(self.embedding(input_ids))

        packed      = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        output, _   = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=input_ids.size(1)
        )
        _, weights = self.attention(output, mask)
        return weights

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
