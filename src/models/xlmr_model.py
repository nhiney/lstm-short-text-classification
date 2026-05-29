"""
xlmr_model.py — XLM-RoBERTa fine-tuned classifier (comparison / upper-bound model).

Used to benchmark the proposed BiLSTM against a state-of-the-art pre-trained
transformer, demonstrating how much is gained by using contextual representations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class XLMRClassifier(nn.Module):
    """XLM-RoBERTa backbone with a linear classification head.

    The [CLS] token representation is passed through dropout → linear layer
    to produce class logits.

    Args:
        model_name  : HuggingFace model identifier
        num_classes : number of output emotion classes
        dropout     : dropout probability before the classifier
    """

    def __init__(
        self,
        model_name: str   = "xlm-roberta-base",
        num_classes: int  = 7,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size   # 768 for base
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            input_ids      : (B, L) tokenised text
            attention_mask : (B, L) 1 = real token, 0 = padding
            token_type_ids : optional segment IDs (not used by XLM-R)

        Returns:
            Raw logits, shape (B, num_classes).
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_emb = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls_emb)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
