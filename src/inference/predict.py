"""
predict.py — Inference pipeline for all three trained models.

Each predictor class:
  1. Loads model weights + associated artefacts (tokenizer / vocabulary / scaler)
  2. Runs the full preprocessing pipeline on raw text
  3. Returns a structured dict: {predicted_label, predicted_name, confidence, probabilities}

Usage::

    from src.inference.predict import LSTMPredictor
    predictor = LSTMPredictor()
    result = predictor.predict("Hôm nay tôi rất vui!!!")
    print(result["predicted_name"], result["confidence"])
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing.clean_text import clean_text
from src.preprocessing.teencode_normalize import normalize_teencode
from src.utils.config import CFG, get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _preprocess(raw: str):
    """Apply cleaning + teencode normalisation. Returns (cleaned, normalised)."""
    cleaned  = clean_text(raw, remove_emoji=False)
    normalised = normalize_teencode(cleaned)
    return cleaned, normalised


def _load_meta() -> dict:
    with open(Path(CFG.data.processed_dir) / "meta.json", encoding="utf-8") as fh:
        return json.load(fh)


def _logits_to_result(logits: torch.Tensor, classes: List[str]) -> dict:
    probs     = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx  = int(probs.argmax())
    pred_code = classes[pred_idx]
    names     = dict(zip(CFG.labels.codes, [getattr(CFG.labels.names, c, c) for c in CFG.labels.codes]))
    return {
        "predicted_label": pred_code,
        "predicted_name":  names.get(pred_code, pred_code),
        "confidence":      round(float(probs[pred_idx]), 4),
        "probabilities":   {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


# ── DNN predictor ─────────────────────────────────────────────────────────────

class DNNPredictor:
    """Inference wrapper for the TF-IDF + DNN baseline model."""

    def __init__(self) -> None:
        from src.models.dnn_tfidf import DNNClassifier

        self._device = get_device()
        meta         = _load_meta()
        self._classes = meta["classes"]
        cfg_m         = CFG.models.dnn

        with open(Path(CFG.outputs.models_dir) / "tfidf.pkl", "rb") as fh:
            self._tfidf = pickle.load(fh)

        self._model = DNNClassifier(
            input_dim=cfg_m.tfidf.max_features,
            hidden_dims=cfg_m.hidden_dims,
            num_classes=CFG.labels.num_classes,
            dropout=cfg_m.dropout,
        ).to(self._device)
        self._model.load_state_dict(
            torch.load(
                Path(CFG.outputs.models_dir) / "dnn_best.pt",
                map_location=self._device,
                weights_only=True,
            )
        )
        self._model.eval()

    def predict(self, text: str) -> dict:
        _, normed = _preprocess(text)
        X = torch.tensor(
            self._tfidf.transform([normed]).toarray(), dtype=torch.float32
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(X)
        return _logits_to_result(logits, self._classes)


# ── LSTM predictor ────────────────────────────────────────────────────────────

class LSTMPredictor:
    """Inference wrapper for the BiLSTM + Attention proposed model."""

    def __init__(self) -> None:
        from src.models.lstm_model import BiLSTMClassifier
        from src.preprocessing.vocabulary import Vocabulary

        self._device = get_device()
        meta          = _load_meta()
        self._classes = meta["classes"]
        cfg_m          = CFG.models.lstm

        self._vocab = Vocabulary.load(
            Path(CFG.data.processed_dir) / "vocabulary.pkl"
        )
        self._max_len = cfg_m.max_len

        self._model = BiLSTMClassifier(
            vocab_size=self._vocab.size,
            embed_dim=cfg_m.embed_dim,
            hidden_size=cfg_m.hidden_size,
            num_layers=cfg_m.num_layers,
            num_classes=CFG.labels.num_classes,
            dropout=cfg_m.dropout,
            bidirectional=cfg_m.bidirectional,
            use_attention=cfg_m.use_attention,
            pad_idx=0,
        ).to(self._device)
        self._model.load_state_dict(
            torch.load(
                Path(CFG.outputs.models_dir) / "lstm_best.pt",
                map_location=self._device,
                weights_only=True,
            )
        )
        self._model.eval()

    def predict(self, text: str) -> dict:
        _, normed = _preprocess(text)
        ids = self._vocab.encode(normed, self._max_len)
        input_ids = torch.tensor([ids], dtype=torch.long).to(self._device)
        with torch.no_grad():
            logits = self._model(input_ids)
        return _logits_to_result(logits, self._classes)

    def get_attention_weights(self, text: str) -> Dict[str, float]:
        """Return per-token attention weight for interpretability.

        Returns:
            Dict mapping token → attention_weight (float).
        """
        from src.preprocessing.vocabulary import Vocabulary

        _, normed = _preprocess(text)
        tokens    = normed.split()[: self._max_len]
        ids       = self._vocab.encode(normed, self._max_len)
        input_ids = torch.tensor([ids], dtype=torch.long).to(self._device)

        with torch.no_grad():
            weights = self._model.get_attention_weights(input_ids)

        if weights is None:
            return {}

        w_arr = weights.squeeze(0).cpu().numpy()[: len(tokens)]
        return {tok: round(float(w), 4) for tok, w in zip(tokens, w_arr)}


# ── XLM-R predictor ───────────────────────────────────────────────────────────

class XLMRPredictor:
    """Inference wrapper for the fine-tuned XLM-RoBERTa model."""

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        from src.models.xlmr_model import XLMRClassifier

        self._device = get_device()
        meta          = _load_meta()
        self._classes = meta["classes"]
        cfg_m          = CFG.models.xlmr

        tok_path = Path(CFG.outputs.models_dir) / "xlmr_tokenizer"
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(tok_path) if tok_path.exists() else cfg_m.model_name
        )
        self._max_len = cfg_m.max_len

        self._model = XLMRClassifier(
            model_name=cfg_m.model_name,
            num_classes=CFG.labels.num_classes,
            dropout=cfg_m.dropout,
        ).to(self._device)
        self._model.load_state_dict(
            torch.load(
                Path(CFG.outputs.models_dir) / "xlmr_best.pt",
                map_location=self._device,
                weights_only=True,
            )
        )
        self._model.eval()

    def predict(self, text: str) -> dict:
        _, normed = _preprocess(text)
        enc = self._tokenizer(
            normed,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self._device)
        mask = enc["attention_mask"].to(self._device)
        with torch.no_grad():
            logits = self._model(ids, mask)
        return _logits_to_result(logits, self._classes)
