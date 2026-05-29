"""
train_lstm.py — Train the proposed BiLSTM + Attention model.

This is the main training script of the thesis:
    "Deep Learning LSTM for Vietnamese Short Text Classification"
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.lstm_model import BiLSTMClassifier
from src.preprocessing.vocabulary import Vocabulary
from src.training.trainer import EarlyStopping, evaluate_model, run_epoch
from src.utils.config import CFG, get_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.evaluation.metrics import compute_metrics

logger = get_logger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LSTMDataset(Dataset):
    """PyTorch Dataset for BiLSTM text classification.

    Converts each normalised text string into a fixed-length integer sequence
    using the pre-built Vocabulary.

    Args:
        df      : DataFrame with ``text_norm`` and ``label_enc`` columns
        vocab   : fitted Vocabulary instance
        max_len : sequence length (truncated / padded to this value)
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int) -> None:
        self.texts  = df["text_norm"].fillna("").tolist()
        self.labels = df["label_enc"].tolist()
        self.vocab  = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label":     torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _batch_fn(batch: dict, device: torch.device):
    """Unpack dict batch into (input_ids, labels) for run_epoch."""
    return batch["input_ids"].to(device), batch["label"].to(device)


# ── Training entry point ──────────────────────────────────────────────────────

def train_lstm() -> dict:
    """Train the BiLSTM + Attention model and return training history."""
    set_seed(CFG.data.random_seed)
    device = get_device()
    cfg_m  = CFG.models.lstm

    out_models = Path(CFG.outputs.models_dir)
    out_logs   = Path(CFG.outputs.logs_dir)
    out_models.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    # ── Load artefacts ────────────────────────────────────────────────────────
    proc = Path(CFG.data.processed_dir)
    vocab = Vocabulary.load(proc / "vocabulary.pkl")
    logger.info("[LSTM] Vocabulary size: %d", vocab.size)

    train_df = pd.read_csv(proc / "train.csv")
    val_df   = pd.read_csv(proc / "val.csv")
    test_df  = pd.read_csv(proc / "test.csv")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        LSTMDataset(train_df, vocab, cfg_m.max_len),
        batch_size=cfg_m.batch_size, shuffle=True, pin_memory=True, num_workers=2,
    )
    val_loader = DataLoader(
        LSTMDataset(val_df, vocab, cfg_m.max_len),
        batch_size=cfg_m.batch_size, num_workers=2,
    )
    test_loader = DataLoader(
        LSTMDataset(test_df, vocab, cfg_m.max_len),
        batch_size=cfg_m.batch_size, num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        vocab_size=vocab.size,
        embed_dim=cfg_m.embed_dim,
        hidden_size=cfg_m.hidden_size,
        num_layers=cfg_m.num_layers,
        num_classes=CFG.labels.num_classes,
        dropout=cfg_m.dropout,
        bidirectional=cfg_m.bidirectional,
        use_attention=cfg_m.use_attention,
        pad_idx=Vocabulary.PAD_IDX,
    ).to(device)

    logger.info("[LSTM] Model architecture:\n%s", model)
    logger.info("[LSTM] Trainable parameters: %s", f"{model.count_parameters():,}")

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=cfg_m.lr, weight_decay=cfg_m.weight_decay
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_m.epochs)
    early_stop = EarlyStopping(
        patience=CFG.training.early_stopping_patience,
        save_path=str(out_models / "lstm_best.pt"),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    logger.info("[LSTM] Training for up to %d epochs on %s ...", cfg_m.epochs, device)
    t0 = time.time()

    for epoch in range(1, cfg_m.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, batch_fn=_batch_fn,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, batch_fn=_batch_fn
        )
        scheduler.step()

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_acc"].append(round(val_acc, 5))

        logger.info(
            "[LSTM] Epoch %3d | train_loss=%.4f | val_loss=%.4f "
            "| train_acc=%.4f | val_acc=%.4f",
            epoch, train_loss, val_loss, train_acc, val_acc,
        )

        if early_stop(val_loss, model, epoch):
            break

    elapsed = time.time() - t0

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(out_models / "lstm_best.pt", map_location=device, weights_only=True)
    )
    preds, labels = evaluate_model(model, test_loader, device, batch_fn=_batch_fn)
    metrics = compute_metrics(labels, preds, CFG.labels.codes)

    logger.info("[LSTM] Test accuracy: %.4f | F1: %.4f", metrics["accuracy"], metrics["f1"])
    logger.info("\n%s", metrics["report"])

    history["test_acc"]        = round(metrics["accuracy"], 5)
    history["test_f1"]         = round(metrics["f1"], 5)
    history["training_time_s"] = round(elapsed, 1)
    history["best_epoch"]      = early_stop.best_epoch

    with open(out_logs / "lstm_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    logger.info("[LSTM] Done in %.1fs. History saved.", elapsed)
    return history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    train_lstm()
