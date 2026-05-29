"""
train_xlmr.py — Fine-tune XLM-RoBERTa for Vietnamese emotion classification.

Serves as the state-of-the-art comparison / upper-bound model for benchmarking
against the proposed BiLSTM.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.models.xlmr_model import XLMRClassifier
from src.training.trainer import EarlyStopping, evaluate_model, run_epoch
from src.utils.config import CFG, get_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.evaluation.metrics import compute_metrics

logger = get_logger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class XLMRDataset(Dataset):
    """PyTorch Dataset for XLM-RoBERTa fine-tuning.

    Args:
        df        : DataFrame with ``text_norm`` and ``label_enc`` columns
        tokenizer : HuggingFace tokenizer
        max_len   : maximum token sequence length
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int) -> None:
        self.texts    = df["text_norm"].fillna("").tolist()
        self.labels   = df["label_enc"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _batch_fn(batch: dict, device: torch.device):
    """Unpack dict batch into (model_inputs_dict, labels) for run_epoch."""
    inputs = {
        "input_ids":      batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    labels = batch["label"].to(device)
    return inputs, labels


# ── Training entry point ──────────────────────────────────────────────────────

def train_xlmr() -> dict:
    """Fine-tune XLM-RoBERTa and return training history."""
    set_seed(CFG.data.random_seed)
    device = get_device()
    cfg_m  = CFG.models.xlmr

    out_models = Path(CFG.outputs.models_dir)
    out_logs   = Path(CFG.outputs.logs_dir)
    out_models.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer & data ─────────────────────────────────────────────────────
    logger.info("[XLM-R] Loading tokenizer: %s", cfg_m.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg_m.model_name)

    proc = Path(CFG.data.processed_dir)
    train_df = pd.read_csv(proc / "train.csv")
    val_df   = pd.read_csv(proc / "val.csv")
    test_df  = pd.read_csv(proc / "test.csv")

    train_loader = DataLoader(
        XLMRDataset(train_df, tokenizer, cfg_m.max_len),
        batch_size=cfg_m.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        XLMRDataset(val_df, tokenizer, cfg_m.max_len),
        batch_size=cfg_m.batch_size, num_workers=2,
    )
    test_loader = DataLoader(
        XLMRDataset(test_df, tokenizer, cfg_m.max_len),
        batch_size=cfg_m.batch_size, num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = XLMRClassifier(
        model_name=cfg_m.model_name,
        num_classes=CFG.labels.num_classes,
        dropout=cfg_m.dropout,
    ).to(device)
    logger.info("[XLM-R] Parameters: %s", f"{model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_m.lr, weight_decay=cfg_m.weight_decay
    )
    total_steps  = len(train_loader) * cfg_m.epochs
    warmup_steps = int(total_steps * cfg_m.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    early_stop = EarlyStopping(
        patience=CFG.training.early_stopping_patience,
        save_path=str(out_models / "xlmr_best.pt"),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    logger.info("[XLM-R] Training for up to %d epochs on %s ...", cfg_m.epochs, device)
    t0 = time.time()

    for epoch in range(1, cfg_m.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scheduler=scheduler, batch_fn=_batch_fn,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, batch_fn=_batch_fn
        )

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_acc"].append(round(val_acc, 5))

        logger.info(
            "[XLM-R] Epoch %2d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch, train_loss, val_loss, val_acc,
        )

        if early_stop(val_loss, model, epoch):
            break

    elapsed = time.time() - t0

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(out_models / "xlmr_best.pt", map_location=device, weights_only=True)
    )
    preds, labels = evaluate_model(model, test_loader, device, batch_fn=_batch_fn)
    metrics = compute_metrics(labels, preds, CFG.labels.codes)

    logger.info("[XLM-R] Test accuracy: %.4f | F1: %.4f", metrics["accuracy"], metrics["f1"])
    logger.info("\n%s", metrics["report"])

    history["test_acc"]        = round(metrics["accuracy"], 5)
    history["test_f1"]         = round(metrics["f1"], 5)
    history["training_time_s"] = round(elapsed, 1)
    history["best_epoch"]      = early_stop.best_epoch

    with open(out_logs / "xlmr_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    # Save tokenizer alongside model weights
    tokenizer.save_pretrained(str(out_models / "xlmr_tokenizer"))

    logger.info("[XLM-R] Done in %.1fs. History saved.", elapsed)
    return history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    train_xlmr()
