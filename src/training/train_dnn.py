"""
train_dnn.py — Train the TF-IDF → DNN baseline model.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset

from src.models.dnn_tfidf import DNNClassifier
from src.training.trainer import EarlyStopping, run_epoch, evaluate_model
from src.utils.config import CFG, get_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.evaluation.metrics import compute_metrics

logger = get_logger(__name__)


def _build_tensor_dataset(X_sparse, y_array: np.ndarray) -> TensorDataset:
    X = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
    y = torch.tensor(y_array, dtype=torch.long)
    return TensorDataset(X, y)


def train_dnn() -> dict:
    """Train TF-IDF + DNN baseline and return training history."""
    set_seed(CFG.data.random_seed)
    device = get_device()
    cfg_m  = CFG.models.dnn

    out_models  = Path(CFG.outputs.models_dir)
    out_logs    = Path(CFG.outputs.logs_dir)
    out_models.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    # ── Load preprocessed data ────────────────────────────────────────────────
    proc = CFG.data.processed_dir
    train_df = pd.read_csv(Path(proc) / "train.csv")
    val_df   = pd.read_csv(Path(proc) / "val.csv")
    test_df  = pd.read_csv(Path(proc) / "test.csv")

    # ── TF-IDF vectorisation ─────────────────────────────────────────────────
    logger.info("[DNN] Fitting TF-IDF vectoriser ...")
    tfidf = TfidfVectorizer(
        max_features=cfg_m.tfidf.max_features,
        ngram_range=tuple(cfg_m.tfidf.ngram_range),
        analyzer="word",
        sublinear_tf=cfg_m.tfidf.sublinear_tf,
    )
    X_train = tfidf.fit_transform(train_df["text_norm"].fillna(""))
    X_val   = tfidf.transform(val_df["text_norm"].fillna(""))
    X_test  = tfidf.transform(test_df["text_norm"].fillna(""))

    y_train = train_df["label_enc"].values
    y_val   = val_df["label_enc"].values
    y_test  = test_df["label_enc"].values

    with open(out_models / "tfidf.pkl", "wb") as fh:
        pickle.dump(tfidf, fh)
    logger.info("[DNN] TF-IDF vocabulary size: %d", len(tfidf.vocabulary_))

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        _build_tensor_dataset(X_train, y_train),
        batch_size=cfg_m.batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        _build_tensor_dataset(X_val, y_val),
        batch_size=cfg_m.batch_size,
    )
    test_loader = DataLoader(
        _build_tensor_dataset(X_test, y_test),
        batch_size=cfg_m.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DNNClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=cfg_m.hidden_dims,
        num_classes=CFG.labels.num_classes,
        dropout=cfg_m.dropout,
    ).to(device)
    logger.info("[DNN] Parameters: %s", f"{model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_m.lr, weight_decay=cfg_m.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_m.epochs)
    early_stop = EarlyStopping(
        patience=CFG.training.early_stopping_patience,
        save_path=str(out_models / "dnn_best.pt"),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    logger.info("[DNN] Training for up to %d epochs on %s ...", cfg_m.epochs, device)
    t0 = time.time()

    for epoch in range(1, cfg_m.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_acc"].append(round(train_acc, 5))
        history["val_acc"].append(round(val_acc, 5))

        logger.info(
            "[DNN] Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch, train_loss, val_loss, val_acc,
        )

        if early_stop(val_loss, model, epoch):
            break

    elapsed = time.time() - t0

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(out_models / "dnn_best.pt", map_location=device, weights_only=True)
    )
    preds, labels = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(labels, preds, CFG.labels.codes)

    logger.info("[DNN] Test accuracy: %.4f | F1: %.4f", metrics["accuracy"], metrics["f1"])
    logger.info("\n%s", metrics["report"])

    history["test_acc"]       = round(metrics["accuracy"], 5)
    history["test_f1"]        = round(metrics["f1"], 5)
    history["training_time_s"] = round(elapsed, 1)
    history["best_epoch"]     = early_stop.best_epoch

    with open(out_logs / "dnn_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    logger.info("[DNN] Done in %.1fs. History saved.", elapsed)
    return history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    train_dnn()
