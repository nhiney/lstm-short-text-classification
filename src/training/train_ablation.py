"""
train_ablation.py — Ablation study for the BiLSTM + Attention model.

Trains three variants to isolate the contribution of each component:
  1. LSTM (unidirectional, no attention)   — baseline LSTM
  2. BiLSTM (bidirectional, no attention)  — contribution of bidirectionality
  3. BiLSTM + Attention                    — full proposed model (already in train_lstm.py)

Results are saved in outputs/logs/ablation_results.json for reporting.

Usage:
    python main.py ablation
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.lstm_model import BiLSTMClassifier
from src.preprocessing.vocabulary import Vocabulary
from src.training.train_lstm import LSTMDataset, _batch_fn
from src.training.trainer import EarlyStopping, evaluate_model, run_epoch
from src.utils.config import CFG, get_device
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.evaluation.metrics import compute_metrics

logger = get_logger(__name__)

ABLATION_VARIANTS = [
    {
        "name":          "LSTM (uni, no-attn)",
        "bidirectional": False,
        "use_attention": False,
        "save_name":     "ablation_lstm_uni.pt",
        "log_name":      "ablation_lstm_uni.json",
    },
    {
        "name":          "BiLSTM (no-attn)",
        "bidirectional": True,
        "use_attention": False,
        "save_name":     "ablation_bilstm_noattn.pt",
        "log_name":      "ablation_bilstm_noattn.json",
    },
    {
        "name":          "BiLSTM + Attention (proposed)",
        "bidirectional": True,
        "use_attention": True,
        "save_name":     "ablation_bilstm_attn.pt",
        "log_name":      "ablation_bilstm_attn.json",
    },
]


def _train_variant(cfg_m, vocab, train_loader, val_loader, test_loader,
                   bidirectional, use_attention, save_path, device) -> dict:
    """Train one ablation variant and return metrics."""
    model = BiLSTMClassifier(
        vocab_size=vocab.size,
        embed_dim=cfg_m.embed_dim,
        hidden_size=cfg_m.hidden_size,
        num_layers=cfg_m.num_layers,
        num_classes=CFG.labels.num_classes,
        dropout=cfg_m.dropout,
        bidirectional=bidirectional,
        use_attention=use_attention,
        pad_idx=Vocabulary.PAD_IDX,
    ).to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg_m.lr, weight_decay=cfg_m.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_m.epochs)
    early_stop = EarlyStopping(patience=CFG.training.early_stopping_patience, save_path=save_path)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(1, cfg_m.epochs + 1):
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, batch_fn=_batch_fn)
        vl, va = run_epoch(model, val_loader, criterion, device, batch_fn=_batch_fn)
        scheduler.step()

        history["train_loss"].append(round(tl, 5))
        history["val_loss"].append(round(vl, 5))
        history["train_acc"].append(round(ta, 5))
        history["val_acc"].append(round(va, 5))

        logger.info("  Epoch %3d | val_loss=%.4f | val_acc=%.4f", epoch, vl, va)
        if early_stop(vl, model, epoch):
            break

    elapsed = time.time() - t0
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    preds, labels = evaluate_model(model, test_loader, device, batch_fn=_batch_fn)
    metrics = compute_metrics(labels, preds, CFG.labels.codes)

    return {
        "accuracy":   metrics["accuracy"],
        "f1":         metrics["f1"],
        "precision":  metrics["precision"],
        "recall":     metrics["recall"],
        "num_params": model.count_parameters(),
        "training_time_s": round(elapsed, 1),
        "best_epoch": early_stop.best_epoch,
        "history":    history,
    }


def run_ablation() -> dict:
    """Train all ablation variants and save comparison to JSON."""
    set_seed(CFG.data.random_seed)
    device  = get_device()
    cfg_m   = CFG.models.lstm

    proc       = Path(CFG.data.processed_dir)
    out_models = Path(CFG.outputs.models_dir)
    out_logs   = Path(CFG.outputs.logs_dir)
    out_models.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    vocab    = Vocabulary.load(proc / "vocabulary.pkl")
    train_df = pd.read_csv(proc / "train.csv")
    val_df   = pd.read_csv(proc / "val.csv")
    test_df  = pd.read_csv(proc / "test.csv")

    make_loader = lambda df, shuffle: DataLoader(
        LSTMDataset(df, vocab, cfg_m.max_len),
        batch_size=cfg_m.batch_size, shuffle=shuffle, num_workers=2,
    )
    train_loader = make_loader(train_df, True)
    val_loader   = make_loader(val_df,   False)
    test_loader  = make_loader(test_df,  False)

    results = {}
    for v in ABLATION_VARIANTS:
        logger.info("\n[Ablation] Training: %s", v["name"])
        metrics = _train_variant(
            cfg_m, vocab, train_loader, val_loader, test_loader,
            bidirectional=v["bidirectional"],
            use_attention=v["use_attention"],
            save_path=str(out_models / v["save_name"]),
            device=device,
        )
        results[v["name"]] = metrics
        logger.info(
            "[Ablation] %s → acc=%.4f | f1=%.4f | params=%s",
            v["name"], metrics["accuracy"], metrics["f1"],
            f"{metrics['num_params']:,}",
        )

    # Save results
    report = {name: {k: v for k, v in m.items() if k != "history"}
              for name, m in results.items()}
    with open(out_logs / "ablation_results.json", "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info("[Ablation] Results saved to %s", out_logs / "ablation_results.json")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    run_ablation()
