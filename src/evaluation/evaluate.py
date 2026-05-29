"""
evaluate.py — Evaluate all three models on the test set and produce a
comparison report (JSON + console).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.utils.config import CFG, get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── DNN evaluation ────────────────────────────────────────────────────────────

def eval_dnn(test_df: pd.DataFrame, device: torch.device) -> dict:
    """Evaluate the TF-IDF + DNN baseline on test_df."""
    from src.models.dnn_tfidf import DNNClassifier

    out_models = Path(CFG.outputs.models_dir)
    cfg_m      = CFG.models.dnn

    logger.info("[Eval] Loading DNN ...")
    with open(out_models / "tfidf.pkl", "rb") as fh:
        tfidf = pickle.load(fh)

    X = torch.tensor(
        tfidf.transform(test_df["text_norm"].fillna("")).toarray(),
        dtype=torch.float32,
    )
    y = torch.tensor(test_df["label_enc"].values, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=128)

    model = DNNClassifier(
        input_dim=len(tfidf.vocabulary_),   # use actual vocab size, not config default
        hidden_dims=cfg_m.hidden_dims,
        num_classes=CFG.labels.num_classes,
        dropout=cfg_m.dropout,
    ).to(device)
    model.load_state_dict(
        torch.load(out_models / "dnn_best.pt", map_location=device, weights_only=True)
    )
    model.eval()

    preds, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend(model(Xb.to(device)).argmax(1).cpu().tolist())
            labels.extend(yb.tolist())
    inf_ms = (time.perf_counter() - t0) / len(labels) * 1_000

    m = compute_metrics(labels, preds, CFG.labels.codes)
    m["inference_ms_per_sample"] = round(inf_ms, 4)
    m["num_params"] = _count_params(model)
    return m


# ── LSTM evaluation ────────────────────────────────────────────────────────────

def eval_lstm(test_df: pd.DataFrame, device: torch.device) -> dict:
    """Evaluate the BiLSTM + Attention model on test_df."""
    from src.models.lstm_model import BiLSTMClassifier
    from src.preprocessing.vocabulary import Vocabulary
    from src.training.train_lstm import LSTMDataset

    out_models = Path(CFG.outputs.models_dir)
    proc       = Path(CFG.data.processed_dir)
    cfg_m      = CFG.models.lstm

    logger.info("[Eval] Loading BiLSTM ...")
    vocab = Vocabulary.load(proc / "vocabulary.pkl")

    loader = DataLoader(
        LSTMDataset(test_df, vocab, cfg_m.max_len),
        batch_size=64,
    )

    model = BiLSTMClassifier(
        vocab_size=vocab.size,
        embed_dim=cfg_m.embed_dim,
        hidden_size=cfg_m.hidden_size,
        num_layers=cfg_m.num_layers,
        num_classes=CFG.labels.num_classes,
        dropout=cfg_m.dropout,
        bidirectional=cfg_m.bidirectional,
        use_attention=cfg_m.use_attention,
        pad_idx=0,
    ).to(device)
    model.load_state_dict(
        torch.load(out_models / "lstm_best.pt", map_location=device, weights_only=True)
    )
    model.eval()

    preds, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            preds.extend(model(ids).argmax(1).cpu().tolist())
            labels.extend(batch["label"].tolist())
    inf_ms = (time.perf_counter() - t0) / len(labels) * 1_000

    m = compute_metrics(labels, preds, CFG.labels.codes)
    m["inference_ms_per_sample"] = round(inf_ms, 4)
    m["num_params"] = _count_params(model)
    return m


# ── XLM-R evaluation ──────────────────────────────────────────────────────────

def eval_xlmr(test_df: pd.DataFrame, device: torch.device) -> dict:
    """Evaluate the fine-tuned XLM-RoBERTa on test_df."""
    from transformers import AutoTokenizer

    from src.models.xlmr_model import XLMRClassifier
    from src.training.train_xlmr import XLMRDataset

    out_models = Path(CFG.outputs.models_dir)
    cfg_m      = CFG.models.xlmr

    logger.info("[Eval] Loading XLM-R ...")
    tok_path  = out_models / "xlmr_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_path) if tok_path.exists() else cfg_m.model_name
    )
    loader = DataLoader(
        XLMRDataset(test_df, tokenizer, cfg_m.max_len),
        batch_size=cfg_m.batch_size,
    )

    model = XLMRClassifier(
        model_name=cfg_m.model_name,
        num_classes=CFG.labels.num_classes,
        dropout=cfg_m.dropout,
    ).to(device)
    model.load_state_dict(
        torch.load(out_models / "xlmr_best.pt", map_location=device, weights_only=True)
    )
    model.eval()

    preds, labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            preds.extend(model(ids, mask).argmax(1).cpu().tolist())
            labels.extend(batch["label"].tolist())
    inf_ms = (time.perf_counter() - t0) / len(labels) * 1_000

    m = compute_metrics(labels, preds, CFG.labels.codes)
    m["inference_ms_per_sample"] = round(inf_ms, 4)
    m["num_params"] = _count_params(model)
    return m


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_evaluation() -> dict:
    """Evaluate all available models and save a comparison JSON report."""
    out_reports = Path(CFG.outputs.reports_dir)
    out_reports.mkdir(parents=True, exist_ok=True)

    device  = get_device()
    test_df = pd.read_csv(Path(CFG.data.processed_dir) / "test.csv")

    eval_fns: Dict[str, Callable] = {
        "DNN+TF-IDF": eval_dnn,
        "BiLSTM":     eval_lstm,
        "XLM-R":      eval_xlmr,
    }

    results = {}
    for name, fn in eval_fns.items():
        try:
            r = fn(test_df, device)
            results[name] = r
            logger.info(
                "[%s] accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | "
                "params=%s | inf=%.2f ms/sample",
                name, r["accuracy"], r["precision"], r["recall"], r["f1"],
                f"{r['num_params']:,}", r["inference_ms_per_sample"],
            )
        except FileNotFoundError as exc:
            logger.warning("[%s] Model not found — skipping. (%s)", name, exc)

    report_path = out_reports / "comparison.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    logger.info("Comparison report saved to %s", report_path)
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    run_evaluation()
