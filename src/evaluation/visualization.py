"""
visualization.py — Generate publication-quality plots for thesis evaluation.

Produces:
  - Training / validation loss and accuracy curves (per model)
  - Confusion matrices (per model)
  - Model comparison bar chart (accuracy, precision, recall, F1)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from src.utils.config import CFG
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":   150,
    "savefig.dpi":  200,
    "savefig.bbox": "tight",
})

_PALETTE = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot loss and accuracy curves for one model.

    Args:
        history    : dict with keys train_loss, val_loss, train_acc, val_acc
        model_name : label used in title
        save_path  : if provided, save figure to this path

    Returns:
        Matplotlib Figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train")
    ax_loss.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Validation")
    ax_loss.set_title(f"{model_name} — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax_acc.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train")
    ax_acc.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Validation")
    ax_acc.set_title(f"{model_name} — Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f"Training History — {model_name}", fontweight="bold")
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: List[List[int]],
    class_labels: List[str],
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix heatmap.

    Args:
        cm           : raw confusion matrix (list of lists)
        class_labels : ordered class names
        model_name   : label used in title
        save_path    : if provided, save figure to this path

    Returns:
        Matplotlib Figure.
    """
    cm_arr    = np.array(cm, dtype=np.float64)
    row_sums  = cm_arr.sum(axis=1, keepdims=True)
    cm_norm   = np.where(row_sums > 0, cm_arr / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ── Model comparison ──────────────────────────────────────────────────────────

def plot_model_comparison(
    results: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side grouped bar chart comparing metric scores.

    Args:
        results   : dict mapping model_name → metrics dict
                    (must have accuracy, precision, recall, f1 keys)
        save_path : if provided, save figure to this path

    Returns:
        Matplotlib Figure.
    """
    model_names = list(results.keys())
    metrics     = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    n_models  = len(model_names)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(model_names, _PALETTE + ["#c44e52"] * 10)):
        vals = [results[name].get(m, 0) for m in metrics]
        bars = ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            vals,
            width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=30,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ── Ablation study chart ─────────────────────────────────────────────────────

def plot_ablation_study(
    ablation_results: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart comparing ablation variants on accuracy and F1.

    Args:
        ablation_results : dict mapping variant_name → {accuracy, f1, ...}
        save_path        : if provided, save figure to this path

    Returns:
        Matplotlib Figure.
    """
    names    = list(ablation_results.keys())
    acc_vals = [ablation_results[n]["accuracy"] for n in names]
    f1_vals  = [ablation_results[n]["f1"]       for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, f1_vals,  width, label="F1 (weighted)",
                   color="#DD8452", edgecolor="white")

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Contribution of BiLSTM Components", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ── Master generator ──────────────────────────────────────────────────────────

def generate_all_figures() -> None:
    """Load all saved artefacts and generate every figure."""
    out_logs    = Path(CFG.outputs.logs_dir)
    out_reports = Path(CFG.outputs.reports_dir)
    out_figs    = Path(CFG.outputs.figures_dir)
    class_labels = CFG.labels.codes

    # Training curves + confusion matrices
    model_map = {
        "DNN+TF-IDF": "dnn_history.json",
        "BiLSTM":     "lstm_history.json",
        "XLM-R":      "xlmr_history.json",
    }
    report_path = out_reports / "comparison.json"
    results = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as fh:
            results = json.load(fh)

    for model_name, fname in model_map.items():
        hist_path = out_logs / fname
        if not hist_path.exists():
            logger.warning("History not found: %s — skipping.", hist_path)
            continue

        with open(hist_path, encoding="utf-8") as fh:
            history = json.load(fh)

        slug = model_name.lower().replace("+", "_").replace("-", "_")
        plot_training_curves(
            history,
            model_name,
            save_path=out_figs / f"{slug}_curves.png",
        )

        if model_name in results:
            cm = results[model_name].get("confusion_matrix")
            if cm:
                plot_confusion_matrix(
                    cm,
                    class_labels,
                    model_name,
                    save_path=out_figs / f"{slug}_cm.png",
                )

    # Comparison chart
    if results:
        plot_model_comparison(results, save_path=out_figs / "model_comparison.png")

    # Ablation study chart (if available)
    ablation_path = out_logs / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path, encoding="utf-8") as fh:
            ablation = json.load(fh)
        plot_ablation_study(ablation, save_path=out_figs / "ablation_study.png")

    logger.info("All figures generated in %s", out_figs)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    generate_all_figures()
