"""
trainer.py — Shared training utilities used by all model-specific trainers.

Provides:
  - run_epoch    : one training or evaluation pass over a DataLoader
  - evaluate_model : collect predictions from a DataLoader without gradient
  - EarlyStopping  : patience-based early stopping with best-checkpoint saving
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Per-epoch pass ────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    batch_fn: Optional[Callable] = None,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """Run one training or evaluation epoch.

    When ``optimizer`` is provided the model is set to ``train()`` mode and
    gradients are computed; otherwise ``eval()`` mode is used.

    Args:
        model      : PyTorch model
        loader     : DataLoader yielding batches
        criterion  : loss function
        device     : computation device
        optimizer  : if provided, performs a gradient update
        scheduler  : optional LR scheduler called per *step* (not per epoch)
        batch_fn   : callable(batch, device) → (inputs, labels).
                     Defaults to the DNN convention ``(X, y) = batch``.
        grad_clip  : gradient clipping max-norm (0 to disable)

    Returns:
        (mean_loss, accuracy) over the epoch.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            if batch_fn is not None:
                inputs, labels = batch_fn(batch, device)
            else:
                # Default: batch = (X_tensor, y_tensor)
                inputs, labels = batch[0].to(device), batch[1].to(device)

            if training:
                optimizer.zero_grad()

            if isinstance(inputs, dict):
                logits = model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                logits = model(*inputs)
            else:
                logits = model(inputs)

            loss = criterion(logits, labels)

            if training:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ── Full inference over a loader ─────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    batch_fn: Optional[Callable] = None,
) -> Tuple[list, list]:
    """Collect predictions and ground-truth labels without computing gradients.

    Args:
        model    : trained PyTorch model in eval() mode
        loader   : DataLoader
        device   : computation device
        batch_fn : same convention as in ``run_epoch``

    Returns:
        (all_preds, all_labels) as Python lists of integers.
    """
    model.eval()
    all_preds:  list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            if batch_fn is not None:
                inputs, labels = batch_fn(batch, device)
            else:
                inputs, labels = batch[0].to(device), batch[1].to(device)

            if isinstance(inputs, dict):
                logits = model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                logits = model(*inputs)
            else:
                logits = model(inputs)

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())

            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.cpu().tolist())
            else:
                all_labels.extend(list(labels))

    return all_preds, all_labels


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Monitor validation loss and save the best model checkpoint.

    Args:
        patience  : number of epochs with no improvement before stopping
        save_path : file path to save the best model state dict
        delta     : minimum change to qualify as an improvement
    """

    def __init__(
        self,
        patience: int,
        save_path: str,
        delta: float = 1e-5,
    ) -> None:
        self.patience  = patience
        self.save_path = save_path
        self.delta     = delta
        self._best     = float("inf")
        self._counter  = 0
        self.best_epoch: int = 0

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """Check for improvement and save checkpoint if improved.

        Returns:
            True  → stop training (patience exhausted)
            False → continue training
        """
        if val_loss < self._best - self.delta:
            self._best     = val_loss
            self._counter  = 0
            self.best_epoch = epoch
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            logger.debug("Checkpoint saved (val_loss=%.4f)", val_loss)
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )
                return True
        return False
