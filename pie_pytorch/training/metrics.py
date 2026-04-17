"""Training metrics.

Two flavours used across the three PIE models:

    Classification (intent):
        - accuracy: fraction of correctly-predicted binary labels
        - f1: F1 score on the positive (crossing) class

    Regression (trajectory, speed):
        - mse: mean squared error across all timesteps and features
        - center_mse (C-MSE): MSE on the (cx, cy) bbox center, matching
          the metric the PIE paper reports for trajectory.

All functions accept raw ``Tensor`` inputs (no numpy) and return a
Python float. Binary-classification helpers accept logits or sigmoid
probabilities via the ``threshold`` argument (default 0.5 -> sigmoid
probs; pass 0.0 for raw logits).

See ``MetricsAccumulator`` for a simple reset/update/compute wrapper
the Trainer uses to aggregate batch-level values across an epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5
) -> float:
    """Binary accuracy.

    ``y_pred``/``y_true`` are any-shaped tensors with matching shapes.
    Predictions are first reduced to 0/1 with ``y_pred > threshold``.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: {y_pred.shape} vs {y_true.shape}")
    preds = (y_pred > threshold).to(torch.float32)
    labels = y_true.to(torch.float32)
    if preds.numel() == 0:
        return float("nan")
    return (preds == labels).float().mean().item()


def f1(
    y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """F1 score for the positive class in binary classification."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: {y_pred.shape} vs {y_true.shape}")
    preds = (y_pred > threshold).to(torch.float32)
    labels = y_true.to(torch.float32)
    tp = (preds * labels).sum().item()
    fp = (preds * (1.0 - labels)).sum().item()
    fn = ((1.0 - preds) * labels).sum().item()
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Elementwise mean squared error."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: {y_pred.shape} vs {y_true.shape}")
    return torch.nn.functional.mse_loss(y_pred, y_true).item()


def center_mse(y_pred_bbox: torch.Tensor, y_true_bbox: torch.Tensor) -> float:
    """MSE on bbox (center_x, center_y). Last-dim is ``[x1, y1, x2, y2]``.

    Matches the C-MSE the PIE paper reports.
    """
    if y_pred_bbox.shape != y_true_bbox.shape:
        raise ValueError(f"shape mismatch: {y_pred_bbox.shape} vs {y_true_bbox.shape}")
    if y_pred_bbox.shape[-1] != 4:
        raise ValueError(
            f"expected last-dim 4 (x1,y1,x2,y2), got {y_pred_bbox.shape[-1]}"
        )
    cx_p = (y_pred_bbox[..., 0] + y_pred_bbox[..., 2]) / 2.0
    cy_p = (y_pred_bbox[..., 1] + y_pred_bbox[..., 3]) / 2.0
    cx_t = (y_true_bbox[..., 0] + y_true_bbox[..., 2]) / 2.0
    cy_t = (y_true_bbox[..., 1] + y_true_bbox[..., 3]) / 2.0
    return ((cx_p - cx_t).pow(2).mean() + (cy_p - cy_t).pow(2).mean()).item() / 2.0


# ---------------------------------------------------------------------------
# Streaming accumulator
# ---------------------------------------------------------------------------
@dataclass
class MetricsAccumulator:
    """Accumulate per-batch predictions and targets in memory, then
    compute named metrics in one shot at epoch end.

    For the PIE models the dataset is small enough that holding all
    epoch predictions on CPU is cheap. If this ever becomes a memory
    concern, swap to running-sum formulas.
    """

    preds: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)

    def reset(self) -> None:
        self.preds.clear()
        self.targets.clear()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.preds.append(y_pred.detach().cpu())
        self.targets.append(y_true.detach().cpu())

    def compute(self, metrics: dict[str, callable]) -> dict[str, float]:
        if not self.preds:
            return {name: float("nan") for name in metrics}
        y_pred = torch.cat([p.reshape(-1, *p.shape[1:]) for p in self.preds], dim=0)
        y_true = torch.cat([t.reshape(-1, *t.shape[1:]) for t in self.targets], dim=0)
        return {name: fn(y_pred, y_true) for name, fn in metrics.items()}


__all__ = [
    "accuracy",
    "f1",
    "mse",
    "center_mse",
    "MetricsAccumulator",
]
