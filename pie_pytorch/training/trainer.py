"""Generic training loop used for all three PIE models.

Features
--------
- RMSprop with ``alpha=0.9`` and ``eps=1e-7`` (Keras RMSprop defaults).
- AMP via ``torch.amp.autocast`` + ``torch.amp.GradScaler`` (auto-enabled
  on CUDA, no-op on CPU).
- Gradient clipping by global norm (``clip_grad_norm``).
- Keras-equivalent ``ReduceLROnPlateau`` + ``EarlyStopping`` callbacks.
- Top-K checkpointing via ``TopKCheckpointManager``.
- Optional W&B logging (lazy-imported so the dep is soft).

Design choices
--------------
- The model takes a ``dict`` batch (what our Datasets produce). The
  caller supplies a ``forward_fn(model, batch) -> (preds, targets)``
  so the loop stays agnostic to intent vs trajectory vs speed shapes.
- ``loss_fn(preds, targets, batch)`` and ``metrics`` are passed in
  explicitly — no heuristics.
- The Trainer has no model-building logic of its own. That lives in
  the CLI / config glue (Phase 3.5).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader

from .callbacks import EarlyStopping, Mode, ReduceLROnPlateau
from .checkpoint import TopKCheckpointManager
from .metrics import MetricsAccumulator


logger = logging.getLogger(__name__)


ForwardFn = Callable[[torch.nn.Module, dict], tuple[torch.Tensor, torch.Tensor]]
LossFn = Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]


@dataclass
class TrainerConfig:
    epochs: int = 100
    lr: float = 1e-5
    weight_decay: float = 0.0           # Keras uses per-layer L2; we apply globally (see notepad D8)
    rmsprop_alpha: float = 0.9          # Keras default 0.9
    rmsprop_eps: float = 1e-7           # Keras default 1e-7
    clip_grad_norm: float | None = 1.0  # None to disable
    use_amp: bool | None = None         # None -> auto (CUDA->True, CPU->False)
    device: str = "auto"                # "auto" | "cpu" | "cuda" | "mps"

    # Callbacks
    monitor: str = "val_loss"
    mode: Mode = "min"
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_min_lr: float = 1e-7
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str | None = None
    keep_top_k: int = 3

    # Logging
    log_every_n_steps: int = 25
    wandb: bool = False                 # set True to enable; we still gracefully no-op if wandb is missing
    wandb_project: str = "pie_pytorch"
    wandb_run_name: str | None = None
    wandb_config: dict = field(default_factory=dict)

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def resolve_amp(self, device: torch.device) -> bool:
        if self.use_amp is not None:
            return self.use_amp
        return device.type == "cuda"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: LossFn,
        metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
        forward_fn: ForwardFn,
        cfg: TrainerConfig | None = None,
    ):
        self.cfg = cfg or TrainerConfig()
        self.device = self.cfg.resolve_device()
        self.use_amp = self.cfg.resolve_amp(self.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.forward_fn = forward_fn

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.cfg.lr,
            alpha=self.cfg.rmsprop_alpha,
            eps=self.cfg.rmsprop_eps,
            weight_decay=self.cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.early_stop = EarlyStopping(
            monitor=self.cfg.monitor,
            mode=self.cfg.mode,
            patience=self.cfg.early_stop_patience,
            min_delta=self.cfg.early_stop_min_delta,
        )
        self.plateau = ReduceLROnPlateau(
            self.optimizer,
            monitor=self.cfg.monitor,
            mode=self.cfg.mode,
            factor=self.cfg.plateau_factor,
            patience=self.cfg.plateau_patience,
            min_lr=self.cfg.plateau_min_lr,
        )
        self.ckpt: TopKCheckpointManager | None = (
            TopKCheckpointManager(
                self.cfg.checkpoint_dir,
                monitor=self.cfg.monitor,
                mode=self.cfg.mode,
                k=self.cfg.keep_top_k,
            )
            if self.cfg.checkpoint_dir
            else None
        )

        self._wandb_run = None
        self._global_step = 0

    # ----------------------------------------------------------------
    # W&B lifecycle (soft dependency)
    # ----------------------------------------------------------------
    def _maybe_init_wandb(self) -> None:
        if not self.cfg.wandb:
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb requested but not installed; falling back to noop")
            return
        self._wandb_run = wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name,
            config={
                **self.cfg.wandb_config,
                "trainer": {
                    k: v for k, v in self.cfg.__dict__.items() if k != "wandb_config"
                },
                "device": str(self.device),
                "amp": self.use_amp,
            },
        )

    def _wandb_log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self._wandb_run is None:
            return
        self._wandb_run.log(data, step=step)

    def _wandb_finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    # ----------------------------------------------------------------
    # Epoch loops
    # ----------------------------------------------------------------
    def _move_batch(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        acc = MetricsAccumulator()
        running_loss, running_n = 0.0, 0
        for i, batch in enumerate(loader):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                preds, targets = self.forward_fn(self.model, batch)
                loss = self.loss_fn(preds, targets, batch)
            self.scaler.scale(loss).backward()
            if self.cfg.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * targets.shape[0]
            running_n += targets.shape[0]
            acc.update(preds, targets)
            self._global_step += 1
            if (i + 1) % self.cfg.log_every_n_steps == 0:
                self._wandb_log(
                    {"train/loss_step": loss.item(), "train/lr": self._lr()},
                    step=self._global_step,
                )

        avg_loss = running_loss / max(running_n, 1)
        computed = acc.compute(self.metrics)
        return {"loss": avg_loss, **computed}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        acc = MetricsAccumulator()
        running_loss, running_n = 0.0, 0
        for batch in loader:
            batch = self._move_batch(batch)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                preds, targets = self.forward_fn(self.model, batch)
                loss = self.loss_fn(preds, targets, batch)
            running_loss += loss.item() * targets.shape[0]
            running_n += targets.shape[0]
            acc.update(preds, targets)
        avg_loss = running_loss / max(running_n, 1)
        computed = acc.compute(self.metrics)
        return {"loss": avg_loss, **computed}

    # ----------------------------------------------------------------
    # Public entry
    # ----------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict[str, Any]:
        """Run the full training loop. Returns a dict with final state."""
        self._maybe_init_wandb()
        history: list[dict[str, float]] = []
        try:
            for epoch in range(1, self.cfg.epochs + 1):
                train_metrics = self._train_one_epoch(train_loader, epoch)
                val_metrics = self._validate(val_loader) if val_loader is not None else {}
                merged = {
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "lr": self._lr(),
                    "epoch": epoch,
                }
                history.append(merged)
                self._wandb_log(merged, step=self._global_step)
                logger.info("epoch %d: %s", epoch, merged)

                # Monitor (default: val/loss) drives plateau + early-stop + ckpt.
                flat = {f"{split}_{k}": v
                        for split, d in (("train", train_metrics), ("val", val_metrics))
                        for k, v in d.items()}
                if self.cfg.monitor not in flat:
                    raise KeyError(
                        f"monitor {self.cfg.monitor!r} not in epoch metrics {sorted(flat)}"
                    )
                self.plateau.update(flat)
                stop = self.early_stop.update(flat)

                if self.ckpt is not None:
                    self.ckpt.save(
                        epoch=epoch,
                        metrics=flat,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=None,  # we use callbacks instead of a scheduler
                        callbacks={"early_stop": self.early_stop, "plateau": self.plateau},
                    )
                if stop:
                    logger.info("EarlyStopping: patience exhausted at epoch %d", epoch)
                    break
        finally:
            self._wandb_finish()
        return {"history": history, "best": self.early_stop.best}

    def _lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


__all__ = ["Trainer", "TrainerConfig", "ForwardFn", "LossFn"]
