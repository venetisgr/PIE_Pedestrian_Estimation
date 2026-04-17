"""Pure-PyTorch equivalents of the Keras callbacks used by the PIE code.

Two minimal implementations, both ``state_dict()``-able so the Trainer
can persist them across resume:

    - EarlyStopping(monitor, mode, patience, min_delta)
    - ReduceLROnPlateau(monitor, mode, factor, patience, min_lr)

Keras semantics (important):
    - ``mode='min'`` considers a lower value "better"; ``'max'`` the opposite.
    - ``patience`` is the number of *non-improving* epochs tolerated.
    - ReduceLROnPlateau multiplies the LR by ``factor`` when patience
      runs out, subject to ``min_lr``.
    - EarlyStopping.should_stop goes True the same epoch that patience
      runs out.

The Trainer is responsible for calling ``update(epoch, metrics)`` once
per epoch and then checking the callback state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


Mode = Literal["min", "max"]


def _improved(current: float, best: float, mode: Mode, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


@dataclass
class EarlyStopping:
    monitor: str = "val_loss"
    mode: Mode = "min"
    patience: int = 5
    min_delta: float = 0.0

    best: float = field(init=False)
    bad_epochs: int = 0
    should_stop: bool = False

    def __post_init__(self):
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def update(self, metrics: dict[str, float]) -> bool:
        if self.monitor not in metrics:
            raise KeyError(
                f"EarlyStopping monitor {self.monitor!r} not in metrics keys "
                f"{sorted(metrics)}"
            )
        value = metrics[self.monitor]
        if _improved(value, self.best, self.mode, self.min_delta):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]
        self.should_stop = state["should_stop"]


@dataclass
class ReduceLROnPlateau:
    optimizer: torch.optim.Optimizer
    monitor: str = "val_loss"
    mode: Mode = "min"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-7
    min_delta: float = 0.0

    best: float = field(init=False)
    bad_epochs: int = 0

    def __post_init__(self):
        assert 0.0 < self.factor < 1.0, "factor must be in (0, 1)"
        self.best = float("inf") if self.mode == "min" else float("-inf")

    def update(self, metrics: dict[str, float]) -> float | None:
        """Return the new LR after this update (a float), or ``None`` if LR
        was not changed."""
        if self.monitor not in metrics:
            raise KeyError(
                f"ReduceLROnPlateau monitor {self.monitor!r} not in metrics "
                f"keys {sorted(metrics)}"
            )
        value = metrics[self.monitor]
        if _improved(value, self.best, self.mode, self.min_delta):
            self.best = value
            self.bad_epochs = 0
            return None
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            new_lr = None
            for pg in self.optimizer.param_groups:
                curr = pg["lr"]
                next_lr = max(self.min_lr, curr * self.factor)
                if next_lr < curr:
                    pg["lr"] = next_lr
                    new_lr = next_lr
            self.bad_epochs = 0
            return new_lr
        return None

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]


__all__ = ["EarlyStopping", "ReduceLROnPlateau", "Mode"]
