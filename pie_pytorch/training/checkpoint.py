"""Top-K checkpoint manager.

Stores model weights as ``safetensors`` (portable, no pickle attack
surface) and trainer state (optimizer, scheduler, callbacks, epoch,
metrics) as a sibling ``.state.pt`` torch file.

Layout per save::

    {dir}/epoch_{N:04d}/model.safetensors
    {dir}/epoch_{N:04d}/state.pt

Plus two symlinks:
    {dir}/best    -> epoch_{K:04d}  (lowest val_loss by default)
    {dir}/latest  -> epoch_{L:04d}

The TopKCheckpointManager keeps at most ``k`` checkpoints and deletes
the worst one when a better run comes along.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file


Mode = Literal["min", "max"]


@dataclass
class CheckpointRecord:
    epoch: int
    metric: float
    path: Path


class TopKCheckpointManager:
    """Keep at most K best checkpoints ranked by a single metric."""

    def __init__(
        self,
        root: str | os.PathLike,
        monitor: str = "val_loss",
        mode: Mode = "min",
        k: int = 3,
    ):
        assert k >= 1
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.k = k
        self.records: list[CheckpointRecord] = []

    # -- public API -----------------------------------------------------
    def save(
        self,
        *,
        epoch: int,
        metrics: dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        callbacks: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save the current run to ``epoch_{N:04d}/`` and update top-K
        bookkeeping. Returns the path to the new epoch directory."""
        if self.monitor not in metrics:
            raise KeyError(
                f"monitor {self.monitor!r} not in metrics {sorted(metrics)}"
            )
        ep_dir = self.root / f"epoch_{epoch:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        save_file(_cpu_state_dict(model), str(ep_dir / "model.safetensors"))
        state = {
            "epoch": epoch,
            "metrics": metrics,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "callbacks": {k: v.state_dict() for k, v in (callbacks or {}).items()},
            "extra": extra or {},
        }
        torch.save(state, ep_dir / "state.pt")

        rec = CheckpointRecord(epoch=epoch, metric=metrics[self.monitor], path=ep_dir)
        self.records.append(rec)
        self._prune()
        self._update_symlinks()
        return ep_dir

    def load(
        self,
        epoch_dir: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        callbacks: dict[str, Any] | None = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Restore into ``model`` (and optionally opt/sched/callbacks).
        Returns the full state dict loaded from ``state.pt``."""
        ep_dir = Path(epoch_dir)
        state_path = ep_dir / "state.pt"
        weights_path = ep_dir / "model.safetensors"
        if not state_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"incomplete checkpoint at {ep_dir}")

        weights = load_file(str(weights_path), device=device)
        model.load_state_dict(weights)

        state = torch.load(state_path, map_location=device, weights_only=False)
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if callbacks and state.get("callbacks"):
            for k, cb in callbacks.items():
                if k in state["callbacks"]:
                    cb.load_state_dict(state["callbacks"][k])
        return state

    def resolve(self, which: str) -> Path:
        """Resolve 'best' / 'latest' / an explicit 'epoch_NNNN' name."""
        p = (self.root / which).resolve()
        if not p.exists():
            raise FileNotFoundError(f"no checkpoint at {p}")
        return p

    # -- internal -------------------------------------------------------
    def _is_better(self, a: float, b: float) -> bool:
        return a < b if self.mode == "min" else a > b

    def _prune(self) -> None:
        # Sort records best-first, keep top K, delete rest.
        self.records.sort(key=lambda r: r.metric, reverse=(self.mode == "max"))
        keep = self.records[: self.k]
        drop = self.records[self.k :]
        for r in drop:
            shutil.rmtree(r.path, ignore_errors=True)
        self.records = keep

    def _update_symlinks(self) -> None:
        if not self.records:
            return
        best = self.records[0].path
        latest = max(self.records, key=lambda r: r.epoch).path
        _symlink_force(self.root / "best", best)
        _symlink_force(self.root / "latest", latest)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _symlink_force(link: Path, target: Path) -> None:
    """Make ``link`` point at ``target`` (directory). Replace if exists.
    Falls back to a small text pointer file on filesystems where
    symlinking dirs is disallowed (e.g. Google Drive)."""
    try:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        os.symlink(target.name, link)
    except (OSError, NotImplementedError):
        # Fallback: write a small text pointer the resolver can read.
        if link.exists():
            link.unlink()
        link.with_suffix(".txt").write_text(target.name)


__all__ = ["TopKCheckpointManager", "CheckpointRecord"]
