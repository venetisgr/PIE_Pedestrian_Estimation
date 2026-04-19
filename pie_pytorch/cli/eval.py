"""Evaluate a checkpoint on a split.

Usage:
    python -m pie_pytorch.cli.eval \
        --config pie_pytorch/configs/trajectory_colab.yaml \
        --checkpoint $PIE_PATH/checkpoints/trajectory_colab/best \
        --split test
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from ..training.checkpoint import TopKCheckpointManager
from ..training.metrics import MetricsAccumulator
from .train import _build_intent, _build_traj_or_speed, load_config


logger = logging.getLogger("pie_pytorch.cli.eval")


def _build(cfg: dict):
    if cfg["task"] == "intent":
        return _build_intent(cfg)
    return _build_traj_or_speed(cfg)


def run(
    cfg: dict,
    checkpoint: str | None = None,
    keras_h5: str | None = None,
    split: str | None = None,
) -> dict[str, float]:
    """Evaluate a model on ``cfg``'s val loader.

    Provide exactly one of:
      - ``checkpoint``: 'best' / 'latest' / 'epoch_XXXX' / abs path.
      - ``keras_h5``: path to a paper .h5 checkpoint (convert-on-the-fly).
    """
    if (checkpoint is None) == (keras_h5 is None):
        raise ValueError("pass exactly one of --checkpoint or --keras-h5")
    if split is not None:
        cfg["data"]["val_split"] = split
    train_loader, val_loader, model, forward_fn, loss_fn, metrics = _build(cfg)

    import os
    from pathlib import Path

    device = cfg["training"].get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if keras_h5 is not None:
        from ..io.keras_to_torch import load_attn_encdec, load_intent

        if cfg["task"] == "intent":
            paper_model, _ = load_intent(keras_h5)
        else:
            paper_model, _ = load_attn_encdec(keras_h5, task=cfg["task"])
        model.load_state_dict(paper_model.state_dict())
        logger.info("loaded paper weights from %s", keras_h5)
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / checkpoint
        if ckpt_path.is_symlink():
            ckpt_path = ckpt_path.resolve()
        mgr = TopKCheckpointManager(Path(cfg["training"]["checkpoint_dir"]))
        state = mgr.load(ckpt_path, model=model, device=device)
        logger.info("loaded checkpoint from epoch %s with metrics %s",
                    state.get("epoch"), state.get("metrics"))

    model = model.to(device).eval()
    acc = MetricsAccumulator()
    running_loss, running_n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            preds, targets = forward_fn(model, batch)
            loss = loss_fn(preds, targets, batch)
            running_loss += loss.item() * targets.shape[0]
            running_n += targets.shape[0]
            acc.update(preds, targets)
    out = {"loss": running_loss / max(running_n, 1), **acc.compute(metrics)}
    return out


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="pie-eval")
    p.add_argument("--config", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint",
        help="'best' | 'latest' | 'epoch_NNNN' | absolute path to our safetensors ckpt dir",
    )
    src.add_argument(
        "--keras-h5",
        help="Path to a paper Keras .h5 checkpoint. Converted on the fly.",
    )
    p.add_argument("--split", default=None, help="Override data.val_split for this run.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotted.key=value assignment (repeatable). Applied after YAML load, "
             "before run(). Same semantics as pie-train --override.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_config(args.config, overrides=args.override)
    results = run(
        cfg,
        checkpoint=args.checkpoint,
        keras_h5=args.keras_h5,
        split=args.split,
    )
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
