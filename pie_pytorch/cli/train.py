"""Unified training CLI for intent / trajectory / speed.

Usage:
    python -m pie_pytorch.cli.train --config pie_pytorch/configs/trajectory_colab.yaml
    python -m pie_pytorch.cli.train --config pie_pytorch/configs/intent_colab.yaml --no-wandb
    python -m pie_pytorch.cli.train --config X.yaml --override training.epochs=2

The config schema is documented in the example YAMLs under
``pie_pytorch/configs/``. Unknown top-level keys raise.

Dotted ``--override key=value`` lets you tweak any field from the
command line without editing the YAML (handy for Colab smoke-runs).
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..data.pie_data import PIE
from ..data.pie_dataset import (
    IntentConfig,
    IntentRawDataset,
    SpeedConfig,
    SpeedDataset,
    TrajectoryConfig,
    TrajectoryDataset,
)
from ..data.subset import SubsetConfig
from ..features.vgg16_features import (
    ExtractorConfig,
    VGG16FeatureExtractor,
    VideoShardCache,
)
from ..features.intent_feature_dataset import (
    IntentFeatureDataset,
    IntentFeatureDatasetConfig,
)
from ..models.intent import IntentConvLSTMEncDec, IntentModelConfig
from ..models.trajectory import AttnEncDec, AttnEncDecConfig
from ..training.metrics import accuracy, center_mse, f1, mse
from ..training.trainer import Trainer, TrainerConfig


logger = logging.getLogger("pie_pytorch.cli.train")


SUPPORTED_TASKS = ("intent", "trajectory", "speed")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} references in strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(x) for x in value]
    return value


def _apply_override(cfg: dict, assignment: str) -> dict:
    """Apply a ``dotted.key=value`` override to cfg, in place."""
    if "=" not in assignment:
        raise ValueError(f"--override expects key=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    # PyYAML's float regex rejects shorthand scientific notation like "5e-3"
    # (it wants "5.0e-3"). Coerce stringy numerics so CLI overrides are forgiving.
    if isinstance(value, str) and value == raw:
        for converter in (int, float):
            try:
                value = converter(raw)
                break
            except ValueError:
                continue
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping")

    # Expand $PIE_PATH etc. after loading so users don't need to
    # interpolate manually.
    cfg = _expand_env(cfg)

    for ov in overrides or []:
        _apply_override(cfg, ov)

    if cfg.get("task") not in SUPPORTED_TASKS:
        raise ValueError(
            f"config.task must be one of {SUPPORTED_TASKS}, got {cfg.get('task')!r}"
        )
    return cfg


# ---------------------------------------------------------------------------
# Data + model construction per task
# ---------------------------------------------------------------------------
def _apply_subset(raw: dict, subset_cfg: dict | None) -> dict:
    if not subset_cfg:
        return raw
    sc = SubsetConfig(
        fraction=subset_cfg.get("fraction"),
        max_tracks=subset_cfg.get("max_tracks"),
        set_ids=subset_cfg.get("set_ids", []),
        seed=subset_cfg.get("seed", 0),
    )
    return sc.apply(raw)


def _build_intent(cfg: dict) -> tuple[DataLoader, DataLoader, torch.nn.Module, Any, Any]:
    pie_path = os.environ[cfg["data"]["pie_path_env"]]
    imdb = PIE(data_path=pie_path)

    data_opts = cfg["data"]["data_opts"]
    train_raw = imdb.generate_data_trajectory_sequence(cfg["data"]["train_split"], **data_opts)
    val_raw = imdb.generate_data_trajectory_sequence(cfg["data"]["val_split"], **data_opts)

    if cfg["data"].get("balance_train"):
        train_raw = imdb.balance_samples_count(train_raw, label_type="intention_binary")

    train_raw = _apply_subset(train_raw, cfg["data"].get("subset"))
    val_raw = _apply_subset(val_raw, cfg["data"].get("subset"))

    feature_cache = VideoShardCache(cfg["data"]["feature_cache_dir"])
    extractor = VGG16FeatureExtractor(ExtractorConfig(device=cfg["training"].get("device", "auto")))

    feat_cfg = IntentFeatureDatasetConfig(
        observe_length=cfg["model"]["observe_length"],
        overlap=data_opts.get("seq_overlap_rate", 0.5),
        cache_root=cfg["data"]["feature_cache_dir"],
    )
    train_ds = IntentFeatureDataset(train_raw, extractor, feature_cache, feat_cfg)
    val_ds = IntentFeatureDataset(val_raw, extractor, feature_cache, feat_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
    )

    model = IntentConvLSTMEncDec(IntentModelConfig(**cfg["model"]))

    def forward_fn(m, batch):
        pred = m(batch["enc_input"], batch["dec_input"]).squeeze(-1)  # (B,)
        return pred, batch["label"]

    def loss_fn(pred, tgt, batch):
        return torch.nn.functional.binary_cross_entropy(pred, tgt)

    metrics = {"acc": accuracy, "f1": f1}
    return train_loader, val_loader, model, forward_fn, loss_fn, metrics


def _build_traj_or_speed(cfg: dict) -> tuple[DataLoader, DataLoader, torch.nn.Module, Any, Any, dict]:
    pie_path = os.environ[cfg["data"]["pie_path_env"]]
    imdb = PIE(data_path=pie_path)

    data_opts = cfg["data"]["data_opts"]
    model_opts = cfg["data"]["model_opts"]
    train_raw = imdb.generate_data_trajectory_sequence(cfg["data"]["train_split"], **data_opts)
    val_raw = imdb.generate_data_trajectory_sequence(cfg["data"]["val_split"], **data_opts)

    train_raw = _apply_subset(train_raw, cfg["data"].get("subset"))
    val_raw = _apply_subset(val_raw, cfg["data"].get("subset"))

    task = cfg["task"]
    if task == "trajectory":
        ds_cls = TrajectoryDataset
        ds_cfg = TrajectoryConfig(
            observe_length=model_opts["observe_length"],
            predict_length=model_opts["predict_length"],
            overlap=model_opts.get("track_overlap", 0.5),
            normalize_bbox=model_opts.get("normalize_bbox", True),
            enc_input_type=tuple(model_opts["enc_input_type"]),
            dec_input_type=tuple(model_opts["dec_input_type"]),
        )
    else:
        ds_cls = SpeedDataset
        ds_cfg = SpeedConfig(
            observe_length=model_opts["observe_length"],
            predict_length=model_opts["predict_length"],
            overlap=model_opts.get("track_overlap", 0.5),
            normalize_bbox=model_opts.get("normalize_bbox", True),
        )

    train_ds = ds_cls(train_raw, ds_cfg)
    val_ds = ds_cls(val_raw, ds_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
    )

    model = AttnEncDec(AttnEncDecConfig(**cfg["model"]))

    def forward_fn(m, batch):
        return m(batch["enc_input"], batch["dec_input"]), batch["target"]

    def loss_fn(pred, tgt, batch):
        return torch.nn.functional.mse_loss(pred, tgt)

    if task == "trajectory":
        metrics = {"mse": mse, "c_mse": center_mse}
    else:
        metrics = {"mse": mse}

    return train_loader, val_loader, model, forward_fn, loss_fn, metrics


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    task = cfg["task"]
    if task == "intent":
        train_loader, val_loader, model, forward_fn, loss_fn, metrics = _build_intent(cfg)
    else:
        train_loader, val_loader, model, forward_fn, loss_fn, metrics = _build_traj_or_speed(cfg)

    t_cfg_src = cfg["training"]
    w_cfg_src = cfg.get("wandb", {})
    trainer_cfg = TrainerConfig(
        epochs=t_cfg_src["epochs"],
        lr=t_cfg_src["lr"],
        weight_decay=t_cfg_src.get("weight_decay", 0.0),
        rmsprop_alpha=t_cfg_src.get("rmsprop_alpha", 0.9),
        rmsprop_eps=t_cfg_src.get("rmsprop_eps", 1e-7),
        clip_grad_norm=t_cfg_src.get("clip_grad_norm", 1.0),
        use_amp=t_cfg_src.get("use_amp"),
        device=t_cfg_src.get("device", "auto"),
        monitor=t_cfg_src.get("monitor", "val_loss"),
        mode=t_cfg_src.get("mode", "min"),
        plateau_factor=t_cfg_src.get("plateau_factor", 0.5),
        plateau_patience=t_cfg_src.get("plateau_patience", 5),
        plateau_min_lr=t_cfg_src.get("plateau_min_lr", 1e-7),
        early_stop_patience=t_cfg_src.get("early_stop_patience", 10),
        early_stop_min_delta=t_cfg_src.get("early_stop_min_delta", 1e-4),
        checkpoint_dir=t_cfg_src.get("checkpoint_dir"),
        keep_top_k=t_cfg_src.get("keep_top_k", 3),
        log_every_n_steps=t_cfg_src.get("log_every_n_steps", 25),
        wandb=w_cfg_src.get("enabled", False),
        wandb_project=w_cfg_src.get("project", "pie_pytorch"),
        wandb_run_name=w_cfg_src.get("run_name"),
        wandb_config=cfg,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        forward_fn=forward_fn,
        cfg=trainer_cfg,
    )
    return trainer.fit(train_loader, val_loader)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pie-train",
        description="Train an intent / trajectory / speed model from a YAML config.",
    )
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotted.key=value assignment (repeatable). Applied after YAML load.",
    )
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B even if config.wandb.enabled=true.")
    p.add_argument("--log-level", default="INFO", help="Python logging level.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_config(args.config, overrides=args.override)
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
