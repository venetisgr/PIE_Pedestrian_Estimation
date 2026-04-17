"""End-to-end CLI + overfit tests that don't need the real PIE dataset.

We monkey-patch PIE.generate_data_trajectory_sequence so the CLI builds
a real Dataset + Trainer pipeline from a synthetic data dict and runs
the full training loop for a couple of epochs. This is the
Phase 3 ``overfit 32-batch`` validation lane without requiring a GPU
or any of the 74 GB of PIE videos.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

from pie_pytorch.cli import train as train_cli


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def test_load_config_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PIE_PATH", "/fake/pie")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
task: speed
data:
  feature_cache_dir: ${PIE_PATH}/features
  pie_path_env: PIE_PATH
  train_split: val
  val_split: val
  data_opts: {}
  model_opts: {observe_length: 15, predict_length: 45, enc_input_type: [obd_speed], dec_input_type: [], prediction_type: [obd_speed], normalize_bbox: true}
model: {observe_length: 14, predict_length: 45, enc_feature_size: 1, dec_feature_size: 0, prediction_size: 1}
training: {epochs: 1, lr: 0.001, batch_size: 4}
"""
    )
    cfg = train_cli.load_config(str(cfg_path))
    assert cfg["data"]["feature_cache_dir"] == "/fake/pie/features"


def test_load_config_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PIE_PATH", "/fake/pie")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
task: speed
data: {pie_path_env: PIE_PATH, train_split: val, val_split: val, data_opts: {}, model_opts: {observe_length: 15, predict_length: 45, enc_input_type: [obd_speed], dec_input_type: [], prediction_type: [obd_speed], normalize_bbox: true}}
model: {observe_length: 14, predict_length: 45, enc_feature_size: 1, dec_feature_size: 0, prediction_size: 1}
training: {epochs: 10, lr: 0.001, batch_size: 4}
"""
    )
    cfg = train_cli.load_config(str(cfg_path), overrides=["training.epochs=2"])
    assert cfg["training"]["epochs"] == 2


def test_load_config_unknown_task(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("task: bogus\n")
    with pytest.raises(ValueError, match="config.task"):
        train_cli.load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# Overfit: speed model, 32 synthetic windows, 15 epochs -> MSE drops
# ---------------------------------------------------------------------------
def _fake_speed_data():
    """Return a fake PIE.generate_data_trajectory_sequence output that
    yields at least one window per track after sliding-window extraction."""
    n_tracks, L = 8, 70
    data = {
        "image": [
            [f"/fake/set05/vid_{t}/{f:05d}.png" for f in range(L)]
            for t in range(n_tracks)
        ],
        "bbox": [
            [[100.0 + f, 50.0 + f, 200.0 + f, 150.0 + f] for f in range(L)]
            for _ in range(n_tracks)
        ],
        "pid": [[[f"ped_{t}"]] * L for t in range(n_tracks)],
        "intention_prob": [[[0.5]] * L for _ in range(n_tracks)],
        "obd_speed": [
            [[5.0 + 0.05 * f + 0.1 * t] for f in range(L)] for t in range(n_tracks)
        ],
    }
    return data


@pytest.fixture
def speed_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("PIE_PATH", str(tmp_path))
    ckpt_dir = tmp_path / "ckpt"
    cfg = {
        "task": "speed",
        "data": {
            "pie_path_env": "PIE_PATH",
            "train_split": "val",
            "val_split": "val",
            "subset": None,
            "data_opts": {"seq_type": "trajectory"},
            "model_opts": {
                "observe_length": 15,
                "predict_length": 45,
                "enc_input_type": ["obd_speed"],
                "dec_input_type": [],
                "prediction_type": ["obd_speed"],
                "normalize_bbox": True,
                "track_overlap": 0.5,
            },
        },
        "model": {
            "observe_length": 14,
            "predict_length": 45,
            "enc_feature_size": 1,
            "dec_feature_size": 0,
            "prediction_size": 1,
            "hidden_size": 32,         # tiny for fast test
            "embed_size": 16,
            "embed_dropout": 0.0,
            "activation": "softsign",
        },
        "training": {
            "epochs": 15,
            "lr": 5e-3,
            "batch_size": 8,
            "num_workers": 0,
            "device": "cpu",
            "use_amp": False,
            "clip_grad_norm": 1.0,
            "monitor": "val_loss",
            "mode": "min",
            "plateau_patience": 100,
            "early_stop_patience": 100,
            "log_every_n_steps": 1000,
            "checkpoint_dir": str(ckpt_dir),
            "keep_top_k": 1,
        },
        "wandb": {"enabled": False},
    }
    return cfg


def test_cli_run_speed_overfit(speed_cfg):
    """Full run() with a mocked PIE -> trainer + checkpoints land."""
    torch.manual_seed(0)

    def _fake_gen(self, split, **opts):
        return _fake_speed_data()

    with mock.patch(
        "pie_pytorch.cli.train.PIE.generate_data_trajectory_sequence",
        new=_fake_gen,
    ):
        out = train_cli.run(speed_cfg)

    # Loss goes down across epochs.
    losses = [h["val/loss"] for h in out["history"]]
    assert losses[0] > losses[-1], f"loss didn't drop: {losses[0]} -> {losses[-1]}"
    # Best val loss is a reasonable improvement, not diverging.
    assert out["best"] < losses[0] * 0.9

    # Checkpoint landed.
    ckpt_root = Path(speed_cfg["training"]["checkpoint_dir"])
    assert any(ckpt_root.glob("epoch_*"))


# ---------------------------------------------------------------------------
# Overfit: trajectory with mocked PIE
# ---------------------------------------------------------------------------
@pytest.fixture
def traj_cfg(speed_cfg):
    """Same shape as speed_cfg but wired for the trajectory task."""
    cfg = {
        **speed_cfg,
        "task": "trajectory",
        "model": {
            "observe_length": 14,
            "predict_length": 45,
            "enc_feature_size": 4,
            "dec_feature_size": 2,
            "prediction_size": 4,
            "hidden_size": 32,
            "embed_size": 16,
            "embed_dropout": 0.0,
            "activation": "softsign",
        },
    }
    cfg["data"] = {
        **speed_cfg["data"],
        "model_opts": {
            "observe_length": 15,
            "predict_length": 45,
            "enc_input_type": ["bbox"],
            "dec_input_type": ["intention_prob", "obd_speed"],
            "prediction_type": ["bbox"],
            "normalize_bbox": True,
            "track_overlap": 0.5,
        },
    }
    return cfg


def test_cli_run_trajectory_overfit(traj_cfg):
    torch.manual_seed(0)

    def _fake_gen(self, split, **opts):
        return _fake_speed_data()

    with mock.patch(
        "pie_pytorch.cli.train.PIE.generate_data_trajectory_sequence",
        new=_fake_gen,
    ):
        out = train_cli.run(traj_cfg)

    losses = [h["val/loss"] for h in out["history"]]
    assert losses[0] > losses[-1]
    assert out["best"] < losses[0] * 0.9


def test_cli_main_no_wandb_flag(tmp_path, monkeypatch):
    """--no-wandb overrides config.wandb.enabled."""
    monkeypatch.setenv("PIE_PATH", str(tmp_path))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
task: speed
data:
  pie_path_env: PIE_PATH
  train_split: val
  val_split: val
  data_opts: {seq_type: trajectory}
  model_opts: {observe_length: 15, predict_length: 45, enc_input_type: [obd_speed], dec_input_type: [], prediction_type: [obd_speed], normalize_bbox: true}
model: {observe_length: 14, predict_length: 45, enc_feature_size: 1, dec_feature_size: 0, prediction_size: 1, hidden_size: 8, embed_size: 4, embed_dropout: 0, activation: softsign}
training: {epochs: 1, lr: 1.0e-3, batch_size: 4, num_workers: 0, device: cpu, use_amp: false, log_every_n_steps: 1000}
wandb: {enabled: true}
"""
    )

    def _fake_gen(self, split, **opts):
        return _fake_speed_data()

    # The only thing we're testing here is that --no-wandb flips the flag.
    with mock.patch(
        "pie_pytorch.cli.train.PIE.generate_data_trajectory_sequence",
        new=_fake_gen,
    ):
        rc = train_cli.main(
            ["--config", str(cfg_path), "--no-wandb", "--log-level", "ERROR"]
        )
    assert rc == 0
