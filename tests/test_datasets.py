"""Unit tests for IntentRawDataset, TrajectoryDataset, SpeedDataset.

Uses synthetic per-track lists so no real PIE dataset is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pie_pytorch.data.pie_dataset import (
    IntentConfig,
    IntentRawDataset,
    SpeedConfig,
    SpeedDataset,
    TrajectoryConfig,
    TrajectoryDataset,
)
from pie_pytorch.data.sequences import compute_stride, windows


# ---------------------------------------------------------------------------
# sequences helpers
# ---------------------------------------------------------------------------
def test_compute_stride():
    assert compute_stride(15, 0.5) == 7  # int(0.5 * 15) = 7
    assert compute_stride(15, 0.0) == 15  # no overlap -> stride == length
    assert compute_stride(10, 0.9) == 1  # floor 1
    with pytest.raises(AssertionError):
        compute_stride(15, 1.0)


def test_windows_count():
    track = list(range(30))
    w = windows(track, 15, 0.5)
    # stride = 7, starts 0,7,14 (0+15=15, 7+15=22, 14+15=29, 21+15=36>30 stop)
    assert [win[0] for win in w] == [0, 7, 14]
    assert all(len(win) == 15 for win in w)


# ---------------------------------------------------------------------------
# IntentRawDataset
# ---------------------------------------------------------------------------
def _intent_synth(n_tracks=3, track_len=30):
    data = {"image": [], "bbox": [], "intention_binary": [], "ped_id": []}
    for t in range(n_tracks):
        imgs = [f"/fake/set01/vid_{t}/{f:05d}.png" for f in range(track_len)]
        boxes = [[f, f, f + 10, f + 10] for f in range(track_len)]
        label = [[t % 2]] * track_len  # per-track constant
        pids = [[f"ped_{t}"]] * track_len
        data["image"].append(imgs)
        data["bbox"].append(boxes)
        data["intention_binary"].append(label)
        data["ped_id"].append(pids)
    return data


def test_intent_dataset_shapes_and_labels():
    data = _intent_synth(n_tracks=4, track_len=30)
    ds = IntentRawDataset(data, IntentConfig(observe_length=15, overlap=0.5))
    # With stride=7 and len=30: 3 windows per track -> 12 total
    assert len(ds) == 12

    sample = ds[0]
    assert sample["bbox_obs"].shape == (15, 4)
    assert sample["bbox_obs"].dtype == torch.float32
    assert sample["dec_input"].shape == (15, 4)
    assert sample["label"].shape == ()
    assert sample["label"].item() in (0.0, 1.0)
    assert len(sample["img_paths"]) == 15
    assert sample["ped_id"].startswith("ped_")


def test_intent_dataset_rejects_missing_keys():
    data = _intent_synth()
    del data["intention_binary"]
    with pytest.raises(AssertionError):
        IntentRawDataset(data)


# ---------------------------------------------------------------------------
# TrajectoryDataset
# ---------------------------------------------------------------------------
def _traj_synth(n_tracks=2, track_len=70):
    data = {
        "image": [],
        "bbox": [],
        "intention_prob": [],
        "obd_speed": [],
    }
    for t in range(n_tracks):
        imgs = [f"/fake/set01/vid_{t}/{f:05d}.png" for f in range(track_len)]
        boxes = [[100.0 + f, 50.0 + f, 200.0 + f, 150.0 + f] for f in range(track_len)]
        ip = [[0.7]] * track_len  # per-track intent prob
        speed = [[5.0 + 0.1 * f] for f in range(track_len)]
        data["image"].append(imgs)
        data["bbox"].append(boxes)
        data["intention_prob"].append(ip)
        data["obd_speed"].append(speed)
    return data


def test_trajectory_dataset_shapes_with_normalization():
    data = _traj_synth(n_tracks=2, track_len=70)
    cfg = TrajectoryConfig(observe_length=15, predict_length=45, overlap=0.5)
    ds = TrajectoryDataset(data, cfg)
    assert len(ds) > 0

    sample = ds[0]
    # normalize_bbox=True -> observation is 14, prediction is 45
    assert sample["enc_input"].shape == (14, 4)
    assert sample["dec_input"].shape == (45, 2)  # intention_prob + obd_speed
    assert sample["target"].shape == (45, 4)

    # First enc row should be (bbox[1] - bbox[0]) = (1, 1, 1, 1) in our synthetic
    assert torch.allclose(sample["enc_input"][0], torch.tensor([1.0, 1.0, 1.0, 1.0]))


def test_trajectory_dataset_without_normalization():
    data = _traj_synth(n_tracks=1, track_len=70)
    cfg = TrajectoryConfig(
        observe_length=15, predict_length=45, overlap=0.5, normalize_bbox=False
    )
    ds = TrajectoryDataset(data, cfg)
    sample = ds[0]
    assert sample["enc_input"].shape == (15, 4)
    assert sample["target"].shape == (45, 4)


# ---------------------------------------------------------------------------
# SpeedDataset
# ---------------------------------------------------------------------------
def test_speed_dataset_shapes():
    data = _traj_synth(n_tracks=2, track_len=70)
    cfg = SpeedConfig(observe_length=15, predict_length=45, overlap=0.5)
    ds = SpeedDataset(data, cfg)
    assert len(ds) > 0

    sample = ds[0]
    assert sample["enc_input"].shape == (14, 1)
    assert sample["dec_input"].shape == (45, 0)  # empty decoder input
    assert sample["target"].shape == (45, 1)
    assert sample["enc_input"].dtype == torch.float32


def test_speed_dataset_dataloader_collate():
    """Dataset must be compatible with torch DataLoader default collate."""
    data = _traj_synth(n_tracks=4, track_len=70)
    ds = SpeedDataset(data)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["enc_input"].shape[0] == 2  # batch dim
    assert batch["enc_input"].shape[1:] == (14, 1)
