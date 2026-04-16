"""Unit tests for SubsetConfig.apply."""

from __future__ import annotations

import pytest

from pie_pytorch.data.subset import SubsetConfig, summary


def _synthetic(n_per_set: dict[str, int]) -> dict[str, list]:
    """Build a fake `generate_data_trajectory_sequence` output.

    Produces ``n_per_set[set_id]`` tracks for each set_id. Each track has
    10 frames.
    """
    image, bbox, pid = [], [], []
    for sid, n in n_per_set.items():
        for t in range(n):
            # Encode track index into the video path so the test can
            # distinguish which tracks survived (guards against false
            # equality when only the frame number differs).
            imgs = [f"/fake/root/{sid}/video_{t:04d}/{f:05d}.png" for f in range(10)]
            boxes = [[f * 1.0, f * 1.0, f + 10.0, f + 10.0] for f in range(10)]
            pids = [[f"{sid}_ped_{t}"] for _ in range(10)]
            image.append(imgs)
            bbox.append(boxes)
            pid.append(pids)
    return {"image": image, "bbox": bbox, "ped_id": pid}


def test_noop():
    data = _synthetic({"set01": 3, "set02": 2})
    cfg = SubsetConfig()
    assert cfg.is_noop()
    out = cfg.apply(data)
    assert out["image"] == data["image"]  # identical order and contents


def test_set_ids_whitelist():
    data = _synthetic({"set01": 3, "set02": 2, "set03": 4})
    out = SubsetConfig(set_ids=["set01", "set03"]).apply(data)
    assert len(out["image"]) == 7
    # Verify by inspecting first path of each surviving track
    seen_sets = {img[0].split("/")[3] for img in out["image"]}
    assert seen_sets == {"set01", "set03"}


def test_fraction_reproducible():
    data = _synthetic({"set01": 100})
    a = SubsetConfig(fraction=0.1, seed=123).apply(data)
    b = SubsetConfig(fraction=0.1, seed=123).apply(data)
    assert a["image"] == b["image"]  # deterministic given seed
    assert len(a["image"]) == 10  # 10% of 100 == 10


def test_fraction_different_seeds_differ():
    data = _synthetic({"set01": 100})
    a = SubsetConfig(fraction=0.1, seed=1).apply(data)
    b = SubsetConfig(fraction=0.1, seed=2).apply(data)
    assert a["image"] != b["image"]


def test_max_tracks_hard_cap():
    data = _synthetic({"set01": 50})
    out = SubsetConfig(max_tracks=7).apply(data)
    assert len(out["image"]) == 7
    # First 7 original tracks (no shuffle since no fraction)
    assert out["image"][:7] == data["image"][:7]


def test_combined_set_then_fraction_then_cap():
    data = _synthetic({"set01": 100, "set02": 100, "set03": 100})
    out = SubsetConfig(
        set_ids=["set02"], fraction=0.2, max_tracks=5, seed=0
    ).apply(data)
    # 100 -> whitelist 100 -> fraction 20 -> cap 5
    assert len(out["image"]) == 5
    seen_sets = {img[0].split("/")[3] for img in out["image"]}
    assert seen_sets == {"set02"}


def test_invalid_fraction():
    data = _synthetic({"set01": 1})
    with pytest.raises(AssertionError):
        SubsetConfig(fraction=0.0).apply(data)
    with pytest.raises(AssertionError):
        SubsetConfig(fraction=1.5).apply(data)


def test_invalid_max_tracks():
    data = _synthetic({"set01": 1})
    with pytest.raises(AssertionError):
        SubsetConfig(max_tracks=0).apply(data)


def test_parallel_list_mismatch_detected():
    data = _synthetic({"set01": 3})
    data["bbox"].pop()  # induce a length mismatch
    with pytest.raises(AssertionError):
        SubsetConfig(max_tracks=2).apply(data)


def test_summary_helper():
    data = _synthetic({"set01": 3, "set02": 2})
    s = summary(data)
    assert s["n_tracks"] == 5
    assert s["set_id_hist"] == {"set01": 3, "set02": 2}
