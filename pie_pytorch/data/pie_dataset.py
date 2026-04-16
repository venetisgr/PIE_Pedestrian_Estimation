"""torch.utils.data.Dataset classes over PIE tracks.

All three classes take the dict returned by
``PIE.generate_data_trajectory_sequence`` (post-SubsetConfig filtering)
and produce fixed-length windows ready for the model.

Phase 1.3 delivers the *raw* datasets (bboxes + image paths + labels).
VGG16 context feature extraction lands in Phase 1.5 via
``IntentContextDataset``, which wraps ``IntentRawDataset`` + a feature
loader. This separation keeps the pure-data layer unit-testable without
torch/vgg.

Shapes returned by __getitem__:
    IntentRawDataset   -> dict(
        bbox_obs:    (T_obs, 4)   float32,
        dec_input:   (T_obs, 4)   float32  # = bbox_obs (decoder_input_type=['bbox'])
        label:       ()           float32  # binary intention
        img_paths:   tuple[str,...]        # length T_obs
        ped_id:      str
    )
    TrajectoryDataset  -> dict(
        enc_input:   (T_obs - 1, F_enc)  float32
        dec_input:   (T_pred, F_dec)     float32
        target:      (T_pred, 4)         float32
    )
    SpeedDataset       -> dict(
        enc_input:   (T_obs - 1, 1)      float32
        dec_input:   (0,)                 (empty; decoder has no exogenous input)
        target:      (T_pred, 1)         float32
    )

See notepad.md D3 for the normalize_bbox "drop first step" quirk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .sequences import windows_over_tracks


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------
@dataclass
class IntentConfig:
    observe_length: int = 15
    overlap: float = 0.5


class IntentRawDataset(Dataset):
    """Raw (no VGG feature) version of the intent dataset.

    Expects a dict with keys: 'image', 'bbox', 'intention_binary', 'ped_id'
    (all per-track lists as produced by pie_data.PIE).
    """

    def __init__(self, data: Mapping[str, list], cfg: IntentConfig | None = None):
        self.cfg = cfg or IntentConfig()
        required = ("image", "bbox", "intention_binary", "ped_id")
        for k in required:
            assert k in data, f"data missing required key {k!r}"

        L, O = self.cfg.observe_length, self.cfg.overlap
        self.img_windows = windows_over_tracks(data["image"], L, O)
        self.bbox_windows = windows_over_tracks(data["bbox"], L, O)
        self.label_windows = windows_over_tracks(data["intention_binary"], L, O)
        self.pid_windows = windows_over_tracks(data["ped_id"], L, O)

        n = len(self.img_windows)
        assert len(self.bbox_windows) == n and len(self.label_windows) == n, (
            "window-count mismatch across parallel lists"
        )

    def __len__(self) -> int:
        return len(self.img_windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bbox_obs = np.asarray(self.bbox_windows[idx], dtype=np.float32)  # (T, 4)
        # intention_binary was broadcast frame-wise but is constant per track;
        # the legacy code takes label[0] (first frame) and the binary label.
        label_frames = self.label_windows[idx]
        label = float(label_frames[0][0])  # first-frame, single-value -> scalar

        return {
            "bbox_obs": torch.from_numpy(bbox_obs),
            "dec_input": torch.from_numpy(bbox_obs.copy()),  # legacy: dec_input = bbox
            "label": torch.tensor(label, dtype=torch.float32),
            "img_paths": tuple(self.img_windows[idx]),
            "ped_id": self.pid_windows[idx][0][0],
        }


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryConfig:
    observe_length: int = 15
    predict_length: int = 45
    overlap: float = 0.5
    normalize_bbox: bool = True
    enc_input_type: tuple[str, ...] = ("bbox",)
    dec_input_type: tuple[str, ...] = ("intention_prob", "obd_speed")


class TrajectoryDataset(Dataset):
    """Trajectory encoder-decoder dataset.

    Legacy behavior (see notepad.md D3):
        1. Window tracks at length ``observe_length + predict_length``.
        2. THEN normalize: for bbox windows, replace ``w`` with
           ``w[1:] - w[0]``; for every other field, drop ``w[0]``.
        3. Split the (L-1)-length window into first ``observe_length - 1``
           frames (encoder) and remaining ``predict_length`` frames
           (decoder target).

    This order matters: windowing before normalizing preserves the
    stride math from the paper/ref impl.
    """

    def __init__(self, data: Mapping[str, list], cfg: TrajectoryConfig | None = None):
        self.cfg = cfg or TrajectoryConfig()
        needed = set(self.cfg.enc_input_type) | set(self.cfg.dec_input_type) | {"bbox"}
        for k in needed:
            assert k in data, f"data missing required key {k!r}"
        assert "bbox" in self.cfg.enc_input_type, (
            "current implementation assumes bbox is an encoder feature"
        )

        L = self.cfg.observe_length + self.cfg.predict_length
        O = self.cfg.overlap

        # Window first
        raw_windows = {k: windows_over_tracks(data[k], L, O) for k in needed}
        n = len(raw_windows["bbox"])
        for k, ws in raw_windows.items():
            assert len(ws) == n, f"window-count mismatch for {k}"

        # Normalize second
        self._windows: dict[str, list] = {k: [] for k in needed}
        for i in range(n):
            for k in needed:
                w = raw_windows[k][i]
                if self.cfg.normalize_bbox:
                    if k == "bbox":
                        arr = np.asarray(w, dtype=np.float32)
                        w = (arr[1:] - arr[0]).tolist()
                    else:
                        w = list(w)[1:]
                self._windows[k].append(w)

    def __len__(self) -> int:
        return len(self._windows["bbox"])

    def _stack(self, idx: int, types: Sequence[str]) -> np.ndarray:
        parts = [np.asarray(self._windows[t][idx], dtype=np.float32) for t in types]
        # All parts are (T, F_k); concat along feature axis
        if len(parts) == 1:
            out = parts[0]
        else:
            out = np.concatenate(parts, axis=-1)
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # After normalize_bbox, each window has length (obs + pred - 1)
        # and the boundary sits at (obs - 1). Without normalization it's
        # the full (obs + pred) with boundary at obs.
        obs_len = self.cfg.observe_length - (1 if self.cfg.normalize_bbox else 0)

        full = self._stack(idx, self.cfg.enc_input_type)  # (L_eff, F_enc)
        enc_input = full[:obs_len]

        if self.cfg.dec_input_type:
            full_dec = self._stack(idx, self.cfg.dec_input_type)  # (L_eff, F_dec)
            dec_input = full_dec[obs_len:]
        else:
            dec_input = np.zeros((self.cfg.predict_length, 0), dtype=np.float32)

        full_bbox = np.asarray(self._windows["bbox"][idx], dtype=np.float32)
        target = full_bbox[obs_len:]

        return {
            "enc_input": torch.from_numpy(enc_input),
            "dec_input": torch.from_numpy(dec_input),
            "target": torch.from_numpy(target),
        }


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
@dataclass
class SpeedConfig:
    observe_length: int = 15
    predict_length: int = 45
    overlap: float = 0.5
    normalize_bbox: bool = True  # kept for API symmetry; first-step-drop matters


class SpeedDataset(Dataset):
    """obd_speed in -> obd_speed out. Dec has no exogenous input."""

    def __init__(self, data: Mapping[str, list], cfg: SpeedConfig | None = None):
        self.cfg = cfg or SpeedConfig()
        assert "obd_speed" in data, "data must include 'obd_speed' per-track lists"

        L = self.cfg.observe_length + self.cfg.predict_length
        raw = windows_over_tracks(data["obd_speed"], L, self.cfg.overlap)
        # Legacy: window first, then drop first step when normalize_bbox
        if self.cfg.normalize_bbox:
            self._windows = [list(w)[1:] for w in raw]
        else:
            self._windows = list(raw)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs_len = self.cfg.observe_length - (1 if self.cfg.normalize_bbox else 0)
        speed = np.asarray(self._windows[idx], dtype=np.float32)  # (L, 1) or (L,)
        if speed.ndim == 1:
            speed = speed[:, None]

        return {
            "enc_input": torch.from_numpy(speed[:obs_len]),
            "dec_input": torch.zeros((self.cfg.predict_length, 0), dtype=torch.float32),
            "target": torch.from_numpy(speed[obs_len:]),
        }
