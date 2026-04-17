"""VGG16 context-feature extractor with on-disk caching.

The original Keras pipeline precomputes VGG16 conv-stack features for
the 2x-enlarged context crop around each pedestrian bbox, pickling one
file per frame per pedestrian. That yields millions of inodes which is
hostile to Google Drive / Colab.

This port keeps the compute pipeline identical in shape:
    image -> enlarged bbox (2x) -> squarify -> pad_resize 224 -> VGG16.features

...but stores features in per-video torch shard files:
    {cache_root}/{set_id}/{vid_id}.pt  -> dict[(frame_key, ped_id)] = tensor(512, 7, 7)

Keras vs torchvision preprocessing differs (see notepad.md D5):
    - Keras vgg16.preprocess_input: RGB -> BGR; subtract Caffe means
      [103.939, 116.779, 123.68]; no /255, no std.
    - torchvision: x/255 -> normalize with mean/std.

For a "faithful + modern" port we default to torchvision preprocessing
(``preprocess='torchvision'``). Pass ``preprocess='keras'`` to reproduce
the legacy numeric pipeline exactly (important for Phase 4 parity).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import VGG16_Weights, vgg16

from ..data.transforms import img_pad, jitter_bbox, load_img, squarify

Preprocess = Literal["torchvision", "keras"]

# Caffe-style means used by keras.applications.vgg16.preprocess_input
_CAFFE_MEAN_BGR = torch.tensor([103.939, 116.779, 123.68]).view(3, 1, 1)
# Standard torchvision ImageNet stats (applied after /255)
_TV_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_TV_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Crop + preprocess
# ---------------------------------------------------------------------------
def context_crop(img_path: str, bbox: list[float], enlarge_ratio: float = 2.0,
                 size: int = 224):
    """Produce the 224x224 padded context crop used by the intent model.

    Matches legacy pie_intent.py::load_images_and_process:
        bbox = jitter_bbox(img_path, [b], 'enlarge', 2)[0]
        bbox = squarify(bbox, 1, img_width)
        bbox = list(map(int, bbox[:4]))
        cropped = img.crop(bbox)
        padded = img_pad(cropped, mode='pad_resize', size=224)
    """
    img = load_img(img_path)
    img_width = img.size[0]
    big = jitter_bbox(img_path, [list(bbox)], "enlarge", enlarge_ratio)[0]
    big = squarify(big, 1, img_width)
    big = list(map(int, big[:4]))
    cropped = img.crop(big)
    return img_pad(cropped, mode="pad_resize", size=size)


def preprocess_image(pil_img, mode: Preprocess = "torchvision") -> torch.Tensor:
    """Convert a PIL RGB image to a VGG16-ready tensor on CPU."""
    # np.asarray(PIL) is read-only; copy so torch.from_numpy is safe to use.
    arr_np = np.array(pil_img, dtype=np.uint8, copy=True)
    arr = torch.from_numpy(arr_np).permute(2, 0, 1).float()
    if mode == "torchvision":
        x = arr / 255.0
        x = (x - _TV_MEAN) / _TV_STD
    elif mode == "keras":
        # RGB -> BGR, subtract Caffe means, no /255
        x = arr[[2, 1, 0], :, :] - _CAFFE_MEAN_BGR
    else:
        raise ValueError(f"unknown preprocess mode {mode!r}")
    return x


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
@dataclass
class ExtractorConfig:
    preprocess: Preprocess = "torchvision"
    device: str = "cpu"  # concrete torch-device string; callers must resolve "auto" first


class VGG16FeatureExtractor(nn.Module):
    """Wraps torchvision VGG16.features (conv stack, no classifier).

    Output shape for a 224x224 input is (B, 512, 7, 7). This matches
    Keras ``vgg16.VGG16(include_top=False, input_shape=(224,224,3)).output_shape``
    = ``(None, 7, 7, 512)`` up to the channel-first/last transpose.
    """

    def __init__(self, cfg: ExtractorConfig | None = None):
        super().__init__()
        self.cfg = cfg or ExtractorConfig()
        if self.cfg.device == "auto":
            raise ValueError(
                "ExtractorConfig.device must be a concrete torch device "
                "(cpu/cuda/mps); resolve 'auto' before constructing "
                "VGG16FeatureExtractor."
            )
        weights = VGG16_Weights.IMAGENET1K_V1
        self._net = vgg16(weights=weights).features  # conv stack only
        for p in self._net.parameters():
            p.requires_grad = False
        self._net.eval()
        # Move weights to the same device as the inputs will be on.
        self._net.to(self.cfg.device)

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return (512, 7, 7)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, 512, 7, 7)."""
        return self._net(x)

    @torch.no_grad()
    def extract_from_path(self, img_path: str, bbox: list[float]) -> torch.Tensor:
        pil = context_crop(img_path, bbox)
        x = preprocess_image(pil, self.cfg.preprocess).unsqueeze(0).to(self.cfg.device)
        return self.forward(x).squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Per-video sharded cache
# ---------------------------------------------------------------------------
class VideoShardCache:
    """On-disk per-video feature cache.

    Layout:
        {cache_root}/{set_id}/{vid_id}.pt  -> dict[(frame_key, ped_id)] = tensor

    - ``frame_key`` is ``Path(img_path).stem`` so keys are portable between
      Linux / Windows / Colab Drive mounts.
    - The shard is loaded once per video and kept in memory for quick
      subsequent lookups (LRU of 1 video per process is enough for typical
      access patterns where Datasets iterate frames of a track contiguously).
    """

    def __init__(self, cache_root: str | os.PathLike):
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._current_key: tuple[str, str] | None = None
        self._current: dict | None = None

    # -- path helpers ---------------------------------------------------
    @staticmethod
    def parse_path(img_path: str) -> tuple[str, str, str]:
        parts = Path(img_path).as_posix().split("/")
        # Last 3 pieces: set_id / vid_id / file.png
        set_id, vid_id, fname = parts[-3], parts[-2], parts[-1]
        return set_id, vid_id, Path(fname).stem

    def shard_path(self, set_id: str, vid_id: str) -> Path:
        return self.root / set_id / f"{vid_id}.pt"

    # -- core API --------------------------------------------------------
    def get(self, img_path: str, ped_id: str) -> torch.Tensor | None:
        set_id, vid_id, frame_key = self.parse_path(img_path)
        self._ensure_loaded(set_id, vid_id)
        return self._current.get((frame_key, ped_id))  # type: ignore[union-attr]

    def put(self, img_path: str, ped_id: str, feature: torch.Tensor) -> None:
        set_id, vid_id, frame_key = self.parse_path(img_path)
        self._ensure_loaded(set_id, vid_id)
        self._current[(frame_key, ped_id)] = feature  # type: ignore[union-attr]

    def flush(self) -> None:
        if self._current_key is None or self._current is None:
            return
        set_id, vid_id = self._current_key
        path = self.shard_path(set_id, vid_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(self._current, tmp)
        os.replace(tmp, path)  # atomic on POSIX

    # -- internal --------------------------------------------------------
    def _ensure_loaded(self, set_id: str, vid_id: str) -> None:
        key = (set_id, vid_id)
        if self._current_key == key:
            return
        if self._current_key is not None:
            self.flush()
        path = self.shard_path(set_id, vid_id)
        self._current = torch.load(path, weights_only=True) if path.exists() else {}
        self._current_key = key

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()


# ---------------------------------------------------------------------------
# High-level helper: extract (and cache) features for a list of (img, bbox, pid)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_and_cache(
    extractor: VGG16FeatureExtractor,
    cache: VideoShardCache,
    img_paths: list[str],
    bboxes: list[list[float]],
    ped_ids: list[str],
    regen: bool = False,
) -> torch.Tensor:
    """Return stacked features (T, 512, 7, 7) for a single track window."""
    feats = []
    for p, b, pid in zip(img_paths, bboxes, ped_ids):
        hit = None if regen else cache.get(p, pid)
        if hit is None:
            hit = extractor.extract_from_path(p, b)
            cache.put(p, pid, hit)
        feats.append(hit)
    cache.flush()
    return torch.stack(feats, dim=0)
