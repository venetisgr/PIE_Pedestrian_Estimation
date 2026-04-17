"""Dataset that materializes VGG16 context features for the intent model.

Wraps ``IntentRawDataset`` (bbox + image paths) and a
``VGG16FeatureExtractor`` + ``VideoShardCache`` so each ``__getitem__``
returns a per-window feature tensor of shape ``(T, 512, 7, 7)`` ready
to feed ``IntentConvLSTMEncDec``.

First-epoch behaviour: cache misses trigger on-the-fly extraction; all
features are persisted to a per-video ``.pt`` shard so subsequent
epochs (and re-runs) hit the cache.

We intentionally skip prefetching / DataLoader workers here because
the cache object maintains a single-video LRU; concurrent readers
would thrash the shard file. For fast training, precompute the full
cache once (see ``pie_pytorch.features.vgg16_features.extract_and_cache``)
and then use this dataset with ``num_workers=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from ..data.pie_dataset import IntentConfig, IntentRawDataset
from .vgg16_features import VGG16FeatureExtractor, VideoShardCache


@dataclass
class IntentFeatureDatasetConfig:
    observe_length: int = 15
    overlap: float = 0.5
    cache_root: str = ""
    use_amp_extract: bool = False


class IntentFeatureDataset(Dataset):
    """``__getitem__`` returns:

        {
            "enc_input": (T, 512, 7, 7) float32   # VGG context features
            "dec_input": (T, 4)         float32   # bbox sequence
            "label":     ()              float32   # crossing probability
            "img_paths": tuple[str, ...]
            "ped_id":    str
        }
    """

    def __init__(
        self,
        raw_data: dict,
        extractor: VGG16FeatureExtractor,
        cache: VideoShardCache,
        cfg: IntentFeatureDatasetConfig | None = None,
    ):
        self.cfg = cfg or IntentFeatureDatasetConfig()
        self._raw = IntentRawDataset(
            raw_data, IntentConfig(observe_length=self.cfg.observe_length, overlap=self.cfg.overlap)
        )
        self.extractor = extractor
        self.cache = cache

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int) -> dict:
        sample = self._raw[idx]
        img_paths = sample["img_paths"]
        ped_id = sample["ped_id"]
        bboxes = sample["bbox_obs"]  # (T, 4) torch

        feats = []
        for t, (p, b) in enumerate(zip(img_paths, bboxes)):
            hit = self.cache.get(p, ped_id)
            if hit is None:
                hit = self.extractor.extract_from_path(p, b.tolist())
                self.cache.put(p, ped_id, hit)
            feats.append(hit)
        # Flush at the end of each window so any new entries persist.
        self.cache.flush()

        enc_input = torch.stack(feats, dim=0)  # (T, 512, 7, 7)
        return {
            "enc_input": enc_input,
            "dec_input": sample["dec_input"],
            "label": sample["label"],
            "img_paths": img_paths,
            "ped_id": ped_id,
        }
