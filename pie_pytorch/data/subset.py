"""Subset / sampling selector for the PIE dataset.

`pie_data.PIE.generate_data_trajectory_sequence` returns a dict of
parallel per-track lists::

    {
        'image':     [[img_paths_for_track_0], [...], ...],
        'bbox':      [[bboxes_for_track_0],    [...], ...],
        'ped_id':    [[[pid],  ...],           [...], ...],
        'intention_binary': ...,
        'intention_prob':   ...,
        'obd_speed':        ...,   # trajectory seq_type only
        ...
    }

`SubsetConfig.apply(data)` filters those lists in place of the original
track order to produce a smaller dataset. Three mutually-exclusive
selectors are supported so Colab users can fit on free-tier disk:

    - ``set_ids``   : whitelist specific set_XX directories
    - ``fraction``  : keep a random fraction of the remaining tracks
    - ``max_tracks``: hard cap on remaining tracks

Evaluation order: set_ids -> fraction -> max_tracks, so ``set_ids=['set03']``
followed by ``fraction=0.1`` keeps 10% of set_03.

Seed ``seed`` to make the ``fraction`` sample reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import numpy as np


@dataclass
class SubsetConfig:
    fraction: Optional[float] = None
    max_tracks: Optional[int] = None
    set_ids: list[str] = field(default_factory=list)
    seed: int = 0

    def is_noop(self) -> bool:
        return self.fraction is None and self.max_tracks is None and not self.set_ids

    # ------------------------------------------------------------------
    def apply(self, data: Mapping[str, list]) -> dict[str, list]:
        """Return a new dict with parallel per-track lists filtered."""
        if self.fraction is not None:
            assert 0.0 < self.fraction <= 1.0, (
                f"fraction must be in (0, 1], got {self.fraction}"
            )
        if self.max_tracks is not None:
            assert self.max_tracks > 0, (
                f"max_tracks must be positive, got {self.max_tracks}"
            )

        keys = list(data.keys())
        assert "image" in keys, "data must contain per-track 'image' paths"
        n_tracks = len(data["image"])

        # All parallel lists must agree on track count.
        for k in keys:
            assert len(data[k]) == n_tracks, (
                f"parallel-list length mismatch: '{k}' has {len(data[k])}, "
                f"'image' has {n_tracks}"
            )

        keep = np.ones(n_tracks, dtype=bool)

        # 1) set_ids whitelist -------------------------------------------------
        if self.set_ids:
            allowed = set(self.set_ids)
            for i, track_imgs in enumerate(data["image"]):
                # first image path of the track is representative of the set_id
                if not track_imgs:
                    keep[i] = False
                    continue
                first = track_imgs[0]
                # PIE path layout: .../set01/video_0001/00123.png
                # Some callers may feed absolute or relative paths; split on '/'.
                parts = str(first).replace("\\", "/").split("/")
                sid = next((p for p in parts if p.startswith("set")), None)
                if sid is None or sid not in allowed:
                    keep[i] = False

        surviving = np.flatnonzero(keep)

        # 2) fraction (random) -------------------------------------------------
        if self.fraction is not None and len(surviving) > 0:
            rng = np.random.default_rng(self.seed)
            target = max(1, int(round(len(surviving) * self.fraction)))
            surviving = np.sort(rng.choice(surviving, size=target, replace=False))

        # 3) max_tracks (hard cap) --------------------------------------------
        if self.max_tracks is not None:
            surviving = surviving[: self.max_tracks]

        out: dict[str, list] = {k: [data[k][i] for i in surviving] for k in keys}
        return out


def _iter_set_ids(data: Mapping[str, list]) -> Iterable[str]:
    """Yield the set_id of each track (for introspection / debugging)."""
    for track_imgs in data.get("image", []):
        if not track_imgs:
            yield ""
            continue
        parts = str(track_imgs[0]).replace("\\", "/").split("/")
        yield next((p for p in parts if p.startswith("set")), "")


def summary(data: Mapping[str, list]) -> dict[str, Any]:
    """Compact stats for logs / notebooks: track count + set_id histogram."""
    sids = list(_iter_set_ids(data))
    hist: dict[str, int] = {}
    for s in sids:
        hist[s] = hist.get(s, 0) + 1
    return {"n_tracks": len(sids), "set_id_hist": hist}
