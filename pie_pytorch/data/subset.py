"""Subset / sampling selector for the PIE dataset.

Supports three mutually-exclusive selectors (Phase 1.4):
    - fraction: float in (0, 1]  -> random fraction of tracks
    - max_tracks: int            -> hard cap on number of tracks
    - set_ids: list[str]         -> keep only tracks from these set_XX dirs

Stub. Implementation lands in Phase 1.4; see plan.md.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubsetConfig:
    fraction: Optional[float] = None
    max_tracks: Optional[int] = None
    set_ids: list[str] = field(default_factory=list)
    seed: int = 0

    def is_noop(self) -> bool:
        return self.fraction is None and self.max_tracks is None and not self.set_ids
