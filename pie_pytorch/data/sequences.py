"""Sliding-window sequence extraction helpers.

The PIE dataset gives us one list per pedestrian track. Both the intent
and trajectory pipelines then chop each track into overlapping
fixed-length windows. This module is the single source of truth for
that chopping logic so the three Dataset classes stay consistent.

Mirrors legacy_tf/pie_intent.py::PIEIntent.get_tracks and
legacy_tf/pie_predict.py::PIEPredict.get_tracks.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def compute_stride(seq_length: int, overlap: float) -> int:
    """Return the sliding-window stride for a given (length, overlap).

    Matches legacy: stride = seq_length if overlap == 0 else
    int((1 - overlap) * seq_length), floored to a minimum of 1.
    """
    assert seq_length > 0, "seq_length must be positive"
    assert 0.0 <= overlap < 1.0, f"overlap must be in [0, 1), got {overlap}"
    stride = seq_length if overlap == 0 else int((1 - overlap) * seq_length)
    return max(1, stride)


def windows(track: Sequence[T], seq_length: int, overlap: float) -> list[Sequence[T]]:
    """Return non-padded sliding windows over a single track."""
    stride = compute_stride(seq_length, overlap)
    return [
        track[i : i + seq_length]
        for i in range(0, len(track) - seq_length + 1, stride)
    ]


def windows_over_tracks(
    tracks: Sequence[Sequence[T]], seq_length: int, overlap: float
) -> list[Sequence[T]]:
    """Flatten: list-of-tracks -> list-of-windows (concatenated across tracks)."""
    out: list[Sequence[T]] = []
    for tr in tracks:
        out.extend(windows(tr, seq_length, overlap))
    return out
