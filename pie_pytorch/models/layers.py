"""Shared layers.

Phase 2.1 - 2.2:
    - ConvLSTM2DCell / ConvLSTM2D (Keras-equivalent defaults; hard_sigmoid gate,
      tanh state, glorot/orthogonal/zeros init).
    - TemporalAttention (mirrors pie_predict.py::attention_temporal).
    - ElementAttention (mirrors pie_predict.py::attention_element).

See notepad.md D9 for the ConvLSTM porting risk notes.
"""

from __future__ import annotations
