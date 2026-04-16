"""TrajectoryAttnEncDec (Phase 2.4).

Port of legacy_tf/pie_predict.py::PIEPredict.pie_encdec with
enc_input_type=['bbox'] and dec_input_type=['intention_prob', 'obd_speed'].
Output: future bbox deltas over predict_length=45.
"""

from __future__ import annotations
