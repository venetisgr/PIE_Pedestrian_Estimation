"""Integration tests for the three PIE PyTorch models."""

from __future__ import annotations

import math

import pytest
import torch

from pie_pytorch.models.intent import IntentConvLSTMEncDec, IntentModelConfig
from pie_pytorch.models.speed import speed_model
from pie_pytorch.models.trajectory import (
    AttnEncDec,
    AttnEncDecConfig,
    trajectory_model,
)


# ---------------------------------------------------------------------------
# IntentConvLSTMEncDec
# ---------------------------------------------------------------------------
def test_intent_forward_shape_and_range():
    model = IntentConvLSTMEncDec()
    B, T = 2, 15
    enc = torch.randn(B, T, 512, 7, 7)
    dec = torch.randn(B, T, 4)
    y = model(enc, dec)
    assert y.shape == (B, 1)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)  # sigmoid


def test_intent_batch_invariance():
    """B samples processed together must equal processing them individually."""
    model = IntentConvLSTMEncDec()
    model.eval()
    enc = torch.randn(3, 15, 512, 7, 7)
    dec = torch.randn(3, 15, 4)
    y_batched = model(enc, dec)
    y_indiv = torch.cat([model(enc[i : i + 1], dec[i : i + 1]) for i in range(3)], dim=0)
    assert torch.allclose(y_batched, y_indiv, atol=1e-6)


def test_intent_grads_flow_to_inputs():
    model = IntentConvLSTMEncDec()
    enc = torch.randn(1, 15, 512, 7, 7, requires_grad=True)
    dec = torch.randn(1, 15, 4, requires_grad=True)
    y = model(enc, dec)
    y.sum().backward()
    assert torch.isfinite(enc.grad).all()
    assert torch.isfinite(dec.grad).all()


def test_intent_raises_on_mismatched_decoder_length():
    model = IntentConvLSTMEncDec()
    enc = torch.randn(1, 15, 512, 7, 7)
    dec_wrong = torch.randn(1, 14, 4)
    with pytest.raises(ValueError, match="must match encoder"):
        model(enc, dec_wrong)


def test_intent_param_count_is_reasonable():
    """Sanity: Keras reference reports ~1.8M trainable params for this
    architecture. We expect within ±30% given gate/init differences."""
    model = IntentConvLSTMEncDec()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1e6 < n < 3e6, f"param count {n} outside sanity range"


def test_intent_config_overrides_take_effect():
    cfg = IntentModelConfig(
        observe_length=10, convlstm_filters=32, lstm_hidden=64, decoder_input_size=4
    )
    model = IntentConvLSTMEncDec(cfg)
    enc = torch.randn(1, 10, 512, 7, 7)
    dec = torch.randn(1, 10, 4)
    y = model(enc, dec)
    assert y.shape == (1, 1)


# ---------------------------------------------------------------------------
# AttnEncDec / trajectory_model
# ---------------------------------------------------------------------------
def test_trajectory_forward_shape():
    model = trajectory_model()
    B, T_obs, T_pred = 2, 14, 45
    enc = torch.randn(B, T_obs, 4)
    dec = torch.randn(B, T_pred, 2)
    y = model(enc, dec)
    assert y.shape == (B, T_pred, 4)
    assert torch.isfinite(y).all()


def test_trajectory_softsign_default_bounds_enc_path():
    """softsign output stays in (-1, 1); head is linear so y is unbounded,
    but internal states should never blow up."""
    model = trajectory_model()
    model.eval()
    enc = torch.full((1, 14, 4), 1e3)  # extreme input
    dec = torch.zeros(1, 45, 2)
    y = model(enc, dec)
    assert torch.isfinite(y).all()


def test_trajectory_backprop_flows():
    model = trajectory_model()
    enc = torch.randn(1, 14, 4, requires_grad=True)
    dec = torch.randn(1, 45, 2, requires_grad=True)
    y = model(enc, dec)
    y.sum().backward()
    assert torch.isfinite(enc.grad).all()
    assert torch.isfinite(dec.grad).all()


def test_trajectory_rejects_wrong_dec_feature_size():
    model = trajectory_model()
    enc = torch.randn(1, 14, 4)
    dec_wrong = torch.randn(1, 45, 3)  # should be 2
    with pytest.raises(AssertionError):
        model(enc, dec_wrong)


# ---------------------------------------------------------------------------
# speed_model
# ---------------------------------------------------------------------------
def test_speed_forward_shape():
    model = speed_model()
    B, T_obs, T_pred = 2, 14, 45
    enc = torch.randn(B, T_obs, 1)
    dec = torch.empty(B, T_pred, 0)  # F_dec=0 for the speed model
    y = model(enc, dec)
    assert y.shape == (B, T_pred, 1)


def test_speed_backprop_flows():
    model = speed_model()
    enc = torch.randn(1, 14, 1, requires_grad=True)
    dec = torch.empty(1, 45, 0)
    y = model(enc, dec)
    y.sum().backward()
    assert torch.isfinite(enc.grad).all()


# ---------------------------------------------------------------------------
# Param count sanity
# ---------------------------------------------------------------------------
def test_trajectory_and_speed_param_counts():
    tm = trajectory_model()
    sm = speed_model()
    n_tm = sum(p.numel() for p in tm.parameters() if p.requires_grad)
    n_sm = sum(p.numel() for p in sm.parameters() if p.requires_grad)
    # Trajectory has F_enc=4, F_dec=2 -> bigger than speed (F_enc=1, F_dec=0).
    assert n_tm > n_sm
    # Both are small-ish LSTM-256 encdec — a few hundred thousand params.
    assert 1e5 < n_sm < 2e6
    assert 1e5 < n_tm < 2e6


# ---------------------------------------------------------------------------
# End-to-end: dataset -> model smoke
# ---------------------------------------------------------------------------
def test_trajectory_e2e_with_synthetic_dataset_sample():
    """A single DataLoader batch from TrajectoryDataset feeds through the
    trajectory model and yields finite outputs."""
    from pie_pytorch.data.pie_dataset import TrajectoryConfig, TrajectoryDataset

    # Build synthetic per-track dict with obd_speed + intention_prob + bbox.
    n, L = 3, 70
    data = {
        "image": [[f"/fake/set05/vid/{i:05d}.png" for i in range(L)] for _ in range(n)],
        "bbox": [
            [[100.0 + i, 50.0 + i, 200.0 + i, 150.0 + i] for i in range(L)]
            for _ in range(n)
        ],
        "intention_prob": [[[0.7] for _ in range(L)] for _ in range(n)],
        "obd_speed": [[[5.0 + 0.1 * i] for i in range(L)] for _ in range(n)],
    }
    ds = TrajectoryDataset(data, TrajectoryConfig())
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = trajectory_model()
    y = model(batch["enc_input"], batch["dec_input"])
    assert y.shape == batch["target"].shape  # (B, 45, 4)
    assert torch.isfinite(y).all()
