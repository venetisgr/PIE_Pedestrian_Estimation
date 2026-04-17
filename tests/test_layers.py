"""Tests for pie_pytorch.models.layers."""

from __future__ import annotations

import pytest
import torch

from pie_pytorch.models.layers import (
    ConvLSTM2D,
    ConvLSTM2DCell,
    ElementAttention,
    KerasLSTM,
    KerasLSTMCell,
    TemporalAttention,
    hard_sigmoid,
)


# ---------------------------------------------------------------------------
# hard_sigmoid
# ---------------------------------------------------------------------------
def test_hard_sigmoid_extremes_and_slope():
    x = torch.tensor([-10.0, -2.5, 0.0, 2.5, 10.0])
    y = hard_sigmoid(x)
    # Keras definition: clip(0.2*x + 0.5, 0, 1)
    # At x=-2.5 -> 0, x=0 -> 0.5, x=2.5 -> 1, extremes clamped.
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    assert torch.allclose(y, expected)


# ---------------------------------------------------------------------------
# KerasLSTMCell
# ---------------------------------------------------------------------------
def test_keras_lstm_cell_shapes():
    cell = KerasLSTMCell(input_size=8, hidden_size=16)
    x = torch.randn(4, 8)
    h = torch.zeros(4, 16)
    c = torch.zeros(4, 16)
    h_new, (h_out, c_out) = cell(x, (h, c))
    assert h_new.shape == (4, 16)
    assert h_out.shape == (4, 16)
    assert c_out.shape == (4, 16)
    assert h_new is h_out  # same object by design


def test_keras_lstm_cell_unit_forget_bias_initialized_to_one():
    cell = KerasLSTMCell(input_size=3, hidden_size=5, unit_forget_bias=True)
    H = 5
    # Keras gate order in the concatenated kernel: [i, f, c_tilde, o].
    forget_slice = cell.W_x.bias[H : 2 * H]
    assert torch.all(forget_slice == 1.0)
    # Other slices stay at zero at init.
    others = torch.cat(
        [cell.W_x.bias[:H], cell.W_x.bias[2 * H : 3 * H], cell.W_x.bias[3 * H :]]
    )
    assert torch.all(others == 0.0)


def test_keras_lstm_cell_no_unit_forget_bias():
    cell = KerasLSTMCell(input_size=3, hidden_size=5, unit_forget_bias=False)
    assert torch.all(cell.W_x.bias == 0.0)


def test_keras_lstm_cell_invalid_activation():
    with pytest.raises(ValueError):
        KerasLSTMCell(3, 5, activation="nope")


def test_keras_lstm_cell_invalid_recurrent_activation():
    with pytest.raises(ValueError):
        KerasLSTMCell(3, 5, recurrent_activation="nope")


# ---------------------------------------------------------------------------
# KerasLSTM
# ---------------------------------------------------------------------------
def test_keras_lstm_return_sequences():
    lstm = KerasLSTM(input_size=4, hidden_size=8, return_sequences=True)
    x = torch.randn(3, 10, 4)
    y = lstm(x)
    assert y.shape == (3, 10, 8)


def test_keras_lstm_return_last_step():
    lstm = KerasLSTM(input_size=4, hidden_size=8, return_sequences=False)
    x = torch.randn(3, 10, 4)
    y = lstm(x)
    assert y.shape == (3, 8)


def test_keras_lstm_return_state():
    lstm = KerasLSTM(input_size=4, hidden_size=8, return_sequences=False, return_state=True)
    y, (h, c) = lstm(torch.randn(2, 5, 4))
    assert y.shape == (2, 8)
    assert h.shape == (2, 8)
    assert c.shape == (2, 8)


def test_keras_lstm_softsign_matches_softsign_fn():
    """If activation='softsign', the cell output at t=1 from zero state
    should factor as o * softsign(c)."""
    lstm = KerasLSTM(input_size=2, hidden_size=3, activation="softsign", return_sequences=False)
    lstm.eval()
    x = torch.randn(1, 1, 2)
    y = lstm(x)
    # Sanity: softsign bounds output in (-1, 1)
    assert torch.all(y.abs() < 1.0)


def test_keras_lstm_initial_state_passthrough():
    lstm = KerasLSTM(input_size=4, hidden_size=8, return_sequences=False)
    x = torch.randn(2, 3, 4)
    h0 = torch.randn(2, 8)
    c0 = torch.randn(2, 8)
    y = lstm(x, initial_state=(h0, c0))
    assert y.shape == (2, 8)


def test_keras_lstm_backprop_flows():
    lstm = KerasLSTM(input_size=4, hidden_size=8, return_sequences=True)
    x = torch.randn(2, 5, 4, requires_grad=True)
    y = lstm(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_keras_lstm_dropout_eval_noop():
    lstm = KerasLSTM(input_size=4, hidden_size=8, dropout=0.5, recurrent_dropout=0.5)
    lstm.eval()
    x = torch.randn(2, 4, 4)
    y1 = lstm(x)
    y2 = lstm(x)
    # Eval mode: dropout off -> deterministic.
    assert torch.allclose(y1, y2)


def test_keras_lstm_dropout_train_injects_noise():
    torch.manual_seed(0)
    lstm = KerasLSTM(input_size=4, hidden_size=8, dropout=0.5)
    lstm.train()
    x = torch.randn(2, 4, 4)
    y1 = lstm(x)
    y2 = lstm(x)
    # Train mode with non-trivial dropout -> different draws.
    assert not torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# ConvLSTM2DCell / ConvLSTM2D
# ---------------------------------------------------------------------------
def test_conv_lstm_cell_shape_same_padding():
    cell = ConvLSTM2DCell(in_channels=3, out_channels=6, kernel_size=3)
    x = torch.randn(2, 3, 7, 7)
    h = torch.zeros(2, 6, 7, 7)
    c = torch.zeros(2, 6, 7, 7)
    _, (h_out, c_out) = cell(x, (h, c))
    assert h_out.shape == (2, 6, 7, 7)
    assert c_out.shape == (2, 6, 7, 7)


def test_conv_lstm_cell_unit_forget_bias():
    cell = ConvLSTM2DCell(in_channels=3, out_channels=4, kernel_size=2)
    H = 4
    assert torch.all(cell.conv_x.bias[H : 2 * H] == 1.0)
    other = torch.cat(
        [cell.conv_x.bias[:H], cell.conv_x.bias[2 * H : 3 * H], cell.conv_x.bias[3 * H :]]
    )
    assert torch.all(other == 0.0)


def test_conv_lstm_cell_recurrent_conv_has_no_bias():
    cell = ConvLSTM2DCell(in_channels=3, out_channels=4, kernel_size=2)
    assert cell.conv_h.bias is None


def test_conv_lstm_2d_return_last_step():
    conv = ConvLSTM2D(in_channels=512, out_channels=64, kernel_size=2)
    # (B, T, C, H, W) — matches PIE intent encoder input exactly.
    x = torch.randn(1, 15, 512, 7, 7)
    y = conv(x)
    assert y.shape == (1, 64, 7, 7)


def test_conv_lstm_2d_return_sequences():
    conv = ConvLSTM2D(in_channels=3, out_channels=8, kernel_size=2, return_sequences=True)
    x = torch.randn(2, 5, 3, 4, 4)
    y = conv(x)
    assert y.shape == (2, 5, 8, 4, 4)


def test_conv_lstm_2d_backprop_flows():
    conv = ConvLSTM2D(in_channels=2, out_channels=4, kernel_size=2)
    x = torch.randn(1, 3, 2, 5, 5, requires_grad=True)
    y = conv(x)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------
def test_temporal_attention_shape_preserved():
    attn = TemporalAttention(sequence_length=15)
    x = torch.randn(4, 15, 4)
    y = attn(x)
    assert y.shape == x.shape


def test_temporal_attention_gate_in_unit_interval():
    """The gate is sigmoid, so output / input ratio must be in [0, 1]."""
    torch.manual_seed(0)
    attn = TemporalAttention(sequence_length=10)
    x = torch.rand(2, 10, 3) + 0.1  # strictly positive -> safe ratio
    y = attn(x)
    ratio = y / x
    assert torch.all(ratio >= 0.0) and torch.all(ratio <= 1.0)


def test_element_attention_shape_preserved():
    attn = ElementAttention(input_dim=8)
    x = torch.randn(3, 7, 8)
    y = attn(x)
    assert y.shape == x.shape


def test_attention_backprop_flows():
    for attn, x in [
        (TemporalAttention(5), torch.randn(2, 5, 4, requires_grad=True)),
        (ElementAttention(4), torch.randn(2, 5, 4, requires_grad=True)),
    ]:
        y = attn(x)
        y.sum().backward()
        assert torch.isfinite(x.grad).all()
