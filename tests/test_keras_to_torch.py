"""Tests for pie_pytorch.io.keras_to_torch.

Verifies:
  - All three paper .h5 checkpoints load without state_dict key errors.
  - Inferred model configs match the paper (hardcoded expected values).
  - ConvLSTM kernel permute is correct (4D axis order).
  - Forward pass runs end-to-end on a fixed input and returns finite output
    with the expected shape (no NaNs from a bad weight transpose).

We do NOT attempt Keras numerical parity in this file — that would
require TF + matching preprocessing, which isn't installed. Instead we
check shapes / finiteness / reproducibility, and rely on the weight
mapping being exact as documented in keras_to_torch.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
H5_INTENT = str(_REPO_ROOT / "data/pie/intention/context_loc_pretrained/model.h5")
H5_TRAJ = str(_REPO_ROOT / "data/pie/trajectory/loc_intent_speed_pretrained/model.h5")
H5_SPEED = str(_REPO_ROOT / "data/pie/speed/speed_pretrained/model.h5")


@pytest.fixture(autouse=True)
def _skip_if_h5_missing():
    for p in (H5_INTENT, H5_TRAJ, H5_SPEED):
        if not Path(p).is_file():
            pytest.skip(f"paper checkpoint not found: {p}")


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------
def test_intent_config_inferred_from_h5():
    from pie_pytorch.io.keras_to_torch import load_intent

    _, cfg = load_intent(H5_INTENT)
    # Paper spec: VGG16 features 512×7×7, ConvLSTM 64 filters 2×2 valid,
    # LSTM hidden 128, bbox decoder input 4, binary head.
    assert cfg.feature_channels == 512
    assert cfg.feature_hw == 7
    assert cfg.convlstm_filters == 64
    assert cfg.convlstm_kernel == 2
    assert cfg.convlstm_padding == "valid"
    assert cfg.convlstm_out_hw() == 6
    assert cfg.lstm_hidden == 128
    assert cfg.decoder_input_size == 4
    assert cfg.output_size == 1


def test_intent_forward_pass_produces_finite_output():
    from pie_pytorch.io.keras_to_torch import load_intent

    model, _ = load_intent(H5_INTENT)
    model.eval()
    torch.manual_seed(0)
    enc = torch.randn(2, 15, 512, 7, 7)
    dec = torch.randn(2, 15, 4)
    with torch.no_grad():
        logits = model(enc, dec)
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_intent_forward_deterministic():
    from pie_pytorch.io.keras_to_torch import load_intent

    m1, _ = load_intent(H5_INTENT)
    m2, _ = load_intent(H5_INTENT)
    m1.eval(); m2.eval()
    enc = torch.randn(1, 15, 512, 7, 7)
    dec = torch.randn(1, 15, 4)
    with torch.no_grad():
        y1 = m1(enc, dec)
        y2 = m2(enc, dec)
    assert torch.allclose(y1, y2)


def test_intent_weight_count_matches_paper():
    """Keras intent model has ~1.84M params; ours should match exactly."""
    from pie_pytorch.io.keras_to_torch import load_intent

    model, _ = load_intent(H5_INTENT)
    total = sum(p.numel() for p in model.parameters())
    assert total == 1837953  # nailed down by the paper checkpoint.


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def test_trajectory_config_inferred():
    from pie_pytorch.io.keras_to_torch import load_attn_encdec

    _, cfg = load_attn_encdec(H5_TRAJ, task="trajectory")
    assert cfg.observe_length == 14  # post-normalize_bbox
    assert cfg.enc_feature_size == 4
    assert cfg.dec_feature_size == 2  # intent_prob + obd_speed
    assert cfg.prediction_size == 4
    assert cfg.hidden_size == 256
    assert cfg.embed_size == 64


def test_trajectory_forward_pass():
    from pie_pytorch.io.keras_to_torch import load_attn_encdec

    model, cfg = load_attn_encdec(H5_TRAJ, task="trajectory")
    model.eval()
    enc = torch.randn(3, cfg.observe_length, cfg.enc_feature_size)
    dec = torch.randn(3, cfg.predict_length, cfg.dec_feature_size)
    with torch.no_grad():
        y = model(enc, dec)
    assert y.shape == (3, cfg.predict_length, cfg.prediction_size)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
def test_speed_config_inferred():
    from pie_pytorch.io.keras_to_torch import load_attn_encdec

    _, cfg = load_attn_encdec(H5_SPEED, task="speed")
    assert cfg.observe_length == 14
    assert cfg.enc_feature_size == 1
    # Note: paper trained speed with a zero-filled 1-dim dec placeholder,
    # not truly empty. Our scratch configs use dec_feature_size=0.
    assert cfg.dec_feature_size == 1
    assert cfg.prediction_size == 1
    assert cfg.hidden_size == 256


def test_speed_forward_pass():
    from pie_pytorch.io.keras_to_torch import load_attn_encdec

    model, cfg = load_attn_encdec(H5_SPEED, task="speed")
    model.eval()
    enc = torch.randn(2, cfg.observe_length, cfg.enc_feature_size)
    dec = torch.zeros(2, cfg.predict_length, cfg.dec_feature_size)
    with torch.no_grad():
        y = model(enc, dec)
    assert y.shape == (2, cfg.predict_length, cfg.prediction_size)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Detailed kernel-transpose sanity
# ---------------------------------------------------------------------------
def test_convlstm_kernel_permute_is_correct():
    """The Conv2d.weight we load must already be ready for a 224x224->7x7
    forward with 512-channel VGG features.

    A common bug: forgetting the (3,2,0,1) permute and loading the raw
    (kH, kW, C_in, 4*C_out) blob into the (4*C_out, C_in, kH, kW) slot.
    That would produce garbage feature maps and shape-mismatch errors.
    """
    from pie_pytorch.io.keras_to_torch import load_intent

    model, _ = load_intent(H5_INTENT)
    w = model.encoder.cell.conv_x.weight
    # Expected (4*64, 512, 2, 2)
    assert w.shape == (256, 512, 2, 2)

    w_rec = model.encoder.cell.conv_h.weight
    assert w_rec.shape == (256, 64, 2, 2)


def test_lstm_kernel_transpose_is_correct():
    from pie_pytorch.io.keras_to_torch import load_intent

    model, _ = load_intent(H5_INTENT)
    # decoder LSTM: Keras kernel is (2308, 512); torch wants (512, 2308)
    w = model.decoder.cell.W_x.weight
    assert w.shape == (512, 2308)
    w_rec = model.decoder.cell.W_h.weight
    assert w_rec.shape == (512, 128)


def test_head_dense_transpose_is_correct():
    from pie_pytorch.io.keras_to_torch import load_intent

    model, _ = load_intent(H5_INTENT)
    assert model.head.weight.shape == (1, 128)
    assert model.head.bias.shape == (1,)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_convert_cli_writes_safetensors_and_config(tmp_path):
    """pie_pytorch.cli.convert end-to-end: .h5 -> .safetensors + .config.json,
    round-tripping the state dict bit-exact."""
    import json

    import safetensors.torch as st

    from pie_pytorch.cli import convert as convert_cli
    from pie_pytorch.io.keras_to_torch import load_intent

    out_path = tmp_path / "intent.safetensors"
    rc = convert_cli.main(
        ["--task", "intent", "--h5", H5_INTENT, "--out", str(out_path)]
    )
    assert rc == 0
    assert out_path.is_file()

    cfg_path = out_path.with_suffix(".config.json")
    payload = json.loads(cfg_path.read_text())
    assert payload["task"] == "intent"
    assert payload["config"]["convlstm_padding"] == "valid"

    # The saved tensors must match what load_intent produces bit-exact.
    expected_model, _ = load_intent(H5_INTENT)
    saved = st.load_file(str(out_path), device="cpu")
    expected = {k: v.detach().cpu() for k, v in expected_model.state_dict().items()}
    assert set(saved) == set(expected)
    for k, v in expected.items():
        assert torch.allclose(saved[k], v)
