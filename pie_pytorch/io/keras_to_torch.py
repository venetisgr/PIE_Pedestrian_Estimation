"""Convert a Keras .h5 model (from the PIE paper) into a PyTorch state dict.

The three pretrained Keras models committed under
``data/pie/{intention,speed,trajectory}/.../model.h5`` were produced by
the TensorFlow 1.9 / Keras 2.2 reference implementation. This module
loads each file via ``h5py``, extracts the layer weights, and remaps
them into the PyTorch modules defined in ``pie_pytorch.models``.

Keras conventions (important, and the source of many off-by-one bugs):

    - ConvLSTM2D kernels: ``(kH, kW, C_in, 4*C_out)``.
      PyTorch ``Conv2d.weight``: ``(4*C_out, C_in, kH, kW)``. Permute
      ``(3, 2, 0, 1)``.
    - ConvLSTM2D uses ``padding`` for the input conv but HARDCODES
      ``padding='same'`` for the recurrent conv. See layers.py docstring.
    - Dense/LSTM ``kernel``: ``(in, out)``; PyTorch Linear.weight is
      ``(out, in)``. Transpose.
    - Gate order in concatenated kernels: ``[i, f, c_tilde, o]`` for both
      Keras and our PyTorch port. No reshuffle needed.
    - Keras LSTM bias has shape ``(4*H,)``. PyTorch W_x.bias has the same
      shape; W_h has no bias. Keras sum of input+recurrent bias collapses
      into W_x.bias — no explicit split.

Supported checkpoints: intent, trajectory, speed. All three models from
pie_predict.py / pie_intent.py.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _h5_read(f: h5py.File, layer: str, weight: str) -> np.ndarray:
    """Read ``model_weights/{layer}/{layer}/{weight}:0`` as a numpy array."""
    path = f"model_weights/{layer}/{layer}/{weight}:0"
    return np.asarray(f[path])


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def _convlstm_kernel_to_torch(k: np.ndarray) -> torch.Tensor:
    """Keras ConvLSTM kernel (kH, kW, C_in, 4*C_out) → torch (4*C_out, C_in, kH, kW)."""
    return _to_tensor(np.transpose(k, (3, 2, 0, 1)))


def _dense_kernel_to_torch(k: np.ndarray) -> torch.Tensor:
    """Keras Dense/LSTM kernel (in, out) → torch Linear.weight (out, in)."""
    return _to_tensor(k.T)


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------
def load_intent_weights(h5_path: str) -> dict[str, torch.Tensor]:
    """Return a state dict usable with ``IntentConvLSTMEncDec.load_state_dict``.

    Mapping (Keras layer → PyTorch parameter):
        conv_lst_m2d_1.kernel           → encoder.cell.conv_x.weight
        conv_lst_m2d_1.recurrent_kernel → encoder.cell.conv_h.weight
        conv_lst_m2d_1.bias             → encoder.cell.conv_x.bias
        decoder_network.kernel          → decoder.cell.W_x.weight
        decoder_network.recurrent_kernel→ decoder.cell.W_h.weight
        decoder_network.bias            → decoder.cell.W_x.bias
        decoder_dense.kernel            → head.weight
        decoder_dense.bias              → head.bias
    """
    with h5py.File(h5_path, "r") as f:
        return {
            "encoder.cell.conv_x.weight": _convlstm_kernel_to_torch(
                _h5_read(f, "conv_lst_m2d_1", "kernel")
            ),
            "encoder.cell.conv_x.bias": _to_tensor(
                _h5_read(f, "conv_lst_m2d_1", "bias")
            ),
            "encoder.cell.conv_h.weight": _convlstm_kernel_to_torch(
                _h5_read(f, "conv_lst_m2d_1", "recurrent_kernel")
            ),
            "decoder.cell.W_x.weight": _dense_kernel_to_torch(
                _h5_read(f, "decoder_network", "kernel")
            ),
            "decoder.cell.W_x.bias": _to_tensor(
                _h5_read(f, "decoder_network", "bias")
            ),
            "decoder.cell.W_h.weight": _dense_kernel_to_torch(
                _h5_read(f, "decoder_network", "recurrent_kernel")
            ),
            "head.weight": _dense_kernel_to_torch(
                _h5_read(f, "decoder_dense", "kernel")
            ),
            "head.bias": _to_tensor(_h5_read(f, "decoder_dense", "bias")),
        }


# ---------------------------------------------------------------------------
# Trajectory + Speed (shared AttnEncDec architecture)
# ---------------------------------------------------------------------------
# The Keras pie_encdec model uses generic layer names ``dense_{N}`` whose
# integer index depends on what else was built in the session. Paper
# checkpoints resolve to:
#   - trajectory (loc_intent_speed_pretrained): dense_22, dense_23, dense_24
#   - speed (speed_pretrained):                  dense_1,  dense_2,  dense_3
# dense_22/dense_1 = temporal attention dense (T, T)
# dense_23/dense_2 = embedding Dense (hidden, embed)
# dense_24/dense_3 = element attention dense (embed+F_dec, embed+F_dec)
def _resolve_attn_dense_names(f: h5py.File) -> tuple[str, str, str]:
    layers = list(f["model_weights"].keys())
    dense = sorted(
        (int(n.split("_")[1]), n) for n in layers if n.startswith("dense_")
    )
    if len(dense) < 3:
        raise ValueError(
            f"expected >=3 dense_ layers for AttnEncDec, found {dense}"
        )
    return dense[0][1], dense[1][1], dense[2][1]


def load_attn_encdec_weights(h5_path: str) -> dict[str, torch.Tensor]:
    """State dict for AttnEncDec (trajectory or speed).

    Mapping:
        encoder_network.*            → encoder.cell.W_{x,h} + bias
        dense_{small} temporal attn  → temporal_attn.fc
        dense_{mid} embedding        → embed
        dense_{large} element attn   → element_attn.fc
        decoder_network.*            → decoder.cell.W_{x,h} + bias
        decoder_dense.*              → head
    """
    with h5py.File(h5_path, "r") as f:
        tattn_name, embed_name, eattn_name = _resolve_attn_dense_names(f)
        return {
            "temporal_attn.fc.weight": _dense_kernel_to_torch(
                _h5_read(f, tattn_name, "kernel")
            ),
            "temporal_attn.fc.bias": _to_tensor(_h5_read(f, tattn_name, "bias")),
            "encoder.cell.W_x.weight": _dense_kernel_to_torch(
                _h5_read(f, "encoder_network", "kernel")
            ),
            "encoder.cell.W_x.bias": _to_tensor(
                _h5_read(f, "encoder_network", "bias")
            ),
            "encoder.cell.W_h.weight": _dense_kernel_to_torch(
                _h5_read(f, "encoder_network", "recurrent_kernel")
            ),
            "embed.weight": _dense_kernel_to_torch(_h5_read(f, embed_name, "kernel")),
            "embed.bias": _to_tensor(_h5_read(f, embed_name, "bias")),
            "element_attn.fc.weight": _dense_kernel_to_torch(
                _h5_read(f, eattn_name, "kernel")
            ),
            "element_attn.fc.bias": _to_tensor(_h5_read(f, eattn_name, "bias")),
            "decoder.cell.W_x.weight": _dense_kernel_to_torch(
                _h5_read(f, "decoder_network", "kernel")
            ),
            "decoder.cell.W_x.bias": _to_tensor(
                _h5_read(f, "decoder_network", "bias")
            ),
            "decoder.cell.W_h.weight": _dense_kernel_to_torch(
                _h5_read(f, "decoder_network", "recurrent_kernel")
            ),
            "head.weight": _dense_kernel_to_torch(
                _h5_read(f, "decoder_dense", "kernel")
            ),
            "head.bias": _to_tensor(_h5_read(f, "decoder_dense", "bias")),
        }


# ---------------------------------------------------------------------------
# High-level loader that also sizes the model from the .h5 shapes
# ---------------------------------------------------------------------------
def load_intent(h5_path: str):
    """Load paper intent weights into a correctly-sized IntentConvLSTMEncDec."""
    from ..models.intent import IntentConvLSTMEncDec, IntentModelConfig

    with h5py.File(h5_path, "r") as f:
        kernel = _h5_read(f, "conv_lst_m2d_1", "kernel")  # (kH, kW, C_in, 4*C_out)
        kH, kW, C_in, four_Cout = kernel.shape
        dec_kernel = _h5_read(f, "decoder_network", "kernel")  # (in, 4*hidden)
        dense = _h5_read(f, "decoder_dense", "kernel")  # (hidden, out)
        hidden = dense.shape[0]
        out = dense.shape[1]

    # Infer spatial input size from the expected flat dim: decoder input
    # = conv_out_hw^2 * conv_filters + decoder_input_size (4 for bbox).
    convlstm_filters = four_Cout // 4
    decoder_input_size = 4
    flat = dec_kernel.shape[0] - decoder_input_size  # 2304 for the paper
    conv_out_hw = int(round(flat / convlstm_filters) ** 0.5)
    # Keras valid: in = out + k - 1
    feature_hw = conv_out_hw + kH - 1

    cfg = IntentModelConfig(
        feature_channels=C_in,
        feature_hw=feature_hw,
        convlstm_filters=convlstm_filters,
        convlstm_kernel=kH,
        convlstm_padding="valid",
        lstm_hidden=hidden,
        decoder_input_size=decoder_input_size,
        output_size=out,
    )
    model = IntentConvLSTMEncDec(cfg)
    state = load_intent_weights(h5_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"load_intent state-dict mismatch\nmissing={missing}\nunexpected={unexpected}"
        )
    return model, cfg


def load_attn_encdec(h5_path: str, *, task: str):
    """Load paper trajectory/speed weights into a correctly-sized AttnEncDec.

    ``task`` must be ``"trajectory"`` or ``"speed"``; it only affects the
    inferred default ``observe_length`` / ``predict_length`` which can be
    overridden by the caller.
    """
    from ..models.trajectory import AttnEncDec, AttnEncDecConfig

    if task not in ("trajectory", "speed"):
        raise ValueError("task must be 'trajectory' or 'speed'")

    with h5py.File(h5_path, "r") as f:
        enc_kernel = _h5_read(f, "encoder_network", "kernel")  # (F_enc, 4*hidden)
        F_enc = enc_kernel.shape[0]
        hidden = enc_kernel.shape[1] // 4
        tattn_name, embed_name, eattn_name = _resolve_attn_dense_names(f)
        embed = _h5_read(f, embed_name, "kernel")  # (hidden, embed)
        embed_size = embed.shape[1]
        eattn = _h5_read(f, eattn_name, "kernel")  # (concat_dim, concat_dim)
        concat_dim = eattn.shape[0]
        F_dec = concat_dim - embed_size
        tattn = _h5_read(f, tattn_name, "kernel")  # (T_obs, T_obs)
        T_obs = tattn.shape[0]
        head = _h5_read(f, "decoder_dense", "kernel")  # (hidden, F_out)
        F_out = head.shape[1]

    cfg = AttnEncDecConfig(
        observe_length=T_obs,
        predict_length=45,
        enc_feature_size=F_enc,
        dec_feature_size=F_dec,
        prediction_size=F_out,
        hidden_size=hidden,
        embed_size=embed_size,
    )
    model = AttnEncDec(cfg)
    state = load_attn_encdec_weights(h5_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"load_attn_encdec state-dict mismatch\n"
            f"missing={missing}\nunexpected={unexpected}"
        )
    return model, cfg


__all__ = [
    "load_intent",
    "load_attn_encdec",
    "load_intent_weights",
    "load_attn_encdec_weights",
]
