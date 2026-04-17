"""Shared layers ported from the PIE Keras models.

Why custom LSTM instead of torch.nn.LSTM?
    - Keras LSTM uses ``hard_sigmoid`` for input/forget/output gates,
      not sigmoid. torch.nn.LSTM hard-codes sigmoid.
    - Keras ``unit_forget_bias=True`` (default) adds 1.0 to the forget
      gate bias at init. torch.nn.LSTM initializes both halves of its
      bias to U(-1/sqrt(H), 1/sqrt(H)).
    - Keras ``activation`` is the cell/output nonlinearity. The paper's
      trajectory/speed models use ``softsign``; torch.nn.LSTM is tanh-only.
    - Keras gate order in the concatenated kernel is [i, f, c_tilde, o];
      PyTorch's is [i, f, g, o] but bias layout differs (ih + hh separate).

To keep the port faithful we replicate Keras behavior here with a
per-timestep Python loop. Not cuDNN-fast, but the PIE models are small
(256 hidden units, sub-second forward on CPU) so throughput is fine.

For a ``modern'' alternative down the line, we can swap these for
``torch.nn.LSTM`` and record the expected numeric drift; see
``notepad.md`` D9.

ConvLSTM has no stdlib PyTorch equivalent so ``ConvLSTM2D`` below is
the only option regardless of parity stance.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Keras-style hard sigmoid: max(0, min(1, 0.2*x + 0.5))."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "softsign": F.softsign,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "linear": lambda x: x,
}


def _resolve_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        return _ACTIVATIONS[name]
    except KeyError as e:
        raise ValueError(
            f"unknown activation {name!r}; supported: {sorted(_ACTIVATIONS)}"
        ) from e


# ---------------------------------------------------------------------------
# KerasLSTMCell / KerasLSTM
# ---------------------------------------------------------------------------
class KerasLSTMCell(nn.Module):
    """Single-step LSTM cell with Keras semantics.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    activation : {"tanh", "softsign"}
        Cell and output nonlinearity.
    recurrent_activation : {"hard_sigmoid", "sigmoid"}
        Gate nonlinearity.
    unit_forget_bias : bool
        If True, initialize forget-gate bias to 1 (Keras default).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "tanh",
        recurrent_activation: str = "hard_sigmoid",
        unit_forget_bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = _resolve_activation(activation)
        if recurrent_activation == "hard_sigmoid":
            self.recurrent_activation = hard_sigmoid
        elif recurrent_activation == "sigmoid":
            self.recurrent_activation = torch.sigmoid
        else:
            raise ValueError(
                f"recurrent_activation must be 'hard_sigmoid' or 'sigmoid', "
                f"got {recurrent_activation!r}"
            )

        # Keras concatenated kernel order: [i, f, c_tilde, o]
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self._init_weights(unit_forget_bias)

    def _init_weights(self, unit_forget_bias: bool) -> None:
        nn.init.xavier_uniform_(self.W_x.weight)  # Keras default
        nn.init.orthogonal_(self.W_h.weight)  # Keras recurrent default
        nn.init.zeros_(self.W_x.bias)
        if unit_forget_bias:
            with torch.no_grad():
                H = self.hidden_size
                self.W_x.bias[H : 2 * H].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        z = self.W_x(x) + self.W_h(h_prev)
        i, f, c_tilde, o = z.chunk(4, dim=-1)
        i = self.recurrent_activation(i)
        f = self.recurrent_activation(f)
        c_tilde = self.activation(c_tilde)
        o = self.recurrent_activation(o)
        c = f * c_prev + i * c_tilde
        h = o * self.activation(c)
        return h, (h, c)


class KerasLSTM(nn.Module):
    """Sequence-level LSTM wrapper around KerasLSTMCell.

    Mirrors the Keras LSTM arg surface used by the PIE models:
        - return_sequences, return_state
        - stateful (not implemented; always False)
        - dropout / recurrent_dropout (applied same as Keras: per-step
          input-gate / recurrent Bernoulli mask during training only).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "tanh",
        recurrent_activation: str = "hard_sigmoid",
        return_sequences: bool = True,
        return_state: bool = False,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        unit_forget_bias: bool = True,
    ):
        super().__init__()
        assert 0.0 <= dropout < 1.0 and 0.0 <= recurrent_dropout < 1.0, (
            "dropout/recurrent_dropout must be in [0, 1)"
        )
        self.cell = KerasLSTMCell(
            input_size,
            hidden_size,
            activation=activation,
            recurrent_activation=recurrent_activation,
            unit_forget_bias=unit_forget_bias,
        )
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        if initial_state is None:
            h = x.new_zeros(B, self.hidden_size)
            c = x.new_zeros(B, self.hidden_size)
        else:
            h, c = initial_state

        # Time-constant dropout masks (Keras variational-style)
        if self.training and self.dropout > 0:
            mask_x = x.new_empty(B, x.size(-1)).bernoulli_(1 - self.dropout) / (
                1 - self.dropout
            )
        else:
            mask_x = None
        if self.training and self.recurrent_dropout > 0:
            mask_h = x.new_empty(B, self.hidden_size).bernoulli_(
                1 - self.recurrent_dropout
            ) / (1 - self.recurrent_dropout)
        else:
            mask_h = None

        outs = [] if self.return_sequences else None
        for t in range(T):
            xt = x[:, t] * mask_x if mask_x is not None else x[:, t]
            ht_in = h * mask_h if mask_h is not None else h
            h, (h, c) = self.cell(xt, (ht_in, c))
            if outs is not None:
                outs.append(h)

        y = torch.stack(outs, dim=1) if outs is not None else h
        if self.return_state:
            return y, (h, c)
        return y


# ---------------------------------------------------------------------------
# ConvLSTM2D
# ---------------------------------------------------------------------------
class ConvLSTM2DCell(nn.Module):
    """Single-step 2D ConvLSTM cell with Keras semantics.

    Input/Output
        x: (B, C_in, H, W)
        state: (h, c) each (B, C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        activation: str = "tanh",
        recurrent_activation: str = "hard_sigmoid",
        unit_forget_bias: bool = True,
        padding: str | int = "same",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Keras ConvLSTM2D uses the user-supplied `padding` for the input
        # conv but hardcodes `padding='same'` for the recurrent conv (the
        # hidden state must keep its spatial size across timesteps).
        # "same" is only supported on stride=1 convs (which we use).
        self.conv_x = nn.Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.conv_h = nn.Conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )

        self.activation = _resolve_activation(activation)
        if recurrent_activation == "hard_sigmoid":
            self.recurrent_activation = hard_sigmoid
        elif recurrent_activation == "sigmoid":
            self.recurrent_activation = torch.sigmoid
        else:
            raise ValueError(
                f"recurrent_activation must be 'hard_sigmoid' or 'sigmoid'"
            )
        self._init_weights(unit_forget_bias)

    def _init_weights(self, unit_forget_bias: bool) -> None:
        nn.init.xavier_uniform_(self.conv_x.weight)
        # orthogonal_ handles 4D conv kernels by flattening internally.
        nn.init.orthogonal_(self.conv_h.weight)
        nn.init.zeros_(self.conv_x.bias)
        if unit_forget_bias:
            with torch.no_grad():
                C = self.out_channels
                self.conv_x.bias[C : 2 * C].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        z = self.conv_x(x) + self.conv_h(h_prev)
        i, f, c_tilde, o = z.chunk(4, dim=1)  # split on channel axis
        i = self.recurrent_activation(i)
        f = self.recurrent_activation(f)
        c_tilde = self.activation(c_tilde)
        o = self.recurrent_activation(o)
        c = f * c_prev + i * c_tilde
        h = o * self.activation(c)
        return h, (h, c)


class ConvLSTM2D(nn.Module):
    """Sequence wrapper around ConvLSTM2DCell.

    Input:  (B, T, C_in, H, W)
    Output: (B, C_out, H, W) if return_sequences=False,
            (B, T, C_out, H, W) otherwise.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        activation: str = "tanh",
        recurrent_activation: str = "hard_sigmoid",
        return_sequences: bool = False,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        padding: str | int = "same",
    ):
        super().__init__()
        assert 0.0 <= dropout < 1.0 and 0.0 <= recurrent_dropout < 1.0
        self.cell = ConvLSTM2DCell(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            recurrent_activation=recurrent_activation,
            padding=padding,
        )
        self.out_channels = out_channels
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        # Run one step of conv_x to discover the output spatial size — this
        # handles both "same" (stays H, W) and "valid" (shrinks by k-1).
        # The first real pass below will overwrite this forward compute.
        with torch.no_grad():
            out_hw = self.cell.conv_x(x[:, 0]).shape[-2:]
        H_out, W_out = out_hw

        h = x.new_zeros(B, self.out_channels, H_out, W_out)
        c = x.new_zeros(B, self.out_channels, H_out, W_out)

        if self.training and self.dropout > 0:
            mask_x = x.new_empty(B, C, H, W).bernoulli_(1 - self.dropout) / (
                1 - self.dropout
            )
        else:
            mask_x = None
        if self.training and self.recurrent_dropout > 0:
            mask_h = x.new_empty(B, self.out_channels, H_out, W_out).bernoulli_(
                1 - self.recurrent_dropout
            ) / (1 - self.recurrent_dropout)
        else:
            mask_h = None

        outs = [] if self.return_sequences else None
        for t in range(T):
            xt = x[:, t] * mask_x if mask_x is not None else x[:, t]
            ht_in = h * mask_h if mask_h is not None else h
            h, (h, c) = self.cell(xt, (ht_in, c))
            if outs is not None:
                outs.append(h)
        return torch.stack(outs, dim=1) if outs is not None else h


# ---------------------------------------------------------------------------
# Attention modules (from pie_predict.py)
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Temporal attention: gate each (time, feature) entry by its
    time-axis mixing of all other timesteps.

    Port of legacy_tf/pie_predict.py::attention_temporal:
        Permute(2,1) -> Dense(T, sigmoid) -> Permute(2,1) -> Multiply
    """

    def __init__(self, sequence_length: int):
        super().__init__()
        self.fc = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        a = x.transpose(1, 2)  # (B, F, T)
        a = torch.sigmoid(self.fc(a))  # (B, F, T)
        a = a.transpose(1, 2)  # (B, T, F)
        return x * a


class ElementAttention(nn.Module):
    """Self-attention over the feature axis at every timestep.

    Port of legacy_tf/pie_predict.py::attention_element:
        Dense(F, sigmoid) -> Multiply
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.fc(x))


__all__ = [
    "hard_sigmoid",
    "KerasLSTMCell",
    "KerasLSTM",
    "ConvLSTM2DCell",
    "ConvLSTM2D",
    "TemporalAttention",
    "ElementAttention",
]
