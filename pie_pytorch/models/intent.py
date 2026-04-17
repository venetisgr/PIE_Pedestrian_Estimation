"""Intent model: ConvLSTM encoder + LSTM decoder + sigmoid head.

Port of legacy_tf/pie_intent.py::PIEIntent.pie_convlstm_encdec.

Architecture (batch first):
    encoder_input  (B, T, 512, 7, 7)   -- VGG16 features per frame
      └─ ConvLSTM2D(filters=64, k=2, return_sequences=False)
         └─ (B, 64, 7, 7)  →  Flatten  →  (B, 64*7*7 = 3136)

    decoder_input  (B, T, 4)           -- bbox sequence (legacy dec_input_type=['bbox'])
      └─ RepeatVector of encoder flat over T  concat with decoder_input
         └─ (B, T, 3136 + 4 = 3140)  →  KerasLSTM(128, tanh, dropout=0.4/0.2,
                                                  return_sequences=False)
         └─ (B, 128)  →  Dense(1, sigmoid)  →  (B, 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import ConvLSTM2D, KerasLSTM


@dataclass
class IntentModelConfig:
    observe_length: int = 15           # shared enc/dec length
    feature_channels: int = 512        # VGG16 feature channels
    feature_hw: int = 7                # VGG16 feature spatial size (224/32)
    convlstm_filters: int = 64
    convlstm_kernel: int = 2
    lstm_hidden: int = 128
    lstm_dropout: float = 0.4
    lstm_recurrent_dropout: float = 0.2
    decoder_input_size: int = 4        # bbox has 4 coords
    output_size: int = 1               # binary crossing probability


class IntentConvLSTMEncDec(nn.Module):
    """ConvLSTM encoder + LSTM decoder model for pedestrian crossing intent.

    Forward inputs
    --------------
    encoder_input : Tensor, shape ``(B, T, C, H, W)``
        Per-frame VGG16 context features. For the default PIE config this
        is ``(B, 15, 512, 7, 7)``.
    decoder_input : Tensor, shape ``(B, T, F_dec)``
        Bounding-box sequence. Defaults to ``(B, 15, 4)``.

    Returns
    -------
    Tensor, shape ``(B, 1)``
        Crossing probability in [0, 1].
    """

    def __init__(self, cfg: IntentModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or IntentModelConfig()

        self.encoder = ConvLSTM2D(
            in_channels=self.cfg.feature_channels,
            out_channels=self.cfg.convlstm_filters,
            kernel_size=self.cfg.convlstm_kernel,
            activation="tanh",
            return_sequences=False,
        )
        hw = self.cfg.feature_hw
        flat_size = self.cfg.convlstm_filters * hw * hw

        self.decoder = KerasLSTM(
            input_size=flat_size + self.cfg.decoder_input_size,
            hidden_size=self.cfg.lstm_hidden,
            activation="tanh",
            return_sequences=False,
            dropout=self.cfg.lstm_dropout,
            recurrent_dropout=self.cfg.lstm_recurrent_dropout,
        )

        self.head = nn.Linear(self.cfg.lstm_hidden, self.cfg.output_size)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C, H, W = encoder_input.shape
        if decoder_input.shape[:2] != (B, T):
            raise ValueError(
                f"decoder_input leading dims {decoder_input.shape[:2]} must "
                f"match encoder (B={B}, T={T})"
            )
        enc = self.encoder(encoder_input)          # (B, F, H, W)
        enc_flat = enc.flatten(start_dim=1)        # (B, F*H*W)

        # RepeatVector(T) + concat along feature axis with decoder_input.
        enc_rep = enc_flat.unsqueeze(1).expand(-1, T, -1)  # (B, T, F*H*W)
        dec_in = torch.cat([enc_rep, decoder_input], dim=-1)  # (B, T, F*H*W + 4)

        dec_out = self.decoder(dec_in)             # (B, hidden)
        return torch.sigmoid(self.head(dec_out))   # (B, output_size)
