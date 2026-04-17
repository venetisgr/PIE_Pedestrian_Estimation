"""Trajectory / speed attention encoder-decoder.

Port of legacy_tf/pie_predict.py::PIEPredict.pie_encdec. One class
``AttnEncDec`` handles both the trajectory model (bbox -> bbox) and the
speed model (speed -> speed); ``trajectory_model`` / ``speed_model``
factory functions below wire up the defaults from the paper.

Architecture (batch first)
--------------------------
    enc_input (B, T_obs, F_enc)
      └─ TemporalAttention(T_obs)                 (B, T_obs, F_enc)
      └─ KerasLSTM(256, softsign,
                   return_sequences=True,
                   return_state=True)
         → out_seq (B, T_obs, 256)  +  (h, c)

    Repeat h over T_pred, embed to embed_size with ReLU, dropout.
    dec_input (B, T_pred, F_dec)
    → concat(embedded_hidden, dec_input, dim=-1)  (B, T_pred, embed+F_dec)
    → ElementAttention(embed + F_dec)
    → KerasLSTM(256, softsign,
                 return_sequences=True,
                 initial_state=(h, c))
      → (B, T_pred, 256)
    → Dense(prediction_size, linear)
      → (B, T_pred, prediction_size)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import ElementAttention, KerasLSTM, TemporalAttention


@dataclass
class AttnEncDecConfig:
    observe_length: int = 14          # T_obs (= 15 - 1 after normalize_bbox)
    predict_length: int = 45          # T_pred
    enc_feature_size: int = 4         # bbox (x1 y1 x2 y2); speed uses 1
    dec_feature_size: int = 2         # intent_prob + obd_speed; speed uses 0
    prediction_size: int = 4          # bbox; speed uses 1
    hidden_size: int = 256
    embed_size: int = 64
    embed_dropout: float = 0.0
    activation: str = "softsign"


class AttnEncDec(nn.Module):
    """Shared attention encoder-decoder used by both trajectory and speed."""

    def __init__(self, cfg: AttnEncDecConfig | None = None):
        super().__init__()
        self.cfg = cfg or AttnEncDecConfig()

        self.temporal_attn = TemporalAttention(sequence_length=self.cfg.observe_length)
        self.encoder = KerasLSTM(
            input_size=self.cfg.enc_feature_size,
            hidden_size=self.cfg.hidden_size,
            activation=self.cfg.activation,
            return_sequences=True,
            return_state=True,
        )

        self.embed = nn.Linear(self.cfg.hidden_size, self.cfg.embed_size)
        self.embed_dropout = nn.Dropout(self.cfg.embed_dropout)

        concat_dim = self.cfg.embed_size + self.cfg.dec_feature_size
        self.element_attn = ElementAttention(input_dim=concat_dim)

        self.decoder = KerasLSTM(
            input_size=concat_dim,
            hidden_size=self.cfg.hidden_size,
            activation=self.cfg.activation,
            return_sequences=True,
        )

        self.head = nn.Linear(self.cfg.hidden_size, self.cfg.prediction_size)

    def forward(
        self,
        enc_input: torch.Tensor,
        dec_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_input : (B, T_obs, F_enc)
        dec_input : (B, T_pred, F_dec)
            Pass a ``(B, T_pred, 0)`` tensor when ``F_dec == 0`` (speed model).

        Returns
        -------
        (B, T_pred, prediction_size)
        """
        B, T_obs, _ = enc_input.shape
        T_pred = self.cfg.predict_length
        assert T_obs == self.cfg.observe_length, (
            f"enc_input T={T_obs} but cfg.observe_length={self.cfg.observe_length}"
        )
        assert dec_input.shape[0] == B, "batch size mismatch"
        assert dec_input.shape[1] == T_pred, (
            f"dec_input T={dec_input.shape[1]} but cfg.predict_length={T_pred}"
        )
        assert dec_input.shape[-1] == self.cfg.dec_feature_size, (
            f"dec_input F={dec_input.shape[-1]} but cfg.dec_feature_size="
            f"{self.cfg.dec_feature_size}"
        )

        # Encoder side
        x = self.temporal_attn(enc_input)                  # (B, T_obs, F_enc)
        _, (h, c) = self.encoder(x)                        # h, c: (B, hidden)

        # Repeat h across the prediction horizon and embed.
        h_rep = h.unsqueeze(1).expand(-1, T_pred, -1)       # (B, T_pred, hidden)
        embedded = torch.relu(self.embed(h_rep))            # (B, T_pred, embed)
        embedded = self.embed_dropout(embedded)

        # Concat with decoder exogenous inputs and self-attend along features.
        dec_concat = torch.cat([embedded, dec_input], dim=-1)  # (B, T_pred, embed + F_dec)
        dec_concat = self.element_attn(dec_concat)

        # Decoder LSTM initialized from encoder final state.
        dec_out = self.decoder(dec_concat, initial_state=(h, c))  # (B, T_pred, hidden)
        return self.head(dec_out)                                  # linear head


# ---------------------------------------------------------------------------
# Factory helpers (paper defaults)
# ---------------------------------------------------------------------------
def trajectory_model(
    observe_length: int = 14,
    predict_length: int = 45,
    dec_feature_size: int = 2,
) -> AttnEncDec:
    """Paper default: bbox in (4) + [intent_prob, obd_speed] decoder in (2) -> bbox out (4)."""
    return AttnEncDec(
        AttnEncDecConfig(
            observe_length=observe_length,
            predict_length=predict_length,
            enc_feature_size=4,
            dec_feature_size=dec_feature_size,
            prediction_size=4,
        )
    )


def speed_model(
    observe_length: int = 14,
    predict_length: int = 45,
) -> AttnEncDec:
    """Paper default: obd_speed in (1) -> obd_speed out (1). No exogenous decoder input."""
    return AttnEncDec(
        AttnEncDecConfig(
            observe_length=observe_length,
            predict_length=predict_length,
            enc_feature_size=1,
            dec_feature_size=0,
            prediction_size=1,
        )
    )
