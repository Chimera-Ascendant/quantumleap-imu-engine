import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class VQStub(nn.Module):
    """A minimal placeholder for a VQ layer that returns passthrough features.
    Returns (quantized, vq_loss, perplexity) to match an expected interface.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        vq_loss = x.new_zeros(())
        perplexity = x.new_ones(()) * x.numel() / max(1, batch)
        return x, vq_loss, perplexity


class PerceptionModel(nn.Module):
    """CNN + (VQ stub) + Transformer with four heads:
    - exercise classification
    - form classification
    - rep boundary probability (per frame)
    - fatigue regression

    Input shape: [B, C=8, T]
    """

    def __init__(self, input_channels: int = 8, hidden_dim: int = 128, codebook_size: int = 256,
                 num_exercises: int = 5, num_forms: int = 4, transformer_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_channels)

        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # VQ placeholder for future swap-in
        self.vq = VQStub()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.exercise_head = nn.Linear(hidden_dim, num_exercises)
        self.form_head = nn.Linear(hidden_dim, num_forms)
        self.rep_head = nn.Linear(hidden_dim, 1)
        self.fatigue_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, C, T]
        x = self.input_norm(x)
        feat = self.encoder(x)  # [B, H, T]
        feat = feat.transpose(1, 2)  # [B, T, H]
        quant, vq_loss, perplexity = self.vq(feat)  # [B, T, H]
        enc = self.transformer(quant)  # [B, T, H]
        pooled = enc.mean(dim=1)  # [B, H]

        exercise_logits = self.exercise_head(pooled)
        form_logits = self.form_head(pooled)
        rep_probs = torch.sigmoid(self.rep_head(enc))  # [B, T, 1]
        fatigue_score = torch.sigmoid(self.fatigue_head(pooled))  # [B, 1]

        return {
            "exercise_logits": exercise_logits,
            "form_logits": form_logits,
            "rep_probs": rep_probs.squeeze(-1),
            "fatigue_score": fatigue_score.squeeze(-1),
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }
