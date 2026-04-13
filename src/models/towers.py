"""
Two-Tower Architecture for Multimodal Contrastive Pre-training.

TextTower  – Bio_ClinicalBERT → [CLS] → Linear → L2-norm → embed_dim
SignalTower – 1D CNN → GlobalAvgPool → Linear → L2-norm → embed_dim

Both towers output L2-normalised vectors of identical dimensionality (EMBED_DIM),
which is required for the InfoNCE / NT-Xent loss used in Phase 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Shared embedding dimension – change here to affect both towers at once.
EMBED_DIM = 128
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


# ---------------------------------------------------------------------------
# Text Tower
# ---------------------------------------------------------------------------
class TextTower(nn.Module):
    """
    Encodes clinical text via Bio_ClinicalBERT.

    Forward input:
        input_ids       (B, L) – token ids, L ≤ 512
        attention_mask  (B, L) – 1 for real tokens, 0 for padding

    Forward output:
        (B, embed_dim) L2-normalised embedding derived from [CLS] token.
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL,
        embed_dim: int = EMBED_DIM,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden = self.bert.config.hidden_size  # 768 for BERT-base

        # Projection head: BERT hidden → embed_dim with a bottleneck
        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, bert_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(bert_hidden // 2),
            nn.Linear(bert_hidden // 2, embed_dim),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768) – [CLS] representation
        projected = self.proj(cls)              # (B, embed_dim)
        return F.normalize(projected, dim=-1)  # L2 normalise


# ---------------------------------------------------------------------------
# Signal Tower
# ---------------------------------------------------------------------------
class SignalTower(nn.Module):
    """
    Encodes a variable-length sequence of (item_type, normalised_value) vitals
    using 1-D convolutions followed by global average pooling.

    Input shape expected:  (B, seq_len, 2)
        channel 0 – item type encoded as float in {0.0, 1.0}  (HR=0, BP=1)
        channel 1 – normalised measurement value in [0, 1]

    Conv1d works on (B, C, L), so the sequence axis is transposed internally.

    Forward output:
        (B, embed_dim) L2-normalised embedding.
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1 – local patterns (kernel=3)
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Block 2 – wider receptive field (kernel=5)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Global average pooling → fixed-size regardless of seq_len
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 2) → (B, 2, seq_len) for Conv1d
        x = x.transpose(1, 2)
        features = self.encoder(x).squeeze(-1)  # (B, 128)
        projected = self.proj(features)          # (B, embed_dim)
        return F.normalize(projected, dim=-1)   # L2 normalise
