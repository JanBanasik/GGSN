"""
Two-Tower Architecture for Multimodal Contrastive Pre-training.

TextTower   – Bio_ClinicalBERT → [CLS] → Linear → L2-norm → embed_dim
SignalTower – nn.Embedding(item_type) ⊕ value → 1D CNN → GAP → Linear → L2-norm

Both towers output L2-normalised vectors of identical EMBED_DIM, required for
the InfoNCE / NT-Xent loss used in Phase 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

EMBED_DIM = 128
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# Defaults aligned with the 14-signal catalog in src/data_prep/extractor.py.
NUM_SIGNAL_TYPES = 14
TYPE_EMBED_DIM = 8
PAD_TYPE_ID = 0  # padding uses item_type_id=0 with value=0; mask handled implicitly


# ---------------------------------------------------------------------------
# Text Tower
# ---------------------------------------------------------------------------
class TextTower(nn.Module):
    """
    Bio_ClinicalBERT [CLS] → projection head → L2-normalised (B, embed_dim).
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL,
        embed_dim: int = EMBED_DIM,
        freeze_bert: bool = False,
        freeze_bottom_layers: int = 0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Safetensors only: transformers+torch<2.6 cannot torch.load pytorch_model.bin (CVE-2025-32434 guard).
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        bert_hidden = self.bert.config.hidden_size  # 768 for BERT-base

        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, bert_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(bert_hidden // 2),
            nn.Dropout(proj_dropout),
            nn.Linear(bert_hidden // 2, embed_dim),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        elif freeze_bottom_layers > 0:
            # Freeze embeddings + first N transformer layers (out of 12 for BERT-base)
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_bottom_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        projected = self.proj(cls)
        return F.normalize(projected, dim=-1)


# ---------------------------------------------------------------------------
# Signal Tower
# ---------------------------------------------------------------------------
class SignalTower(nn.Module):
    """
    Encodes a sequence of (item_type_id, normalised_value, hours, delta) events.

    Forward inputs:
        item_ids:    (B, seq_len) long, values in [0, num_item_types)
        values:      (B, seq_len) float in [0, 1]
        mask:        (B, seq_len) float, 1 for real events, 0 for padding
        hours:       (B, seq_len) float in [0, 1] — event time / 24 (position
                     in the 24h ICU window). None → no-time encoding.
        delta_hours: (B, seq_len) float — Δt between event and the paired
                     note's note_time, normalised by ``delta_norm_hours`` to
                     roughly [-1, +1]. None → no-delta encoding.

    Forward output:
        (B, embed_dim) L2-normalised embedding.
    """

    def __init__(
        self,
        num_item_types: int = NUM_SIGNAL_TYPES,
        type_embed_dim: int = TYPE_EMBED_DIM,
        embed_dim: int = EMBED_DIM,
        use_hours: bool = True,
        use_delta: bool = True,
        delta_norm_hours: float = 24.0,
    ) -> None:
        super().__init__()

        self.type_embed = nn.Embedding(num_item_types, type_embed_dim)
        self.use_hours = use_hours
        self.use_delta = use_delta
        self.delta_norm_hours = float(delta_norm_hours)

        # type_embed + value (+ hours if enabled) (+ delta if enabled)
        in_channels = type_embed_dim + 1 + (1 if use_hours else 0) + (1 if use_delta else 0)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
        delta_hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        type_emb = self.type_embed(item_ids)                      # (B, L, type_embed_dim)
        chans = [type_emb, values.unsqueeze(-1)]
        if self.use_hours:
            if hours is None:
                hours = torch.zeros_like(values)
            chans.append(hours.unsqueeze(-1))
        if self.use_delta:
            if delta_hours is None:
                delta_hours = torch.zeros_like(values)
            else:
                # Normalise to roughly [-1, +1]; clip extremes for stability.
                delta_hours = (delta_hours / self.delta_norm_hours).clamp(-1.5, 1.5)
            chans.append(delta_hours.unsqueeze(-1))
        x = torch.cat(chans, dim=-1)                               # (B, L, in_channels)
        x = x.transpose(1, 2)                                      # (B, C, L)
        h = self.encoder(x)                                        # (B, 128, L)

        if mask is not None:
            # Mask out padded positions before pooling
            m = mask.unsqueeze(1)                                  # (B, 1, L)
            h = h * m
            denom = m.sum(dim=-1).clamp(min=1.0)                   # (B, 1)
            pooled = h.sum(dim=-1) / denom                         # (B, 128)
        else:
            pooled = h.mean(dim=-1)                                # (B, 128)

        projected = self.proj(pooled)                              # (B, embed_dim)
        return F.normalize(projected, dim=-1)
