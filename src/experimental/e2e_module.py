"""LightningModule for end-to-end BERT + GNN fine-tuning (EXPERIMENTAL).

Negative result documented in RAPORT.md §5.3: catastrophic forgetting at any
practical lr_bert. Kept around for ablation reproducibility — not part of the
production pipeline.

Train/val/test plumbing lives in `lightning_base.MortalityModuleBase`.
"""

from __future__ import annotations

import torch

from src.experimental.e2e import TemporalPatientGNNE2E
from src.models.lightning_base import MortalityModuleBase


class E2EMortalityModule(MortalityModuleBase):
    """
    Two optimizer param groups:
        GNN params  → lr (default 1e-3)
        BERT params → lr_bert (default 5e-7, 2000× lower to prevent catastrophic forgetting)
    """

    def __init__(
        self,
        text_tower_path: str,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "dual",
        freeze_bert_layers: int = 8,
        lr: float = 1e-3,
        lr_bert: float = 5e-7,
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TemporalPatientGNNE2E(
            text_tower_path=text_tower_path,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            freeze_bert_layers=freeze_bert_layers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.gnn_parameters(), "lr": self.hparams.lr},
                {"params": self.model.bert_parameters(), "lr": self.hparams.lr_bert},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"},
        }
