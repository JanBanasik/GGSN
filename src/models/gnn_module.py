"""Phase 2 — LightningModule for in-hospital mortality prediction (frozen embeddings).

Train/val/test plumbing lives in `lightning_base.MortalityModuleBase`.
This module wires up `TemporalPatientGNN` and a plain Adam + ReduceLROnPlateau.
"""

from __future__ import annotations

import torch

from src.models.lightning_base import MortalityModuleBase
from src.models.tgnn_model import TemporalPatientGNN


class GNNMortalityModule(MortalityModuleBase):
    """
    Frozen-embedding variant of the mortality GNN.

    Hyperparameters are saved to hparams.yaml automatically.
    Set pos_weight from DataModule.pos_weight (train neg/pos ratio).

    Logging (from base):
        train_loss   — per step, focal loss
        val_loss     — per epoch, plain BCE
        val_auroc    — per epoch (monitored for early stopping + checkpointing)
        val_auprc    — per epoch
        test_*       — full suite (AUROC, AUPRC, Brier, sens@95spec) on best ckpt
    """

    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",
        use_demo: bool = False,
        use_icd: bool = False,
        lr: float = 1e-3,
        focal_gamma: float = 0.0,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TemporalPatientGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            use_demo=use_demo,
            use_icd=use_icd,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auroc"},
        }
