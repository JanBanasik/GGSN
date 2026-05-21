"""Shared Lightning scaffolding for mortality-prediction modules.

`MortalityModuleBase` factors out the validation/test plumbing common to
`GNNMortalityModule` (frozen embeddings) and `E2EMortalityModule`
(end-to-end BERT + GNN). Subclasses provide:

    - `self.model` — a nn.Module taking a PyG batch and returning logits (B,)
    - `configure_optimizers()` — the only step that differs meaningfully
                                  between frozen and e2e variants

The base class implements the standard binary-classification loop:
    - Focal loss with pos_weight on training (gamma=0 ⇒ plain BCE)
    - BCE on validation (raw loss without focal reweighting)
    - val_auroc / val_auprc logged per epoch (val_auroc monitored for ckpt + ES)
    - test_auroc, test_auprc, test_brier, test_sens@95spec on best ckpt
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn

from src.training.loss import FocalLoss
from src.utils import metrics as M


class MortalityModuleBase(L.LightningModule):
    """Base class with shared train/val/test logic.

    Subclasses set `self.model` (any nn.Module mapping PyG batch → logits)
    and must implement `configure_optimizers`. Hparams `pos_weight` and
    `focal_gamma` are read from `self.hparams`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pw = torch.tensor([self.hparams.pos_weight], device=self.device)
        loss = FocalLoss(gamma=self.hparams.focal_gamma, pos_weight=pw)(self(batch), batch.y)
        self.log("train_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch.y)
        self.log("val_loss", loss, batch_size=batch.num_graphs, prog_bar=True)
        self._val_logits.append(logits.cpu())
        self._val_labels.append(batch.y.cpu())

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_logits).numpy()
        labels = torch.cat(self._val_labels).numpy().astype(int)
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        self._val_logits.clear()
        self._val_labels.clear()
        try:
            self.log("val_auroc", M.auroc(labels, probs), prog_bar=True)
            self.log("val_auprc", M.auprc(labels, probs))
        except ValueError:
            self.log("val_auroc", 0.5, prog_bar=True)
            self.log("val_auprc", 0.5)

    def test_step(self, batch, batch_idx):
        self._val_logits.append(self(batch).cpu())
        self._val_labels.append(batch.y.cpu())

    def on_test_epoch_end(self):
        logits = torch.cat(self._val_logits).numpy()
        labels = torch.cat(self._val_labels).numpy().astype(int)
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        self._val_logits.clear()
        self._val_labels.clear()
        results = {
            "test_auroc": round(M.auroc(labels, probs), 4),
            "test_auprc": round(M.auprc(labels, probs), 4),
            "test_brier": round(M.brier_score(labels, probs), 4),
            "test_sens@95spec": round(M.sens_at_spec(labels, probs, 0.95), 4),
        }
        self.log_dict(results)
        print("\nFinal metrics (best checkpoint):")
        for k, v in results.items():
            print(f"  {k}: {v}")
