"""LightningDataModule for Phase 2 GNN training."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch_geometric.loader import DataLoader

from src.utils.gnn_dataset import build_datasets


class MIMICGraphDataModule(L.LightningDataModule):
    """
    Wraps build_datasets() for Lightning-compatible data loading.

    Call setup() before accessing pos_weight — needed to construct GNNMortalityModule.

    Optional features:
        icd_path           — Charlson ICD CSV from extract_icd.py
        all_stay_path      — full-stay signals CSV from extract_all_stay_signals.py

    Example:
        dm = MIMICGraphDataModule(csv_path=..., embeddings_path=...,
                                  icd_path=..., all_stay_path=...)
        dm.setup()
        model = GNNMortalityModule(pos_weight=dm.pos_weight, use_icd=True)
        trainer.fit(model, dm)
    """

    def __init__(
        self,
        csv_path: str,
        embeddings_path: str,
        *,
        signal_only: bool = False,
        demo_path: str | None = None,
        icd_path: str | None = None,
        all_stay_path: str | None = None,
        max_signals: int | None = None,
        train_ratio: float = 0.8,
        seed: int = 42,
        cache_path: str | None = None,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._train: list | None = None
        self._val: list | None = None
        self.pos_weight: float = 1.0

    def setup(self, stage: str | None = None) -> None:
        self._train, self._val, self.pos_weight = build_datasets(
            Path(self.hparams.csv_path),
            Path(self.hparams.embeddings_path),
            signal_only=self.hparams.signal_only,
            demo_path=Path(self.hparams.demo_path) if self.hparams.demo_path else None,
            icd_path=Path(self.hparams.icd_path) if self.hparams.icd_path else None,
            all_stay_signals_path=(
                Path(self.hparams.all_stay_path) if self.hparams.all_stay_path else None
            ),
            max_signals=self.hparams.max_signals,
            train_ratio=self.hparams.train_ratio,
            seed=self.hparams.seed,
            cache_path=Path(self.hparams.cache_path) if self.hparams.cache_path else None,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )
