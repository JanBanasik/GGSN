"""
Phase 2 — End-to-end BERT + GNN fine-tuning for in-hospital mortality.

Loads tokenised graphs (text stored as input_ids, not pre-computed embeddings)
and jointly fine-tunes Bio_ClinicalBERT + TemporalGNN.

Usage:
    uv run python -m src.experimental.train_e2e [options]

Key differences vs train_gnn.py:
    - Graphs contain raw token IDs, BERT runs live each forward pass
    - Two LR groups: GNN (lr=1e-3) and BERT (lr_bert=5e-7)
    - batch_size=4 default (VRAM constraint on 8GB GPU)
    - precision=bf16-mixed default (required for VRAM budget)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from src.experimental.e2e import build_datasets_e2e
from src.experimental.e2e_module import E2EMortalityModule

torch.set_float32_matmul_precision("high")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Phase 2 E2E BERT+GNN")
    p.add_argument("--csv-path", default="data/processed/pairs_all-icus_note_level.csv")
    p.add_argument("--text-tower-path", default="data/embeddings/text_tower.pt")
    p.add_argument("--cache-path", default="data/processed/graphs_cache_e2e_k10.pt")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pooling", default="dual", choices=["mean", "attention", "dual"])
    p.add_argument(
        "--freeze-bert-layers",
        type=int,
        default=8,
        help="Freeze bottom N BERT layers (0=all trainable, 12=all frozen)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-bert", type=float, default=5e-7)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="32", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()

    print("Building e2e graph dataset (tokenised notes)…")
    train_graphs, val_graphs, pos_weight = build_datasets_e2e(
        csv_path=Path(args.csv_path),
        max_len=args.max_len,
        train_ratio=args.train_ratio,
        seed=args.seed,
        cache_path=Path(args.cache_path),
    )
    print(f"  pos_weight = {pos_weight:.3f}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = E2EMortalityModule(
        text_tower_path=args.text_tower_path,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        freeze_bert_layers=args.freeze_bert_layers,
        lr=args.lr,
        lr_bert=args.lr_bert,
        focal_gamma=args.focal_gamma,
        pos_weight=pos_weight,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        callbacks=[
            EarlyStopping(monitor="val_auroc", patience=args.patience, mode="max"),
            ModelCheckpoint(monitor="val_auroc", mode="max", save_top_k=1, filename="best"),
        ],
        logger=CSVLogger("data/snapshots", name="gnn"),
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
