"""
Phase 2 — Train Temporal GNN for in-hospital mortality prediction.

Usage:
    uv run python -m src.training.train_gnn [options]

Key flags:
    --signal-only       Drop all note nodes (signal-only ablation).
    --demo-path PATH    Add demographic features (age, gender, admission type).
    --icd-path PATH     Add Charlson ICD-10 comorbidity node (node_type=2).
                        Generate with: uv run python -m src.data_prep.extract_icd
    --all-stay-path PATH  Use full-stay signals instead of ±2h paired signals.
                        Generate with: uv run python -m src.data_prep.extract_all_stay_signals
    --pooling           mean | attention | dual  (default: mean)
    --focal-gamma       0 = plain BCE, 2 = focal loss
    --precision         32 | 16-mixed | bf16-mixed  (16-mixed enables AMP)

Outputs land in data/snapshots/gnn/<version>/  (CSVLogger default).
Best checkpoint is automatically saved and reloaded for final test metrics.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", ".*does not have many workers.*")

import lightning as L
import torch

torch.set_float32_matmul_precision("high")
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data_prep.graph_datamodule import MIMICGraphDataModule
from src.models.gnn_module import GNNMortalityModule


def _resolve_cache(args: argparse.Namespace) -> str | None:
    """Derive a cache filename that encodes the active ablation flags."""
    if not args.cache_path:
        return None
    base = Path(args.cache_path)
    suffix = (
        ("_signal_only" if args.signal_only else "")
        + ("_demo" if args.demo_path else "")
        + ("_icd" if args.icd_path else "")
        + (f"_allstay{args.max_signals or 'full'}" if args.all_stay_path else "")
    )
    return str(base.parent / f"graphs_cache{suffix}.pt") if suffix else str(base)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Phase 2 Temporal GNN")
    # Data
    p.add_argument("--csv-path", default="data/processed/pairs_all-icus_note_level.csv")
    p.add_argument("--embeddings-path", default="data/embeddings/node_embeddings.pt")
    p.add_argument("--cache-path", default="data/processed/graphs_cache.pt")
    # Ablations / features
    p.add_argument(
        "--signal-only",
        action="store_true",
        help="Drop all note nodes — signal-only temporal GNN ablation",
    )
    p.add_argument(
        "--demo-path",
        default=None,
        help="Path to demographics.csv (age_norm, gender_f, is_emergency, is_elective)",
    )
    p.add_argument(
        "--icd-path",
        default=None,
        help="Path to icd_charlson_*.csv — adds Charlson comorbidity node per stay",
    )
    p.add_argument(
        "--all-stay-path",
        default=None,
        help="Path to all_stay_signals_*.csv — replaces ±2h paired signals with full stay",
    )
    p.add_argument(
        "--max-signals",
        type=int,
        default=None,
        help="Cap signals per stay to EARLIEST N events (use with --all-stay-path "
        "to prevent memory explosion AND temporal leakage from terminal vitals)",
    )
    # Model
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pooling", default="mean", choices=["mean", "attention", "dual"])
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma (0=plain BCE, 2=standard focal)",
    )
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--precision",
        default="32",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Mixed precision: 16-mixed enables AMP",
    )
    args = p.parse_args()

    input_paths = {
        "--csv-path": args.csv_path,
        "--embeddings-path": args.embeddings_path,
        "--demo-path": args.demo_path,
        "--icd-path": args.icd_path,
        "--all-stay-path": args.all_stay_path,
    }
    missing = [
        f"{flag}={path}" for flag, path in input_paths.items() if path and not Path(path).is_file()
    ]
    if missing:
        p.error("input file(s) not found: " + ", ".join(missing))

    dm = MIMICGraphDataModule(
        csv_path=args.csv_path,
        embeddings_path=args.embeddings_path,
        signal_only=args.signal_only,
        demo_path=args.demo_path,
        icd_path=args.icd_path,
        all_stay_path=args.all_stay_path,
        max_signals=args.max_signals,
        train_ratio=args.train_ratio,
        seed=args.seed,
        cache_path=_resolve_cache(args),
        batch_size=args.batch_size,
    )
    dm.setup()

    model = GNNMortalityModule(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        use_demo=args.demo_path is not None,
        use_icd=args.icd_path is not None,
        lr=args.lr,
        focal_gamma=args.focal_gamma,
        pos_weight=dm.pos_weight,
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
    trainer.fit(model, dm)
    trainer.test(model, dm.val_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    main()
