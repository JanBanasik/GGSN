"""
Phase 2 — Train Temporal GNN for in-hospital mortality prediction.

Usage:
    uv run python -m src.training.train_gnn [options]

Default paths assume cwd = GGSN_Projektowe/.

Key ablation flags:
    --signal-only   Drop all note nodes; test signal-only temporal GNN.
    --e2e           Fine-tune Phase 1 text_tower jointly with the GNN.
    --pooling       mean | attention | dual  (default: mean)
    --focal-gamma   0 = plain BCE, 2 = focal loss
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.models.tgnn_model import TemporalPatientGNN, TemporalPatientGNNE2E
from src.training.loss import FocalLoss
from src.utils import metrics as M
from src.utils.gnn_dataset import build_datasets, build_datasets_e2e


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float, float]:
    """Returns (val_loss, val_auroc, val_auprc). Uses unweighted BCE for val_loss."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        total_loss += criterion(logits, batch.y).item()
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    val_loss = total_loss / max(len(loader), 1)

    try:
        val_auroc = M.auroc(labels_np, probs)
        val_auprc = M.auprc(labels_np, probs)
    except ValueError:
        val_auroc = val_auprc = 0.5

    return val_loss, val_auroc, val_auprc


def _compute_final_metrics(model: nn.Module, loader, device: torch.device) -> dict:
    """Full metric suite (AUROC, AUPRC, Brier, sens@95spec) on val set."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_logits.append(model(batch).cpu())
            all_labels.append(batch.y.cpu())
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    return {
        "auroc": round(M.auroc(labels_np, probs), 4),
        "auprc": round(M.auprc(labels_np, probs), 4),
        "brier": round(M.brier_score(labels_np, probs), 4),
        "sens@95spec": round(M.sens_at_spec(labels_np, probs, 0.95), 4),
    }


# ── Training ───────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("data/snapshots") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_id}")

    # ── Data ─────────────────────────────────────────────────────────────
    if args.e2e:
        e2e_cache = Path(args.cache_path).parent / "graphs_cache_e2e.pt"
        train_graphs, val_graphs, pos_weight = build_datasets_e2e(
            Path(args.csv_path),
            tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
            max_len=args.max_len,
            train_ratio=args.train_ratio,
            seed=args.seed,
            cache_path=e2e_cache,
        )
    else:
        if args.signal_only:
            cache = Path(args.cache_path).parent / "graphs_cache_signal_only.pt"
        else:
            cache = Path(args.cache_path) if args.cache_path else None

        train_graphs, val_graphs, pos_weight = build_datasets(
            Path(args.csv_path),
            Path(args.embeddings_path),
            signal_only=args.signal_only,
            train_ratio=args.train_ratio,
            seed=args.seed,
            cache_path=cache,
        )

    print(f"Graphs: {len(train_graphs):,} train | {len(val_graphs):,} val")

    # ── Model ────────────────────────────────────────────────────────────
    if args.e2e:
        model = TemporalPatientGNNE2E(
            text_tower_path=args.text_tower_path,
            node_dim=64,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            freeze_bert_layers=args.freeze_bert_layers,
        ).to(device)
        optimizer = torch.optim.Adam([
            {"params": model.gnn_parameters(), "lr": args.lr, "weight_decay": 1e-4},
            {"params": model.bert_parameters(), "lr": args.lr_bert, "weight_decay": 0.0},
        ])
    else:
        model = TemporalPatientGNN(
            node_dim=64,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=1e-4
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Loaders / loss / scheduler ───────────────────────────────────────
    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pw_tensor)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    cfg = {**vars(args), "run_id": run_id, "pos_weight": round(pos_weight, 3)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ── Training loop ────────────────────────────────────────────────────
    best_auroc = 0.0
    patience_count = 0
    log_rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_auroc, val_auprc = evaluate(model, val_loader, device)
        scheduler.step(val_auroc)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Ep {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| AUROC {val_auroc:.4f} | AUPRC {val_auprc:.4f} "
            f"| lr {lr_now:.1e} | {time.time()-t0:.1f}s"
        )
        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "val_auroc": round(val_auroc, 4),
            "val_auprc": round(val_auprc, 4), "lr": lr_now,
        })

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_count = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print(f"  ★  new best AUROC {best_auroc:.4f}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Final evaluation ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    final_metrics = _compute_final_metrics(model, val_loader, device)

    print("\nFinal val metrics (best checkpoint):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    (run_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))

    with open(run_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "lr"]
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nBest val AUROC: {best_auroc:.4f}")
    print(f"Run saved: {run_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train Phase 2 Temporal GNN")
    # Data
    p.add_argument("--csv-path", default="data/processed/pairs_all-icus_note_level.csv")
    p.add_argument("--embeddings-path", default="data/embeddings/node_embeddings.pt")
    p.add_argument("--cache-path", default="data/processed/graphs_cache.pt")
    # Ablations
    p.add_argument("--signal-only", action="store_true",
                   help="Drop all note nodes — test signal-only temporal GNN")
    # Model
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pooling", default="mean", choices=["mean", "attention", "dual"])
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="Focal loss gamma (0=plain BCE, 2=standard focal)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    # E2E fine-tuning
    p.add_argument("--e2e", action="store_true",
                   help="Fine-tune Phase 1 text_tower jointly with GNN")
    p.add_argument("--text-tower-path", default="data/embeddings/text_tower.pt")
    p.add_argument("--lr-bert", type=float, default=1e-5)
    p.add_argument("--freeze-bert-layers", type=int, default=8)
    p.add_argument("--max-len", type=int, default=256)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
