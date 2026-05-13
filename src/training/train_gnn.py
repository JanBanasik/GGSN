"""
Phase 2 — Train Temporal GNN for in-hospital mortality prediction.

Usage:
    uv run python -m src.training.train_gnn [options]

Default paths assume cwd = GGSN_Projektowe/.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.models.tgnn_model import TemporalPatientGNN
from src.utils import metrics as M
from src.utils.graph_builder import build_patient_graph, load_note_embeddings


# ── CSV pre-processing ─────────────────────────────────────────────────────

def _preprocess_csv(csv_path: Path) -> tuple[dict, dict, dict, dict]:
    """Parse pairs CSV into per-stay Python dicts for O(1) graph construction.

    Returns:
        notes_by_stay:    {stay_id -> [{note_id, note_time}]}
        signals_by_stay:  {stay_id -> [{norm_value, item_type_id, event_hours_from_intime}]}
        mortality_by_stay:{stay_id -> int}
        subject_by_stay:  {stay_id -> int}
    """
    df = pl.read_csv(csv_path)

    # note_hours_from_intime = event_hours_from_intime + delta_hours_to_note
    # (consistent across all rows for the same note_id)
    df = df.with_columns(
        (pl.col("event_hours_from_intime") + pl.col("delta_hours_to_note")).alias("note_hours")
    )

    # One row per (stay_id, note_id)
    notes_df = (
        df.group_by(["stay_id", "note_id"])
        .agg(pl.first("note_hours"))
    )

    # One row per unique signal event (stay_id, time, type)
    signals_df = (
        df.select(["stay_id", "event_hours_from_intime", "item_type_id", "norm_value"])
        .unique(["stay_id", "event_hours_from_intime", "item_type_id"])
    )

    meta_df = (
        df.group_by("stay_id")
        .agg(pl.first("mortality"), pl.first("subject_id"))
    )

    notes_by_stay: dict[int, list] = defaultdict(list)
    for r in notes_df.iter_rows(named=True):
        notes_by_stay[r["stay_id"]].append({
            "note_id": r["note_id"],
            "note_time": float(r["note_hours"]),
        })

    signals_by_stay: dict[int, list] = defaultdict(list)
    for r in signals_df.iter_rows(named=True):
        signals_by_stay[r["stay_id"]].append({
            "norm_value": float(r["norm_value"]),
            "item_type_id": int(r["item_type_id"]),
            "event_hours_from_intime": float(r["event_hours_from_intime"]),
        })

    mortality_by_stay: dict[int, int] = {}
    subject_by_stay: dict[int, int] = {}
    for r in meta_df.iter_rows(named=True):
        mortality_by_stay[r["stay_id"]] = int(r["mortality"])
        subject_by_stay[r["stay_id"]] = int(r["subject_id"])

    return notes_by_stay, signals_by_stay, mortality_by_stay, subject_by_stay


def _build_graph_list(
    stay_ids: list[int],
    notes_by_stay: dict,
    signals_by_stay: dict,
    mortality_by_stay: dict,
    note_embeddings: dict,
) -> list:
    graphs, skipped = [], 0
    for sid in stay_ids:
        data = build_patient_graph(
            sid,
            notes_by_stay.get(sid, []),
            signals_by_stay.get(sid, []),
            note_embeddings,
        )
        if data is None:
            skipped += 1
            continue
        data.y = torch.tensor(float(mortality_by_stay[sid]), dtype=torch.float32)
        graphs.append(data)
    if skipped:
        print(f"  [builder] skipped {skipped}/{len(stay_ids)} stays (no valid nodes)")
    return graphs


# ── Dataset builder (with disk cache) ─────────────────────────────────────

def build_datasets(
    csv_path: Path,
    embeddings_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    cache_path: Path | None = None,
) -> tuple[list, list, float]:
    """Build subject-disjoint train/val graph lists.

    Returns (train_graphs, val_graphs, pos_weight).
    """
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached graphs from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached["train"], cached["val"], cached["pos_weight"]

    print("Pre-processing CSV…")
    notes_by_stay, signals_by_stay, mortality_by_stay, subject_by_stay = (
        _preprocess_csv(csv_path)
    )
    print(f"  {len(mortality_by_stay):,} stays | {len(notes_by_stay):,} with notes")

    print(f"Loading embeddings from {embeddings_path}…")
    note_embeddings = load_note_embeddings(embeddings_path)
    print(f"  {len(note_embeddings):,} note embeddings")

    # Subject-disjoint train/val split
    all_subjects = sorted(set(subject_by_stay.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_subjects)
    n_train = int(len(all_subjects) * train_ratio)
    train_subj = set(all_subjects[:n_train])

    train_stays = [s for s, subj in subject_by_stay.items() if subj in train_subj]
    val_stays = [s for s, subj in subject_by_stay.items() if subj not in train_subj]
    print(f"  split: {len(train_stays):,} train stays | {len(val_stays):,} val stays")

    print("Building train graphs…")
    train_graphs = _build_graph_list(
        train_stays, notes_by_stay, signals_by_stay, mortality_by_stay, note_embeddings
    )
    print("Building val graphs…")
    val_graphs = _build_graph_list(
        val_stays, notes_by_stay, signals_by_stay, mortality_by_stay, note_embeddings
    )

    n_pos = sum(int(g.y.item()) for g in train_graphs)
    n_neg = len(train_graphs) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(
        f"  train: {n_pos:,} pos / {n_neg:,} neg  |  pos_weight = {pos_weight:.2f}"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching to {cache_path}…")
        torch.save(
            {"train": train_graphs, "val": val_graphs, "pos_weight": pos_weight},
            cache_path,
        )

    return train_graphs, val_graphs, pos_weight


# ── Loss ───────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """BCE with focal down-weighting of easy examples.

    gamma=0 → plain BCE (with pos_weight).
    gamma=2 → standard focal loss; hard positives dominate, easy negatives
              contribute little → directly improves AUPRC on imbalanced data.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if self.gamma == 0.0:
            return bce.mean()
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (((1 - p_t) ** self.gamma) * bce).mean()


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Returns (val_loss, val_auroc). Uses unweighted BCE for val_loss."""
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
    except ValueError:
        val_auroc = 0.5  # fallback if only one class in batch

    return val_loss, val_auroc


# ── Training ───────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("data/snapshots") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_id}")

    cache = Path(args.cache_path) if args.cache_path else None
    train_graphs, val_graphs, pos_weight = build_datasets(
        Path(args.csv_path),
        Path(args.embeddings_path),
        train_ratio=args.train_ratio,
        seed=args.seed,
        cache_path=cache,
    )
    print(f"Graphs: {len(train_graphs):,} train | {len(val_graphs):,} val")

    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = TemporalPatientGNN(
        node_dim=64,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    pw_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pw_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    cfg = {**vars(args), "run_id": run_id, "pos_weight": round(pos_weight, 3)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

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
        val_loss, val_auroc = evaluate(model, val_loader, device)
        scheduler.step(val_auroc)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Ep {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| AUROC {val_auroc:.4f} | lr {lr_now:.1e} | {time.time()-t0:.1f}s"
        )
        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "val_auroc": round(val_auroc, 4),
            "lr": lr_now,
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

    # Additional metrics on val set with best model
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    _, best_auroc_check = evaluate(model, val_loader, device)

    # Full metric suite
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            all_logits.append(model(batch).cpu())
            all_labels.append(batch.y.cpu())
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    final_metrics = {
        "auroc": round(M.auroc(labels_np, probs), 4),
        "auprc": round(M.auprc(labels_np, probs), 4),
        "brier": round(M.brier_score(labels_np, probs), 4),
        "sens@95spec": round(M.sens_at_spec(labels_np, probs, 0.95), 4),
    }
    print("\nFinal val metrics (best checkpoint):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")

    (run_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))

    with open(run_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_auroc","lr"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nBest val AUROC: {best_auroc:.4f}")
    print(f"Run saved: {run_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train Phase 2 Temporal GNN")
    p.add_argument("--csv-path", default="data/processed/pairs_all-icus_note_level.csv")
    p.add_argument("--embeddings-path", default="data/embeddings/node_embeddings.pt")
    p.add_argument("--cache-path", default="data/processed/graphs_cache.pt",
                   help="Cache built graph list to this path (speeds up reruns)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pooling", default="mean", choices=["mean", "attention", "dual"])
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="Focal loss gamma (0=plain BCE, 2=standard focal)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
