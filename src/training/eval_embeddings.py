"""
Evaluate embedding quality for the contrastive pre-training (Phase 1).

Compares two checkpoints (by default epoch 0 vs epoch 16) on the val split:
  - Linear probe AUROC  (LogisticRegression, mortality label)
  - KMeans cluster purity (k=2)
  - Diagonal similarity gap
  - UMAP 2D scatter (coloured by mortality)

Also plots the val-loss learning curve with reference baselines.

Usage:
    uv run python -m src.training.eval_embeddings \\
        --run-dir data/snapshots/run_<timestamp> \\
        --epoch-a 0 --epoch-b 16 \\
        --out data/plots/embedding_comparison_v4.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_LN64 = 4.1589  # ln(64) — effective batch 64


def load_val_epoch(run_dir: Path, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) for val set from a per-epoch snapshot."""
    folder = run_dir / f"epoch_{epoch:03d}"
    emb: dict[str, torch.Tensor] = torch.load(folder / "val_embeddings.pt", map_location="cpu")
    labels: dict[str, int] = json.loads((folder / "val_labels.json").read_text())

    stay_ids = sorted(emb.keys())
    X = torch.stack([emb[k] for k in stay_ids]).numpy().astype(np.float32)
    y = np.array([labels[k] for k in stay_ids], dtype=np.int32)
    return X, y


def linear_probe_auroc(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """5-fold stratified CV AUROC on the provided set (standard embedding eval protocol)."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)),
        ]
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aurocs = []
    for tr_idx, te_idx in skf.split(X, y):
        pipe.fit(X[tr_idx], y[tr_idx])
        proba = pipe.predict_proba(X[te_idx])[:, 1]
        aurocs.append(roc_auc_score(y[te_idx], proba))
    return float(np.mean(aurocs))


def kmeans_purity(X: np.ndarray, y: np.ndarray, k: int = 2) -> float:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(X)
    total = len(y)
    purity = 0.0
    for c in range(k):
        mask = clusters == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(y[mask], minlength=2)
        purity += counts.max()
    return float(purity / total)


def diagonal_gap(X: np.ndarray) -> float:
    """Mean cosine similarity of the matrix diag vs off-diag (proxy using text-only embeddings)."""
    norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T  # (N, N)
    n = len(sim)
    diag_mean = float(np.diag(sim).mean())
    mask = ~np.eye(n, dtype=bool)
    offdiag_mean = float(sim[mask].mean())
    return diag_mean - offdiag_mean


def umap_2d(X: np.ndarray, y: np.ndarray, n_sample: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    import umap as umap_lib

    rng = np.random.default_rng(42)
    if len(X) > n_sample:
        idx = rng.choice(len(X), n_sample, replace=False)
        X, y = X[idx], y[idx]
    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    return reducer.fit_transform(X), y


def plot_comparison(
    run_dir: Path,
    epoch_a: int,
    epoch_b: int,
    metrics_a: dict,
    metrics_b: dict,
    umap_a: tuple[np.ndarray, np.ndarray],  # (coords, labels)
    umap_b: tuple[np.ndarray, np.ndarray],
    out_path: Path,
) -> None:
    log_csv = run_dir / "log.csv"
    log = pd.read_csv(log_csv) if log_csv.exists() else None

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle(
        "Phase 1 — Contrastive Pre-training: Embedding Quality", fontsize=14, fontweight="bold"
    )

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_auroc = fig.add_subplot(gs[0, 1])
    ax_purity = fig.add_subplot(gs[0, 2])
    ax_umap_a = fig.add_subplot(gs[1, 0])
    ax_umap_b = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[1, 2])

    if log is not None:
        ax_curve.plot(log["epoch"], log["train_loss"], label="train", color="#4C72B0", lw=2)
        ax_curve.plot(log["epoch"], log["val_loss"], label="val", color="#DD8452", lw=2)
        ax_curve.axhline(
            BASELINE_LN64, ls="--", color="gray", lw=1, label=f"baseline ln(64)={BASELINE_LN64:.2f}"
        )
        ax_curve.axvline(epoch_b, ls=":", color="#2ca02c", lw=1.5, label=f"best ep.{epoch_b}")
        ax_curve.set_xlabel("Epoch")
        ax_curve.set_ylabel("InfoNCE Loss")
        ax_curve.set_title("Val Loss — Phase 1 v4 (note-level, all-icus)")
        ax_curve.legend(fontsize=8)
        ax_curve.set_ylim(bottom=min(log["val_loss"].min() * 0.95, 2.2))

    auroc_vals = [metrics_a["auroc"], metrics_b["auroc"]]
    auroc_labels = [f"Epoch {epoch_a}\n(pre-training)", f"Epoch {epoch_b}\n(best)"]
    colors_auroc = ["#9ecae1", "#2171b5"]
    bars = ax_auroc.bar(auroc_labels, auroc_vals, color=colors_auroc, width=0.5)
    ax_auroc.axhline(0.5, ls=":", color="gray", lw=1, label="random 0.50")
    for bar, val in zip(bars, auroc_vals, strict=False):
        ax_auroc.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_auroc.set_ylim(0.45, min(max(auroc_vals) + 0.07, 1.0))
    ax_auroc.set_ylabel("AUROC (mortality)")
    ax_auroc.set_title("Linear Probe AUROC")
    ax_auroc.legend(fontsize=8)

    purity_vals = [metrics_a["purity"], metrics_b["purity"]]
    bars_p = ax_purity.bar(auroc_labels, purity_vals, color=colors_auroc, width=0.5)
    ax_purity.axhline(1 - 0.1246, ls=":", color="gray", lw=1, label="majority-class 0.875")
    for bar, val in zip(bars_p, purity_vals, strict=False):
        ax_purity.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_purity.set_ylim(0.5, 1.0)
    ax_purity.set_ylabel("KMeans Purity (k=2)")
    ax_purity.set_title("Cluster Purity")
    ax_purity.legend(fontsize=7)

    cmap = {0: "#4878D0", 1: "#EE854A"}
    for ax, (coords, lbls), ep in [(ax_umap_a, umap_a, epoch_a), (ax_umap_b, umap_b, epoch_b)]:
        for lbl, color in cmap.items():
            mask = lbls == lbl
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=4,
                alpha=0.4,
                label=("Died" if lbl == 1 else "Survived"),
                rasterized=True,
            )
        ax.set_title(f"UMAP — Epoch {ep}")
        ax.legend(fontsize=8, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_table.axis("off")
    table_data = [
        ["Metric", f"Ep. {epoch_a}", f"Ep. {epoch_b} (best)"],
        ["InfoNCE val", "—", f"{metrics_b.get('infonce_val', '—')}"],
        ["Baseline ln(64)", "—", f"{BASELINE_LN64:.3f}"],
        ["% reduction", "—", f"{metrics_b.get('reduction_pct', '—')}"],
        ["AUROC", f"{metrics_a['auroc']:.3f}", f"{metrics_b['auroc']:.3f}"],
        ["KMeans purity", f"{metrics_a['purity']:.3f}", f"{metrics_b['purity']:.3f}"],
        ["Diag gap", f"{metrics_a['diag_gap']:.3f}", f"{metrics_b['diag_gap']:.3f}"],
    ]
    tbl = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax_table.set_title("Summary", pad=14)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    epoch_a, epoch_b = args.epoch_a, args.epoch_b
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_csv = run_dir / "log.csv"
    best_val = None
    if log_csv.exists():
        log = pd.read_csv(log_csv)
        best_row = log.loc[log["val_loss"].idxmin()]
        best_val = float(best_row["val_loss"])
        print(f"Best val loss: {best_val:.4f} at epoch {int(best_row['epoch'])}")
        print(f"InfoNCE reduction: {(BASELINE_LN64 - best_val) / BASELINE_LN64 * 100:.1f}%")

    print(f"\nLoading val embeddings: epoch {epoch_a} …")
    X_a, y_a = load_val_epoch(run_dir, epoch_a)
    print(f"Loading val embeddings: epoch {epoch_b} …")
    X_b, y_b = load_val_epoch(run_dir, epoch_b)

    assert np.array_equal(y_a, y_b), "Labels mismatch between epochs — unexpected"

    print(f"\nVal set: {len(y_a)} stays, mortality rate = {y_a.mean():.3f}")
    print(f"Embedding dim: {X_a.shape[1]}")

    print("\nComputing metrics for epoch A (5-fold CV) …")
    auroc_a = linear_probe_auroc(X_a, y_a)
    purity_a = kmeans_purity(X_a, y_a)
    dgap_a = diagonal_gap(X_a)

    print("Computing metrics for epoch B (5-fold CV) …")
    auroc_b = linear_probe_auroc(X_b, y_b)
    purity_b = kmeans_purity(X_b, y_b)
    dgap_b = diagonal_gap(X_b)

    metrics_a = {"auroc": auroc_a, "purity": purity_a, "diag_gap": dgap_a}
    metrics_b = {
        "auroc": auroc_b,
        "purity": purity_b,
        "diag_gap": dgap_b,
        "infonce_val": f"{best_val:.4f}" if best_val else "—",
        "reduction_pct": (
            f"{(BASELINE_LN64 - best_val) / BASELINE_LN64 * 100:.1f}%" if best_val else "—"
        ),
    }

    print(f"\n{'Metric':<20} {'Epoch ' + str(epoch_a):<16} {'Epoch ' + str(epoch_b):<16}")
    print("-" * 52)
    print(f"{'AUROC':<20} {auroc_a:<16.4f} {auroc_b:<16.4f}")
    print(f"{'KMeans purity':<20} {purity_a:<16.4f} {purity_b:<16.4f}")
    print(f"{'Diagonal gap':<20} {dgap_a:<16.4f} {dgap_b:<16.4f}")
    if best_val:
        print(
            f"\nInfoNCE val (best ep.{epoch_b}): {best_val:.4f}  "
            f"[{(BASELINE_LN64 - best_val) / BASELINE_LN64 * 100:.1f}% reduction vs baseline]"
        )

    print("\nRunning UMAP (this takes ~1-2 min) …")
    coords_a, y_umap_a = umap_2d(X_a, y_a)
    coords_b, y_umap_b = umap_2d(X_b, y_b)

    plot_comparison(
        run_dir=run_dir,
        epoch_a=epoch_a,
        epoch_b=epoch_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        umap_a=(coords_a, y_umap_a),
        umap_b=(coords_b, y_umap_b),
        out_path=out_path,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Phase 1 embedding quality")
    p.add_argument(
        "--run-dir",
        required=True,
        help="Path to a training run directory containing epoch_XXX/ subdirectories.",
    )
    p.add_argument(
        "--epoch-a", type=int, default=0, help="'Before' epoch (default: 0 = pre-training)"
    )
    p.add_argument(
        "--epoch-b",
        type=int,
        default=16,
        help="'After' epoch (default: 16 = best val InfoNCE of v4)",
    )
    p.add_argument(
        "--out", default="data/plots/embedding_comparison_v4.png", help="Output PNG path"
    )
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        p.error(f"run directory does not exist: {run_dir}")
    return args


if __name__ == "__main__":
    main(_parse_args())
