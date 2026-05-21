"""
Graph dataset construction for Phase 2 GNN training.

Public API:
    build_datasets(csv_path, embeddings_path, *, ...) -> (train, val, pos_weight)

Optional features:
    all_stay_signals_path  — full-stay signals from extract_all_stay_signals.py
                             replaces the ±2h paired signals with the full stay trajectory
    icd_path               — Charlson comorbidity CSV from extract_icd.py
                             adds ICD node (node_type=2) to each graph

End-to-end variant (build_datasets_e2e) has been moved to src/experimental/e2e.py.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data_prep.extract_icd import CHARLSON_COLS
from src.utils.graph_builder import (
    DEMO_DIM,
    N_CHARLSON,
    build_patient_graph,
    load_note_embeddings,
)


def _preprocess_csv(csv_path: Path) -> tuple[dict, dict, dict, dict]:
    """Parse pairs CSV into per-stay Python dicts.

    Returns:
        notes_by_stay:    {stay_id -> [{note_id, note_time}]}
        signals_by_stay:  {stay_id -> [{norm_value, item_type_id, event_hours_from_intime}]}
        mortality_by_stay:{stay_id -> int}
        subject_by_stay:  {stay_id -> int}
    """
    df = pl.read_csv(csv_path)
    df = df.with_columns(
        (pl.col("event_hours_from_intime") + pl.col("delta_hours_to_note")).alias("note_hours")
    )

    notes_df = df.group_by(["stay_id", "note_id"]).agg(pl.first("note_hours"))
    signals_df = df.select(
        ["stay_id", "event_hours_from_intime", "item_type_id", "norm_value"]
    ).unique(["stay_id", "event_hours_from_intime", "item_type_id"])
    meta_df = df.group_by("stay_id").agg(pl.first("mortality"), pl.first("subject_id"))

    notes_by_stay: dict[int, list] = defaultdict(list)
    for r in notes_df.iter_rows(named=True):
        notes_by_stay[r["stay_id"]].append(
            {
                "note_id": r["note_id"],
                "note_time": float(r["note_hours"]),
            }
        )

    signals_by_stay: dict[int, list] = defaultdict(list)
    for r in signals_df.iter_rows(named=True):
        signals_by_stay[r["stay_id"]].append(
            {
                "norm_value": float(r["norm_value"]),
                "item_type_id": int(r["item_type_id"]),
                "event_hours_from_intime": float(r["event_hours_from_intime"]),
            }
        )

    mortality_by_stay: dict[int, int] = {}
    subject_by_stay: dict[int, int] = {}
    for r in meta_df.iter_rows(named=True):
        mortality_by_stay[r["stay_id"]] = int(r["mortality"])
        subject_by_stay[r["stay_id"]] = int(r["subject_id"])

    return notes_by_stay, signals_by_stay, mortality_by_stay, subject_by_stay


def _load_demo(demo_path: Path) -> dict[int, torch.Tensor]:
    """Load demographics CSV → {stay_id -> Tensor(DEMO_DIM,)}."""
    df = pl.read_csv(demo_path)
    demo: dict[int, torch.Tensor] = {}
    for r in df.iter_rows(named=True):
        feat = torch.tensor(
            [
                r["age_norm"],
                float(r["gender_f"]),
                float(r["is_emergency"]),
                float(r["is_elective"]),
            ],
            dtype=torch.float32,
        )
        demo[int(r["stay_id"])] = feat
    print(f"  {len(demo):,} stays with demographic features")
    return demo


def _load_icd(icd_path: Path) -> dict[int, torch.Tensor]:
    """Load Charlson comorbidity CSV → {stay_id -> Tensor(N_CHARLSON,)}."""
    df = pl.read_csv(icd_path)
    icd: dict[int, torch.Tensor] = {}
    for r in df.iter_rows(named=True):
        feat = torch.tensor(
            [float(r[c]) for c in CHARLSON_COLS],
            dtype=torch.float32,
        )
        icd[int(r["stay_id"])] = feat
    n_any = sum(1 for v in icd.values() if v.sum() > 0)
    print(
        f"  {len(icd):,} stays with ICD data | {n_any:,} ({n_any / max(len(icd), 1):.1%}) with ≥1 Charlson code"
    )
    return icd


def _load_all_stay_signals(path: Path) -> dict[int, list[dict]]:
    """Load full-stay signals CSV → {stay_id -> [{norm_value, item_type_id, event_hours_from_intime}]}."""
    df = pl.read_csv(path)
    signals: dict[int, list] = defaultdict(list)
    for r in df.iter_rows(named=True):
        signals[int(r["stay_id"])].append(
            {
                "norm_value": float(r["norm_value"]),
                "item_type_id": int(r["item_type_id"]),
                "event_hours_from_intime": float(r["event_hours_from_intime"]),
            }
        )
    print(
        f"  {sum(len(v) for v in signals.values()):,} signal events across {len(signals):,} stays"
    )
    return signals


def _build_graph_list(
    stay_ids: list[int],
    notes_by_stay: dict,
    signals_by_stay: dict,
    mortality_by_stay: dict,
    note_embeddings: dict,
    *,
    signal_only: bool = False,
    demo: dict[int, torch.Tensor] | None = None,
    icd: dict[int, torch.Tensor] | None = None,
    all_stay_signals: dict[int, list] | None = None,
    max_signals: int | None = None,
) -> list:
    """Build a PyG Data list for the given stays."""
    graphs, skipped = [], 0
    _zero_demo = torch.zeros(DEMO_DIM) if demo is not None else None
    _zero_icd = torch.zeros(N_CHARLSON) if icd is not None else None

    for sid in stay_ids:
        note_rows = [] if signal_only else notes_by_stay.get(sid, [])
        emb_dict = {} if signal_only else note_embeddings
        demo_feat = demo.get(sid, _zero_demo) if demo is not None else None
        icd_feat = icd.get(sid, _zero_icd) if icd is not None else None

        # Use full-stay signals if provided, otherwise fall back to paired signals
        sig_rows = (
            all_stay_signals.get(sid, [])
            if all_stay_signals is not None
            else signals_by_stay.get(sid, [])
        )

        # When using all-stay signals: keep only first max_signals events (earliest,
        # not latest) so the model predicts from early-stay data, not terminal vitals.
        if max_signals is not None and all_stay_signals is not None and len(sig_rows) > max_signals:
            sig_rows = sorted(sig_rows, key=lambda r: r["event_hours_from_intime"])[:max_signals]

        data = build_patient_graph(
            sid,
            note_rows,
            sig_rows,
            emb_dict,
            demo_feat=demo_feat,
            icd_feat=icd_feat,
        )
        if data is None:
            skipped += 1
            continue
        data.y = torch.tensor(float(mortality_by_stay[sid]), dtype=torch.float32)
        graphs.append(data)

    if skipped:
        print(f"  [builder] skipped {skipped}/{len(stay_ids)} stays (no valid nodes)")
    return graphs


def _subject_split(
    subject_by_stay: dict[int, int],
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    all_subjects = sorted(set(subject_by_stay.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_subjects)
    n_train = int(len(all_subjects) * train_ratio)
    train_subj = set(all_subjects[:n_train])
    train_stays = [s for s, subj in subject_by_stay.items() if subj in train_subj]
    val_stays = [s for s, subj in subject_by_stay.items() if subj not in train_subj]
    return train_stays, val_stays


def build_datasets(
    csv_path: Path,
    embeddings_path: Path,
    *,
    signal_only: bool = False,
    demo_path: Path | None = None,
    icd_path: Path | None = None,
    all_stay_signals_path: Path | None = None,
    max_signals: int | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    cache_path: Path | None = None,
) -> tuple[list, list, float]:
    """Build subject-disjoint train/val graph lists (frozen embeddings mode).

    Args:
        signal_only:            Drop note nodes (signal-only ablation).
        demo_path:              Demographics CSV → demographic features appended after pooling.
        icd_path:               Charlson ICD CSV → ICD node (node_type=2) added per stay.
        all_stay_signals_path:  Full-stay signals CSV → replaces paired signals.
                                If None, uses ±2h paired signals from csv_path.

    Returns (train_graphs, val_graphs, pos_weight).

    NOTE: Existing caches built with SIGNAL_RAW_DIM=10 are incompatible with
    the current SIGNAL_RAW_DIM=16. Delete old .pt caches before first run.
    """
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached graphs from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached["train"], cached["val"], cached["pos_weight"]

    print("Pre-processing CSV…")
    notes_by_stay, signals_by_stay, mortality_by_stay, subject_by_stay = _preprocess_csv(csv_path)
    print(f"  {len(mortality_by_stay):,} stays | {len(notes_by_stay):,} with notes")

    note_embeddings: dict = {}
    if not signal_only:
        print(f"Loading embeddings from {embeddings_path}…")
        note_embeddings = load_note_embeddings(embeddings_path)
        print(f"  {len(note_embeddings):,} note embeddings")
    else:
        print("  signal-only mode — skipping note embeddings")

    demo: dict | None = None
    if demo_path is not None:
        print(f"Loading demographics from {demo_path}…")
        demo = _load_demo(demo_path)

    icd: dict | None = None
    if icd_path is not None:
        print(f"Loading ICD comorbidities from {icd_path}…")
        icd = _load_icd(icd_path)

    all_stay_signals: dict | None = None
    if all_stay_signals_path is not None:
        print(f"Loading full-stay signals from {all_stay_signals_path}…")
        all_stay_signals = _load_all_stay_signals(all_stay_signals_path)

    train_stays, val_stays = _subject_split(subject_by_stay, train_ratio, seed)
    print(f"  split: {len(train_stays):,} train | {len(val_stays):,} val")

    if max_signals is not None and all_stay_signals_path is not None:
        print(f"  capping signals to earliest {max_signals} per stay")
    print("Building train graphs…")
    train_graphs = _build_graph_list(
        train_stays,
        notes_by_stay,
        signals_by_stay,
        mortality_by_stay,
        note_embeddings,
        signal_only=signal_only,
        demo=demo,
        icd=icd,
        all_stay_signals=all_stay_signals,
        max_signals=max_signals,
    )
    print("Building val graphs…")
    val_graphs = _build_graph_list(
        val_stays,
        notes_by_stay,
        signals_by_stay,
        mortality_by_stay,
        note_embeddings,
        signal_only=signal_only,
        demo=demo,
        icd=icd,
        all_stay_signals=all_stay_signals,
        max_signals=max_signals,
    )

    n_pos = sum(int(g.y.item()) for g in train_graphs)
    n_neg = len(train_graphs) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  train: {n_pos:,} pos / {n_neg:,} neg  |  pos_weight = {pos_weight:.3f}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching to {cache_path}…")
        torch.save(
            {"train": train_graphs, "val": val_graphs, "pos_weight": pos_weight},
            cache_path,
        )

    return train_graphs, val_graphs, pos_weight
