"""
Graph dataset construction for Phase 2 GNN training.

Public API:
    build_datasets(csv_path, embeddings_path, *, signal_only, ...) -> (train, val, pos_weight)
    build_datasets_e2e(csv_path, ...) -> (train, val, pos_weight)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.utils.graph_builder import (
    build_patient_graph,
    build_patient_graph_e2e,
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

    notes_df = (
        df.group_by(["stay_id", "note_id"])
        .agg(pl.first("note_hours"))
    )
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
    *,
    signal_only: bool = False,
) -> list:
    """Build a PyG Data list for the given stays.

    signal_only=True drops all note nodes — useful for the signal-only ablation.
    """
    graphs, skipped = [], 0
    for sid in stay_ids:
        note_rows = [] if signal_only else notes_by_stay.get(sid, [])
        emb_dict = {} if signal_only else note_embeddings
        data = build_patient_graph(
            sid,
            note_rows,
            signals_by_stay.get(sid, []),
            emb_dict,
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
    train_ratio: float = 0.8,
    seed: int = 42,
    cache_path: Path | None = None,
) -> tuple[list, list, float]:
    """Build subject-disjoint train/val graph lists (frozen embeddings mode).

    signal_only=True: build graphs with only signal nodes (no Phase 1 embeddings).
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

    note_embeddings: dict = {}
    if not signal_only:
        print(f"Loading embeddings from {embeddings_path}…")
        note_embeddings = load_note_embeddings(embeddings_path)
        print(f"  {len(note_embeddings):,} note embeddings")
    else:
        print("  signal-only mode — skipping note embeddings")

    train_stays, val_stays = _subject_split(subject_by_stay, train_ratio, seed)
    print(f"  split: {len(train_stays):,} train | {len(val_stays):,} val")

    print("Building train graphs…")
    train_graphs = _build_graph_list(
        train_stays, notes_by_stay, signals_by_stay, mortality_by_stay,
        note_embeddings, signal_only=signal_only,
    )
    print("Building val graphs…")
    val_graphs = _build_graph_list(
        val_stays, notes_by_stay, signals_by_stay, mortality_by_stay,
        note_embeddings, signal_only=signal_only,
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


def build_datasets_e2e(
    csv_path: Path,
    tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_len: int = 256,
    train_ratio: float = 0.8,
    seed: int = 42,
    cache_path: Path | None = None,
) -> tuple[list, list, float]:
    """Build train/val graph lists with tokenised note text (for e2e fine-tuning).

    Returns (train_graphs, val_graphs, pos_weight).
    """
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached e2e graphs from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached["train"], cached["val"], cached["pos_weight"]

    from transformers import AutoTokenizer

    print(f"Loading tokenizer {tokenizer_name}…")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Pre-processing CSV…")
    notes_by_stay, signals_by_stay, mortality_by_stay, subject_by_stay = (
        _preprocess_csv(csv_path)
    )

    print("Collecting note texts…")
    df = pl.read_csv(csv_path).with_columns(
        (pl.col("event_hours_from_intime") + pl.col("delta_hours_to_note")).alias("note_hours")
    )
    note_text_df = (
        df.group_by(["stay_id", "note_id"])
        .agg(pl.first("text"), pl.first("note_hours"))
    )
    note_texts: dict[str, str] = {
        r["note_id"]: r["text"] or ""
        for r in note_text_df.iter_rows(named=True)
    }

    print(f"  {len(note_texts):,} unique notes — tokenising (max_len={max_len})…")
    note_ids = list(note_texts.keys())
    enc = tokenizer(
        [note_texts[nid] for nid in note_ids],
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    note_tokens: dict[str, tuple] = {
        nid: (enc["input_ids"][i], enc["attention_mask"][i])
        for i, nid in enumerate(note_ids)
    }
    print("  tokenisation done.")

    notes_with_tokens: dict[int, list] = defaultdict(list)
    for stay_id, nrows in notes_by_stay.items():
        for r in nrows:
            nid = r["note_id"]
            if nid not in note_tokens:
                continue
            ids, mask = note_tokens[nid]
            notes_with_tokens[stay_id].append({
                "note_id": nid,
                "note_time": r["note_time"],
                "input_ids": ids,
                "attn_mask": mask,
            })

    train_stays, val_stays = _subject_split(subject_by_stay, train_ratio, seed)
    print(f"  split: {len(train_stays):,} train | {len(val_stays):,} val")

    def _build(stay_ids: list[int]) -> list:
        graphs, skipped = [], 0
        for sid in stay_ids:
            data = build_patient_graph_e2e(
                sid,
                notes_with_tokens.get(sid, []),
                signals_by_stay.get(sid, []),
            )
            if data is None:
                skipped += 1
                continue
            data.y = torch.tensor(float(mortality_by_stay[sid]), dtype=torch.float32)
            graphs.append(data)
        if skipped:
            print(f"  [builder] skipped {skipped} stays (no valid nodes)")
        return graphs

    print("Building train graphs…")
    train_graphs = _build(train_stays)
    print("Building val graphs…")
    val_graphs = _build(val_stays)

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
