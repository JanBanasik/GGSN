"""
Phase 1 — Dataset for contrastive pre-training.

Extracted from train_contrastive.py for modularity.

Public API:
    ContrastivePairsDataset  — torch Dataset, one sample per unique note_id
    split_indices_by_subject — subject-disjoint train/val split
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from torch.utils.data import Dataset


class ContrastivePairsDataset(Dataset):
    """
    Reads pairs_*.csv (one row per note × signal pair) and collapses to
    one sample per unique note_id.

    Each sample:
        input_ids       (max_text_len,) long
        attention_mask  (max_text_len,) long
        item_ids        (seq_len,)      long  — signal type indices
        values          (seq_len,)      float — normalised values in [0, 1]
        signal_mask     (seq_len,)      float — 1 for real event, 0 for pad
        hours           (seq_len,)      float — event_hours_from_intime / 24
        deltas          (seq_len,)      float — Δt to note_time (hours, signed)
        note_id         str
        subject_id      int
        mortality       int
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        max_text_len: int = 256,
        seq_len: int = 64,
    ) -> None:
        df = pl.read_csv(csv_path)
        has_hours = "event_hours_from_intime" in df.columns
        has_delta = "delta_hours_to_note" in df.columns

        per_note: dict[str, dict] = {}
        for row in df.sort("event_time").iter_rows(named=True):
            nid = row["note_id"]
            entry = per_note.setdefault(
                nid,
                {
                    "text": row["text"],
                    "subject_id": int(row["subject_id"]),
                    "mortality": int(row["mortality"]),
                    "items": [],
                    "values": [],
                    "hours": [],
                    "deltas": [],
                },
            )
            entry["items"].append(int(row["item_type_id"]))
            entry["values"].append(float(row["norm_value"]))
            entry["hours"].append(
                float(row["event_hours_from_intime"]) / 24.0 if has_hours else 0.0
            )
            entry["deltas"].append(float(row["delta_hours_to_note"]) if has_delta else 0.0)

        self.max_text_len = max_text_len
        self.seq_len = seq_len
        self.has_hours = has_hours
        self.has_delta = has_delta

        self.note_ids: list[str] = []
        self.encodings: list[dict] = []
        self.item_ids: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.hours: list[torch.Tensor] = []
        self.deltas: list[torch.Tensor] = []
        self.signal_masks: list[torch.Tensor] = []
        self.subject_ids: list[int] = []
        self.mortality: list[int] = []

        print(
            f"  Tokenising {len(per_note)} unique notes… (hours: {has_hours}, delta: {has_delta})"
        )
        for nid, entry in per_note.items():
            enc = tokenizer(
                entry["text"],
                max_length=max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            items = entry["items"][:seq_len]
            values = entry["values"][:seq_len]
            hours_seq = entry["hours"][:seq_len]
            deltas_seq = entry["deltas"][:seq_len]
            real_n = len(items)
            pad_n = seq_len - real_n
            if pad_n > 0:
                items = items + [0] * pad_n
                values = values + [0.0] * pad_n
                hours_seq = hours_seq + [0.0] * pad_n
                deltas_seq = deltas_seq + [0.0] * pad_n
            mask = [1.0] * real_n + [0.0] * pad_n

            self.note_ids.append(nid)
            self.encodings.append(
                {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                }
            )
            self.item_ids.append(torch.tensor(items, dtype=torch.long))
            self.values.append(torch.tensor(values, dtype=torch.float32))
            self.hours.append(torch.tensor(hours_seq, dtype=torch.float32))
            self.deltas.append(torch.tensor(deltas_seq, dtype=torch.float32))
            self.signal_masks.append(torch.tensor(mask, dtype=torch.float32))
            self.subject_ids.append(entry["subject_id"])
            self.mortality.append(entry["mortality"])

    def __len__(self) -> int:
        return len(self.note_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "item_ids": self.item_ids[idx],
            "values": self.values[idx],
            "hours": self.hours[idx],
            "deltas": self.deltas[idx],
            "signal_mask": self.signal_masks[idx],
            "note_id": self.note_ids[idx],
            "subject_id": self.subject_ids[idx],
            "mortality": self.mortality[idx],
        }


def split_indices_by_subject(
    dataset: ContrastivePairsDataset, val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """
    Train/val split keeping all notes from one subject in the same split
    (prevents subject-level leakage during contrastive training).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    subjects = sorted(set(dataset.subject_ids))
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_frac))
    val_subjects = set(subjects[:n_val])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, sid in enumerate(dataset.subject_ids):
        (val_idx if sid in val_subjects else train_idx).append(i)
    return train_idx, val_idx
