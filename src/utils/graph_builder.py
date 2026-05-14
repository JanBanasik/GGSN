"""
Phase 2 — Graph builder: turn one ICU stay into a PyG Data object.

Node types (both projected to NODE_DIM before passing to GNN):
  - Signal nodes  (~52 / stay median): raw features [type_emb(8) | norm_value | hours/24]
  - Note nodes    (~2  / stay median): text_tower embedding (128-D), loaded from embeddings file

Edges: directed temporal — (i → j) for every pair where time_i < time_j.
Edge attr: (t_j - t_i) / 24  ∈ [0, 1]  (normalized hours).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data


NODE_DIM = 64          # projected node feature dim (shared for both node types)
SIGNAL_RAW_DIM = 10    # type_emb(8) + norm_value(1) + hours(1)  [frozen-emb mode]
SIGNAL_VAL_DIM = 2     # norm_value(1) + hours(1)                [e2e mode]
NOTE_EMB_DIM = 128     # text_tower output dim
DEMO_DIM = 4           # age_norm, gender_f, is_emergency, is_elective


def build_patient_graph(
    stay_id: int,
    note_rows: list[dict],
    signal_rows: list[dict],
    note_embeddings: dict[str, torch.Tensor],
    demo_feat: torch.Tensor | None = None,
) -> Data | None:
    """
    Build a temporal graph for one ICU stay.

    Args:
        stay_id:         ICU stay identifier (int).
        note_rows:       List of dicts with keys: note_id, note_time (float hours from intime).
        signal_rows:     List of dicts with keys: norm_value, item_type_id,
                         event_hours_from_intime.
        note_embeddings: {note_id -> Tensor(128,)} from Phase 1 text_tower.
        demo_feat:       Optional Tensor(DEMO_DIM,) of graph-level demographic features.
                         Stored as shape (1, DEMO_DIM) so PyG batches it to (B, DEMO_DIM).

    Returns:
        PyG Data with:
            x           (N, NOTE_EMB_DIM) — note nodes full / signal nodes zero-padded
            node_type   (N,) int  0=signal  1=note
            timestamps  (N,) float hours/24
            edge_index  (2, E) directed temporal edges
            edge_attr   (E, 1) Δt / 24
            demo        (1, DEMO_DIM) if demo_feat provided, else absent
        Returns None if stay has no valid nodes.
    """
    events: list[dict] = []

    # Note nodes
    for r in note_rows:
        nid = r["note_id"]
        if nid not in note_embeddings:
            continue
        events.append({
            "type": 1,
            "time": float(r["note_time"]),
            "feat": note_embeddings[nid],         # Tensor(128,)
        })

    # Signal nodes
    for r in signal_rows:
        feat = torch.zeros(SIGNAL_RAW_DIM)
        type_id = int(r["item_type_id"])
        feat[type_id % 8] = 1.0                  # crude one-hot for first 8 dims
        feat[8] = float(r["norm_value"])
        feat[9] = float(r["event_hours_from_intime"]) / 24.0
        events.append({
            "type": 0,
            "time": float(r["event_hours_from_intime"]),
            "feat": feat,                         # Tensor(10,)
        })

    if not events:
        return None

    # Sort chronologically
    events.sort(key=lambda e: e["time"])
    n = len(events)

    # Build node feature tensors (keep raw dims; GNN model projects them)
    node_type = torch.tensor([e["type"] for e in events], dtype=torch.long)
    timestamps = torch.tensor([e["time"] / 24.0 for e in events], dtype=torch.float32)

    # Pad all features to max(NOTE_EMB_DIM, SIGNAL_RAW_DIM) = 128
    # Signal nodes: pad 10-D → 128-D with zeros
    # Note nodes:   already 128-D
    x = torch.zeros(n, NOTE_EMB_DIM, dtype=torch.float32)
    for i, e in enumerate(events):
        feat = e["feat"]
        x[i, :feat.shape[0]] = feat

    # Directed temporal edges: all (i → j) where time_i < time_j
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            src.append(i)
            dst.append(j)

    if not src:
        # Single node — no edges; GNN will just pool it
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        dt = timestamps[dst] - timestamps[src]   # already normalized /24
        edge_attr = dt.unsqueeze(1)

    kwargs: dict = dict(
        x=x,
        node_type=node_type,
        timestamps=timestamps,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    if demo_feat is not None:
        # shape (1, DEMO_DIM) so PyG cat-batches it to (B, DEMO_DIM)
        kwargs["demo"] = demo_feat.unsqueeze(0)

    return Data(**kwargs)


def load_note_embeddings(embeddings_path: Path) -> dict[str, torch.Tensor]:
    """Load {note_id -> Tensor(128,)} from Phase 1 export."""
    raw = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    return {k: v for k, v in raw.items()}


def build_patient_graph_e2e(
    stay_id: int,
    note_rows: list[dict],
    signal_rows: list[dict],
) -> Data | None:
    """
    Build a temporal graph for end-to-end fine-tuning.

    Note nodes carry pre-tokenised text tensors instead of pre-computed embeddings.
    Signal nodes carry [norm_value, hours/24] as 2-D features plus item_type_id.

    Args:
        stay_id:     ICU stay identifier.
        note_rows:   List of dicts with keys:
                       note_id, note_time (float hours from intime),
                       input_ids (Tensor[max_len] long),
                       attn_mask (Tensor[max_len] long).
        signal_rows: List of dicts with keys:
                       norm_value, item_type_id, event_hours_from_intime.

    Returns:
        PyG Data with:
            x              (N, 2)          signal: [norm_val, hours/24]; note: zeros
            item_type_ids  (N,) long       signal type IDs; 0 for note nodes
            note_input_ids (N_note, max_len) long
            note_attn_mask (N_note, max_len) long
            node_type      (N,) long       0=signal  1=note
            timestamps     (N,) float
            edge_index     (2, E)
            edge_attr      (E, 1)
        Returns None if stay has no valid nodes.

    PyG DataLoader batches note_input_ids / note_attn_mask by concatenating along
    dim-0 (default __cat_dim__), giving (N_note_total_batch, max_len). The
    note_mask applied to batched node_type indexes the same rows in the same order.
    """
    events: list[dict] = []

    for r in note_rows:
        if "input_ids" not in r:
            continue
        events.append({
            "type": 1,
            "time": float(r["note_time"]),
            "input_ids": r["input_ids"],    # Tensor(max_len,)
            "attn_mask": r["attn_mask"],    # Tensor(max_len,)
            "type_id": 0,                   # sentinel — not used for note nodes
            "val": 0.0,
        })

    for r in signal_rows:
        events.append({
            "type": 0,
            "time": float(r["event_hours_from_intime"]),
            "input_ids": None,
            "attn_mask": None,
            "type_id": int(r["item_type_id"]),
            "val": float(r["norm_value"]),
        })

    if not events:
        return None

    events.sort(key=lambda e: e["time"])
    n = len(events)

    node_type = torch.tensor([e["type"] for e in events], dtype=torch.long)
    timestamps = torch.tensor([e["time"] / 24.0 for e in events], dtype=torch.float32)
    item_type_ids = torch.tensor([e["type_id"] for e in events], dtype=torch.long)

    # Signal feature: [norm_value, hours/24]; zeros for note nodes
    x = torch.zeros(n, SIGNAL_VAL_DIM, dtype=torch.float32)
    for i, e in enumerate(events):
        if e["type"] == 0:
            x[i, 0] = e["val"]
            x[i, 1] = e["time"] / 24.0

    # Note token tensors stacked into (N_note, max_len)
    note_evts = [e for e in events if e["type"] == 1]
    if note_evts:
        note_input_ids = torch.stack([e["input_ids"] for e in note_evts])
        note_attn_mask = torch.stack([e["attn_mask"] for e in note_evts])
    else:
        max_len = 256
        note_input_ids = torch.zeros(0, max_len, dtype=torch.long)
        note_attn_mask = torch.zeros(0, max_len, dtype=torch.long)

    # Directed temporal edges: all (i → j) where time_i < time_j
    if n > 1:
        rows_t, cols_t = torch.triu_indices(n, n, offset=1)
        edge_index = torch.stack([rows_t, cols_t])
        dt = timestamps[cols_t] - timestamps[rows_t]
        edge_attr = dt.unsqueeze(1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)

    return Data(
        x=x,
        item_type_ids=item_type_ids,
        note_input_ids=note_input_ids,
        note_attn_mask=note_attn_mask,
        node_type=node_type,
        timestamps=timestamps,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
