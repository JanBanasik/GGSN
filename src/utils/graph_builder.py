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
SIGNAL_RAW_DIM = 10    # type_emb(8) + norm_value(1) + hours(1)
NOTE_EMB_DIM = 128     # text_tower output dim


def build_patient_graph(
    stay_id: int,
    note_rows: list[dict],
    signal_rows: list[dict],
    note_embeddings: dict[str, torch.Tensor],
) -> Data | None:
    """
    Build a temporal graph for one ICU stay.

    Args:
        stay_id:         ICU stay identifier (int).
        note_rows:       List of dicts with keys: note_id, note_time (float hours from intime).
        signal_rows:     List of dicts with keys: norm_value, item_type_id,
                         event_hours_from_intime.
        note_embeddings: {note_id -> Tensor(128,)} from Phase 1 text_tower.

    Returns:
        PyG Data with:
            x           (N, NOTE_EMB_DIM) for note nodes  OR
                        (N, SIGNAL_RAW_DIM) for signal nodes
                        → caller projects to NODE_DIM inside the GNN model
            node_type   (N,) int  0=signal  1=note
            timestamps  (N,) float hours/24 — used for edge construction
            edge_index  (2, E) directed temporal edges
            edge_attr   (E, 1) Δt / 24
            y           scalar float  mortality label
            stay_id     int
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

    return Data(
        x=x,
        node_type=node_type,
        timestamps=timestamps,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def load_note_embeddings(embeddings_path: Path) -> dict[str, torch.Tensor]:
    """Load {note_id -> Tensor(128,)} from Phase 1 export."""
    raw = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    # keys are "stay_XXXXX" in v4 (note_level still uses stay_id as key per dataset)
    return {k: v for k, v in raw.items()}
