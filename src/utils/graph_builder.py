"""
Phase 2 — Graph builder: turn one ICU stay into a PyG Data object.

Node types (all projected to NODE_DIM before passing to GNN):
  - node_type=0  Signal nodes: [one_hot(type_id, 14) | norm_value | hours/24]
                 ~52/stay median, spaced every 15-30 min (full stay or 24h)
  - node_type=1  Note nodes:   text_tower embedding (128-D) from Phase 1
                 ~2/stay median
  - node_type=2  ICD node:     Charlson comorbidity binary vector (N_CHARLSON-D)
                 0 or 1 node per stay, placed at timestamp=-1 (before all events)
                 so it has directed edges TO every signal/note node

Edges: directed temporal — (i → j) for every pair where time_i < time_j.
Edge attr: (t_j - t_i) / 24  (normalized hours, can exceed 1 for full-stay graphs).

End-to-end builder (build_patient_graph_e2e) is in src/experimental/e2e.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

NODE_DIM = 64  # projected node feature dim (shared for all node types)
SIGNAL_RAW_DIM = 16  # one_hot(type_id, 14) + norm_value(1) + hours/24(1)
SIGNAL_VAL_DIM = 2  # norm_value(1) + hours(1)  [e2e mode only]
NOTE_EMB_DIM = 128  # text_tower output dim (= max feature width, used for padding)
DEMO_DIM = 4  # age_norm, gender_f, is_emergency, is_elective
N_CHARLSON = 19  # Charlson comorbidity categories (see extract_icd.py)
ICD_NODE_DIM = N_CHARLSON


def build_patient_graph(
    stay_id: int,
    note_rows: list[dict],
    signal_rows: list[dict],
    note_embeddings: dict[str, torch.Tensor],
    demo_feat: torch.Tensor | None = None,
    icd_feat: torch.Tensor | None = None,
) -> Data | None:
    """
    Build a temporal graph for one ICU stay.

    Args:
        stay_id:         ICU stay identifier (int).
        note_rows:       [{note_id, note_time (hours from intime)}]
        signal_rows:     [{norm_value, item_type_id, event_hours_from_intime}]
        note_embeddings: {note_id -> Tensor(128,)} from Phase 1 text_tower.
        demo_feat:       Optional Tensor(DEMO_DIM,) — appended after pooling.
                         Shape (1, DEMO_DIM) so PyG batches to (B, DEMO_DIM).
        icd_feat:        Optional Tensor(N_CHARLSON,) — Charlson comorbidity vector.
                         Added as a single ICD node at timestamp=-1 (before all events).

    Returns:
        PyG Data with:
            x           (N, NOTE_EMB_DIM)  — padded features
            node_type   (N,) int           0=signal  1=note  2=icd
            timestamps  (N,) float         hours/24 (icd → -1/24)
            edge_index  (2, E)
            edge_attr   (E, 1)  Δt / 24
            demo        (1, DEMO_DIM)  if demo_feat provided
        Returns None if stay has no valid nodes.
    """
    events: list[dict] = []

    # ICD node (node_type=2) placed at timestamp=-1h so it precedes all events
    if icd_feat is not None:
        events.append(
            {
                "type": 2,
                "time": -1.0,  # before all real events
                "feat": icd_feat,  # Tensor(N_CHARLSON,)
            }
        )

    # Note nodes (node_type=1)
    for r in note_rows:
        nid = r["note_id"]
        if nid not in note_embeddings:
            continue
        events.append(
            {
                "type": 1,
                "time": float(r["note_time"]),
                "feat": note_embeddings[nid],  # Tensor(128,)
            }
        )

    # Signal nodes (node_type=0)
    for r in signal_rows:
        feat = torch.zeros(SIGNAL_RAW_DIM)
        type_id = int(r["item_type_id"])
        if 0 <= type_id < 14:  # proper one-hot over 14 signal types
            feat[type_id] = 1.0
        feat[14] = float(r["norm_value"])
        feat[15] = float(r["event_hours_from_intime"]) / 24.0
        events.append(
            {
                "type": 0,
                "time": float(r["event_hours_from_intime"]),
                "feat": feat,  # Tensor(16,)
            }
        )

    if not events:
        return None

    events.sort(key=lambda e: e["time"])
    n = len(events)

    node_type = torch.tensor([e["type"] for e in events], dtype=torch.long)
    timestamps = torch.tensor([e["time"] / 24.0 for e in events], dtype=torch.float32)

    # Pad all features to NOTE_EMB_DIM=128
    x = torch.zeros(n, NOTE_EMB_DIM, dtype=torch.float32)
    for i, e in enumerate(events):
        feat = e["feat"]
        x[i, : feat.shape[0]] = feat

    # Directed temporal edges: all (i → j) where time_i < time_j
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if timestamps[i] < timestamps[j]:
                src.append(i)
                dst.append(j)

    if not src:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        dt = timestamps[dst] - timestamps[src]
        edge_attr = dt.unsqueeze(1)

    kwargs: dict = dict(
        x=x,
        node_type=node_type,
        timestamps=timestamps,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    if demo_feat is not None:
        kwargs["demo"] = demo_feat.unsqueeze(0)  # (1, DEMO_DIM) → batches to (B, DEMO_DIM)

    return Data(**kwargs)


def load_note_embeddings(embeddings_path: Path) -> dict[str, torch.Tensor]:
    """Load {note_id -> Tensor(128,)} from Phase 1 export."""
    raw = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    return {k: v for k, v in raw.items()}
