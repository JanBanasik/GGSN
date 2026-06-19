"""
End-to-end fine-tuning variant — EXPERIMENTAL, NOT PRODUCTION.

Two previous attempts failed:
  - All-pairs O(n²) edges → ~57 min/epoch (too slow for meaningful LR search)
  - lr_bert=1e-5 caused catastrophic forgetting of Phase 1 representations

Required fixes before retry:
  - Sparse edges: K=10 nearest-predecessor instead of all-pairs (O(n²) → O(n·K))
  - AMP: uv run python -m src.training.train_gnn --precision 16-mixed (already wired)
  - lr_bert=5e-7 (100× lower than previous attempt)
  - Expected: ~5-8 min/epoch, viable for 50+ epoch runs

This file contains the moved classes:
  - TemporalPatientGNNE2E  (was src/models/tgnn_model.py)
  - build_patient_graph_e2e (was src/utils/graph_builder.py)
  - build_datasets_e2e      (was src/utils/gnn_dataset.py)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from src.models.towers import EMBED_DIM, NUM_SIGNAL_TYPES, TYPE_EMBED_DIM, TextTower
from src.utils.gnn_dataset import _preprocess_csv, _subject_split
from src.utils.graph_builder import NODE_DIM, SIGNAL_VAL_DIM

# E2E Graph builder


def build_patient_graph_e2e(
    stay_id: int,
    note_rows: list[dict],
    signal_rows: list[dict],
) -> Data | None:
    """
    Build a temporal graph for end-to-end fine-tuning.

    Note nodes carry pre-tokenised text tensors instead of pre-computed embeddings.
    Signal nodes carry [norm_value, hours/24] as 2-D features plus item_type_id.

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
    """
    events: list[dict] = []

    for r in note_rows:
        if "input_ids" not in r:
            continue
        events.append(
            {
                "type": 1,
                "time": float(r["note_time"]),
                "input_ids": r["input_ids"],
                "attn_mask": r["attn_mask"],
                "type_id": 0,
                "val": 0.0,
            }
        )

    for r in signal_rows:
        events.append(
            {
                "type": 0,
                "time": float(r["event_hours_from_intime"]),
                "input_ids": None,
                "attn_mask": None,
                "type_id": int(r["item_type_id"]),
                "val": float(r["norm_value"]),
            }
        )

    if not events:
        return None

    events.sort(key=lambda e: e["time"])
    n = len(events)

    node_type = torch.tensor([e["type"] for e in events], dtype=torch.long)
    timestamps = torch.tensor([e["time"] / 24.0 for e in events], dtype=torch.float32)
    item_type_ids = torch.tensor([e["type_id"] for e in events], dtype=torch.long)

    x = torch.zeros(n, SIGNAL_VAL_DIM, dtype=torch.float32)
    for i, e in enumerate(events):
        if e["type"] == 0:
            x[i, 0] = e["val"]
            x[i, 1] = e["time"] / 24.0

    note_evts = [e for e in events if e["type"] == 1]
    if note_evts:
        note_input_ids = torch.stack([e["input_ids"] for e in note_evts])
        note_attn_mask = torch.stack([e["attn_mask"] for e in note_evts])
    else:
        max_len = 256
        note_input_ids = torch.zeros(0, max_len, dtype=torch.long)
        note_attn_mask = torch.zeros(0, max_len, dtype=torch.long)

    # K=10 nearest predecessors per node (O(n·K) instead of O(n²))
    K = 10
    src_list, dst_list, dt_list = [], [], []
    for j in range(1, n):
        for i in range(max(0, j - K), j):
            src_list.append(i)
            dst_list.append(j)
            dt_list.append(timestamps[j].item() - timestamps[i].item())
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(dt_list, dtype=torch.float32).unsqueeze(1)
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


# E2E Dataset builder


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
    notes_by_stay, signals_by_stay, mortality_by_stay, subject_by_stay = _preprocess_csv(csv_path)

    print("Collecting note texts…")
    df = pl.read_csv(csv_path).with_columns(
        (pl.col("event_hours_from_intime") + pl.col("delta_hours_to_note")).alias("note_hours")
    )
    note_text_df = df.group_by(["stay_id", "note_id"]).agg(pl.first("text"), pl.first("note_hours"))
    note_texts: dict[str, str] = {
        r["note_id"]: r["text"] or "" for r in note_text_df.iter_rows(named=True)
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
        nid: (enc["input_ids"][i], enc["attention_mask"][i]) for i, nid in enumerate(note_ids)
    }
    print("  tokenisation done.")

    notes_with_tokens: dict[int, list] = defaultdict(list)
    for stay_id, nrows in notes_by_stay.items():
        for r in nrows:
            nid = r["note_id"]
            if nid not in note_tokens:
                continue
            ids, mask = note_tokens[nid]
            notes_with_tokens[stay_id].append(
                {
                    "note_id": nid,
                    "note_time": r["note_time"],
                    "input_ids": ids,
                    "attn_mask": mask,
                }
            )

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


# E2E Model


class TemporalPatientGNNE2E(nn.Module):
    """
    End-to-end variant: Phase 1 TextTower fine-tuned jointly with the GNN.

    Note nodes are encoded live via text_tower(input_ids, attn_mask) instead of
    using pre-computed frozen embeddings. Signal nodes use a proper nn.Embedding
    for type IDs instead of the crude mod-8 one-hot.

    Optimizer should use two param groups:
        gnn_parameters() → lr ~1e-3
        bert_parameters() → lr ~5e-7  (100× lower than previous attempt)
    """

    def __init__(
        self,
        text_tower_path: str,
        node_dim: int = NODE_DIM,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "dual",
        freeze_bert_layers: int = 8,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pooling = pooling

        self.text_tower = TextTower(freeze_bottom_layers=freeze_bert_layers)
        sd = torch.load(text_tower_path, map_location="cpu", weights_only=True)
        self.text_tower.load_state_dict(sd, strict=False)

        self.type_emb = nn.Embedding(NUM_SIGNAL_TYPES, TYPE_EMBED_DIM)  # 14 × 8
        val_proj_dim = node_dim - TYPE_EMBED_DIM  # 64 - 8 = 56
        self.val_proj = nn.Linear(SIGNAL_VAL_DIM, val_proj_dim)  # 2 → 56
        self.note_proj = nn.Linear(EMBED_DIM, node_dim)  # 128 → 64

        def _mlp(in_d: int, out_d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.LayerNorm(out_d),
            )

        self.convs = nn.ModuleList()
        self.convs.append(GINEConv(_mlp(node_dim, hidden_dim), edge_dim=1))
        for _ in range(n_layers - 1):
            self.convs.append(GINEConv(_mlp(hidden_dim, hidden_dim), edge_dim=1))

        self.drop = nn.Dropout(dropout)

        if pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            )
            clf_in = hidden_dim
        elif pooling == "dual":
            clf_in = hidden_dim * 2
        else:
            clf_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data) -> torch.Tensor:
        node_type = data.node_type
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        device = node_type.device

        sig_mask = node_type == 0
        note_mask = node_type == 1
        h = torch.zeros(node_type.size(0), self.node_dim, device=device, dtype=data.x.dtype)

        if sig_mask.any():
            type_e = self.type_emb(data.item_type_ids[sig_mask])
            val_e = self.val_proj(data.x[sig_mask])
            h[sig_mask] = torch.cat([type_e, val_e], dim=-1)

        if note_mask.any():
            note_emb = self.text_tower(data.note_input_ids, data.note_attn_mask)
            h[note_mask] = self.note_proj(note_emb)

        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr))
            h = self.drop(h)

        B = int(batch.max().item()) + 1

        if self.pooling == "attention":
            g = self.pool(h, batch)
        elif self.pooling == "dual":
            note_g = global_mean_pool(h[note_mask], batch[note_mask], size=B)
            sig_g = global_mean_pool(h[sig_mask], batch[sig_mask], size=B)
            g = torch.cat([note_g, sig_g], dim=-1)
        else:
            g = global_mean_pool(h, batch)

        return self.classifier(g).squeeze(-1)

    def gnn_parameters(self) -> list:
        """All parameters except text_tower — train at lr ~1e-3."""
        tt_ids = {id(p) for p in self.text_tower.parameters()}
        return [p for p in self.parameters() if id(p) not in tt_ids]

    def bert_parameters(self) -> list:
        """text_tower parameters only — train at lr ~5e-7."""
        return list(self.text_tower.parameters())
