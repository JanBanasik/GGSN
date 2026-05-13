"""
Phase 2 — Temporal Patient GNN for in-hospital mortality prediction.

Heterogeneous node types:
  - Signal nodes (node_type=0): x is 128-D but only first SIGNAL_RAW_DIM=10 dims are filled
  - Note nodes   (node_type=1): x is full 128-D text_tower embedding from Phase 1

Both types projected to NODE_DIM=64 via separate linear layers before GINEConv message
passing. Edge attr is Δt/24 (scalar temporal distance).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from src.utils.graph_builder import NODE_DIM, NOTE_EMB_DIM, SIGNAL_RAW_DIM


class TemporalPatientGNN(nn.Module):
    def __init__(
        self,
        node_dim: int = NODE_DIM,       # 64
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",          # "mean" | "attention"
    ):
        super().__init__()

        # Separate projections for the two node types
        self.sig_proj = nn.Linear(SIGNAL_RAW_DIM, node_dim)   # 10 → 64
        self.note_proj = nn.Linear(NOTE_EMB_DIM, node_dim)    # 128 → 64

        def _mlp(in_d: int, out_d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.LayerNorm(out_d),
            )

        # Layer 0: node_dim → hidden_dim; layers 1..n-1: hidden_dim → hidden_dim
        self.convs = nn.ModuleList()
        self.convs.append(GINEConv(_mlp(node_dim, hidden_dim), edge_dim=1))
        for _ in range(n_layers - 1):
            self.convs.append(GINEConv(_mlp(hidden_dim, hidden_dim), edge_dim=1))

        self.drop = nn.Dropout(dropout)

        # Pooling: mean or learned attention gate
        self.pooling = pooling
        if pooling == "attention":
            self.pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x                    # (N, 128)
        node_type = data.node_type    # (N,)
        edge_index = data.edge_index  # (2, E)
        edge_attr = data.edge_attr    # (E, 1)
        batch = data.batch            # (N,)

        # Project heterogeneous features → shared node_dim
        h = torch.zeros(x.size(0), self.sig_proj.out_features, device=x.device)
        sig_mask = node_type == 0
        note_mask = node_type == 1
        if sig_mask.any():
            h[sig_mask] = self.sig_proj(x[sig_mask, :SIGNAL_RAW_DIM])
        if note_mask.any():
            h[note_mask] = self.note_proj(x[note_mask])

        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr))
            h = self.drop(h)

        if self.pooling == "attention":
            g = self.pool(h, batch)
        else:
            g = global_mean_pool(h, batch)

        return self.classifier(g).squeeze(-1)  # (B,) logits
