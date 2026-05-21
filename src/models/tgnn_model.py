"""
Phase 2 — Temporal Patient GNN for in-hospital mortality prediction.

Heterogeneous node types (all projected to NODE_DIM=64 before GINEConv):
  - node_type=0  Signal: [one_hot(14) | norm_value | hours/24]  → sig_proj  (16→64)
  - node_type=1  Note:   text_tower embedding 128-D              → note_proj (128→64)
  - node_type=2  ICD:    Charlson comorbidity binary vector       → icd_proj  (19→64)
                         One node per stay, placed at t=-1 (prior knowledge)

Edge attr: Δt/24 (directed earlier→later; ICD node edges have Δt≥0 to all others).

Pooling options:
  "mean"      — global mean over all nodes
  "attention" — learned gate weights before pooling
  "dual"      — separate mean pools for note and signal nodes, concatenated

E2E fine-tuning variant (TemporalPatientGNNE2E) is in src/experimental/e2e.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

from src.utils.graph_builder import DEMO_DIM, ICD_NODE_DIM, NODE_DIM, NOTE_EMB_DIM, SIGNAL_RAW_DIM


class TemporalPatientGNN(nn.Module):
    def __init__(
        self,
        node_dim: int = NODE_DIM,  # 64
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",  # "mean" | "attention" | "dual"
        use_demo: bool = False,  # append DEMO_DIM demographic features after pooling
        use_icd: bool = False,  # include ICD comorbidity node (node_type=2)
    ):
        super().__init__()

        self.pooling = pooling
        self.use_demo = use_demo
        self.use_icd = use_icd

        self.sig_proj = nn.Linear(SIGNAL_RAW_DIM, node_dim)  # 16 → 64
        self.note_proj = nn.Linear(NOTE_EMB_DIM, node_dim)  # 128 → 64
        if use_icd:
            self.icd_proj = nn.Linear(ICD_NODE_DIM, node_dim)  # 19 → 64

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
            classifier_in = hidden_dim
        elif pooling == "dual":
            classifier_in = hidden_dim * 2
        else:
            classifier_in = hidden_dim

        if use_demo:
            classifier_in += DEMO_DIM

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x  # (N, NOTE_EMB_DIM)
        node_type = data.node_type  # (N,)
        edge_index = data.edge_index  # (2, E)
        edge_attr = data.edge_attr  # (E, 1)
        batch = data.batch  # (N,)

        h = torch.zeros(x.size(0), self.sig_proj.out_features, device=x.device, dtype=x.dtype)
        sig_mask = node_type == 0
        note_mask = node_type == 1
        icd_mask = node_type == 2

        if sig_mask.any():
            h[sig_mask] = self.sig_proj(x[sig_mask, :SIGNAL_RAW_DIM])
        if note_mask.any():
            h[note_mask] = self.note_proj(x[note_mask])
        if self.use_icd and icd_mask.any():
            h[icd_mask] = self.icd_proj(x[icd_mask, :ICD_NODE_DIM])

        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr))
            h = self.drop(h)

        B = int(batch.max().item()) + 1

        if self.pooling == "attention":
            g = self.pool(h, batch)
        elif self.pooling == "dual":
            # Pool only signal and note nodes; ICD node already propagated its info via GNN
            sig_mask | note_mask
            note_g = global_mean_pool(h[note_mask], batch[note_mask], size=B)
            sig_g = global_mean_pool(h[sig_mask], batch[sig_mask], size=B)
            g = torch.cat([note_g, sig_g], dim=-1)
        else:
            g = global_mean_pool(h, batch)

        if self.use_demo:
            g = torch.cat([g, data.demo], dim=-1)

        return self.classifier(g).squeeze(-1)  # (B,) logits
