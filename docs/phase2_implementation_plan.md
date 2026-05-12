# Phase 2 Implementation Plan
**Stan na 2026-05-12. Cel: in-hospital mortality AUROC > 0.88 na MIMIC-IV all-icus.**

---

## Dlaczego Phase 1 było konieczne

Phase 2 GNN potrzebuje sensownych node features. Mamy trzy opcje:

| Node features            | Linear probe AUROC | Co to znaczy                          |
|--------------------------|--------------------|---------------------------------------|
| Random (128-D)           | ~0.50              | Sieć musi nauczyć się wszystkiego od zera |
| Bio_ClinicalBERT (ep.0)  | 0.704              | Pretrenowany BERT, bez contrastive    |
| **Phase 1 v2 note-level**| **~0.73 (target)** | Contrastive text↔signal alignment    |

Różnica Bio_ClinicalBERT → Phase 1: **model nauczył się dopasowywać język kliniczny do mierzonych parametrów**. Notatka "SpO2 dropping, increased RR" powinna być blisko sygnałów desaturacji i tachypnei — i po Phase 1 jest.

Ablacja którą pokażemy w finalnym projekcie: zamień Phase 1 embeddingi na random lub czysty BERT → AUROC spada. To jest dowód że Phase 1 miało sens.

---

## Krok 1 — Re-trening Phase 1 v4 (note-level, od zera)

### Dlaczego od zera, nie resume v2

v2 (`run_20260504_193437`) plateau po ep.11 z val 3.054 (eff.batch 64). Mamy teraz wiedzę z v3 co działa lepiej.

### Zmiany vs v2

| Parametr          | v2 (stary)    | v4 (nowy)     | Powód                          |
|-------------------|---------------|---------------|--------------------------------|
| Effective batch   | 64            | **128**       | 2× więcej in-batch negatives   |
| Temperature τ     | 0.07          | **0.07**      | v3 potwierdził τ=0.07 optymalny|
| Δt 4. kanał       | ❌            | **✅**        | Dodaje temporal alignment      |
| CSV               | note_level    | **note_level**| Granularność dla Phase 2 grafu |
| Epochs            | 13            | **25-30**     | Dać szansę na dłuższe uczenie  |
| grad_accum_steps  | 4             | **8**         | batch 8 × accum 8 = eff. 64... |

**⚠️ Re-ekstrakcja wymagana.** Istniejący `pairs_all-icus_note_level.csv` nie ma kolumny `delta_hours_to_note` (był wyekstrahowany przed dodaniem tej kolumny w v3). Extractor.py już obsługuje Δt dla note-level — wystarczy ponowna ekstrakcja (~30-60 min):

```bash
uv run python -m src.data_prep.extractor --full --cohort all-icus --pair-strategy note_level
```

Nadpisze stary plik. Nowy CSV będzie miał kolumnę `delta_hours_to_note` ∈ [−2h, +2h] (bounded przez `pair_window_hours=2.0`).

### Komenda startowa

```bash
uv run python -m src.training.train_contrastive \
    --epochs 30 \
    --batch-size 8 \
    --grad-accum-steps 8 \
    --temperature 0.07 \
    --max-text-len 256 \
    --seq-len 64 \
    --csv-path data/processed/pairs_all-icus_note_level.csv \
    --max-time-hours 10
```

`max-text-len 256` (nie 512) bo note-level notatki są krótsze niż stay-level concat; `seq-len 64` wystarczy dla ±2h okna.

### Spodziewany wynik

Baseline ln(128) = 4.852. Target val ~3.4-3.6 → ~27-30% redukcji.
(v2 notevel miał 26.5% przy eff.batch 64 — większy batch + Δt powinno dać minimalny boost.)

---

## Krok 2 — Budowa grafów per stay

### Dane wejściowe

- `data/processed/pairs_all-icus_note_level.csv` — każdy wiersz = 1 signal event dla danej notatki
- `data/embeddings/text_tower.pt` (v4 best) — do embedowania notatek
- Etykieta: kolumna `mortality` (in-hospital, `hospital_expire_flag`)

### Struktura węzłów

**Węzeł typu Signal** (~52 na stay, median):
```python
features = cat([
    item_type_embedding[item_type_id],   # Embedding(14, 8)  → R^8
    tensor([norm_value]),                # R^1,  ∈ [0, 1]
    tensor([event_hours / 24.0]),        # R^1,  ∈ [0, 1]
])  # → R^10
node_feature = Linear(10, NODE_DIM)(features)   # R^64
```

**Węzeł typu Note** (~2 na stay, median):
```python
with torch.no_grad():
    z_text = text_tower(note_text)        # R^128  (zamrożony Phase 1)
node_feature = Linear(128, NODE_DIM)(z_text)    # R^64
```

Oba typy wprojektowane do wspólnego `NODE_DIM = 64`. Opcjonalnie: dodaj binarną flagę `is_note ∈ {0, 1}` jako dodatkową cechę węzła.

### Struktura krawędzi

```python
# Directed temporal: wszystkie krawędzie i → j gdzie time_i < time_j
# edge_attr: Δt znormalizowane
edges = [(i, j) for i in range(n) for j in range(n) if times[i] < times[j]]
edge_attr = [(times[j] - times[i]) / 24.0 for i, j in edges]
```

Median ~54 węzłów → ~54×53/2 ≈ 1431 krawędzi per stay. Przy batch_size=32: ~46k krawędzi — zarządzalne dla PyG.

**Plik**: `src/utils/graph_builder.py` (szkielet już istnieje, uzupełnić).

```python
def build_patient_graph(
    stay_id: int,
    note_rows: pl.DataFrame,        # wiersze z note_level CSV dla tego stay
    text_tower: TextTower,
    tokenizer,
    node_dim: int = 64,
    device: torch.device = cpu,
) -> torch_geometric.data.Data:
    ...
    return Data(x=node_features, edge_index=edge_index,
                edge_attr=edge_attr, y=mortality)
```

---

## Krok 3 — GNN architektura

```python
class TemporalPatientGNN(nn.Module):
    def __init__(self, node_dim=64, hidden_dim=128, n_layers=3, dropout=0.3):
        self.sig_proj  = nn.Linear(10, node_dim)       # signal nodes
        self.note_proj = nn.Linear(128, node_dim)      # note nodes (Phase 1 embedding)
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(node_dim + 1, hidden_dim),  # +1 = edge_attr dim
                    nn.ReLU(),
                    nn.Linear(hidden_dim, node_dim),
                ),
                edge_dim=1,
            )
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(node_dim, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        x = data.x                          # (N_nodes, node_dim) — już zprojektowane
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch) # (B, node_dim)
        return self.classifier(x).squeeze(-1)
```

**Plik**: `src/models/tgnn_model.py` (szkielet już istnieje, uzupełnić).

---

## Krok 4 — Trening GNN

**Plik**: `src/training/train_gnn.py` (szkielet już istnieje)

```python
# Loss
pos_weight = torch.tensor([(1 - 0.125) / 0.125])  # ≈ 7.0 dla mortality 12.5%
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Early stopping na val AUROC (nie loss — ważniejsze przy imbalance)
# 5-fold CV split po subject_id
```

**Fazy fine-tuningu**:
1. Epoki 1-5: zamrożony text_tower (trenuj tylko GNN + projections)
2. Epoki 6+: unfreeze text_tower z lr×0.1 (fine-tune end-to-end)

---

## Krok 5 — Ablacje (dowód że Phase 1 ma sens)

| Eksperyment               | Node features          | Oczekiwany AUROC |
|---------------------------|------------------------|------------------|
| **Baseline**              | Random 64-D            | ~0.70-0.72       |
| **BERT only**             | Bio_ClinicalBERT ep.0  | ~0.74-0.76       |
| **Phase 1 v4 frozen**     | v4 note-level pretrain | ~0.80-0.84       |
| **Phase 1 v4 fine-tuned** | v4 end-to-end          | **~0.85-0.90**   |

Każdy wyższy rząd = konkretny wkład Phase 1. To jest narracja projektu.

---

## Harmonogram (2026-05-12 → koniec czerwca)

| Tydzień       | Zadanie                                              |
|---------------|------------------------------------------------------|
| 12-19 maja    | Re-ekstrakcja note_level z Δt (jeśli brak kolumny); nocny trening Phase 1 v4 |
| 19-26 maja    | `graph_builder.py` + `train_gnn.py`; pierwsze GNN runy na cardio subset |
| 26 maja–2 cze | Pełny trening GNN all-icus; 5-fold CV; ablacje       |
| 2-9 cze       | Hyperparameter tuning GNN; wizualizacje              |
| 9-23 cze      | Finalny trening; raport; slajdy                      |
| 23-30 cze     | Bufor / poprawki                                     |

---

## Pliki do zaimplementowania

| Plik                            | Status   | Priorytet |
|---------------------------------|----------|-----------|
| `src/utils/graph_builder.py`    | szkielet | P0        |
| `src/models/tgnn_model.py`      | szkielet | P0        |
| `src/training/train_gnn.py`     | szkielet | P0        |
| `src/utils/metrics.py`          | pusty    | P0        |
| `src/training/eval_embeddings.py` | ✅ gotowy | —       |
| `src/training/sweep_temp.sh`    | ✅ gotowy | —        |

---

## Kluczowe decyzje podjęte

- **Phase 1**: note-level pairing (v4) jako source embeddingów dla Phase 2 — właściwa granularność (1 nota = 1 węzeł grafu)
- **Graph**: heterogeniczne węzły (signal + note), chronologiczne krawędzie, edge_attr = Δt
- **GNN**: GINEConv (edge attributes) × 3 layers, global_mean_pool, BCE + pos_weight
- **Stay-level v3 embeddingi**: zostają jako canonical dla celów porównawczych (AUROC 0.784); Phase 2 używa v4 note-level
