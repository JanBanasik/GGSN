# Roadmapa ulepszeń — droga od PoC do finalnego projektu

**Stan na 2026-04-28** (po pierwszych pełnych runach Fazy 1)

---

## Co zrobiliśmy do tej pory

Faza 1 została rozszerzona z PoC do "v1":
- Pełna kohorta CCU/CVICU (10 775 stays, 9 749 unique notes, 314k par)
- 14 typów sygnałów (7 vital + 7 lab) zamiast 2 (HR + BP)
- Naprawiona etykieta in-hospital mortality (`hospital_expire_flag` z `admissions`) — 15.2% vs PoC 53.5%
- SignalTower z `nn.Embedding(14, 8)` dla heterogenicznych typów eventów
- AMP, train/val split per `subject_id`, run_config + git SHA, per-epoch snapshoty
- Cleaner.py filtruje 13 phrasów leakage
- Manim animacje (similarity matrix, UMAP, two-tower diagram)

**Co zaobserwowaliśmy**:
- Run 1 (full fine-tune BERT, lr=2e-5): overfit od epoki 3, best val 2.515 (9% redukcji vs baseline 2.77)
- Run 2 (freeze 8/12 layers, lr=5e-6, dropout=0.2): wolniejszy overfit, best val 2.493 (10% redukcji), train-val gap rośnie
- Mimo plateau val_loss, **diagonal-vs-offdiag gap na similarity matrix dalej rośnie** (epoka 9: gap=0.108) — animacja wygląda dobrze
- Run 3 (full freeze BERT, grad accum 4, effective batch 64) — w trakcie, oczekiwany trening na noc

**Diagnoza problemów**:
1. Małe batche → mało in-batch negatives → InfoNCE jest słaby learner
2. Notatki radiologiczne w CCU są tematycznie homogeniczne (chest x-ray, edema)
3. Pairing ±2h sprawia że sąsiednie notatki tego samego pacjenta dostają niemal identyczne sygnały
4. 8 740 train notek to mało dla 110M-parameter BERTa
5. Train/val skale loss różnią się gdy używamy grad accum (val nie buforuje)

---

## Co zrobić do finalnego projektu (priorytety)

### P0 — must-have (krytyczne dla AUROC > 0.88 w Fazie 2)

#### 1. Skalowanie datasetu
**Problem**: 8 740 train notek to 50× za mało dla porządnego contrastive.

**Rozwiązania**:
- (a) **Trenować Fazę 1 na całym MIMIC-IV** (94 458 stays, ~330k notatek discharge + radiology). Embeddingi to "kliniczna semantyka" — uniwersalne. W Fazie 2 GNN bierze tylko CCU/CVICU subset.
- (b) Dodać **discharge summaries** (po cleanerze) — to ~7-8× więcej tekstu. UWAGA: discharge są pisane przy wypisie, więc trzeba je sztucznie sparować z sygnałami z 24h pobytu (np. losowy moment z okna). Albo trenować osobny model na discharge i osobny na radiology.
- (c) Dodać **nursing notes** jeśli są dostępne w MIMIC-IV-Note (sprawdzić; w PoC nie były).

**Implementacja**: rozszerzyć `extractor.py` o flag `--all-icus` i `--include-discharge`. Cleanera trzeba znacząco wzmocnić dla discharge (one prawie zawsze mają informację o wypisie/zgonie).

#### 2. Lepszy pairing — jeden stay = jeden sample
**Problem**: 1 pacjent z 5 notatkami w 24h dostaje 5 prób ze tymi samymi (lub niemal) sygnałami, model nie ma jak ich odróżnić.

**Rozwiązania**:
- **Stay-level contrastive**: zamiast (note_t1, sig±2h) jako positive pair, użyć (concatenated all notes for stay, all signals for stay 24h) jako positive. Każdy stay = 1 sample. Mniej próbek (~5k zamiast ~9k) ale każda jest "bogatsza" i rozróżnialna.
- Albo **stricter window** (±30min zamiast ±2h) + drop notatki bez wystarczającego pokrycia sygnałami.

**Implementacja**: nowy mode w `extractor.py`: `pair_strategy ∈ {note_level, stay_level}`.

#### 3. Hard negative mining
**Problem**: in-batch negatives są losowe — większość to "łatwe" (np. tekst o płucach vs sygnały zdrowego pacjenta). Sieć nie uczy się subtelnych różnic.

**Rozwiązania**:
- **Pre-compute embeddingi raz za N epok**, znajdź top-k najtrudniejsze negatives per sample, sample te do batcha. Standard MoCo/SimCLR.
- **Sampling po podobieństwie pacjentów**: w batchu zawsze są pacjenci z podobnymi diagnozami (ICD-10) — wtedy negatives są naturalnie trudne.

**Implementacja**: zmodyfikowany `Sampler` w DataLoader, pre-compute table per epoch.

#### 4. Mocniejszy SignalTower
**Problem**: 1D CNN na sekwencji (item_type, value) traci informację czasową (event_time nie jest podawany).

**Rozwiązania**:
- Dodać **time encoding** — pozycyjny embedding z `event_time - intime` (godziny od przyjęcia)
- Zamienić CNN na **mały Transformer** (4 warstwy, 4 heads, 64-dim) — lepsza obsługa heterogenicznych sekwencji
- Albo **gated GRU** — prostszy ale dobry dla zmiennych długości

**Implementacja**: nowa klasa `SignalTransformer` w `towers.py`, A/B testowanie.

### P1 — should-have (poprawia jakość ale nie blokuje)

#### 5. Hyperparameter sweep
- Temperature τ ∈ {0.05, 0.07, 0.10, 0.15, 0.20} — krytyczne dla contrastive
- Effective batch ∈ {32, 64, 128} (przez grad accum)
- LR_HEAD ∈ {1e-4, 2e-4, 5e-4}
- Embed dim ∈ {64, 128, 256}

**Implementacja**: skrypt `src/training/sweep.py` z prostym grid search, pisze wyniki do CSV.

#### 6. Reproducibility / experiment tracking
- Dodać **wandb** (free dla studentów) lub **MLflow** lokalny
- Logować: wszystkie hyperparams, sample similarity matrices co N epok, embedding clusters, sample top-5 nearest neighbors
- Per-run sysinfo (GPU, CUDA, python, package versions)

**Implementacja**: `src/training/tracker.py` jako abstrakcja, init na początku `train()`.

#### 7. Naprawić skale train/val loss przy grad accum
**Problem**: train liczony na effective_batch (np. 64), val na raw batch (16). Różne baseliny ln(N), trudne porównanie.

**Rozwiązanie**: w `evaluate_val_loss` zrobić ten sam buffer co w treningu — concat micro-batchy, InfoNCE na effective val batch.

#### 8. Augmentacja danych
- **Tekst**: random masking 15% tokenów (à la MLM), synonim substitution
- **Sygnały**: temporal jitter (przesunięcie ±5min), value noise (gaussian σ=0.02)
- Augmentacja dwóch view per sample → SimCLR-style (text+aug_text positive, signal+aug_signal positive)

### P2 — nice-to-have

#### 9. Self-supervised pre-training SignalTower
Zanim contrastive z tekstem, pre-trenować SignalTower przez masked vital prediction (random mask 15% values, predict valuenum). Pomaga w warm-start.

#### 10. Multi-task pre-training
Dodać auxiliary task: predict next vital from past signals (jak language modeling dla sygnałów). To wymusza temporal reasoning.

#### 11. Eval na trzymanej kohorcie
Trzymać 5% subjects na test (nie tylko val), nigdy nie tknięte podczas hyperparameter tuning. Final reported metrics na test.

---

## Faza 2 — co musimy zaimplementować

### `src/utils/graph_builder.py`
Per-stay graph:
- Nodes: notatki (z embeddings z Fazy 1) + opcjonalnie sygnały-jako-osobne-nodes z embeddings ze SignalTower
- Edges: chronologiczne (i→j gdy event_time[i] < event_time[j])
- Edge attr: Δt = (event_time[j] - event_time[i]).total_seconds() / 3600 (godziny)

```python
def build_patient_graph(stay_id: int, embeddings: dict, events: pl.DataFrame) -> torch_geometric.data.Data:
    ...
```

### `src/models/tgnn_model.py`
GINEConv layers (PyG):
```python
class TemporalGNN(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=256, n_layers=3):
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU()), edge_dim=1)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)
```

### `src/training/train_gnn.py`
- BCEWithLogitsLoss z `pos_weight = (n_neg / n_pos)` (~5.5 dla mortality 15%)
- Optimizer: Adam lr=1e-3, scheduler ReduceLROnPlateau
- Early stopping na val AUPRC (nie AUROC — ważniejsze przy class imbalance)
- 5-fold cross-validation per subject_id

### `src/utils/metrics.py`
- AUROC, AUPRC, Brier score, sensitivity@95% specificity
- Calibration curve, decision curve analysis

### Ablations dla papier/raport
- Frozen vs fine-tuned embeddings
- Random embeddings vs Phase 1 embeddings (baseline — pokaż że contrastive coś daje)
- GINEConv vs SAGEConv (z i bez edge_attr)
- Bez Δt vs z Δt (czyli z edge_attr=1.0 vs edge_attr=Δt)
- Liczba warstw GNN: 1, 2, 3, 5
- Dataset size (50%, 100% pacjentów)

---

## Sugerowany harmonogram (8 tygodni do finalnego)

| Tydzień | Faza 1 | Faza 2 | Inne |
|---|---|---|---|
| 1 (teraz) | środowa prezentacja na obecnym v1 | — | — |
| 2 | P0.1 (full MIMIC), P0.2 (stay-level pairing) | — | wandb setup |
| 3 | P0.3 (hard negatives), P0.4 (transformer SignalTower) | szkielet `graph_builder.py` | — |
| 4 | P1.5 (sweep), wybór finalnych hyperparams | `tgnn_model.py` + `train_gnn.py` | — |
| 5 | finalny trening Fazy 1 (long, na docelowym datasecie) | pierwszy run GNN, debugging | — |
| 6 | — | hyperparameter tuning GNN | ablations |
| 7 | — | finalny trening + 5-fold CV | wykresy, statystyki |
| 8 | rezerwa | rezerwa | raport, slajdy |

---

## Pytania otwarte / do dyskusji

- Czy używać tylko notatek radiologicznych, czy też discharge / nursing? Trade-off: więcej danych ale ryzyko leakage przez discharge.
- Czy zostać przy in-hospital mortality czy zmienić na 30-day mortality (więcej pozytywów, częstszy benchmark)?
- Czy graf w Fazie 2 powinien zawierać node'y dla sygnałów osobno, czy tylko notatki (sygnały tylko w embeddingach)?
- Czy pre-trenować na MIMIC-III (większy, ale starszy schema) i fine-tune na MIMIC-IV?
- Test set: hold-out subjects czy hold-out admissions w czasie (temporal split, bardziej realistyczne)?
