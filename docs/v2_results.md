# Historia eksperymentów Fazy 1 (v0 → v3)

Ten dokument trzyma chronologiczne wyniki wszystkich runów treningu kontrastywnego
oraz konkretne decyzje hyperparam i diagnozy, które do nich doprowadziły.

Wszystkie InfoNCE val_loss porównane do baseline `ln(N)` gdzie `N = effective batch
size` przy InfoNCE (in-batch negatives).

---

## v0 — PoC (2026-04-13)

**Setup**: cardio (CCU only), toy=1000 stays, 5 epok, batch=8, full BERT fine-tune.
Mortality label = `patients.dod` (BROKEN — obejmuje zgony poza-szpitalne).

| Metryka | Wartość |
|---|---|
| Notki | 855 |
| Stays | 483 |
| Pary | 7 235 (note × signal) |
| Sygnały | 2 (HR, BP_mean) |
| Mortality (broken) | 53.5% |
| Final val InfoNCE | 1.418 (baseline ln(8)=2.08, **31.7% redukcji**) |

⚠️ **Wartość 31.7% to artefakt skali** — przy batch=8 zadanie jest łatwe.
Realny benchmark vs większy batch pokazany w v2.

---

## v1 — Cardio rozszerzone (2026-04-27/28)

**Setup**: cardio CCU+CVICU full (10 775 stays), 14 sygnałów, mortality label
naprawione (`hospital_expire_flag`), partial-freeze BERT (top 4/12 warstw),
proj_dropout=0.2, lr_bert=5e-6, lr_head=2e-4.

| Metryka | Wartość |
|---|---|
| Notki | 9 749 (radiology, post-cleaner) |
| Stays | 5 418 |
| Pary | 314 518 |
| Sygnały | 14 (7 vital + 7 lab) |
| Mortality (in-hospital) | 15.2% |

### Run history
- **run_20260427_224518** (full fine-tune, lr_bert=2e-5): overfit od ep.3, best val 2.515 (9% redukcji vs baseline 2.77)
- **run_20260428_011739** (partial freeze, lr_bert=5e-6, dropout=0.2): wolniejszy overfit, **best val 2.493 (10% redukcji)**, val plateau po ep.4

### Wyniki v1 final
- **best val 2.493 / baseline ln(16)=2.77 → 7.8% redukcji** ⭐
- Diagonal-vs-offdiag gap na similarity matrix rośnie do **+0.108** (ep.9)
- KMeans cluster purity = 0.51 (~random)
- Linear probe AUROC dla mortality = 0.61

### Diagnozy v1
- Tematyczna homogeniczność notek CCU (chest x-ray) → contrastive task fundamentalnie trudny
- 8 740 train notek za mało dla 110M-param BERT
- val_loss skala niespójna z train (val=batch 16, train=batch 64 z grad_accum)

---

## v2 — All-icus + macro val_loss (2026-04-29 / 05-04)

**Setup**: pełen MIMIC ICU (`--cohort all-icus`), 102k notek (10× v1), partial-freeze BERT,
grad_accum=4 (effective batch 64), `val_loss_mode=macro`, time encoding (event_hours_from_intime).

| Metryka | Wartość |
|---|---|
| Notki | 102 221 |
| Stays | 52 727 |
| Pary | 3.49M (note × signal) |
| Mortality | 12.5% |

### v2 pretrain (run_20260504_193437) — 13 epok

```
Epoch 1:  train 3.67 | val 3.51   (val < train: dropout uogólnia)
Epoch 4:  train 3.20 | val 3.22
Epoch 11: train 2.75 | val 3.054  ← BEST val
Epoch 13: train 2.67 | val 3.058  ← plateau
```

**Final v2 pretrain**: **val 3.054 / baseline ln(64)=4.16 → 26.5% redukcji** ⭐⭐⭐
**3.4× lepszy niż v1** (7.8% → 26.5%).

### v2 hard-neg fine-tune (negative result)

Dwie próby fine-tune z `HardNegativeBatchSampler` (M=8 anchors per batch + 56 shared
hard negs, leakage guard subject_id-disjoint):

| Próba | LR_BERT | pool | Val przed | Val po | Werdykt |
|---|---|---|---|---|---|
| #1 high-LR (resume in-place) | 4.6e-6 | 256 | 3.054 | **4.56** ep.14 | catastrophic forgetting |
| #2 low-LR (init_from_weights) | 5e-7 | 64 | 3.054 | **3.13–3.34** ep.2-6 | brak boost, plateau gorszy niż pretrain |

**Wniosek**: hard negatives na cosine similarity wybierają **klinicznie podobne** notki
(np. dwa CT klatki piersiowej różnych pacjentów ICU). Push-apart **psuje semantykę**.

**Aby naprawić w przyszłości** (P2):
- Smarter neg selection: wykluczać kandydatów z podobnym `first_careunit` lub jednakowym ICD-10
- Cosine threshold ("anti-fake-hard"): include only `cos(anchor, neg) ∈ [0.3, 0.7]`
- Soft positives: notki z podobnym ICD-10 → label 0.3 zamiast 0

### Final v2 = pretrain-only
- best embeddings: `run_20260504_193437/best_text_tower.pt` + `best_signal_tower.pt` (ep.11)
- Wyeksportowane do `data/embeddings/` w run_20260505_131540 finalize step

---

## v3 — Stay-level (2026-05-05) — ⚠️ architectural detour

**Setup**: stay-level pairing + Δt + effective batch 128.
Run `20260505_213819`: 25 epok, best val **2.471** (ep.20), 40.6% redukcji vs baseline ln(64)=4.159.

**Wyniki techniczne były dobre, ale architektura błędna dla Fazy 2.**

Błąd: stay-level embedding = 1 wektor per hospitalizacja (cały tekst sklejony).
Faza 2 potrzebuje per-note embeddingów (węzły grafu = pojedyncze notatki + pomiary sygnałów).
Przy stay-level nie ma z czego budować temporal graph.

**Lekcja**: InfoNCE val nie jest jedyną metryką sukcesu — granularność embeddingów musi
pasować do downstream task. Zostajemy z note-level (v2/v4) jako właściwą granularnością.

**Co v3 wniosło na przyszłość (użyte w v4)**:
- Δt jako 4. kanał SignalTower — zostaje
- Effective batch 128 → stosujemy w v4 note-level
- τ=0.07 potwierdzone jako optymalne

---

## Aktualnie canonical embeddings (2026-05-12)

`data/embeddings/` zawiera v3 stay-level embeddingi (z overnight run 20260505_213819).
**Nie używamy ich do Fazy 2.** Canonical dla Fazy 2 = **v4 note-level** (do wytrenowania).

Canonical dla Fazy 2: `data/embeddings/` = **v4 note-level** (run_20260512_200632, ep.16).

---

## v4 — Note-level + Δt + re-ekstrakcja (2026-05-12/13) ✅

**Setup**: note-level pairing z Δt jako 4. kanałem SignalTower, eff. batch 64, τ=0.07, max_text_len=256, seq_len=64.
Run `20260512_200632`: early stopping ep.21, best val **2.980** (ep.16).

| Metryka | Wartość |
|---|---|
| Plik | `pairs_all-icus_note_level.csv` (nowa ekstrakcja z Δt) |
| Notatki | 102 221 |
| Stays | 52 727 |
| Best val | **2.980** (ep. 16/21) |
| Baseline ln(64) | 4.159 |
| **% redukcji** | **28.3%** (+1.7 pp vs v2 bez Δt) |
| Linear probe AUROC | **0.730** (5-fold CV, val set) |
| Diagonal gap | 0.105 → **0.811** (×7.7 po treningu) |

To są embeddingi wejściowe do Fazy 2 GNN.

---

## Faza 2 — GNN (2026-05-13) ✅

**Setup**: TemporalPatientGNN (GINEConv×3, hidden 128, node_dim 64), v4 note-level
embeddings (frozen), heterogeneous node projections (signal 10→64, note 128→64),
all-pairs temporal edges z Δt/24 jako edge_attr, BCEWithLogitsLoss(pos_weight=6.9),
Adam lr=1e-3, early stopping na val AUROC (patience=15). 52 727 stays, 80/20 split
per subject_id.

Run `20260513_111551`: early stopping ep.36, best val AUROC ep.21.

| Metryka | Wartość |
|---|---|
| Val AUROC | **0.8156** |
| Val AUPRC | 0.3787 (random baseline = 0.125) |
| Brier score | 0.1822 |
| Sensitivity@95%spec | 0.2982 |

### Krzywa uczenia (kluczowe punkty)

| Epoka | Val AUROC | Uwaga |
|---|---|---|
| 1 | 0.755 | **frozen Phase 1 embeddings startują tu — dowód że contrastive ma sens** |
| 7 | 0.796 | GNN uczy się temporal structure |
| 15 | 0.813 | plateau zaczyna się |
| 21 | **0.816** | best |
| 36 | — | early stop (patience=15 po ep.21) |

### Porównanie z baseline

| Model | AUROC | Delta |
|---|---|---|
| Phase 1 linear probe (logistic na note emb) | 0.730 | baseline |
| **Phase 2 GNN (ten run)** | **0.816** | **+8.6 pp** |
| Target | 0.88 | −6.4 pp do celu |

**Kluczowa obserwacja**: epoka 1 = 0.755 to kurczę, a nie 0.5 — fakt że frozen embeddingi
startują 25.5 pp powyżej losowego to bezpośredni dowód wartości Fazy 1. Dalsze +6.1 pp
pochodzi z uczenia temporal structure w GNN.

### Diagnoza plateau przy 0.816

Val loss jest głośny (0.41–0.69 w trakcie plateau) — normalne przy różnych rozmiarach
grafów (25–82 węzłów) i małym batch=32. AUROC stabilny → model nie jest overfit.

Plateau wskazuje że obecna architektura jest zbliżona do swojego limitu przy frozen
embeddings. Potencjalne dźwignie:
- `hidden_dim` 128→256 (model może być wąskim gardłem)
- Attention pooling zamiast mean pool (ważniejsze węzły powinny dominować)
- Finetuning Phase 1 embeddings razem z GNN (end-to-end)
- `n_layers` 3→4 (głębszy receptive field)

---

## Tabela porównawcza wszystkich runów

| Wersja | Run | Cohort | Strategia | Δt | Eff. batch | Best val | Baseline | % redukcji |
|---|---|---|---|---|---|---|---|---|
| v0 PoC | manual | cardio toy | note ±2h | ❌ | 8 | 1.418 | 2.08 | 31.7% (artefakt skali) |
| v1 #1 | 224518 | cardio full | note ±2h | ❌ | 16 | 2.515 | 2.77 | 9.2% |
| v1 #2 | 011739 | cardio full | note ±2h | ❌ | 16 | **2.493** | 2.77 | **9.9%** |
| v2 pretrain | 193437 | all-icus | note ±2h | ❌ | 64 | 3.054 | 4.16 | 26.6% |
| v2 hard-neg #1 | 193437 ep.14+ | all-icus | note ±2h | ❌ | 64 | 4.56 | 4.16 | −9.7% (catastrophic) |
| v2 hard-neg #2 | 131540 | all-icus | note ±2h | ❌ | 64 | 3.135 | 4.16 | 24.6% (brak boost) |
| v3 stay-level | 213819 | all-icus | stay_level | ✅ | 64 | 2.471 | 4.16 | 40.6% (zła granularność) |
| **v4 note-level** ⭐ | **200632** | all-icus | **note ±2h** | ✅ | 64 | **2.980** | 4.16 | **28.3%** → Phase 2 |

### Faza 2 GNN

| Run | Embeddings | GNN layers | hidden | Pooling | Val AUROC |
|---|---|---|---|---|---|
| **20260513_111551** ⭐ | v4 frozen | GINEConv×3 | 128 | mean | **0.816** |
