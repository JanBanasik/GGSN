# Raport projektowy — Temporal GNN + Contrastive Learning dla predykcji śmiertelności szpitalnej

**Przedmiot:** Głębokie i grafowe sieci neuronowe  
**Dane:** MIMIC-IV 3.1 (52 727 pobytów na OIOM, 94 429 pacjentów)  
**Zadanie:** Predykcja śmiertelności wewnątrzszpitalnej (in-hospital mortality)  
**Architektura:** Two-phase — Phase 1: Contrastive pre-training, Phase 2: Temporal GNN

---

## 1. Architektura systemu

### Phase 1 — Contrastive Pre-training (Two-Tower)

Cel: nauczyć wspólną przestrzeń embeddingów dla notatek klinicznych i sygnałów fizjologicznych,
tak żeby para (notatka, sygnał) z tego samego pobytu była blisko siebie w przestrzeni 128-dim.

#### TextTower

```
input_ids (B, 256) ──► Bio_ClinicalBERT ──► [CLS] token (B, 768)
                                                    │
                                          Linear(768 → 384)
                                          GELU
                                          LayerNorm(384)
                                          Dropout
                                          Linear(384 → 128)
                                                    │
                                          L2-normalize ──► (B, 128)
```

- Backbone: `emilyalsentzer/Bio_ClinicalBERT` (BERT-base, 110M params, 12 warstw, hidden=768)
- Projection head: 768 → 384 → 128 z GELU i LayerNorm
- Wyjście: L2-znormalizowany wektor 128-dim

#### SignalTower

```
item_type_ids (B, L) ──► Embedding(14, 8) ──┐
values        (B, L) ─────────────────────── concat ──► (B, L, 11)
hours         (B, L) ─────────────────────── │            │ transpose
delta_hours   (B, L) ─────────────────────── │       (B, 11, L)
                                              │            │
                                     Conv1d(11→64, k=3) + BN + ReLU
                                     Conv1d(64→128, k=5) + BN + ReLU
                                     Conv1d(128→128, k=3) + BN + ReLU
                                                    │
                                          masked mean pooling (B, 128)
                                          Linear(128 → 128) + LayerNorm
                                                    │
                                          L2-normalize ──► (B, 128)
```

- Embedding typów sygnałów: `nn.Embedding(14, 8)`
- Wejście per event: [type_embed(8) | norm_value(1) | hours/24(1) | delta_hours/24(1)] = 11 cech
- Encoder: 3× Conv1d z rosnącymi kernelami (3, 5, 3) + BatchNorm + ReLU
- Pooling: masked mean (ignoruje padding)

#### Strata — InfoNCE (NT-Xent)

```
L = (CE(sim_matrix / τ, labels) + CE(sim_matrix.T / τ, labels)) / 2
```

- `sim_matrix[i,j] = dot(z_text_i, z_signal_j)` — macierz podobieństwa w batchu
- `labels = [0, 1, 2, ..., B-1]` — para diagonalna = pozytywna
- τ = 0.07 (temperatura)
- Gradient accumulation: macro-batch InfoNCE (effective batch=64 = 16 × 4) dla stabilności przy 8GB VRAM

#### Konfiguracja treningowa Phase 1

| Hiperparametr | Wartość |
|---|---|
| Backbone | `emilyalsentzer/Bio_ClinicalBERT` |
| Zamrożone warstwy BERT | dolne 8/12 (top 4 fine-tunowane) |
| LR (BERT) | 5×10⁻⁶ |
| LR (głowice projekcji) | 2×10⁻⁴ |
| Effective batch size | 64 (batch=16 × grad_accum=4) |
| Max text length | 256 tokenów |
| Epoki | 25 (early stopping, patience=5) |
| Filtr leakage | `src/data_prep/cleaner.py` — regex `expired|deceased|DNR|...` |

#### Wyniki Phase 1 (run `run_20260512_200632`, v4)

| Metryka | Wartość |
|---|---|
| Best val InfoNCE | **2.980** (epoka 16) |
| Baseline ln(64) | 4.159 |
| **Redukcja vs baseline** | **28.3%** |
| Linear probe AUROC (mortality) | **0.730** |
| Diagonal similarity gap | 0.105 → **0.811** (×7.7) |

Diagonal gap rośnie monotonicznie — model uczy się dopasowywać notatkę do
jej kontekstu fizjologicznego. Wizualizacja: `data/plots/embedding_comparison_v4.png`.

**Negatywny wynik — hard negative mining:** dwie próby (cosine-similarity-based)
nie poprawiły wyniku. Wysokie LR (4.6×10⁻⁶) → catastrophic forgetting (val 3.054 → 4.56).
Niskie LR (5×10⁻⁷) → brak zmian (val 3.054 → 3.135). Przyczyna: cosine similarity
wybiera klinicznie podobne notatki z różnych pobytów (np. dwa CT klatki), a
rozsuwanie ich niszczy nauczoną semantykę. Smarter selection (ICD/care-unit
disjoint) byłby wymagany. Flaga `--hard-negatives` istnieje w kodzie ale w
finalnym runie jest **wyłączona**.

Artefakty: `data/embeddings/node_embeddings.pt` (102 221 embeddingów notatek,
128-dim), `data/embeddings/text_tower.pt`, `data/embeddings/signal_tower.pt`.

---

### Phase 2 — Temporal Heterogeneous GNN

#### Konstrukcja grafu pacjenta

Dla każdego pobytu na OIOM tworzony jest jeden graf skierowany:

**Typy nodów (w grafie):**

| node_type | Opis | Cechy wejściowe | Projekcja |
|---|---|---|---|
| 0 | Sygnał (vital sign / lab) | `[one_hot(14) \| norm_val \| hours/24]` = 16-dim | `Linear(16→64)` |
| 1 | Notatka kliniczna | embedding 128-dim z Phase 1 | `Linear(128→64)` |
| 2 | ICD node (Charlson) | 19-dim binarny wektor | `Linear(19→64)` |

**Demografika — graph-level feature (nie node):**

Cechy `[age_norm, gender_f, is_emergency, is_elective]` (4-dim) są przechowywane jako
atrybut `data.demo` całego grafu, **nie jako node**. Konkatenowane są do wektora
grafu *po* poolingu, tuż przed klasyfikatorem. Uzasadnienie: demografika to
statyczna charakterystyka pobytu, nie zdarzenie w czasie — dodanie jej jako node
do grafu temporalnego byłoby niespójne architektonicznie.

```
Po poolingu:  g ∈ R^(B×128)   +   demo ∈ R^(B×4)   →   concat → R^(B×132)
                                                              ↓
                                                      Linear(132→32) → ...
```

**Kodowanie sygnałów (16-dim):**
```
dims  0-13: one-hot typ sygnału (HR, SpO2, MAP, Temp, RR, GCS, ...)
dim    14:  znormalizowana wartość ∈ [0,1]
dim    15:  czas od przyjęcia / 24h ∈ [0,1]
```

**Krawędzie:**
- Skierowane: wcześniejszy node → późniejszy node (porządek czasowy)
- Atrybut krawędzi: Δt/24 (różnica czasu w dobach)
- ICD node umieszczony w t=-1h → krawędzie do wszystkich innych nodów (wiedza a priori)
- Wszystkie pary (i→j) gdzie t_i < t_j — graf gęsty, O(n²) krawędzi per stay

#### Architektura GNN

```
Nody (heterogeniczne) ──► projekcja per typ ──► h ∈ R^(N×64)
                                                   │
                              ┌────────────────────┤
                              │   GINEConv(64→128) │
                              │   + ReLU           │ × 3 warstwy
                              │   + Dropout(0.3)   │
                              └────────────────────┘
                                                   │
                                            h ∈ R^(N×128)
                                                   │
                          ┌────────────────────────┼─────────────────┐
                       "mean"                 "attention"           "dual"
                    global_mean_pool      AttentionalAggregation   mean_notes ⊕ mean_signals
                    (B, 128)              (B, 128)                  (B, 256)
                                                   │
                              [opcjonalnie ⊕ demografika (4-dim)]
                                                   │
                                        Linear(128/256 → 32)
                                        ReLU + Dropout(0.3)
                                        Linear(32 → 1)
                                                   │
                                              logit (B,)
```

**GINEConv** (Graph Isomorphism Network with Edge features):
```
h_v' = MLP( h_v + Σ_{u∈N(v)} (h_u + edge_attr_uv) )
```
MLP wewnątrz konwolucji: `Linear → ReLU → LayerNorm`

**AttentionalAggregation (pooling attention):**
```
α_v = softmax( MLP_gate(h_v) )
g = Σ_v α_v · h_v
```
gate_nn: `Linear(128→64) → ReLU → Linear(64→1)`

#### Strata treningowa

- **Focal Loss** z `pos_weight = n_neg / n_pos = 6.897`:
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```
γ=2.0 w najlepszym modelu — skupia gradient na trudnych przypadkach

#### Parametry najlepszego modelu (demo_attention_focal2)

| Komponent | Parametry |
|---|---|
| sig_proj (16→64) | 1 024 + 64 = 1 088 |
| note_proj (128→64) | 8 192 + 64 = 8 256 |
| GINEConv × 3 | ~47 000 |
| gate_nn (attention) | ~8 300 |
| classifier (132→32→1) | ~4 200 |
| **Łącznie** | **~64 700** |

Demografika (4-dim) konkatenowana po poolingu → classifier_in = 128 + 4 = 132.

---

## 2. Refaktoryzacja kodu

Projekt przeszedł pełny refaktor do PyTorch Lightning + modularnej struktury.

### Zmiany w istniejących plikach

| Plik | Przed | Po | Co zmieniono |
|---|---|---|---|
| `train_gnn.py` | 287 linii | 121 linii | Rewrite na Lightning Trainer, flagi CLI |
| `tgnn_model.py` | 256 linii | 131 linii | Usunięto E2E, dodano ICD projection head |
| `graph_builder.py` | 244 linii | 147 linii | **Naprawa błędu one-hot**, ICD node |
| `gnn_dataset.py` | 325 linii | 281 linii | ICD + all-stay paths, max_signals cap |
| `train_contrastive.py` | 1276 linii | 620 linii | Logika wyekstrahowana do nowych modułów |

### Naprawa krytycznego błędu — one-hot encoding

Stary kod używał `feat[type_id % 8] = 1.0` dla 14 typów sygnałów, co powodowało kolizje
dla typów 8-13 (np. typ 8 i typ 0 miały tę samą reprezentację). Naprawiono na:

```python
# PRZED (błędne):
feat[type_id % 8] = 1.0   # SIGNAL_RAW_DIM = 10, kolizje dla type_id >= 8

# PO (poprawne):
if 0 <= type_id < 14:
    feat[type_id] = 1.0   # SIGNAL_RAW_DIM = 16 (14 one-hot + value + hours)
```

### Nowe pliki

| Plik | Opis |
|---|---|
| `src/models/gnn_module.py` | `GNNMortalityModule(LightningModule)` — wrapper treningowy |
| `src/data_prep/graph_datamodule.py` | `MIMICGraphDataModule(LightningDataModule)` |
| `src/data_prep/contrastive_dataset.py` | `CardiacPairsDataset`, `split_indices_by_subject` |
| `src/training/snapshot.py` | Funkcje ewaluacji i snapshotów Phase 1 |
| `src/training/loss.py` | `FocalLoss`, `info_nce_loss` |
| `src/data_prep/extract_icd.py` | Ekstrakcja Charlson ICD-10 z diagnoses_icd |
| `src/data_prep/extract_all_stay_signals.py` | Full-stay vitals + labs |
| `src/models/e2e_module.py` | `E2EMortalityModule` dla end-to-end BERT fine-tuning |
| `src/training/train_e2e.py` | Skrypt treningowy E2E |
| `src/experimental/e2e.py` | Martwy kod E2E z dokumentacją (dlaczego nie działa) |

---

## 3. Nowe funkcjonalności

### 3.1 ICD-10 Charlson Comorbidity Node

Źródło: `hosp/diagnoses_icd.csv.gz`, filtr do 19 kategorii Charlson (Quan et al. 2005).

Kategorie: zawał serca (MI), niewydolność serca (CHF), choroba naczyń obwodowych (PVD),
udar (CVD), demencja, POChP, CTD, choroba wrzodowa, łagodna/ciężka choroba wątroby,
cukrzyca (z/bez powikłań), hemiplegia, CKD, guzy lite, białaczka, chłoniak, przerzuty, AIDS.

**Zabezpieczenie przed data leakage:** kody ICD-10 w MIMIC-IV przypisywane są przy wypisie,
nie przy przyjęciu. Używamy wyłącznie kategorii Charlson — przewlekłe schorzenia istniejące
przed hospitalizacją, kodowane niezależnie od wyniku leczenia.

Generowanie: `uv run python -m src.data_prep.extract_icd`  
Użycie w treningu: `--icd-path data/processed/icd_charlson_all-icus.csv`

### 3.2 Full-stay signals

Zamiast sygnałów w oknie ±2h od każdej notatki — wszystkie zdarzenia z pełnego pobytu
`[intime, outtime]`.

Skala: 55 740 432 wierszy, mediana 315 sygnałów/pobyt, maksimum 30 312.

**Analiza data leakage:** sygnały z `[intime, outtime]` dla predykcji śmiertelności
wewnątrzszpitalnej = brak leakage (dane z pobytu dla wyniku z pobytu).

**Wykryty temporal leakage:** pierwotna implementacja brała *ostatnie* N sygnałów
(najbliższe śmierci). Wyniki: AUROC 0.931 — podejrzanie wysoki.
Naprawiono na *pierwsze* N sygnałów (wczesna faza pobytu, klinicznie sensowna predykcja).
Po naprawie: AUROC 0.820.

Generowanie: `uv run python -m src.data_prep.extract_all_stay_signals`  
Użycie w treningu: `--all-stay-path data/processed/all_stay_signals_all-icus.csv --max-signals 50`

### 3.3 End-to-end BERT fine-tuning

Wspólny trening Bio_ClinicalBERT + GNN zamiast zamrożonych embeddingów.

Kluczowe decyzje:
- K=10 poprzedników per node (zamiast O(n²) all-pairs — było 57 min/epokę)
- Dwie grupy LR: GNN `lr=1e-3`, BERT `lr_bert=5e-7` (2000× wolniej)
- 8 z 12 warstw BERT zamrożonych (tylko top 4 fine-tunowane)

---

## 4. Zabezpieczenia przed data leakage — podsumowanie

| Mechanizm | Implementacja | Status |
|---|---|---|
| Subject-disjoint split | `_subject_split()` w `gnn_dataset.py` — pacjent nigdy w obu zbiorach | ✅ |
| Text leakage filter | `cleaner.py` — regex `expired\|deceased\|DNR\|...` usuwa notatki z frazami wskazującymi zgon | ✅ |
| ICD Charlson whitelist | Tylko 19 kategorii przewlekłych chorób, nie kody wyniku | ✅ |
| Full-stay signals | `[intime, outtime]` = dane pobytu dla wyniku pobytu | ✅ |
| Early-N signal cap | `--max-signals N` trzyma **najwcześniejsze** N zdarzeń, nie końcowe — chroni przed agonalnymi vitalami | ✅ |
| Temporal ordering w grafie | Krawędzie tylko do przodu w czasie | ✅ |
| ICD node placement | t=-1h (przed wszystkimi zdarzeniami) — wiedza a priori | ✅ |

---

## 5. Eksperymenty i wyniki

### 5.1 Setup

- **Podział danych:** 80/20 train/val (subject-disjoint)
- **Imbalans klas:** 5 343 pozytywnych / 36 853 negatywnych = 1:6.9
- **Epoki:** max 100, early stopping patience=15 na val_auroc
- **Optymizator:** Adam + ReduceLROnPlateau (factor=0.5, patience=5)
- **Hardware:** NVIDIA RTX 4060 Laptop GPU (8GB VRAM)

### 5.2 Wyniki wszystkich eksperymentów

| # | Experiment | AUROC | AUPRC | Brier | sens@95spec | Epoki |
|---|---|---|---|---|---|---|
| 1 | signal_only (ablacja) | 0.784 | 0.381 | 0.181 | 0.301 | 77 |
| 2 | e2e fine-tuning | 0.790 | 0.303 | 0.204 | 0.193 | 17 |
| 3 | baseline_icd | 0.823 | 0.377 | 0.202 | 0.311 | 42 |
| 4 | baseline_allstay (first-50) | 0.820 | 0.396 | 0.171 | 0.327 | 65 |
| 5 | **baseline** | **0.829** | **0.403** | **0.180** | **0.326** | 41 |
| 6 | baseline_focal2 | 0.827 | 0.408 | 0.182 | 0.336 | 55 |
| 7 | baseline_4layers | 0.822 | 0.376 | 0.195 | 0.307 | — |
| 8 | baseline_2layers | 0.835 | 0.438 | 0.190 | 0.352 | — |
| 9 | demo_4layers | 0.836 | 0.408 | 0.203 | 0.330 | — |
| 10 | baseline_demo | 0.841 | 0.429 | 0.172 | 0.352 | 41 |
| 11 | demo_focal2 | 0.843 | 0.426 | 0.170 | 0.353 | — |
| 12 | demo_attention | 0.844 | 0.448 | 0.197 | 0.378 | — |
| 13 | demo_dual | 0.846 | 0.436 | 0.170 | 0.352 | — |
| 14 | **demo_attention_focal2** | **0.850** | **0.465** | 0.195 | **0.388** | — |

*AUPRC random baseline ≈ 0.127 (prevalence 12.7%) → model 3.6× powyżej przypadku*

### 5.3 Kluczowe obserwacje

**Co działa:**
- Notatki kliniczne są kluczowe: signal_only 0.784 → baseline 0.829 (+0.045 AUROC)
- Dane demograficzne: +0.012 AUROC vs baseline
- Attention pooling > mean pooling (model uczy się ważniejszych momentów czasu)
- Dual pooling (mean+attention) daje najlepszy Brier score (kalibracja)
- Focal loss γ=2 poprawia sens@95spec (czułość przy wysokiej specyficzności — klinicznie ważne)
- Phase 1 contrastive pre-training tworzy dobre zamrożone reprezentacje

**Co nie działa:**
- ICD Charlson node nie pomaga (rzadkie wektory binarne, 56.4% stays = zero)
- Full-stay signals nie biją ±2h przy max_signals=50
- E2E fine-tuning → catastrophic forgetting (0.790 < 0.829 baseline)
- 4 warstwy GNN gorsze od 3 (baseline_4layers 0.822 vs baseline 0.829; demo_4layers 0.836 vs demo 0.841) — głębszy GNN nie pomaga przy tych danych

---

## 6. Najlepszy model

**`demo_attention_focal2`** — AUROC 0.850, AUPRC 0.465

Konfiguracja:
```bash
uv run python -m src.training.train_gnn \
  --demo-path data/processed/demographics.csv \
  --pooling attention \
  --focal-gamma 2.0 \
  --hidden-dim 128 \
  --n-layers 3 \
  --dropout 0.3
```

Checkpoint: `data/snapshots/gnn/version_10/checkpoints/best.ckpt`
(re-run jako `version_14` z identyczną konfiguracją dał 0.848 / 0.468 — wynik
stabilny między seedami przy `seed=42`, ale ~±0.002 AUROC szumu między runami).

#### Reproducibility

- `seed=42` we wszystkich eksperymentach (`np.random`, `torch`, `torch.cuda`)
- Pełny suite ablacyjny: `bash run_experiments.sh`
- Surowe metryki: `data/snapshots/gnn/version_{0..15}/metrics.csv`
- Hyperparametry per run: `data/snapshots/gnn/version_*/hparams.yaml`
- Hardware: NVIDIA RTX 4060 Laptop (8GB VRAM); typowy run = 30-60 min
  (early stopping ~40 epok, batch=32 grafów)

---

## 7. Cel vs rzeczywistość

**Cel semestralny:** pobić SOTA 0.88 AUROC  
**Osiągnięty wynik:** 0.850 AUROC, 0.465 AUPRC

**Dlaczego nie 0.88:**
- SOTA używa end-to-end BERT fine-tuning z multi-GPU + gradient checkpointing
- Zamrożone embeddingi = bottleneck informacji z notatek
- SOTA modele mają dedykowane encodery per modalność z osobnymi LR

**Dlaczego wynik jest dobry mimo to:**
- 56K params GNN + frozen BERT osiąga 0.850 vs SOTA ~0.88 → gap tylko 0.03
- AUPRC 0.465 przy baseline 0.127 = 3.6× powyżej losowego
- E2E fine-tuning próbowany i udokumentowany jako negatywny wynik (wartość badawcza)
- Systematyczne ablacje pokazują który komponent co wnosi

---

## 8. Wnioski

1. **Heterogeniczny GNN łączący sygnały fizjologiczne z notatkami klinicznymi jest skuteczny** — ablacja signal_only potwierdza że obie modalności są potrzebne

2. **Phase 1 contrastive pre-training dostarcza wartościowych zamrożonych reprezentacji** — E2E fine-tuning niszczy te reprezentacje zamiast je ulepszać (catastrophic forgetting)

3. **Dane demograficzne są prostym ale skutecznym uzupełnieniem** — 4 cechy (wiek, płeć, typ przyjęcia) dają +0.012 AUROC

4. **Temporal leakage jest subtelny i łatwy do przeoczenia** — używanie ostatnich sygnałów przed wypisem daje AUROC 0.931 (artefakt), pierwsze sygnały dają 0.820 (realny wynik)

5. **ICD Charlson jako node grafowy nie działa przy tej skali** — 19-dim rzadki wektor binarny za mało informatywny dla 57K-param sieci przy imbalansie 1:7

---

## 9. Możliwe ulepszenia (future work)

- **E2E z gradient checkpointing** — umożliwiłoby fine-tuning BERT bez OOM na 8GB VRAM
- **ICD jako cechy demograficzne** (concatenate po poolingu) zamiast osobnego nodu
- **Temporal attention** — transformer zamiast GINEConv dla lepszego modelowania czasu
- **Ensemble** baseline_demo + demo_attention_focal2
- **Larger hidden_dim** (256/512) z regularyzacją


---

## 10. Bibliografia

- Alsentzer, E. et al. (2019). Publicly Available Clinical BERT Embeddings.
  *NAACL-HLT Clinical NLP Workshop*.
- Hu, W. et al. (2020). Strategies for Pre-training Graph Neural Networks.
  *ICLR 2020* (GINEConv).
- Johnson, A.E.W. et al. (2023). MIMIC-IV, a freely accessible electronic
  health record dataset. *Scientific Data*.
- Oord, A. van den et al. (2018). Representation Learning with Contrastive
  Predictive Coding. *arXiv:1807.03748* (InfoNCE).
- Quan, H. et al. (2005). Coding Algorithms for Defining Comorbidities in
  ICD-9-CM and ICD-10 Administrative Data. *Medical Care* 43:1130-1139
  (Charlson Comorbidity Index).
