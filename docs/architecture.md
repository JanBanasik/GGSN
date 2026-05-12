# Architektura Systemu – Temporal GNN + Contrastive Learning

Poniższe diagramy opisują pełen pipeline predykcji śmiertelności szpitalnej
na danych MIMIC-IV (pierwsze 24h pobytu).

> **Status (2026-05-12)**: dokument zaktualizowany do stanu v4 (aktualny kierunek).
> Faza 1 używa **note-level pairing** — każda notatka = osobna para z sygnałami ±2h.
> Stay-level (v3) było testem który poprawił InfoNCE ale miał złą granularność dla Fazy 2.
> Pełna historia eksperymentów: [v2_results.md](v2_results.md).
>
> Główne różnice względem PoC v0:
> - **Cohort**: `--cohort all-icus` (pełny ICU MIMIC-IV, 52 727 stays, 102k notatek)
> - **Pairing**: `--pair-strategy note_level` — note-level granularity dla węzłów grafu
> - **Mortality**: `admissions.hospital_expire_flag` (in-hospital), nie `patients.dod`
> - **Sygnały**: 14 typów (7 vital + 7 lab) zamiast 2 (HR + BP)
> - **SignalTower input**: 4 kanały — `(item_type_embed, value, hours_from_intime, delta_to_note)`

---

## Diagram 1 – Ogólny przepływ danych (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAW MIMIC-IV DATA                               │
│                                                                         │
│  icu/icustays.csv.gz   hosp/patients.csv.gz   note/radiology.csv.gz     │
│  icu/chartevents.csv.gz                                                 │
└───────────────────────┬─────────────────────────────────────────────────┘
                        │  src/data_prep/extractor.py
                        │  ├── load_cohort(cohort)  → CCU/CVICU lub all-icus + mortality
                        │  ├── load_notes()         → radiology notes [t₀, t₀+24h] + cleaner
                        │  ├── load_vitals()        → 7 vital itemids (HR, BP×3, SpO2, RR, Temp)
                        │  ├── load_labs()          → 7 lab itemids (Trop, BNP, Cr, Lac, K, Hgb, WBC)
                        │  └── pair_notes_signals(strategy) → note_level (±2h) lub stay_level (24h concat)
                        ▼
              ┌────────────────────────────────────┐
              │  pairs_<cohort>_<strategy>.csv     │  ← cardio_note: 9 749 notek / 314k par
              │  (data/processed/)                 │     all-icus_note: 102k notek / 3.49M par
              │                                    │     all-icus_stay: 53 380 stays / 9.53M par
              │                                    │     mortality (in-hosp): 12.5-15.2%
              └────────────────────┬───────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   ┌─────────────┐               ┌─────────────┐
   │  TEXT path  │               │ SIGNAL path │
   │  (notatki)  │               │  (vitale)   │
   └──────┬──────┘               └──────┬──────┘
          │                             │
          │   FAZA 1: Contrastive Pre-training (self-supervised, bez etykiet)
          │                             │
          ▼                             ▼
   ┌─────────────┐               ┌─────────────┐
   │ TextTower   │               │ SignalTower  │
   │ (BERT+proj) │               │ (CNN+proj)  │
   └──────┬──────┘               └──────┬──────┘
          │  z_text (128-D)             │  z_signal (128-D)
          └──────────────┬──────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │  InfoNCE Loss    │  maximize sim(z_text[i], z_signal[i])
               │  (NT-Xent)       │  over in-batch negatives j≠i
               └──────────────────┘
                         │
                         │ (po wytrenowaniu – zamrożony TextTower)
                         ▼
              ┌──────────────────────┐
              │  node_embeddings.pt  │  ← {note_id → Tensor(128,)}
              │  (data/embeddings/)  │     v2: 9 749–102 221 wektorów (cohort-zależne)
              └──────────┬───────────┘
                         │
          FAZA 2: Temporal Graph Neural Network (supervised)
                         │
                         ▼
              ┌──────────────────────┐
              │  Graph Builder       │  src/utils/graph_builder.py
              │  (per patient stay)  │
              │  x = embeddingi      │
              │  edge_index = chrono │
              │  edge_attr = Δt[h]   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   T-GNN Model        │  src/models/tgnn_model.py
              │   GINEConv layers    │  uwzględnia edge_attr (Δt)
              │   GlobalMeanPool     │
              │   Linear → Sigmoid   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  P(śmiertelność)     │  cel: AUROC > 0.88
              │  [0, 1]              │       AUPRC wysoki (klasa ~10-15%)
              └──────────────────────┘
```

---

## Diagram 2 – Faza 1: Two-Tower Architecture (szczegółowo)

```
WEJŚCIE: jeden batch B = 8 par (notatka_i, sygnał_i)

╔══════════════════════════════════════════════════════╗
║                    TEXT TOWER                        ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Tokenizer (Bio_ClinicalBERT, max_len=256)           ║
║  "CHEST RADIOGRAPH..." → [CLS] T1 T2 ... [SEP] PAD  ║
║                                                      ║
║  ┌─────────────────────────────────────────────┐    ║
║  │  Bio_ClinicalBERT  (110M params, frozen→)   │    ║
║  │  12 Transformer layers, hidden_size=768     │    ║
║  └──────────────────────┬──────────────────────┘    ║
║                         │ [CLS] token: (B, 768)      ║
║  ┌──────────────────────▼──────────────────────┐    ║
║  │  Projection Head                            │    ║
║  │  Linear(768→384) → GELU → LayerNorm(384)   │    ║
║  │  → Linear(384→128)                         │    ║
║  └──────────────────────┬──────────────────────┘    ║
║                         │ (B, 128)                   ║
║                    L2-Normalize                      ║
║                         │                            ║
║                    z_text (B, 128)                   ║
╚═════════════════════════╪════════════════════════════╝
                          │
                          │         InfoNCE Loss
                          ◄────────────────────────────►
                          │
╔═════════════════════════╪════════════════════════════╗
║                    z_signal (B, 128)                 ║
║                    L2-Normalize                      ║
║                         │                            ║
║  ┌──────────────────────▼──────────────────────┐    ║
║  │  Projection Head                            │    ║
║  │  Linear(128→128) → LayerNorm(128)           │    ║
║  └──────────────────────┬──────────────────────┘    ║
║                         │ (B, 128)                   ║
║  ┌──────────────────────▼──────────────────────┐    ║
║  │  1D CNN Encoder (v3)                        │    ║
║  │                                             │    ║
║  │  Input: 4 channels × seq_len events         │    ║
║  │    ch0..7: type_embed = nn.Embedding(14, 8) │    ║
║  │    ch8:    norm_value ∈ [0, 1]             │    ║
║  │    ch9:    hours_from_intime ∈ [0, 1]      │    ║
║  │    ch10:   delta_to_note (clipped /24)     │    ║
║  │                                             │    ║
║  │  Transpose → (B, 11, seq_len)              │    ║
║  │  Conv1d(11→64,  k=3) → BN → ReLU          │    ║
║  │  Conv1d(64→128, k=5) → BN → ReLU          │    ║
║  │  Conv1d(128→128,k=3) → BN → ReLU          │    ║
║  │  Masked-mean pool → (B, 128)               │    ║
║  └─────────────────────────────────────────────┘    ║
║                                                      ║
║  seq_len: 64 (note_level) | 128 (stay_level)         ║
║  14 typów: HR, BP_mean/sys/dia, SpO2, RR, Temp,      ║
║           Trop_I, NTproBNP, Cr, Lac, K, Hgb, WBC     ║
║                                                      ║
║                   SIGNAL TOWER                       ║
╚══════════════════════════════════════════════════════╝

InfoNCE Loss (NT-Xent, temperature τ=0.07):
┌─────────────────────────────────────────────────────┐
│  Similarity matrix S:  S[i,j] = z_text[i]·z_sig[j]/τ│
│                                                     │
│       sig₀   sig₁   sig₂  ... sig₇                 │
│  txt₀ [★    ·      ·         ·  ]   ← positive: i=j │
│  txt₁ [·    ★      ·         ·  ]   negative: i≠j   │
│  ...                                                │
│  txt₇ [·    ·      ·         ★  ]                   │
│                                                     │
│  L = [CE(S, diag) + CE(Sᵀ, diag)] / 2              │
└─────────────────────────────────────────────────────┘
```

---

## Diagram 3 – Faza 2: Temporal GNN (szczegółowo)

```
WEJŚCIE: jeden pacjent (stay_id), ~54 zdarzeń (mediana), posortowanych chronologicznie

  ~52 pomiary sygnałów + ~2 notatki kliniczne w oknie 24h

  Węzły dwóch typów:
  ┌─ Signal node (~52/stay) ─────────────────────────────────────────┐
  │  [item_type_id → Embedding(14,8)] ++ [norm_value] ++ [hours/24]  │
  │  → Linear(10 → 64) → z_sig ∈ R^64                               │
  └──────────────────────────────────────────────────────────────────┘
  ┌─ Note node (~2/stay) ────────────────────────────────────────────┐
  │  text → text_tower(frozen, Faza 1) → 128-D                       │
  │  → Linear(128 → 64) → z_note ∈ R^64                             │
  └──────────────────────────────────────────────────────────────────┘

  Krawędzie skierowane (chronologiczne): edge (i→j) dla wszystkich i < j
  Edge attributes: edge_attr[i→j] = (t_j - t_i) / 24   [znormalizowane godziny]

╔══════════════════════════════════════════════════════╗
║              TEMPORAL GNN (T-GNN)                    ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Input Graph per patient:                            ║
║  ┌───────────────────────────────────────────────┐  ║
║  │  x:          (N, 64)   node features          │  ║
║  │  edge_index: (2, E)    directed edges         │  ║
║  │  edge_attr:  (E, 1)    Δt/24 (normalized)     │  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  GINEConv Layer 1  (supports edge_attr!)      │  ║
║  │  h_i = MLP([h_i + Σⱼ∈N(i) (h_j + W·e_ij)])  │  ║
║  │  → ReLU → (N, 128)                           │  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  GINEConv Layer 2 + Layer 3                   │  ║
║  │  → ReLU → (N, 128)                           │  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  Global Mean Pooling                          │  ║
║  │  (N, 128) → (1, 128)  jeden wektor na pacjenta│  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  Classifier Head                              │  ║
║  │  Linear(128→32) → ReLU → Dropout(0.3)        │  ║
║  │  → Linear(32→1) → Sigmoid                    │  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║                P(mortality) ∈ [0, 1]                 ║
║                                                      ║
║  Loss: BCEWithLogitsLoss (weighted, bo ~10-15% pos.) ║
║  Metryki: AUROC (cel >0.88), AUPRC                   ║
╚══════════════════════════════════════════════════════╝

Dlaczego GINEConv, a nie SAGEConv?
  SAGEConv ignoruje edge_attr → tracimy informację o Δt.
  GINEConv: h_i' = MLP(h_i + Σ(h_j + edge_attr_ij))
            → Δt jest wprost dodawane do reprezentacji sąsiada.
```

---

## Diagram 4 – Struktura projektu i przepływ kodu

```
GGSN_Projektowe/
│
├── data/
│   ├── raw/              ← surowe CSV z MIMIC (gitignored)
│   ├── processed/
│   │   ├── pairs_<cohort>_<strategy>.csv   (cohort-aware output, zawiera event_hours_from_intime + delta_hours_to_note)
│   │   └── signal_metadata_<cohort>_<strategy>.json
│   └── embeddings/
│       ├── node_embeddings.pt  (v3 stay-level, tylko referencja; wejście GNN = v4 note-level)
│       ├── text_tower.pt       (wagi TextTower po Fazie 1 — po v4: aktualizować)
│       └── signal_tower.pt
│
├── src/
│   ├── data_prep/
│   │   ├── extractor.py    → load_cohort, load_notes, load_vitals,
│   │   │                     pair_notes_vitals, run_extraction
│   │   ├── cleaner.py      → (TODO) usuwanie Data Leakage
│   │   └── preprocessor.py → (TODO) normalizacja, augmentacja
│   │
│   ├── models/
│   │   ├── towers.py       → TextTower (BERT), SignalTower (CNN)
│   │   └── tgnn_model.py   → (TODO) GINEConv + Classifier
│   │
│   ├── training/
│   │   ├── train_contrastive.py → CardiacPairsDataset, info_nce_loss, train()
│   │   ├── hard_negatives.py    → HardNegativeBatchSampler (nie używamy aktywnie)
│   │   ├── eval_embeddings.py   → linear probe AUROC, UMAP, diagonal gap
│   │   └── train_gnn.py         → (TODO) GraphDataset, GNN training loop
│   │
│   └── utils/
│       ├── graph_builder.py → (TODO) Data(x, edge_index, edge_attr)
│       └── metrics.py       → (TODO) AUROC, AUPRC
│
└── notebooks/
    ├── 01_extraction.ipynb          ← eksploracja danych i pipeline
    └── 02_contrastive_training.ipynb ← trening Fazy 1, podgląd embeddingów

─────────────────────────────────────
Zależności między plikami:
─────────────────────────────────────
extractor.py
    └──► cardio_pairs.csv
              └──► train_contrastive.py
                        ├──► towers.py (TextTower, SignalTower)
                        └──► node_embeddings.pt
                                  └──► graph_builder.py
                                            └──► train_gnn.py
                                                      └──► tgnn_model.py
                                                                └──► metrics.py
```

---

## Kluczowe decyzje projektowe

| Decyzja | Wybór | Uzasadnienie |
|---|---|---|
| Encoder tekstu | Bio_ClinicalBERT | Pre-trenowany na notatkach klinicznych (MIMIC-III) |
| Encoder sygnałów | 1D CNN | Zachowuje lokalną strukturę czasową; lekki (~50k params) |
| Wymiar embeddingu | 128 | Kompromis: wystarczający dla GNN, nie za duży do batch=8 |
| Loss Fazy 1 | InfoNCE (τ=0.07) | SimCLR default; in-batch negatives → O(B²) par |
| GNN | GINEConv | Jedyna standardowa warstwa w PyG obsługująca `edge_attr` |
| Krawędzie grafu | chronologiczne (i→j, i<j) | Modeluje przyczynowość; Δt jako edge_attr |
| BERT LR | 2e-5 | Fine-tuning: mały LR by nie zniszczyć pre-treningowych wag |
| Phase separation | TAK | BERT i GNN nigdy trenowane razem (per README) |
