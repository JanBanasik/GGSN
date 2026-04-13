# Architektura Systemu – Temporal GNN + Contrastive Learning

Poniższe diagramy opisują pełen pipeline predykcji śmiertelności szpitalnej
na danych MIMIC-IV (pierwsze 24h pobytu na CCU/CVICU).

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
                        │  ├── load_cohort()      → CCU/CVICU stays + mortality
                        │  ├── load_notes()       → notatki w oknie [t₀, t₀+24h]
                        │  ├── load_vitals()      → HR(220045) + BP(220181)
                        │  └── pair_notes_vitals()→ pary w oknie ±2h
                        ▼
              ┌─────────────────────┐
              │  cardio_pairs.csv   │  ← 7 235 par (855 unikalnych notatek)
              │  (data/processed/)  │     483 stay_id, mortality rate ~X%
              └──────────┬──────────┘
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
              │  (data/embeddings/)  │     855 wektorów węzłów
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
║  │  1D CNN Encoder                             │    ║
║  │                                             │    ║
║  │  Input: (B, seq=32, 2)                      │    ║
║  │    ch0: item_type ∈ {0.0=HR, 1.0=BP}       │    ║
║  │    ch1: norm_value ∈ [0, 1]                │    ║
║  │                                             │    ║
║  │  Transpose → (B, 2, 32)                    │    ║
║  │  Conv1d(2→64,  k=3) → BN → ReLU           │    ║
║  │  Conv1d(64→128, k=5) → BN → ReLU          │    ║
║  │  Conv1d(128→128,k=3) → BN → ReLU          │    ║
║  │  AdaptiveAvgPool1d(1) → (B, 128)           │    ║
║  └─────────────────────────────────────────────┘    ║
║                                                      ║
║  Wejście: (B, seq_len, 2) – vitale posortowane       ║
║           czasowo, padding zerami                    ║
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
WEJŚCIE: jeden pacjent (stay_id) z N zdarzeniami klinicznych

  Zdarzenia posortowane chronologicznie:
  event₀(t=0h) → event₁(t=3h) → event₂(t=7h) → event₃(t=18h)

  Każdy węzeł:
  x_i = embedding z Fazy 1 (128-D, zamrożony)

  Krawędzie skierowane (chronologiczne):
  edge (i→j) istnieje gdy i < j (lub tylko i→i+1 dla łańcucha)

  Edge attributes (kluczowy element T-GNN!):
  edge_attr[i→j] = Δt[i,j] = t_j - t_i   [w godzinach]

╔══════════════════════════════════════════════════════╗
║              TEMPORAL GNN (T-GNN)                    ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Input Graph per patient:                            ║
║  ┌───────────────────────────────────────────────┐  ║
║  │  x:          (N, 128)  node embeddings        │  ║
║  │  edge_index: (2, E)    directed edges         │  ║
║  │  edge_attr:  (E, 1)    Δt in hours            │  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  GINEConv Layer 1  (supports edge_attr!)      │  ║
║  │  h_i = MLP([h_i + Σⱼ∈N(i) (h_j + W·e_ij)])  │  ║
║  │  → ReLU → (N, 256)                           │  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  GINEConv Layer 2                             │  ║
║  │  → ReLU → (N, 256)                           │  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  Global Mean Pooling                          │  ║
║  │  (N, 256) → (1, 256)  jeden wektor na pacjenta│  ║
║  └──────────────────────┬────────────────────────┘  ║
║                         │                            ║
║  ┌──────────────────────▼────────────────────────┐  ║
║  │  Classifier Head                              │  ║
║  │  Linear(256→64) → ReLU → Dropout(0.3)        │  ║
║  │  → Linear(64→1) → Sigmoid                    │  ║
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
│   │   └── cardio_pairs.csv   (7 235 par, wyjście ekstraktora)
│   └── embeddings/
│       ├── node_embeddings.pt  (855 wektorów 128-D, wejście GNN)
│       ├── text_tower.pt       (wagi TextTower po Fazie 1)
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
