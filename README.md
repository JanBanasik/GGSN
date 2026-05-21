# Temporal GNN + Contrastive Learning — predykcja śmiertelności na MIMIC-IV

Projekt semestralny GGSN (Głębokie i Grafowe Sieci Neuronowe).
Pełny opis architektury, eksperymentów i wyników: **[RAPORT.md](RAPORT.md)**.

## Co to robi

Predykcja in-hospital mortality na podstawie danych z pełnego pobytu na OIOM
(MIMIC-IV 3.1, 52 727 pobytów). Pipeline dwufazowy:

1. **Phase 1 — Contrastive pre-training (Two-Tower)**: Bio_ClinicalBERT
   (notatki radiologiczne) + 1D-CNN (sygnały fizjologiczne) trenowane
   InfoNCE na parach (notatka, sygnały ±2h). Wynik: 128-dim embeddingi notatek.
2. **Phase 2 — Temporal Heterogeneous GNN**: graf per pobyt z węzłami
   signal / note / ICD-Charlson, GINEConv×3 z atrybutami czasowymi krawędzi,
   demografika konkatenowana po poolingu, klasyfikator BCE/Focal.

Najlepszy model: **AUROC 0.850, AUPRC 0.465** (`demo_attention_focal2`,
checkpoint `data/snapshots/gnn/version_10/`).

## Struktura

```
src/
  data_prep/       # extractor.py, cleaner.py, extract_{icd,demographics,all_stay_signals}.py
                   # graph_datamodule.py, contrastive_dataset.py
  models/          # towers.py (Phase 1), tgnn_model.py + gnn_module.py (Phase 2)
  training/        # train_contrastive.py, train_gnn.py, loss.py, snapshot.py
  utils/           # graph_builder.py, gnn_dataset.py, metrics.py
  experimental/    # e2e.py, e2e_module.py, train_e2e.py — historyczny E2E (catastrophic forgetting)
                   # hard_negatives.py wciąż w training/ — opcjonalny flag, off-by-default
  visualization/   # Manim animacje (similarity matrix, UMAP, two-tower)
data/
  raw/             # symlink/CSV z MIMIC-IV (gitignored)
  processed/       # CSV-e par + cache grafów .pt (gitignored)
  embeddings/      # node_embeddings.pt, text_tower.pt, signal_tower.pt
  snapshots/       # CSVLogger runów Phase 2 (gnn/version_{0..15}/)
  plots/           # embedding_comparison_v4.png
notebooks/
  03_demo.ipynb            # cienki notebook demo (kod = source of truth w src/)
run_experiments.sh         # cały suite ablacji Phase 2
RAPORT.md                  # pełny raport projektowy
```

## Uruchomienie

```bash
# Setup
uv sync

# Phase 1 (Bio_ClinicalBERT + SignalTower, ~3-4h na RTX 4060 8GB)
uv run python -m src.data_prep.extractor --cohort all-icus --pair-strategy note_level
uv run python -m src.training.train_contrastive \
  --csv-path data/processed/pairs_all-icus_note_level.csv

# Phase 2 — najlepszy model
uv run python -m src.training.train_gnn \
  --demo-path data/processed/demographics.csv \
  --pooling attention --focal-gamma 2.0

# Pełny ablacyjny suite (~2-3h)
bash run_experiments.sh
```

## Zabezpieczenia przed data leakage

- Subject-disjoint split (pacjent nigdy w obu zbiorach)
- `cleaner.py` filtruje frazy `expired|deceased|DNR|...` w notatkach
- ICD: tylko 19 kategorii Charlson (przewlekłe choroby), nie kody wyniku
- `--max-signals N` trzyma **najwcześniejsze** N sygnałów (nie agonalne końcowe)
- Krawędzie grafu tylko do przodu w czasie

Patrz [RAPORT.md §4](RAPORT.md#4-zabezpieczenia-przed-data-leakage--podsumowanie).

## Najważniejsze pliki

- **[RAPORT.md](RAPORT.md)** — pełny raport projektowy z architekturą, ablacjami i wynikami
- `run_experiments.sh` — pełen suite Phase 2 (~2-3h na RTX 4060)
- `data/snapshots/gnn/version_10/checkpoints/best.ckpt` — najlepszy model
- `data/snapshots/run_20260512_200632/` — canonical Phase 1 v4
