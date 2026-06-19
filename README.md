# Temporal GNN and Contrastive Learning for ICU Mortality Prediction

This project predicts in-hospital mortality from MIMIC-IV ICU stays using a
two-stage multimodal pipeline:

1. **Contrastive pre-training** aligns Bio_ClinicalBERT representations of
   radiology notes with 1D-CNN representations of physiological signals.
2. **Temporal heterogeneous GNN** combines the learned note embeddings,
   vital/lab events, Charlson comorbidities, and demographics.

The best evaluated configuration reached **0.850 AUROC** and **0.465 AUPRC**.
The complete architecture, ablations, and discussion are documented in the
[project report](RAPORT.md) (Polish).

> This is a research and educational project, not a clinical device. It must
> not be used for diagnosis or patient-care decisions.

## Repository contents

```text
src/
  data_prep/       MIMIC-IV extraction, leakage filtering, and datasets
  models/          two-tower encoders and the temporal GNN
  training/        Phase 1/Phase 2 training, losses, and evaluation
  utils/           graph construction and metrics
  experimental/    archived end-to-end experiment (negative result)
  visualization/   optional Manim/UMAP visualizations
docs/presentation/ final slides and their Python generator
tests/              fast tests that use synthetic data only
run_experiments.sh Phase 2 ablation runner
RAPORT.md           full project report
```

Raw MIMIC data, processed datasets, embeddings, experiment logs, and model
checkpoints are intentionally excluded from Git. Consequently, the reported
metrics can be reviewed from the report, but reproducing them requires
credentialed access to the datasets and running the training pipeline.

## Prerequisites and data access

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- Access to [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/)
  and [MIMIC-IV-Note 2.2](https://physionet.org/content/mimic-iv-note/2.2/)
- A CUDA-capable GPU is strongly recommended for contrastive pre-training

PhysioNet requires credentialing and a data use agreement. Do not commit or
redistribute MIMIC data under this repository's MIT license.

Install the training environment:

```bash
uv sync
```

Optional environments:

```bash
uv sync --extra notebook       # Jupyter demo work
uv sync --extra presentation   # regenerate the PowerPoint deck
uv sync --extra viz            # UMAP utilities; Manim requires its own environment
uv sync --group dev            # linting and tests
```

By default the extractors expect this local, ignored layout:

```text
data/raw/mimiciv/3.1/
  hosp/{admissions,patients,labevents,diagnoses_icd}.csv.gz
  icu/{icustays,chartevents}.csv.gz
data/raw/mimic-iv-note/2.2/
  note/radiology.csv.gz
```

The roots can instead be provided explicitly:

```bash
export MIMIC_IV_ROOT=/path/to/mimiciv/3.1
export MIMIC_IV_NOTE_ROOT=/path/to/mimic-iv-note/2.2
```

## Running the pipeline

Prepare Phase 1 note-signal pairs:

```bash
uv run python -m src.data_prep.extractor \
  --cohort all-icus \
  --pair-strategy note_level
```

Train the two-tower model (approximately 3–4 hours on an RTX 4060 8 GB):

```bash
uv run python -m src.training.train_contrastive \
  --csv-path data/processed/pairs_all-icus_note_level.csv
```

Prepare demographics and Charlson features, then train the best Phase 2
configuration:

```bash
uv run python -m src.data_prep.extract_demographics
uv run python -m src.data_prep.extract_icd --cohort all-icus
uv run python -m src.training.train_gnn \
  --demo-path data/processed/demographics.csv \
  --icd-path data/processed/icd_charlson_all-icus.csv \
  --pooling attention \
  --focal-gamma 2.0
```

Run the complete Phase 2 ablation suite (approximately 2–3 hours):

```bash
bash run_experiments.sh
```

Evaluate a Phase 1 run by naming its output directory explicitly:

```bash
uv run python -m src.training.eval_embeddings \
  --run-dir data/snapshots/run_<timestamp>
```

Use `--help` on each module for the complete CLI. Outputs are written below
`data/`, which remains ignored by Git.

## Leakage controls

- Patient-disjoint train/validation/test splits
- Filtering of notes containing explicit outcome and end-of-life phrases
- Charlson chronic-disease categories instead of outcome codes
- Earliest-event selection when limiting physiological signals
- Directed graph edges that only move forward in time

See [section 4 of the report](RAPORT.md#4-zabezpieczenia-przed-data-leakage--podsumowanie)
for the full methodology.

## Development and release checks

```bash
uv run ruff check .
uv run pytest
uv run python -m compileall -q src
bash -n run_experiments.sh
```

The test suite uses synthetic inputs and never requires MIMIC data. GitHub
Actions runs the same lint and test checks on Python 3.11.

## License

Source code is released under the [MIT License](LICENSE). MIMIC-IV and
MIMIC-IV-Note remain governed by their respective PhysioNet terms.

## Authors

- [Jan Banasik](https://github.com/JanBanasik)
- [Antoni Pater](https://github.com/antoniopater)
