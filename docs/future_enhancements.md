# Future Enhancements

Current best: AUROC 0.832, AUPRC 0.407, Brier 0.171 (Phase 1 + GNN + demographics).
Written 2026-05-15 after project reach its current state.

---

## 1. All-stay signal nodes
**Expected gain:** +2–3 pp AUROC  
**Effort:** Medium (data pipeline only, no model changes)  
**Risk:** Low

Currently signal nodes only include measurements within ±2h of a radiology note — the pairing window inherited from Phase 1. The full ICU stay has chartevents every 15–30 minutes for its entire duration. Adding all of them as signal nodes makes the temporal graph represent the actual patient trajectory instead of snapshots near note-writing moments.

What needs to change:
- `extractor.py`: extract ALL chartevents per stay (not just paired ones), save to a new CSV column or separate file
- `gnn_dataset.py`: load full-stay signals alongside the paired ones
- `graph_builder.py`: no changes needed, signal node construction is generic
- New cache: `graphs_cache_all_signals.pt` (will be larger, ~2× nodes per graph)
- Training: same command + `--cache-path data/processed/graphs_cache_all_signals.pt`

---

## 2. ICD-10 diagnosis codes as a third node type
**Expected gain:** +1–3 pp AUROC  
**Effort:** Medium (new node type in model + graph builder)  
**Risk:** Low-medium

MIMIC has ICD-10 diagnosis codes per admission (`hosp/diagnoses_icd.csv.gz`). These represent chronic conditions — diabetes, CHF, COPD, cancer — that are strong mortality predictors the GNN currently has no access to.

Architecture: add a third node type with a learned embedding per ICD code. Since diagnoses have no timestamp, connect them to all other nodes with a special edge type (or just connect to the first/last node). The heterogeneous graph already supports multiple node types.

What needs to change:
- `extractor.py` or new `extract_icd.py`: load top-N most common ICD-10 codes per stay (N ~20)
- `graph_builder.py`: add `icd_nodes` — each code is a node, features = learned embedding
- `tgnn_model.py`: add `icd_proj` projection layer (embedding_dim → 64)
- Model handles the new node type the same way as signal/note nodes

---

## 3. Weak mortality supervision in Phase 1 (semi-supervised)
**Expected gain:** +2–4 pp AUROC  
**Effort:** Medium (Phase 1 retraining required)  
**Risk:** Medium — may destabilize contrastive alignment

The fundamental bottleneck: Phase 1 embeddings encode clinical similarity (InfoNCE objective), not mortality signal. Fix: add a small auxiliary classification head during Phase 1 contrastive training.

```
L_total = L_InfoNCE + 0.1 × L_mortality_bce
```

The mortality head would operate on the averaged (text + signal) embedding with a stop-gradient on the signal branch to avoid leakage. The weight (0.1) keeps InfoNCE dominant so alignment is preserved. This gives the TextTower a mortality gradient during pre-training without destroying the cross-modal structure.

Requires: Phase 1 full rerun (~21 epochs × 2-3h = overnight). If it works, Phase 2 starts from a much stronger prior.

---

## 4. Nursing and physician notes (richer text modality)
**Expected gain:** +3–5 pp AUROC (speculative)  
**Effort:** High (data prep + leakage filtering is hard)  
**Risk:** High — nursing notes contain strong leakage phrases

Currently TextTower only sees radiology reports (written once, fairly formulaic). Nursing progress notes and physician notes are written continuously and contain the most direct mortality signal. Including them in Phase 1 would fundamentally change what the embeddings learn.

Blockers:
- Much stronger leakage filtering needed (`cleaner.py` must catch more phrases)
- Notes are noisier, longer, require more aggressive truncation
- Phase 1 would need longer training (more pairs)
- Data: `mimic-iv-note/note/discharge.csv.gz`, `note/radiology.csv.gz` (already used)

---

## 5. E2E fine-tuning with sparse edges + AMP
**Expected gain:** +3–5 pp AUROC (if it doesn't catastrophically forget)  
**Effort:** High  
**Risk:** High — two previous attempts failed

The current e2e run failed because:
1. All-pairs O(n²) edges → ~57 min/epoch (too slow for meaningful LR search)
2. lr_bert=1e-5 causes catastrophic forgetting of Phase 1 representations

Required fixes to make it viable:
- Sparse edges: K=10 nearest-predecessor edges instead of all-pairs. Reduces O(n²) → O(n·K). Requires `build_patient_graph_e2e` change + cache rebuild.
- AMP (mixed precision): `torch.cuda.amp.autocast()` in train loop → ~2× speedup
- lr_bert=5e-7 (100× lower than current) — barely touches BERT weights per step
- Combined: ~5-8 min/epoch, viable for 50+ epoch runs

In `train_gnn.py`: add `--sparse-k` flag, in `graph_builder.py`: replace all-pairs loop with K-nearest construction using sorted timestamps.

---

## 6. Richer demographic features
**Expected gain:** +0.5–1 pp AUROC  
**Effort:** Low (extend `extract_demographics.py`)  
**Risk:** Very low

Current DEMO_DIM=4: age_norm, gender_f, is_emergency, is_elective.

Candidates to add:
- `first_careunit` one-hot (MICU, SICU, CCU, CVICU, TSICU, other) → 6 features
- `insurance` ordinal (Medicare/Medicaid/Other) → 2 features
- `marital_status` binary (married vs not) → 1 feature

New DEMO_DIM ~13. All are in `admissions.csv.gz` or `icustays.csv.gz` already loaded by `extract_demographics.py`. No model architecture changes beyond updating `DEMO_DIM` constant in `graph_builder.py`.

---

## 7. Multi-task learning (auxiliary targets)
**Expected gain:** +1–2 pp AUROC  
**Effort:** Medium  
**Risk:** Low-medium

Train the GNN simultaneously on multiple related targets instead of mortality alone. Auxiliary tasks regularize the representation and can improve the main task.

Candidates:
- **ICU length of stay** (regression): available from `icustays.los`
- **30-day readmission**: derivable from `admissions` table
- **Sepsis flag**: derivable from ICD codes (A40.*, A41.*)

Architecture: shared GNN backbone, separate classifier heads per task.
```
loss = L_mortality + 0.3 × L_los + 0.2 × L_readmission
```

---

## 8. Transformer over temporal sequence (alternative to GNN)
**Expected gain:** Unknown — could be better or worse  
**Effort:** High (new model class)  
**Risk:** Medium

Instead of a GNN, treat the ICU stay as a sequence of events and apply a causal transformer (GPT-style) over them. Each event (signal or note) is a token; position encoding encodes absolute time. The [CLS]-equivalent at the end is passed to the mortality classifier.

Advantage over GNN: transformers naturally handle variable-length sequences and learn arbitrary temporal patterns without explicit edge construction. Disadvantage: loses the explicit temporal graph structure and Δt edge attributes.

Would require a new `TemporalTransformer` model class and a sequence-format dataset builder (simpler than graph construction — just sort events chronologically).

---

## Priority order (recommended)

| # | Enhancement | Gain | Effort | Do it? |
|---|---|---|---|---|
| 1 | All-stay signal nodes | +2–3 pp | Medium | **Yes — next** |
| 2 | ICD-10 node type | +1–3 pp | Medium | Yes |
| 3 | Richer demographics | +0.5–1 pp | Low | Yes (quick win) |
| 4 | Weak mortality in Phase 1 | +2–4 pp | Medium | Yes (overnight run) |
| 5 | Multi-task learning | +1–2 pp | Medium | Maybe |
| 6 | Nursing notes | +3–5 pp | High | Only if time |
| 7 | E2E sparse + AMP | +3–5 pp | High | Only if time |
| 8 | Transformer alternative | Unknown | High | Research interest only |
