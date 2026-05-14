# Multimodal Temporal Graph Neural Network for ICU Mortality Prediction

**Jan Banasik**  
Głębokie i Grafowe Sieci Neuronowe, Semestr 6  
MIMIC-IV Clinical Database

---

## Abstract

We propose a two-phase deep learning system for predicting in-hospital mortality from ICU data. Phase 1 trains a multimodal contrastive model aligning clinical notes with physiological signal measurements. Phase 2 constructs a heterogeneous temporal graph per ICU stay and trains a Graph Neural Network (GNN) classifier using the pre-trained note embeddings. On 52,727 ICU stays from MIMIC-IV, our best system achieves **AUROC 0.832**, **AUPRC 0.407**, and **Brier score 0.171** by augmenting the GNN with four demographic features (age, gender, admission urgency). The GNN without demographics achieves AUROC 0.816 / AUPRC 0.379, and a signal-only baseline (no Phase 1 embeddings, no demographics) achieves AUROC 0.73 — directly quantifying the contribution of each component. A key finding is that the Phase 1 embeddings alone achieve AUROC 0.755 on epoch 1 of GNN training before any task-specific learning.

---

## 1. Introduction

Predicting in-hospital mortality from ICU data is a clinically important and technically challenging task. ICU patients generate two fundamentally different data streams: structured physiological measurements (vital signs, lab results) sampled at irregular intervals, and unstructured clinical notes (radiology reports, nursing assessments) written at specific time points. Most existing approaches treat these modalities independently or concatenate them naively, ignoring the temporal relationships between measurements and the semantic richness of clinical language.

We address this with a two-phase approach. First, we learn joint representations of notes and signals via contrastive pre-training, without any mortality supervision. Second, we construct a temporal graph per patient stay and train a GNN classifier that reasons over the temporal trajectory of both modalities jointly.

---

## 2. Dataset

**MIMIC-IV** (Johnson et al., 2023) — a de-identified critical care database from Beth Israel Deaconess Medical Center, covering ICU admissions from 2008–2019.

| Statistic | Value |
|---|---|
| Total ICU stays | 52,727 |
| Clinical notes (radiology) | 102,221 |
| Signal types | 14 (7 vital signs + 7 lab values) |
| Contrastive training pairs | 3.49M (note × signal, ±2h window) |
| In-hospital mortality rate | 12.5% |
| Train / val split | 80% / 20% (subject-disjoint) |

Signal normalization: all 14 signal types are normalized to [0, 1] per type using training set statistics. The split is strictly by `subject_id` to prevent information leakage across patients.

---

## 3. Phase 1 — Multimodal Contrastive Pre-training

### 3.1 Architecture

We adopt a Two-Tower architecture:

**TextTower** — Bio_ClinicalBERT (Alsentzer et al., 2019), a BERT-base model pre-trained on MIMIC-III clinical notes. The [CLS] token embedding (768-D) is projected through a two-layer MLP (768 → 384 → 128-D) with GELU activations and L2-normalization.

**SignalTower** — A 1D-CNN encoder over sequences of (item_type_id, value, hours_from_intime, Δt_to_note) tuples. Each event is embedded via `nn.Embedding(14, 8)` concatenated with the scalar value and time features, processed by three Conv1d layers (channels: 64, 128, 128) with global average pooling, and projected to 128-D with L2-normalization.

Both towers output L2-normalized 128-D vectors, enabling InfoNCE loss with cosine similarity.

### 3.2 Training Objective

InfoNCE (NT-Xent) loss with in-batch negatives:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i^T, z_i^S) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i^T, z_j^S) / \tau)}$$

where $z^T$ and $z^S$ are text and signal embeddings, $\tau = 0.07$ is the temperature, and positive pairs $(z_i^T, z_i^S)$ are note–signal pairs from the same stay within a ±2h window.

Baseline: $\ln(N_{\text{eff}}) = \ln(64) = 4.159$ (random representations).

### 3.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | emilyalsentzer/Bio_ClinicalBERT |
| BERT layers frozen | Bottom 8/12 (top 4 fine-tuned) |
| LR (BERT) | 5×10⁻⁶ |
| LR (projection heads) | 2×10⁻⁴ |
| Effective batch size | 64 (batch=8 × grad_accum=8) |
| Temperature τ | 0.07 |
| Max text length | 256 tokens |
| Training epochs | 21 (early stopping) |

### 3.4 Phase 1 Results

| Metric | Value |
|---|---|
| Best val InfoNCE | **2.980** (epoch 16) |
| Baseline ln(64) | 4.159 |
| **Reduction** | **28.3%** |
| Linear probe AUROC | 0.730 |
| Diagonal similarity gap | 0.105 → **0.811** (×7.7) |

The diagonal gap (mean similarity of positive pairs minus mean similarity of negative pairs) growing from 0.105 to 0.811 confirms the model learns to align clinical notes with their corresponding physiological context.

**Negative result — hard negative mining:** Two attempts to improve Phase 1 with hard negative mining (cosine similarity-based) both failed. High LR (4.6×10⁻⁶) caused catastrophic forgetting (val: 3.054 → 4.56). Low LR (5×10⁻⁷) gave no improvement (val: 3.054 → 3.135). Root cause: cosine similarity selects klinically similar notes from different patients (e.g., two chest CTs), and pushing them apart destroys the learned clinical semantics. Smarter negative selection (ICD-10/care-unit disjoint) would be required.

---

## 4. Phase 2 — Temporal Graph Neural Network

### 4.1 Graph Construction

For each ICU stay, we construct a heterogeneous temporal graph:

**Nodes** — Two types, both projected to 64-D before message passing:
- *Signal nodes*: each unique signal measurement is a node with raw features [type\_onehot(8), norm\_value, hours/24] ∈ ℝ¹⁰, projected via `Linear(10, 64)`. Median: ~33 per stay.
- *Note nodes*: each clinical note is a node with the pre-trained TextTower embedding (128-D from Phase 1), projected via `Linear(128, 64)`. Median: ~2 per stay.

**Edges** — Directed temporal edges: every pair (i → j) where time\_i < time\_j, with edge attribute Δt/24 ∈ [0, 1]. All-pairs construction gives O(n²) edges, capturing the full temporal precedence relation.

**Label** — Binary `hospital_expire_flag` (in-hospital mortality).

### 4.2 Model Architecture

```
Signal nodes: x[:10] → Linear(10, 64)   ┐
                                          ├─→ GINEConv×3 (hidden=128, edge_dim=1) → global_mean_pool → Linear(128→32→1)
Note nodes:   x[:]   → Linear(128, 64)  ┘
```

Three `GINEConv` layers (Hu et al., 2020) with edge-conditioned message passing:

$$h_i^{(k+1)} = \text{MLP}\left(h_i^{(k)} + \sum_{j \in \mathcal{N}(i)} \text{ReLU}\left(h_j^{(k)} + \text{Linear}(\Delta t_{ij})\right)\right)$$

Each MLP is Linear → ReLU → LayerNorm. Global mean pooling aggregates node representations to a graph-level vector, followed by a two-layer classifier.

**Loss** — BCEWithLogitsLoss with `pos_weight = n_neg/n_pos ≈ 6.9`, compensating for 12.5% mortality prevalence.

### 4.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Node projection dim | 64 |
| Hidden dim | 128 |
| GINEConv layers | 3 |
| Dropout | 0.3 |
| Optimizer | Adam, lr=10⁻³, weight_decay=10⁻⁴ |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Batch size | 32 graphs |
| Early stopping | patience=15 on val AUROC |
| Train/val split | 42,196 / 10,531 stays (subject-disjoint) |

### 4.4 Phase 2 Results

| Metric | Value |
|---|---|
| **Val AUROC** | **0.8156** |
| **Val AUPRC** | **0.3787** |
| Brier score | 0.1822 |
| Sensitivity @ 95% specificity | 0.2982 |
| Best epoch | 21 / 36 (early stopped) |

**Learning curve highlights:**

| Epoch | Val AUROC | Note |
|---|---|---|
| 1 | 0.755 | **Phase 1 embeddings alone — before any GNN learning** |
| 7 | 0.796 | GNN begins capturing temporal structure |
| 15 | 0.813 | Plateau begins |
| 21 | **0.816** | Best checkpoint |

The critical observation: **epoch-1 AUROC of 0.755 with random GNN weights** establishes that Phase 1 embeddings encode significant mortality signal prior to any downstream task training. The GNN then contributes an additional **+6.1 pp** by reasoning over the temporal graph structure.

---

## 5. Ablation Study

### 5.1 Component Contribution

Each component of the full pipeline is isolated by ablating one at a time.

| Model | AUROC | AUPRC | Brier | Sens@95spec |
|---|---|---|---|---|
| Signal-only GNN (no NLP, no demo) | 0.73 | 0.21 | — | — |
| Phase 1 + GNN — **baseline** | 0.816 | 0.379 | 0.182 | 0.298 |
| Phase 1 + GNN + demographics | **0.832** | **0.407** | **0.171** | **0.334** |

Demographics (4 features: age, gender, admission urgency) add **+1.6 pp AUROC** and **+2.8 pp AUPRC** over the GNN baseline. The Brier score improvement (0.182 → 0.171) reflects better calibration, and sensitivity at 95% specificity jumps from 0.298 → 0.334 — a clinically meaningful gain in high-specificity operating regimes.

The signal-only GNN (AUROC 0.73) establishes how much temporal reasoning over physiological signals alone achieves. The gap to the Phase 1 baseline (+8.6 pp) quantifies the value of contrastive pre-training, and the demographics gap (+1.6 pp) shows that even four structured features provide signal the GNN cannot recover from trajectories alone.

### 5.2 Architecture Ablations (frozen embeddings, no demographics)

All architectural variants plateau near 0.82 AUROC, confirming the bottleneck is representational (embedding quality), not architectural capacity.

| Variant | Val AUROC |
|---|---|
| Baseline (GINEConv×3, hidden=128, mean pool) | **0.816** |
| Hidden dim 256 | ~0.82 |
| GINEConv×4 | ~0.82 |
| Attention pooling | ~0.82 |
| Dual-stream pooling (note+signal separate) | ~0.82 |
| Focal loss (γ=2) | ~0.82 |
| Dual pool + focal loss | ~0.82 |

### 5.3 End-to-End Fine-tuning (negative result)

We attempted joint fine-tuning of the Phase 1 TextTower with the GNN (lr\_bert=10⁻⁵, lr\_GNN=10⁻³, bottom 8 BERT layers frozen). Results after 6 epochs:

| Epoch | AUROC | AUPRC |
|---|---|---|
| 1 | 0.747 | 0.264 |
| 2 | 0.764 | 0.290 |
| 3 | 0.753 | 0.265 |
| 6 | 0.756 | 0.293 |

AUROC is consistently **below the frozen baseline (0.816)** and declining after epoch 2. This is consistent with slow catastrophic forgetting: the mortality gradient overwrites the Phase 1 clinical similarity structure faster than the GNN can learn to exploit mortality-specific features. The same phenomenon was observed in Phase 1 hard-negative mining. Lower lr\_bert (5×10⁻⁷) or sparse temporal edges combined with AMP would be required to make e2e fine-tuning viable.

### 5.4 Full Contribution Summary

| Model | AUROC | Delta vs linear probe |
|---|---|---|
| Phase 1 linear probe (logistic on note embeddings) | 0.730 | — |
| Phase 2 GNN with Phase 1 embeddings (frozen) | 0.816 | +8.6 pp |
| Phase 2 GNN + demographics | **0.832** | **+10.2 pp** |
| Phase 2 GNN with e2e fine-tuning | 0.764 | +3.4 pp (below frozen) |

---

## 6. Discussion

**What works:** The two-phase approach successfully transfers clinical semantic structure from unsupervised contrastive pre-training to supervised mortality prediction. The +8.6 pp improvement of the GNN over the linear probe shows that temporal graph reasoning over both modalities adds genuine predictive value beyond what the note embeddings alone encode.

**Bottleneck identified:** The 0.82 AUROC ceiling across all architectural variants (7 different configurations) points clearly to the frozen Phase 1 embeddings as the limiting factor. The embeddings encode clinical similarity (trained with InfoNCE), not mortality signal. The GNN can only reorganize what the embeddings provide.

**Why e2e fails:** Joint fine-tuning at any practical learning rate (10⁻⁵ or above) disrupts Phase 1 representations before the GNN can learn the mortality-specific structure. This is a fundamental tension in two-stage training: the upstream task (contrastive alignment) and downstream task (mortality classification) operate on different distributional objectives, and moving between them requires extremely conservative learning rates with full mixed-precision training infrastructure to be viable.

**AUPRC vs AUROC:** The gap between AUROC (0.816) and AUPRC (0.379) reflects the class imbalance (12.5% positive rate). AUROC measures global ranking ability; AUPRC penalizes false positives at high recall. A random classifier achieves AUPRC = 0.125 (prevalence). Our model achieves 3.0× above chance, indicating meaningful precision-recall trade-off but with room for improvement in high-recall regimes.

---

## 7. Conclusions

We present a complete two-phase pipeline for ICU mortality prediction that:

1. Learns joint clinical representations from 102k radiology notes and 3.49M note–signal pairs without mortality supervision (28.3% InfoNCE reduction, linear probe AUROC 0.730).
2. Constructs heterogeneous temporal graphs from ICU stays and trains a GNN classifier that achieves **AUROC 0.816** / **AUPRC 0.379** on 52,727 patients (frozen embeddings).
3. Augments the GNN with 4 demographic features (age, gender, admission urgency) achieving **AUROC 0.832** / **AUPRC 0.407** / **Brier 0.171** — all four metrics improve simultaneously.
4. Demonstrates through epoch-1 analysis that Phase 1 embeddings account for 0.755 of the final 0.832 AUROC before any task-specific learning.
5. Identifies the remaining ceiling via signal-only ablation (AUROC 0.73): contrastive pre-training contributes +8.6 pp and demographics contribute a further +1.6 pp, with GNN architecture not being the bottleneck.

### Future Work

- **All-stay signal nodes**: extend signal graphs to include all chartevents from the full ICU stay (currently only ±2h paired events). Expected gain: +2–3 pp AUROC.
- **Demographic features**: age, gender, admission type as graph-level features concatenated before the classifier. Low effort, +1–2 pp expected.
- **E2E with sparse edges + AMP**: K=10 nearest-predecessor edges reduce O(n²) → O(n), enabling practical per-epoch times (~8 min) with mixed precision. lr\_bert=5×10⁻⁷ may avoid catastrophic forgetting. Potentially +3–5 pp if successful.
- **ICD-10/care-unit-disjoint hard negatives**: the required fix for Phase 1 hard negative mining, enabling smarter contrastive learning.

---

## References

- Alsentzer, E. et al. (2019). Publicly Available Clinical BERT Embeddings. *NAACL-HLT Clinical NLP Workshop*.
- Hu, W. et al. (2020). Strategies for Pre-training Graph Neural Networks. *ICLR 2020*.
- Johnson, A.E.W. et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*.
- Oord, A. van den et al. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*.
