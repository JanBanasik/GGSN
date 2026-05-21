#!/usr/bin/env bash
# run_experiments.sh — Full ablation suite for Phase 2 Temporal GNN
#
# Runs all experiments sequentially; logs to data/snapshots/gnn/<version>/
# Results printed at end via: grep -r test_ data/snapshots/gnn/
#
# Usage:
#   bash run_experiments.sh           # all experiments
#   bash run_experiments.sh --dry-run # print commands only
#
# Runtime estimate: ~5-15 min per experiment depending on GPU/CPU.
# Total: ~14 experiments = 1-3 hours.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# MIMIC-IV data lives one level above GGSN_Projektowe
MIMIC_ROOT="$(dirname "$(pwd)")/datasets_mimic/mimic/files/mimiciv/3.1"
MIMIC_NOTE_ROOT="$(dirname "$(pwd)")/datasets_mimic/mimic-notes/files/mimic-iv-note/2.2"
export MIMIC_IV_ROOT="$MIMIC_ROOT"
export MIMIC_IV_NOTE_ROOT="$MIMIC_NOTE_ROOT"
echo "MIMIC_IV_ROOT=$MIMIC_IV_ROOT"

CSV="data/processed/pairs_all-icus_note_level.csv"
EMB="data/embeddings/node_embeddings.pt"
CACHE="data/processed/graphs_cache.pt"
DEMO="data/processed/demographics.csv"
ICD="data/processed/icd_charlson_all-icus.csv"
ALL_STAY="data/processed/all_stay_signals_all-icus.csv"

LOG="data/experiments_log.txt"
mkdir -p data/snapshots/gnn

run() {
    local name="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Experiment: $name"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
    echo "  CMD: uv run python -m src.training.train_gnn $*"
    echo ""

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY RUN — skipping]"
        return
    fi

    local start=$SECONDS
    # shellcheck disable=SC2068
    uv run python -m src.training.train_gnn "$@" 2>&1 | tee -a "$LOG"
    local elapsed=$(( SECONDS - start ))
    echo "  [done in ${elapsed}s]" | tee -a "$LOG"
}

echo "=== GGSN GNN Experiment Suite (continuing from #5) ===" | tee -a "$LOG"
echo "Resumed: $(date)" | tee -a "$LOG"

# ─── Experiments 1-5 already completed — continuing from #6 ──────────────────
# Results so far:
#   baseline:           AUROC=0.8293  AUPRC=0.4033  Brier=0.1799  sens@95spec=0.3265
#   signal_only:        AUROC=0.7837  AUPRC=0.3812  Brier=0.1814  sens@95spec=0.3013
#   baseline_demo:      AUROC=0.8407  AUPRC=0.4288  Brier=0.1716  sens@95spec=0.3525  ← best
#   baseline_icd:       AUROC=0.8230  AUPRC=0.3773  Brier=0.2017  sens@95spec=0.3108
#   baseline_allstay50: AUROC=0.8197  AUPRC=0.3958  Brier=0.1705  sens@95spec=0.3273

# ─── Experiments ─────────────────────────────────────────────────────────────

# ICD and all-stay didn't improve over baseline — focus remaining experiments
# on demo (best single feature) + pooling, loss, and architecture ablations.

# ─── Pooling ablations on demo baseline ──────────────────────────────────────

# 6. Attention pooling + demo
run "demo_attention" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --demo-path "$DEMO" \
    --pooling attention

# 7. Dual pooling + demo
run "demo_dual" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --demo-path "$DEMO" \
    --pooling dual

# ─── Loss ablations ──────────────────────────────────────────────────────────

# 8. Focal loss gamma=2, baseline
run "baseline_focal2" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --focal-gamma 2.0

# 9. Focal loss gamma=2 + demo
run "demo_focal2" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --demo-path "$DEMO" \
    --focal-gamma 2.0

# 10. Focal gamma=2 + demo + attention (best combo candidate)
run "demo_attention_focal2" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --demo-path "$DEMO" \
    --pooling attention \
    --focal-gamma 2.0

# ─── Architecture depth ablation ─────────────────────────────────────────────

# 11. Shallower: 2 GNN layers
run "baseline_2layers" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --n-layers 2

# 12. Deeper: 4 GNN layers
run "baseline_4layers" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --n-layers 4

# 13. Demo + 4 layers
run "demo_4layers" \
    --csv-path "$CSV" \
    --embeddings-path "$EMB" \
    --cache-path "$CACHE" \
    --demo-path "$DEMO" \
    --n-layers 4

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════"
echo "  All experiments done: $(date)"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Test metrics (AUROC, AUPRC, Brier, sens@95spec):"
grep -r "test_" data/snapshots/gnn/ 2>/dev/null | grep -v "\.csv:" | sort || echo "  (no results yet)"
echo ""
echo "CSVLogger results in: data/snapshots/gnn/"
echo "Full log: $LOG"
