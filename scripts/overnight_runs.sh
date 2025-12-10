#!/bin/bash
# =============================================================================
# OVERNIGHT SAE TRAINING RUNS
# =============================================================================
#
# Purpose: Test whether silence filtering was causing the steering issues
#
# Runs:
#   1. v2_no_filter      - Primary fix: train on ALL timesteps (164K vs 58K)
#   2. v2_32x_no_filter  - Smaller model: 32× expansion (2048 features)
#   3. v1_arch_v2_data   - V1 architecture on V2 data for comparison
#   4. v2_topk64         - Larger topk with 64× expansion
#
# Expected time: ~6 hours total (1.5h each for 100K steps)
#
# Usage:
#   chmod +x scripts/overnight_runs.sh
#   DYLD_FALLBACK_LIBRARY_PATH=/usr/local/ffmpeg7/lib ./scripts/overnight_runs.sh
#
# =============================================================================

set -e  # Exit on error

# Configuration
LOG_DIR="experiments/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# =============================================================================
# RUN 1: Primary Fix — No Silence Filtering
# =============================================================================
# Hypothesis: Training on ALL timesteps (164K) should give SAE knowledge of
# what silence latents look like, preventing garbage output at inference.

run_v2_no_filter() {
    log "=========================================="
    log "RUN 1/5: v2_no_filter (64×, topk=32, NO filter)"
    log "=========================================="

    python scripts/03_train_sae.py \
        --experiment_name v2_no_filter \
        --expansion_factor 64 \
        --topk 32 \
        --no_filter_silence \
        --num_steps 100000

    log "RUN 1 COMPLETE: v2_no_filter"
}

# =============================================================================
# RUN 2: Smaller Model — 32× Expansion, No Filtering
# =============================================================================
# Hypothesis: 4096 features may be overkill for 164K samples.
# 2048 features with same sparsity might generalize better.

run_v2_32x_no_filter() {
    log "=========================================="
    log "RUN 2/5: v2_32x_no_filter (32×, topk=32, NO filter)"
    log "=========================================="

    python scripts/03_train_sae.py \
        --experiment_name v2_32x_no_filter \
        --expansion_factor 32 \
        --topk 32 \
        --no_filter_silence \
        --num_steps 100000

    log "RUN 2 COMPLETE: v2_32x_no_filter"
}

# =============================================================================
# RUN 3: V1 Architecture on V2 Data
# =============================================================================
# Hypothesis: V1's expansion=16, topk=64 actually worked better for steering.
# Let's compare it on v2 data without filtering.

run_v1_arch_v2_data() {
    log "=========================================="
    log "RUN 3/5: v1_arch_v2_data (16×, topk=64, NO filter)"
    log "=========================================="

    python scripts/03_train_sae.py \
        --experiment_name v1_arch_v2_data \
        --expansion_factor 16 \
        --topk 64 \
        --no_filter_silence \
        --num_steps 50000

    log "RUN 3 COMPLETE: v1_arch_v2_data"
}

# =============================================================================
# RUN 4: Larger TopK with 64× Expansion
# =============================================================================
# Hypothesis: topk=32 with 4096 features might be too sparse.
# topk=64 (like v1) might give better reconstruction and steering.

run_v2_topk64() {
    log "=========================================="
    log "RUN 4/5: v2_topk64 (64×, topk=64, NO filter)"
    log "=========================================="

    python scripts/03_train_sae.py \
        --experiment_name v2_topk64 \
        --expansion_factor 64 \
        --topk 64 \
        --no_filter_silence \
        --num_steps 100000

    log "RUN 4 COMPLETE: v2_topk64"
}

# =============================================================================
# RUN 5: AdamW Optimizer — Weight Decay Regularization
# =============================================================================
# Hypothesis: AdamW with weight decay may help prevent feature collapse
# and improve generalization compared to vanilla Adam.

run_v2_adamw_no_filter() {
    log "=========================================="
    log "RUN 5/5: v2_adamw_no_filter (64×, topk=32, AdamW optimizer)"
    log "=========================================="

    python scripts/03_train_sae.py \
        --experiment_name v2_adamw_no_filter \
        --expansion_factor 64 \
        --topk 32 \
        --optimizer adamw \
        --no_filter_silence \
        --num_steps 100000

    log "RUN 5 COMPLETE: v2_adamw_no_filter"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log "=========================================="
    log "STARTING OVERNIGHT SAE TRAINING RUNS"
    log "=========================================="
    log "Log directory: $LOG_DIR"
    log "Estimated time: ~7 hours (5 runs)"
    log ""

    START_TIME=$(date +%s)

    # Run 1: Primary fix
    run_v2_no_filter

    # Run 2: Smaller model
    run_v2_32x_no_filter

    # Run 3: V1 architecture
    run_v1_arch_v2_data

    # Run 4: Larger topk
    run_v2_topk64

    # Run 5: AdamW optimizer
    run_v2_adamw_no_filter

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))

    log "=========================================="
    log "ALL RUNS COMPLETE!"
    log "=========================================="
    log "Total time: ${HOURS}h ${MINUTES}m"
    log ""
    log "Checkpoints saved to:"
    log "  - experiments/v2_no_filter/checkpoints/"
    log "  - experiments/v2_32x_no_filter/checkpoints/"
    log "  - experiments/v1_arch_v2_data/checkpoints/"
    log "  - experiments/v2_topk64/checkpoints/"
    log "  - experiments/v2_adamw_no_filter/checkpoints/"
    log ""
    log "Next steps:"
    log "  1. Check W&B for training curves"
    log "  2. Run evaluation script on each checkpoint"
    log "  3. Test steering in demo"

    # Create summary
    echo "
# Overnight Runs Summary — $(date)

## Experiments Run
| Name | Expansion | TopK | Optimizer | Filter | Steps | Status |
|------|-----------|------|-----------|--------|-------|--------|
| v2_no_filter | 64× | 32 | Adam | NO | 100K | ✓ |
| v2_32x_no_filter | 32× | 32 | Adam | NO | 100K | ✓ |
| v1_arch_v2_data | 16× | 64 | Adam | NO | 50K | ✓ |
| v2_topk64 | 64× | 64 | Adam | NO | 100K | ✓ |
| v2_adamw_no_filter | 64× | 32 | AdamW | NO | 100K | ✓ |

## Total Time
${HOURS}h ${MINUTES}m

## What to Check
1. Compare R² across runs (all should be >0.90)
2. Check dead feature % (<5% is good)
3. Test steering on each checkpoint
4. The winner should have BOTH good metrics AND good steering
5. Compare AdamW vs Adam (weight decay may help generalization)
" > "$LOG_DIR/summary.md"

    log "Summary saved to $LOG_DIR/summary.md"
}

# Run main
main
