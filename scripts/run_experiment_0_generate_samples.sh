#!/bin/bash
# ============================================================
# Experiment 0: Generate Samples (Answers + BLEURT Scores)
# ============================================================
# This script generates model answers for TruthfulQA questions
# and computes BLEURT scores as ground truth labels, **if they
# do not already exist**.
#
#
# This script is self-contained and inlines the logic from
# `scripts/01_generate_samples.sh` (no external script calls).
# ============================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "Experiment 0: Generate Samples (Answers + BLEURT Scores)"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# Configuration (copied from 01_generate_samples.sh)
# ------------------------------------------------------------
MODEL_NAME="gpt-neo-2.7B"
DATASET="tqa"
BATCH_SIZE=32
NUM_EXEMPLARS=16

# Navigate to project root
cd "$PROJECT_ROOT"

# Ensure Python prints are unbuffered so Colab shows output in real time
export PYTHONUNBUFFERED=1

echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Project Root: $PROJECT_ROOT"
echo ""

# ------------------------------------------------------------
# Generate answers (if missing)
# ------------------------------------------------------------
ANSWER_DIR="./save_for_eval/${DATASET}_hal_det/answers"

# If the answers directory exists and has any .npy files, assume answers are already generated
if [ -d "$ANSWER_DIR" ]; then
    EXISTING_COUNT=$(ls -1 "$ANSWER_DIR"/*.npy 2>/dev/null | wc -l || echo 0)
else
    EXISTING_COUNT=0
fi

if [ "$EXISTING_COUNT" -gt 0 ]; then
    echo "✓ Answer files already exist in $ANSWER_DIR ($EXISTING_COUNT files)"
else
    echo "Generating answers..."
    python tsv_main.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET" \
        --gene 1 \
        --most_likely 1 \
        --batch_size "$BATCH_SIZE" \
        --num_exemplars "$NUM_EXEMPLARS"
    echo "✓ Answers generated"
fi

# ------------------------------------------------------------
# Generate BLEURT scores (if missing)
# ------------------------------------------------------------
BLEURT_FILE="./ml_${DATASET}_bleurt_score.npy"
if [ -f "$BLEURT_FILE" ]; then
    echo "✓ BLEURT scores already exist: $BLEURT_FILE"
else
    echo "Generating BLEURT scores..."
    python tsv_main.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET" \
        --generate_gt 1 \
        --most_likely 1 \
        --batch_size "$BATCH_SIZE"
    echo "✓ BLEURT scores generated"
fi

echo ""
echo "============================================================"
echo "Experiment 0 Complete!"
echo "============================================================"
echo "Generated files:"
echo "  - $ANSWER_DIR/*.npy"
echo "  - $BLEURT_FILE"
echo ""
