#!/bin/bash
# ============================================================
# Experiment 1: Train TSV Logistic Regression
# ============================================================
# This script trains the TSV logistic regression model on the full dataset
# as shown in Cell 4 of CS762-Final.ipynb
#
# Usage:
#   ./run_experiment_1_train_tsv_logreg.sh
# ============================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "Experiment 1: Train TSV Logistic Regression"
echo "============================================================"
echo ""

# Navigate to the experiment directory
cd "$PROJECT_ROOT/experiments/probe_controlled_tsv"

# Set environment variables
export PYTHONUNBUFFERED=1

# Run the training script with parameters from Cell 4
echo "Running: train_tsv_logreg_full.py"
echo "Parameters:"
echo "  --model_name EleutherAI/gpt-neo-2.7B"
echo "  --layer_id 17"
echo "  --C 0.1"
echo "  --threshold 0.5"
echo "  --test_size 0.2"
echo "  --batch_size 16"
echo "  --output_dir ../../artifacts"
echo "  --device cuda"
echo ""

python3 train_tsv_logreg_full.py \
  --model_name EleutherAI/gpt-neo-2.7B \
  --layer_id 17 \
  --C 0.1 \
  --threshold 0.5 \
  --test_size 0.2 \
  --batch_size 16 \
  --output_dir ../../artifacts \
  --device cuda

echo ""
echo "============================================================"
echo "Experiment 1 Complete!"
echo "============================================================"
echo "Output saved to: ../../artifacts/gpt-neo-2.7B_logreg_tsv_817.pt"
echo ""

