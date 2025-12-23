#!/bin/bash
# ============================================================
# Experiment 2: Train Probe
# ============================================================
# This script trains the hallucination probe on the full dataset
# as shown in Cell 5 of CS762-Final.ipynb
#
# Usage:
#   ./run_experiment_2_train_probe.sh
# ============================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "Experiment 2: Train Probe"
echo "============================================================"
echo ""

# Navigate to the project root
cd "$PROJECT_ROOT"

# Set environment variables
export PYTHONUNBUFFERED=1

# Run the training script with parameters from Cell 5
echo "Running: train_probe_full.py"
echo "Parameters:"
echo "  --model_name EleutherAI/gpt-neo-2.7B"
echo "  --layer_id 17"
echo "  --probe_type mlp"
echo "  --epochs 100"
echo "  --lr 0.001"
echo "  --batch_size 16"
echo "  --threshold 0.5"
echo "  --split_path artifacts/gpt-neo-2.7B_split.npz"
echo "  --output_dir artifacts"
echo "  --bleurt_path ml_tqa_bleurt_score.npy"
echo "  --data_dir save_for_eval/tqa_hal_det/answers"
echo "  --device cuda"
echo ""

python3 experiments/probe_controlled_tsv/train_probe_full.py \
  --model_name EleutherAI/gpt-neo-2.7B \
  --layer_id 17 \
  --probe_type mlp \
  --epochs 100 \
  --lr 0.001 \
  --batch_size 16 \
  --threshold 0.5 \
  --split_path artifacts/gpt-neo-2.7B_split.npz \
  --output_dir artifacts \
  --bleurt_path ml_tqa_bleurt_score.npy \
  --data_dir save_for_eval/tqa_hal_det/answers \
  --device cuda

echo ""
echo "============================================================"
echo "Experiment 2 Complete!"
echo "============================================================"
echo "Output saved to: artifacts/gpt-neo-2.7B_probe_817.pt"
echo ""

