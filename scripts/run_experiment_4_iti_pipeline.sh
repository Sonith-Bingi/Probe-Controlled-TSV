#!/bin/bash
# ============================================================
# Experiment 4: ITI Pipeline
# ============================================================
# This script runs the ITI (Inverse Truthfulness Intervention) pipeline
# as shown in Cell 9 of CS762-Final.ipynb
#
# Usage:
#   ./run_experiment_4_iti_pipeline.sh
# ============================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "Experiment 4: ITI Pipeline"
echo "============================================================"
echo ""

# Navigate to the project root
cd "$PROJECT_ROOT"

# Set environment variables
export PYTHONUNBUFFERED=1

# Run the ITI pipeline with parameters from Cell 9
echo "Running: ITI.run_iti_pipeline"
echo "Parameters:"
echo "  --model EleutherAI/gpt-neo-2.7B"
echo "  --split artifacts/gpt-neo-2.7B_split.npz"
echo "  --layer 17"
echo "  --num_heads 20"
echo "  --top_k 10"
echo "  --alpha 7"
echo "  --directions ITI/directions"
echo "  --out_dir results/full_experiment_comparison"
echo "  --max_new_tokens 50"
echo "  --device cuda"
echo "  --run_eval"
echo ""

python3 -m ITI.run_iti_pipeline \
  --model EleutherAI/gpt-neo-2.7B \
  --split artifacts/gpt-neo-2.7B_split.npz \
  --layer 17 \
  --num_heads 20 \
  --top_k 10 \
  --alpha 7 \
  --directions ITI/directions \
  --out_dir results/full_experiment_comparison \
  --max_new_tokens 50 \
  --device cuda \
  --run_eval

echo ""
echo "Running ITI evaluation metrics..."

# Run the evaluation script to compute and display metrics
python3 experiments/probe_controlled_tsv/eval_iti_metrics.py \
  --model_name EleutherAI/gpt-neo-2.7B \
  --iti_gens_path results/full_experiment_comparison/iti_generations.json \
  --full_results_path results/full_experiment_comparison/full_experiment_results.json \
  --tsv_logistic_path artifacts/gpt-neo-2.7B_logreg_tsv_817.pt \
  --layer_id 17 \
  --device cuda

echo ""
echo "============================================================"
echo "Experiment 4 Complete!"
echo "============================================================"
echo "Results saved to: results/full_experiment_comparison/"
echo "  - ITI generations: iti_generations.json"
echo "  - Full results: full_experiment_results.json"
echo ""

