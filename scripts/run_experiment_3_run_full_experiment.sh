#!/bin/bash
# ============================================================
# Experiment 3: Run Full Experiment
# ============================================================
# This script runs the full comparative experiment
# as shown in Cells 6, 7, and 8 of CS762-Final.ipynb
#
# Usage:
#   ./run_experiment_3_run_full_experiment.sh
# ============================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "Experiment 3: Run Full Experiment"
echo "============================================================"
echo ""

# Navigate to the project root
cd "$PROJECT_ROOT"

# Set environment variables
export PYTHONUNBUFFERED=1

# Run the full experiment script with parameters from Cells 6, 7, 8
echo "Running: run_full_experiment.py"
echo "Parameters:"
echo "  --max_new_tokens 50"
echo "  --layer_id 17"
echo "  --alpha_max 1.5"
echo ""

python3 experiments/probe_controlled_tsv/run_full_experiment.py \
  --max_new_tokens 50 \
  --layer_id 17 \
  --alpha_max 1.5

echo ""
echo "============================================================"
echo "Experiment 3 Complete!"
echo "============================================================"
echo "Results saved to: results/full_experiment_comparison/"
echo ""

