#!/bin/bash
# ============================================================
# Run All Experiments: Complete Pipeline
# ============================================================
# This script runs all experiments from CS762-Final.ipynb:
# 0. Generate samples + BLEURT scores (if missing)
# 1. Train TSV Logistic Regression
# 2. Train Probe
# 3. Run Full Experiment
# 4. ITI Pipeline
#
# Usage:
#   ./run_all_experiments.sh           # Run all experiments
#   ./run_all_experiments.sh --skip-0  # Skip data generation (Experiment 0)
#   ./run_all_experiments.sh --skip-1  # Skip experiment 1
#   ./run_all_experiments.sh --skip-2  # Skip experiment 2
#   ./run_all_experiments.sh --skip-3  # Skip experiment 3
#   ./run_all_experiments.sh --skip-4  # Skip experiment 4
# ============================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
SKIP_0=false
SKIP_1=false
SKIP_2=false
SKIP_3=false
SKIP_4=false

for arg in "$@"; do
    case $arg in
        --skip-0)
            SKIP_0=true
            shift
            ;;
        --skip-1)
            SKIP_1=true
            shift
            ;;
        --skip-2)
            SKIP_2=true
            shift
            ;;
        --skip-3)
            SKIP_3=true
            shift
            ;;
        --skip-4)
            SKIP_4=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--skip-1] [--skip-2] [--skip-3] [--skip-4]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "CS762 Final Project: Running All Experiments"
echo "============================================================"
echo ""
echo "This script will run all experiments from CS762-Final.ipynb"
echo ""

# Track start time
START_TIME=$(date +%s)

# Experiment 0: Generate Samples + BLEURT scores (if missing)
if [ "$SKIP_0" = true ]; then
    echo ">>> Skipping Experiment 0: Generate Samples + BLEURT scores"
else
    echo ">>> Running Experiment 0: Generate Samples + BLEURT scores"
    bash "$SCRIPT_DIR/run_experiment_0_generate_samples.sh"
    if [ $? -ne 0 ]; then
        echo "ERROR: Experiment 0 failed!"
        exit 1
    fi
fi
echo ""

# Experiment 1: Train TSV Logistic Regression
if [ "$SKIP_1" = true ]; then
    echo ">>> Skipping Experiment 1: Train TSV Logistic Regression"
else
    echo ">>> Running Experiment 1: Train TSV Logistic Regression"
    bash "$SCRIPT_DIR/run_experiment_1_train_tsv_logreg.sh"
    if [ $? -ne 0 ]; then
        echo "ERROR: Experiment 1 failed!"
        exit 1
    fi
fi
echo ""

# Experiment 2: Train Probe
if [ "$SKIP_2" = true ]; then
    echo ">>> Skipping Experiment 2: Train Probe"
else
    echo ">>> Running Experiment 2: Train Probe"
    bash "$SCRIPT_DIR/run_experiment_2_train_probe.sh"
    if [ $? -ne 0 ]; then
        echo "ERROR: Experiment 2 failed!"
        exit 1
    fi
fi
echo ""

# Experiment 3: Run Full Experiment
if [ "$SKIP_3" = true ]; then
    echo ">>> Skipping Experiment 3: Run Full Experiment"
else
    echo ">>> Running Experiment 3: Run Full Experiment"
    bash "$SCRIPT_DIR/run_experiment_3_run_full_experiment.sh"
    if [ $? -ne 0 ]; then
        echo "ERROR: Experiment 3 failed!"
        exit 1
    fi
fi
echo ""

# Experiment 4: ITI Pipeline
if [ "$SKIP_4" = true ]; then
    echo ">>> Skipping Experiment 4: ITI Pipeline"
else
    echo ">>> Running Experiment 4: ITI Pipeline"
    bash "$SCRIPT_DIR/run_experiment_4_iti_pipeline.sh"
    if [ $? -ne 0 ]; then
        echo "ERROR: Experiment 4 failed!"
        exit 1
    fi
fi
echo ""

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "============================================================"
echo "All Experiments Complete!"
echo "============================================================"
echo ""
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Summary of outputs:"
echo "  - TSV Model: artifacts/gpt-neo-2.7B_logreg_tsv_817.pt"
echo "  - Probe Model: artifacts/gpt-neo-2.7B_probe_817.pt"
echo "  - Full Experiment Results: results/full_experiment_comparison/"
echo "  - ITI Results: results/full_experiment_comparison/"
echo ""
echo "All experiments have been completed successfully!"
echo ""

