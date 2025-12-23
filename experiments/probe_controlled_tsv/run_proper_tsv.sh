#!/bin/bash

# Run Proper TSV Implementation Following Original Paper
# This script trains and evaluates the corrected TSV method

set -e

echo "=================================="
echo "PROPER TSV PAPER IMPLEMENTATION"
echo "=================================="

# Create results directory
mkdir -p results/proper_tsv_evaluation

# Step 1: Train proper TSV following paper methodology
echo "Step 1: Training proper TSV with paper-exact methodology..."
python train_tsv_proper_paper_method.py \
    --model_name "EleutherAI/gpt-neo-2.7B" \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --save_path "artifacts/proper_tsv_model.pt" \
    --seed 42

echo "✓ Proper TSV training completed"

# Step 2: Evaluate proper TSV detection
echo "Step 2: Evaluating proper TSV detection performance..."
python evaluate_proper_tsv_detection.py \
    --model_name "EleutherAI/gpt-neo-2.7B" \
    --tsv_model_path "artifacts/proper_tsv_model.pt" \
    --max_samples 200 \
    --output_path "results/proper_tsv_evaluation/detection_results.json"

echo "✓ Proper TSV evaluation completed"

# Step 3: Compare with original approximation method
echo "Step 3: Comparison summary..."
echo "=================================="

# Display results
if [ -f "results/proper_tsv_evaluation/detection_results.json" ]; then
    echo "Proper TSV Results:"
    python -c "
import json
with open('results/proper_tsv_evaluation/detection_results.json', 'r') as f:
    results = json.load(f)
print(f\"  AUROC: {results['auroc']:.4f}\")
print(f\"  Accuracy: {results['accuracy']:.4f}\")
print(f\"  κ parameter: {results['model_parameters']['kappa']:.4f}\")
print(f\"  Samples: {results['num_samples']}\")
"
fi

# Compare with original method if results exist
if [ -f "results/auroc_evaluation/hallucination_auroc_results.json" ]; then
    echo ""
    echo "Original TSV (approximation method):"
    python -c "
import json
with open('results/auroc_evaluation/hallucination_auroc_results.json', 'r') as f:
    results = json.load(f)
for method_name, method_data in results.items():
    if 'Original TSV' in method_name:
        print(f\"  AUROC: {method_data['auroc']:.4f}\")
        break
"
fi

echo ""
echo "=================================="
echo "KEY IMPROVEMENTS IMPLEMENTED:"
echo "=================================="
echo "✓ von Mises-Fisher distribution with learnable κ parameter"
echo "✓ Proper prototype learning with EMA updates (Equation 6)" 
echo "✓ Paper-exact detection scoring (Equation 13)"
echo "✓ MLE objective with cross-entropy loss (Equation 3)"
echo "✓ Optimal transport for pseudo-label assignment"
echo ""
echo "Expected improvement: 54.92% → 65-75% AUROC"
echo "Files created:"
echo "  - train_tsv_proper_paper_method.py (proper training)"
echo "  - evaluate_proper_tsv_detection.py (proper evaluation)" 
echo "  - artifacts/proper_tsv_model.pt (trained model)"
echo "  - results/proper_tsv_evaluation/ (results)"