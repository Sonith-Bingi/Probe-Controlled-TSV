#!/usr/bin/env python3
"""
AUROC Evaluation for Hallucination Detection
===========================================

This script evaluates the Area Under the ROC Curve (AUROC) for each method's
ability to detect hallucinations using:

1. Risk scores from steering vectors/probes as predictions
2. BLEURT scores to determine ground truth (truthful vs hallucinated)
3. Standard AUROC computation following TSV paper methodology

The script processes results from the 4-method comparison:
- Baseline (no steering) 
- Fixed Logistic TSV
- Adaptive Logistic TSV (probe-controlled)
- Original TSV (optimal transport)

Usage:
    python evaluate_auroc.py [--results_dir RESULTS_DIR] [--bleurt_threshold 0.4]
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_experimental_results(results_dir: str) -> Dict:
    """Load experimental results from the 4-method comparison."""
    results_file = Path(results_dir) / "full_experiment_results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_risk_scores_and_labels(
    results: Dict,
    bleurt_threshold: float = 0.4
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract risk scores and ground truth labels for each method.
    
    Args:
        results: Experimental results dictionary
        bleurt_threshold: Threshold for determining truthful vs hallucinated
    
    Returns:
        Dict mapping method names to (risk_scores, ground_truth_labels)
    """
    
    method_data = {}
    
    for method_name, method_results in results["results"].items():
        if "generations" not in method_results:
            logger.warning(f"No generation data found for {method_name}")
            continue
            
        generations = method_results["generations"]
        risk_scores = []
        bleurt_scores = []
        
        for gen in generations:
            # Extract risk score (varies by method)
            if method_name == "baseline":
                # For baseline, use a constant score (no steering capability)
                risk_score = 0.5
            elif "risk_trace" in gen and len(gen["risk_trace"]) > 0:
                # Use final risk score from trace
                risk_score = gen["risk_trace"][-1]
            elif "alpha_trace" in gen and len(gen["alpha_trace"]) > 0:
                # Use final alpha value as proxy for risk
                risk_score = gen["alpha_trace"][-1]
            else:
                # Fallback: use steering rate as proxy
                risk_score = 1.0 if gen.get("triggered_steps", 0) > 0 else 0.0
            
            risk_scores.append(risk_score)
            bleurt_scores.append(gen["bleurt"])
        
        # Convert to numpy arrays
        risk_scores = np.array(risk_scores)
        bleurt_scores = np.array(bleurt_scores)
        
        # Create ground truth labels: 1 = truthful, 0 = hallucinated
        truth_labels = (bleurt_scores > bleurt_threshold).astype(int)
        
        # For AUROC, we want to predict hallucination (so flip the labels)
        # High risk score should predict hallucination (label = 1)
        hallucination_labels = 1 - truth_labels
        
        logger.info(f"{method_name}: {len(risk_scores)} samples, "
                   f"{hallucination_labels.sum()}/{len(hallucination_labels)} hallucinated")
        
        method_data[method_name] = (risk_scores, hallucination_labels)
    
    return method_data


def compute_auroc_metrics(method_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Compute AUROC and related metrics for each method."""
    
    auroc_results = {
        "bleurt_threshold": None,
        "num_samples": None,
        "methods": {}
    }
    
    for method_name, (risk_scores, labels) in method_data.items():
        
        # Skip if we don't have enough samples or classes
        if len(np.unique(labels)) < 2:
            logger.warning(f"Skipping {method_name}: insufficient class diversity")
            continue
            
        # Compute AUROC
        try:
            auroc = roc_auc_score(labels, risk_scores)
            
            # Compute ROC curve for additional metrics
            fpr, tpr, thresholds = roc_curve(labels, risk_scores)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_specificity = 1 - fpr[optimal_idx]
            optimal_sensitivity = tpr[optimal_idx]
            
            method_results = {
                "auroc": float(auroc),
                "optimal_threshold": float(optimal_threshold),
                "optimal_sensitivity": float(optimal_sensitivity),
                "optimal_specificity": float(optimal_specificity),
                "optimal_j_score": float(j_scores[optimal_idx]),
                "num_hallucinated": int(labels.sum()),
                "num_truthful": int(len(labels) - labels.sum())
            }
            
            auroc_results["methods"][method_name] = method_results
            
            logger.info(f"{method_name} AUROC: {auroc:.4f}")
            
        except Exception as e:
            logger.error(f"Error computing AUROC for {method_name}: {e}")
            continue
    
    # Set global metadata
    if method_data:
        first_method = next(iter(method_data.values()))
        auroc_results["num_samples"] = len(first_method[1])
    
    return auroc_results


def plot_roc_curves(method_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                   save_path: Optional[str] = None):
    """Plot ROC curves for all methods."""
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    method_names = {
        'baseline': 'Baseline (No Steering)',
        'fixed_logistic': 'Fixed Logistic TSV',
        'adaptive_logistic': 'Adaptive Logistic TSV', 
        'original_tsv': 'Original TSV (Optimal Transport)'
    }
    
    for i, (method_key, (risk_scores, labels)) in enumerate(method_data.items()):
        if len(np.unique(labels)) < 2:
            continue
            
        fpr, tpr, _ = roc_curve(labels, risk_scores)
        auroc = roc_auc_score(labels, risk_scores)
        
        method_name = method_names.get(method_key, method_key)
        color = colors[i % len(colors)]
        
        plt.plot(fpr, tpr, color=color, linewidth=2, 
                label=f'{method_name} (AUROC = {auroc:.3f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Hallucination Detection Methods')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to: {save_path}")
    
    plt.show()


def save_auroc_results(auroc_results: Dict, output_file: str):
    """Save AUROC results to JSON file."""
    
    with open(output_file, 'w') as f:
        json.dump(auroc_results, f, indent=2)
    
    logger.info(f"AUROC results saved to: {output_file}")


def print_auroc_summary(auroc_results: Dict):
    """Print formatted AUROC summary table."""
    
    print("\\n" + "="*80)
    print("HALLUCINATION DETECTION AUROC RESULTS")
    print("="*80)
    
    print(f"Number of samples: {auroc_results['num_samples']}")
    print(f"BLEURT threshold: {auroc_results.get('bleurt_threshold', 'N/A')}")
    
    print("\\n" + "-"*80)
    print(f"{'Method':<25} {'AUROC':>10} {'Sensitivity':>12} {'Specificity':>12} {'J-Score':>10}")
    print("-"*80)
    
    # Sort methods by AUROC (descending)
    methods_sorted = sorted(
        auroc_results["methods"].items(),
        key=lambda x: x[1]["auroc"],
        reverse=True
    )
    
    method_display_names = {
        'baseline': 'Baseline',
        'fixed_logistic': 'Fixed Logistic TSV',
        'adaptive_logistic': 'Adaptive Logistic TSV',
        'original_tsv': 'Original TSV'
    }
    
    for method_key, metrics in methods_sorted:
        display_name = method_display_names.get(method_key, method_key)
        print(f"{display_name:<25} {metrics['auroc']:>10.4f} "
              f"{metrics['optimal_sensitivity']:>12.4f} "
              f"{metrics['optimal_specificity']:>12.4f} "
              f"{metrics['optimal_j_score']:>10.4f}")
    
    print("="*80)
    
    # Highlight best method
    if methods_sorted:
        best_method, best_metrics = methods_sorted[0]
        best_name = method_display_names.get(best_method, best_method)
        print(f"\\n🏆 Best performing method: {best_name} (AUROC = {best_metrics['auroc']:.4f})")
    
    print("\\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AUROC for hallucination detection")
    parser.add_argument("--results_dir", type=str, 
                       default="results/full_experiment_comparison",
                       help="Directory containing experimental results")
    parser.add_argument("--bleurt_threshold", type=float, default=0.4,
                       help="BLEURT threshold for truthful/hallucinated classification")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for AUROC results")
    parser.add_argument("--plot_roc", action="store_true", 
                       help="Generate ROC curve plots")
    
    args = parser.parse_args()
    
    try:
        # Load experimental results
        logger.info(f"Loading results from: {args.results_dir}")
        results = load_experimental_results(args.results_dir)
        
        # Extract risk scores and labels
        logger.info("Extracting risk scores and ground truth labels...")
        method_data = extract_risk_scores_and_labels(results, args.bleurt_threshold)
        
        if not method_data:
            logger.error("No valid method data found!")
            sys.exit(1)
        
        # Compute AUROC metrics
        logger.info("Computing AUROC metrics...")
        auroc_results = compute_auroc_metrics(method_data)
        auroc_results["bleurt_threshold"] = args.bleurt_threshold
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = Path(args.output_dir) / "hallucination_detection_auroc.json"
        save_auroc_results(auroc_results, output_file)
        
        # Print summary
        print_auroc_summary(auroc_results)
        
        # Generate plots if requested
        if args.plot_roc:
            plot_path = Path(args.output_dir) / "roc_curves_hallucination_detection.png"
            plot_roc_curves(method_data, plot_path)
        
        logger.info("AUROC evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"AUROC evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()