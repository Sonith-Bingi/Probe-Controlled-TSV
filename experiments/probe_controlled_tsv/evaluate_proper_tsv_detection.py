#!/usr/bin/env python
"""
Proper TSV Detection using Original Paper Methodology

Implements exact detection scoring from Equation 13:
S(x') = exp(κμ_truthful^T r_v_test) / Σ_c' exp(κμ_c'^T r_v_test)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Import proper TSV trainer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from train_tsv_proper_paper_method import ProperTSVTrainer, extract_hidden_states_proper, load_tqa_data_proper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProperTSVDetector:
    """
    Proper TSV detector implementing paper-exact methodology
    """
    
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        
        # Load trained TSV model
        logger.info(f"Loading proper TSV model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.prototypes = checkpoint['prototypes'].to(device)
        self.kappa = checkpoint['kappa'].to(device)
        self.hidden_dim = checkpoint['hidden_dim']
        
        logger.info(f"Loaded prototypes shape: {self.prototypes.shape}")
        logger.info(f"Loaded κ value: {self.kappa.item():.4f}")
        
        # Initialize TSV trainer for inference
        self.tsv_trainer = ProperTSVTrainer(
            hidden_dim=self.hidden_dim, 
            num_classes=2
        ).to(device)
        self.tsv_trainer.prototypes.data = self.prototypes
        self.tsv_trainer.kappa.data = self.kappa
        self.tsv_trainer.eval()
    
    def compute_detection_score_batch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute detection scores for batch using Equation 13
        
        Args:
            hidden_states: [N, hidden_dim] batch of hidden representations
            
        Returns:
            scores: [N] truthfulness scores for each sample
        """
        with torch.no_grad():
            # Normalize to unit sphere
            hidden_norm = F.normalize(hidden_states, p=2, dim=1)  # [N, hidden_dim]
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)  # [2, hidden_dim]
            
            # Compute similarities to each prototype
            similarities = torch.matmul(hidden_norm, proto_norm.T)  # [N, 2]
            
            # Apply learned concentration parameter κ
            logits = self.kappa * similarities  # [N, 2]
            
            # Softmax to get probabilities (Equation 13)
            probabilities = F.softmax(logits, dim=1)  # [N, 2]
            
            # Return probability of truthful class (index 1)
            return probabilities[:, 1]
    
    def detect_hallucinations(
        self,
        model: nn.Module,
        tokenizer,
        prompts: List[str],
        layer_id: int = -1,
        batch_size: int = 8
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect hallucinations using proper TSV methodology
        
        Returns:
            scores: Detection scores for each prompt
            details: Additional diagnostic information
        """
        
        # Extract hidden states
        logger.info("Extracting hidden states for detection...")
        hidden_states = extract_hidden_states_proper(
            model, tokenizer, prompts, self.device, layer_id, batch_size
        ).to(self.device)
        
        # Compute detection scores using paper methodology
        logger.info("Computing detection scores with proper TSV...")
        detection_scores = self.compute_detection_score_batch(hidden_states)
        
        # Additional diagnostics
        with torch.no_grad():
            hidden_norm = F.normalize(hidden_states, p=2, dim=1)
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)
            
            # Similarities to each prototype
            similarities = torch.matmul(hidden_norm, proto_norm.T)
            sim_halluc = similarities[:, 0].cpu().numpy()
            sim_truthful = similarities[:, 1].cpu().numpy()
            
            # Raw logits before softmax
            logits = self.kappa * similarities
            logits_halluc = logits[:, 0].cpu().numpy()
            logits_truthful = logits[:, 1].cpu().numpy()
        
        details = {
            'kappa_value': self.kappa.item(),
            'similarity_to_halluc_proto': sim_halluc.tolist(),
            'similarity_to_truthful_proto': sim_truthful.tolist(),
            'logits_halluc': logits_halluc.tolist(),
            'logits_truthful': logits_truthful.tolist(),
            'score_range': [float(detection_scores.min()), float(detection_scores.max())],
            'score_mean': float(detection_scores.mean()),
            'score_std': float(detection_scores.std())
        }
        
        return detection_scores.cpu().numpy(), details


def evaluate_proper_tsv_detection(
    model_name: str = "EleutherAI/gpt-neo-2.7B",
    tsv_model_path: str = "artifacts/proper_tsv_model.pt",
    device: str = "auto",
    layer_id: int = -1,
    batch_size: int = 8,
    max_samples: int = 200
) -> Dict:
    """
    Evaluate proper TSV detection performance
    """
    
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    logger.info(f"Evaluating proper TSV detection on device: {device}")
    
    # Load language model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None
    )
    model.eval()
    
    # Load TSV detector
    detector = ProperTSVDetector(tsv_model_path, device)
    
    # Load test data
    logger.info("Loading test data...")
    all_prompts, all_labels = load_tqa_data_proper(
        model_name=model_name.split("/")[-1],
        threshold=0.5,
        max_samples=max_samples
    )
    
    # Use last portion as test set (not used in training)
    test_start = max(0, len(all_prompts) - max_samples // 2)
    test_prompts = all_prompts[test_start:]
    test_labels = all_labels[test_start:]
    
    logger.info(f"Testing on {len(test_prompts)} samples")
    logger.info(f"  Truthful: {sum(test_labels)}")
    logger.info(f"  Hallucinated: {len(test_labels) - sum(test_labels)}")
    
    # Run detection
    detection_scores, details = detector.detect_hallucinations(
        model, tokenizer, test_prompts, layer_id, batch_size
    )
    
    # Compute metrics
    test_labels_np = np.array(test_labels)
    
    # AUROC
    auroc = roc_auc_score(test_labels_np, detection_scores)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(test_labels_np, detection_scores)
    optimal_idx = np.argmax(tpr - fpr)  # Youden's index
    optimal_threshold = thresholds[optimal_idx]
    
    # Accuracy at optimal threshold
    predictions = (detection_scores >= optimal_threshold).astype(int)
    accuracy = accuracy_score(test_labels_np, predictions)
    
    # Per-class performance
    truthful_mask = test_labels_np == 1
    halluc_mask = test_labels_np == 0
    
    truthful_scores = detection_scores[truthful_mask]
    halluc_scores = detection_scores[halluc_mask]
    
    results = {
        'method': 'Proper TSV (Paper Implementation)',
        'auroc': float(auroc),
        'accuracy': float(accuracy),
        'optimal_threshold': float(optimal_threshold),
        'num_samples': len(test_prompts),
        'num_truthful': int(sum(test_labels)),
        'num_hallucinated': int(len(test_labels) - sum(test_labels)),
        'score_statistics': {
            'overall_mean': float(np.mean(detection_scores)),
            'overall_std': float(np.std(detection_scores)),
            'truthful_mean': float(np.mean(truthful_scores)) if len(truthful_scores) > 0 else None,
            'truthful_std': float(np.std(truthful_scores)) if len(truthful_scores) > 0 else None,
            'hallucinated_mean': float(np.mean(halluc_scores)) if len(halluc_scores) > 0 else None,
            'hallucinated_std': float(np.std(halluc_scores)) if len(halluc_scores) > 0 else None,
        },
        'model_parameters': {
            'kappa': details['kappa_value'],
            'prototype_shape': list(detector.prototypes.shape),
            'hidden_dim': detector.hidden_dim
        },
        'detection_details': details
    }
    
    # Log results
    logger.info("=" * 60)
    logger.info("PROPER TSV DETECTION RESULTS")
    logger.info("=" * 60)
    logger.info(f"AUROC: {auroc:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")
    logger.info(f"κ parameter: {details['kappa_value']:.4f}")
    logger.info(f"Score range: [{details['score_range'][0]:.4f}, {details['score_range'][1]:.4f}]")
    
    if len(truthful_scores) > 0 and len(halluc_scores) > 0:
        logger.info(f"Class separation:")
        logger.info(f"  Truthful scores: {np.mean(truthful_scores):.4f} ± {np.std(truthful_scores):.4f}")
        logger.info(f"  Hallucinated scores: {np.mean(halluc_scores):.4f} ± {np.std(halluc_scores):.4f}")
        logger.info(f"  Effect size (Cohen's d): {(np.mean(truthful_scores) - np.mean(halluc_scores)) / np.sqrt((np.var(truthful_scores) + np.var(halluc_scores)) / 2):.4f}")
    
    return results


def compare_detection_methods(
    model_name: str = "EleutherAI/gpt-neo-2.7B",
    tsv_model_path: str = "artifacts/proper_tsv_model.pt",
    original_tsv_path: str = "../../artifacts/gpt-neo-2.7B_tqa_tsv.pt",
    device: str = "auto"
) -> Dict:
    """
    Compare proper TSV vs original approximation method
    """
    
    results = {}
    
    # Evaluate proper TSV
    logger.info("Evaluating proper TSV implementation...")
    try:
        results['proper_tsv'] = evaluate_proper_tsv_detection(
            model_name, tsv_model_path, device
        )
    except Exception as e:
        logger.error(f"Error evaluating proper TSV: {e}")
        results['proper_tsv'] = {'error': str(e)}
    
    # TODO: Add comparison with original method if needed
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate proper TSV detection")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--tsv_model_path", type=str, default="artifacts/proper_tsv_model.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--layer_id", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_path", type=str, default="results/proper_tsv_detection_results.json")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_proper_tsv_detection(
        model_name=args.model_name,
        tsv_model_path=args.tsv_model_path,
        device=args.device,
        layer_id=args.layer_id,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()