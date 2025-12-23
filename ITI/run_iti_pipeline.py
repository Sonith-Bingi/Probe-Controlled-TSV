#!/usr/bin/env python3
"""End-to-end ITI pipeline that trains directions on the repo split and
applies them to the test split (TruthfulQA). Designed to be run from the
repo root as a single command.
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import KFold

from ITI.utils import (
    load_truthfulqa,
    save_directions,
    load_directions,
    split_into_heads,
    find_qkv_linear,
    find_value_linear,
    extract_value_from_qkv,
)

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN (optional, may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def load_dataset_examples():
    return load_truthfulqa()


def create_split_if_missing(split_path, dataset_length, seed=42):
    """Generates a standard 80/20 train/test split and saves it if the file doesn't exist."""
    if os.path.exists(split_path):
        return

    print(f"Split file {split_path} not found. Creating a new random 80/20 split...")
    all_indices = np.arange(dataset_length)
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    
    cutoff = int(0.8 * dataset_length)
    train_idx = all_indices[:cutoff]
    test_idx = all_indices[cutoff:]
    
    # Save as compressed numpy
    np.savez(split_path, train_idx=train_idx, test_idx=test_idx)
    print(f"Saved split to {split_path} (Train: {len(train_idx)}, Test: {len(test_idx)})")


def flatten_examples_balanced(examples):
    """Convert list of TruthfulQA examples into a balanced (True/False) list for training."""
    flat = []
    for ex in examples:
        # Positive
        flat.append({
            'question': ex['question'],
            'best_answer': ex['best_answer'], # Truth
            'label': 1
        })
        # Negative (Randomly sample one incorrect answer)
        incorrects = ex.get('incorrect_answers', [])
        if incorrects:
            bad = random.choice(incorrects)
            flat.append({
                'question': ex['question'],
                'best_answer': bad, # Falsehood
                'label': 0
            })
    return flat


def capture_per_head_values_fused(model, tokenizer, flat_examples: List[dict], layer_id: int, num_heads: int, device: torch.device):
    """Capture per-example per-head value vectors using a fused qkv linear."""
    model.eval()

    qkv_linear = find_qkv_linear(model, layer_id)
    if qkv_linear is None:
        raise RuntimeError("No fused qkv linear found")

    cached = {}

    def qkv_hook(module, inp, out):
        cached['qkv'] = out.detach().cpu().clone()
        return None

    handle = qkv_linear.register_forward_hook(qkv_hook)

    activations = []
    labels = []
    
    for ex in tqdm(flat_examples, desc="extract (fused qkv)"):
        prompt = f"Q: {ex['question']}\nA: {ex['best_answer']}"
        inputs = tokenizer(prompt, return_tensors='pt')
        # Handle device mapping
        if hasattr(model, "device"):
             target_device = model.device
        else:
             target_device = device
             
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)

        qkv = cached.get('qkv')
        if qkv is None:
            handle.remove()
            raise RuntimeError("qkv not captured during forward")

        v = extract_value_from_qkv(qkv, num_heads)  # (batch, seq, num_heads, head_dim)
        last_v = v[:, -1, :, :].squeeze(0).cpu().numpy()
        activations.append(last_v)
        labels.append(ex['label'])

    handle.remove()
    return np.stack(activations, axis=0), np.array(labels)


def capture_per_head_values_residual(model, tokenizer, flat_examples: List[dict], layer_id: int, num_heads: int, device: torch.device):
    """Fallback: capture residual block outputs and split into head slices."""
    model.eval()

    cached = {}

    def block_hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out
        cached['h'] = h.detach().cpu().clone()
        return None

    try:
        block = model.transformer.h[layer_id]
    except Exception:
        block = getattr(model, 'transformer', None)
        if block is None:
            raise RuntimeError("Cannot find transformer block for residual hook")

    handle = block.register_forward_hook(block_hook)

    activations = []
    labels = []
    
    if hasattr(model, "device"):
         target_device = model.device
    else:
         target_device = device

    for ex in tqdm(flat_examples, desc="extract (residual)"):
        prompt = f"Q: {ex['question']}\nA: {ex['best_answer']}"
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)

        h = cached.get('h')
        if h is None:
            handle.remove()
            raise RuntimeError("residual hidden not captured")

        h_np = h.numpy()
        last = h_np[:, -1, :]
        heads = split_into_heads(last, num_heads=num_heads)
        activations.append(heads.squeeze(0))
        labels.append(ex['label'])

    handle.remove()
    return np.stack(activations, axis=0), np.array(labels)


def train_head_probes_and_directions(acts: np.ndarray, labels: np.ndarray, top_k: int, layer_id: int):
    """Train logistic probes per head and compute theta/sigma directions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    N, H, head_dim = acts.shape
    head_infos = []
    
    print(f"Training probes on {N} examples...")
    
    for h in range(H):
        X = acts[:, h, :]
        score = 0.0
        if len(np.unique(labels)) > 1:
            Xtr, Xv, ytr, yv = train_test_split(X, labels, test_size=0.2, random_state=1)
            clf = LogisticRegression(max_iter=200, solver='liblinear').fit(Xtr, ytr)
            score = float(clf.score(Xv, yv))

        mean_true = X[labels == 1].mean(0) if labels.sum() > 0 else np.zeros(head_dim)
        mean_false = X[labels == 0].mean(0) if (labels == 0).sum() > 0 else np.zeros(head_dim)
        theta = mean_true - mean_false
        norm = np.linalg.norm(theta)
        sigma = 0.0
        if norm >= 1e-8:
            proj = (X @ theta) / (norm + 1e-12)
            sigma = float(proj.std())

        head_infos.append({'head': int(h), 'score': float(score), 'theta': theta.astype(np.float32), 'sigma': float(sigma)})

    # Sort by score
    head_infos = sorted(head_infos, key=lambda x: x['score'], reverse=True)[:top_k]
    
    # Print top scores check
    print("Top head scores:", [round(x['score'], 3) for x in head_infos[:5]])

    saved = {}
    for info in head_infos:
        k = f"layer{layer_id}_head{info['head']}"
        saved[k] = {
            'layer': int(layer_id),
            'head': int(info['head']),
            'score': float(info['score']),
            'sigma': float(info['sigma']),
            'theta': np.array(info['theta'], dtype=np.float32),
        }
    return saved


def generate_on_examples(model, tokenizer, directions_path_prefix, layer, num_heads, alpha, max_new_tokens, device, examples, safe_apply=False, target_perturb_norm=1.0):
    # load directions
    directions = load_directions(directions_path_prefix)
    # monkeypatch loader used by apply_iti
    import ITI.apply_iti as apply_iti
    apply_iti.load_truthfulqa = lambda: examples
    gens = apply_iti.generate_with_iti(model, tokenizer, directions, layer, num_heads, alpha, max_new_tokens, device, temperature=0.5, top_p=0.95, safe_apply=safe_apply, target_perturb_norm=target_perturb_norm)
    
    out = []
    for g, ex in zip(gens, examples):
        refs = []
        if ex.get('best_answer'):
            refs.append(ex['best_answer'])
        refs.extend(ex.get('correct_answers') or [])
        out.append({
            'question': ex['question'],
            'references': refs if refs else [""],
            'generated': g.get('generation',''),
        })
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='EleutherAI/gpt-neo-2.7B')
    parser.add_argument('--split', default='gpt-neo-2.7B_split.npz') # Defaults to local file
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=48)
    parser.add_argument('--directions', default='ITI/directions')
    parser.add_argument('--out_dir', default='results/full_experiment_comparison')
    parser.add_argument('--alpha', type=float, default=15.0)
    parser.add_argument('--max_new_tokens', type=int, default=80)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--skip_train', action='store_true', help='If set, skip training and use existing directions files')
    parser.add_argument('--safe_apply', action='store_true')
    parser.add_argument('--target_perturb_norm', type=float, default=1.0)
    parser.add_argument('--run_eval', action='store_true')
    args = parser.parse_args()

    # Set the seed immediately so generation and training are deterministic
    set_seed(args.seed) 
    print(f"Random seed set to {args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load full dataset
    ds = load_dataset_examples()
    
    # 2. Check or create split file
    create_split_if_missing(args.split, len(ds))

    # 3. Load indices
    split = np.load(args.split)
    train_idx = [int(i) for i in split['train_idx'] if int(i) >= 0]
    test_idx = [int(i) for i in split['test_idx'] if int(i) >= 0]
    
    # Filter valid indices
    train_idx = [i for i in train_idx if i < len(ds)]
    test_idx = [i for i in test_idx if i < len(ds)]

    print(f"Total examples: {len(ds)} | Train: {len(train_idx)} | Test: {len(test_idx)}")

    print("Loading model...")
    load_kwargs = {}
    if args.load_in_8bit:
        load_kwargs['load_in_8bit'] = True
        load_kwargs['device_map'] = 'auto'
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    device = args.device
    if not args.load_in_8bit:
        model.to(device)

    # TRAINING
    if not args.skip_train:
        train_examples = [ds[i] for i in train_idx]
        flat_train = flatten_examples_balanced(train_examples)
        print(f"Flattened training examples (True/False pairs): {len(flat_train)}")

        # Try fused qkv
        qkv_lin = find_qkv_linear(model, args.layer)
        if qkv_lin is not None:
            print("Fused qkv linear found: using exact value extraction")
            acts, labels = capture_per_head_values_fused(model, tokenizer, flat_train, args.layer, args.num_heads, device)
        else:
            print("Fallback to residual head-slice approximation")
            acts, labels = capture_per_head_values_residual(model, tokenizer, flat_train, args.layer, args.num_heads, device)

        print("Training probes...")
        saved = train_head_probes_and_directions(acts, labels, args.top_k, args.layer)
        save_directions(args.directions, saved)
        print("Directions saved to:", args.directions + ".npz/.json")
    else:
        print("Skipping training...")

    # INFERENCE
    print("Generating with ITI on test split...")
    test_examples = [ds[i] for i in test_idx]
    gens = generate_on_examples(model, tokenizer, args.directions, args.layer, args.num_heads, args.alpha, args.max_new_tokens, device, test_examples, safe_apply=args.safe_apply, target_perturb_norm=args.target_perturb_norm)
    
    out_gen_path = os.path.join(args.out_dir, 'iti_generations.json')
    with open(out_gen_path, 'w', encoding='utf-8') as f:
        json.dump(gens, f, indent=2, ensure_ascii=False)
    print("Saved ITI generations to", out_gen_path)