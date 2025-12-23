"""Train linear probes per attention-head (approx) and compute ITI directions.

Prototype approach (head approximation):
 - Register a forward hook on model.transformer.h[layer_id] to capture the
   block output hidden states for the last token.
 - Split the last-token residual vector into `num_heads` equal slices and treat
   each slice as an attention-head activation approximation.
 - Build a dataset of activations for (question, answer) pairs labeled true/false
   using TruthfulQA golds (we use `best_answer` as canonical truthful answer).
 - Train a logistic regression probe per head and compute validation accuracy.
 - For top-K heads, compute mass-mean shift direction = mean(true) - mean(false)
   and compute sigma = std of activations projected on that direction.
 - Save directions to disk (npz + json metadata).

This is intended as a starting point you can refine to hook into actual
head value vectors for a specific model implementation.
"""
import argparse
import numpy as np
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ITI.utils import load_truthfulqa, split_into_heads, save_directions, find_qkv_linear, extract_value_from_qkv


def capture_value_activations_for_examples(model, tokenizer, examples, layer_id, device, num_heads):
    """Return arrays of shape (N, num_heads, head_dim) containing per-head value vectors for the last token.

    This attempts to hook into the model's qkv linear projection and extract the value vectors exactly.
    If the model does not expose a combined qkv linear, this function will raise RuntimeError.
    """
    model.to(device)
    model.eval()

    qkv_linear = find_qkv_linear(model, layer_id)
    if qkv_linear is None:
        raise RuntimeError("Could not find combined qkv linear layer on model; cannot extract exact head values.")

    cached = {}

    def qkv_hook(module, inp, out):
        # out is qkv: (batch, seq, 3*embed)
        cached['qkv'] = out.detach().cpu()
        return None

    handle = qkv_linear.register_forward_hook(qkv_hook)

    activations = []
    questions = []
    for ex in tqdm(examples, desc="extract"):
        q = ex['question']
        a = ex['best_answer']
        prompt = f"Q: {q}\nA: {a}"
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        qkv = cached.get('qkv')
        if qkv is None:
            handle.remove()
            raise RuntimeError("qkv hook did not capture outputs; model forward may not have called the linear module")
        # extract value portion and per-head view
        v = extract_value_from_qkv(qkv, num_heads)
        # take last token's head vectors and move to numpy
        last_v = v[:, -1, :, :].squeeze(0).cpu().numpy()
        activations.append(last_v)
        questions.append(ex)

    handle.remove()
    activations = np.stack(activations, axis=0)  # (N, H, head_dim)
    return activations, questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='EleutherAI/gpt-neo-2.7B')
    parser.add_argument('--layer_id', type=int, default=10, help='layer to probe')
    parser.add_argument('--num_heads', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=48)
    parser.add_argument('--out_path', default='ITI/directions')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading TruthfulQA...")
    data = load_truthfulqa()
    # Build a simple labeled dataset: if example's best_answer in correct_answers -> truthful
    labeled = []
    for ex in data:
        label = 1 if ex['best_answer'] in ex.get('correct_answers', []) else 0
        labeled.append({**ex, 'label': label})

    print(f"Found {len(labeled)} examples")

    print("Loading model/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    print("Extracting per-head VALUE activations (exact) — this may be slow")
    acts, examples = capture_value_activations_for_examples(model, tokenizer, labeled, args.layer_id, args.device, args.num_heads)
    # acts shape: (N, num_heads, head_dim)
    N, H, head_dim = acts.shape
    print(f"Collected activations: {acts.shape} (N, H, head_dim)")

    if H != args.num_heads:
        print(f"Warning: detected num_heads={H} but requested num_heads={args.num_heads}; updating")
        args.num_heads = H

    labels = np.array([ex['label'] for ex in examples])

    directions = {}

    # For each head, train a small logistic regression probe
    print("Training probes per head")
    head_scores = []
    head_infos = []
    for h in range(args.num_heads):
        X = acts[:, h, :]
        # small train/val split
        Xtr, Xv, ytr, yv = train_test_split(X, labels, test_size=0.2, random_state=args.seed)
        clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
        score = clf.score(Xv, yv)
        head_scores.append((h, score))
        # compute mass means for direction using all data
        mean_true = X[labels == 1].mean(0) if (labels == 1).sum() else np.zeros(X.shape[1])
        mean_false = X[labels == 0].mean(0) if (labels == 0).sum() else np.zeros(X.shape[1])
        theta = mean_true - mean_false
        # compute sigma along theta
        if np.linalg.norm(theta) < 1e-8:
            sigma = 0.0
        else:
            proj = (X @ theta) / (np.linalg.norm(theta) + 1e-12)
            sigma = proj.std()
        head_infos.append({'head': h, 'score': score, 'theta': theta, 'sigma': float(sigma)})

    # rank heads by score
    head_infos = sorted(head_infos, key=lambda x: x['score'], reverse=True)
    selected = head_infos[:args.top_k]

    # prepare object to save
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    saved = {}
    for info in selected:
        k = f"layer{args.layer_id}_head{info['head']}"
        saved[k] = {
            'layer': args.layer_id,
            'head': int(info['head']),
            'score': float(info['score']),
            'sigma': float(info['sigma']),
            'theta': np.array(info['theta'], dtype=np.float32),
        }

    print(f"Saving {len(saved)} directions to {args.out_path}(.npz/.json)")
    save_directions(args.out_path, saved)
    print("Done")


if __name__ == '__main__':
    main()
