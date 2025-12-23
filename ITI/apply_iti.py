"""Apply Inference-Time Intervention (ITI) at inference time.

This script loads saved directions from `train_directions.py` output and
registers a forward hook that injects alpha * sigma * theta into selected
head slices (approximated) at the specified layer for every generated token.

Usage example:
  python ITI/apply_iti.py --model_name EleutherAI/gpt-neo-2.7B --directions ITI/directions --alpha 15 --layer 10

"""
import argparse
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ITI.utils import load_truthfulqa, load_directions, split_into_heads, find_qkv_linear, extract_value_from_qkv
from tqdm import tqdm


def make_hook(directions, num_heads, device, alpha):
    # directions: dict keyed by 'layerX_headY' with keys 'layer','head','theta','sigma'
    # assume all directions are for same layer and theta shape = head_dim
    # Build an array of zeros shaped (embed_dim,) with head slices containing theta* sigma
    layer_dirs = {}
    for k, v in directions.items():
        head = int(v['head'])
        theta = torch.tensor(v['theta'], device=device)
        sigma = float(v['sigma'])
        layer_dirs[head] = (theta, sigma)

    def hook_on_qkv(module, inp, out):
        """Hook for a fused qkv linear: out shape (batch, seq, 3*embed).
        We will extract the value slice, add the intervention to the last-token
        value vectors for selected heads, and reconstruct the qkv output.
        """
        qkv = out
        if not isinstance(qkv, torch.Tensor):
            # unexpected, fall back to no-op
            return out
        # extract v as (batch, seq, num_heads, head_dim)
        try:
            v = extract_value_from_qkv(qkv, num_heads)
        except Exception:
            return out

        batch, seq = v.shape[0], v.shape[1]
        head_dim = v.shape[-1]

        # build additive tensor for values
        add = torch.zeros_like(v, device=v.device)
        for head_idx, (theta, sigma) in layer_dirs.items():
            if head_idx >= num_heads:
                continue
            th = torch.tensor(theta, device=v.device).view(1, 1, head_dim)
            add_val = (alpha * float(sigma)) * th
            # add only to last token across batch for this head
            add[:, -1, head_idx, :] = add[:, -1, head_idx, :] + add_val

        v_mod = v + add

        # put v_mod back into qkv tensor
        # v_mod shape -> (batch, seq, num_heads, head_dim) -> reshape to (batch, seq, embed)
        embed = head_dim * num_heads
        v_flat = v_mod.reshape(batch, seq, embed)
        # replace the value slice in qkv
        total = qkv.shape[-1]
        single = total // 3
        new_qkv = torch.cat([qkv[..., : 2 * single], v_flat], dim=-1)
        return new_qkv

    def hook_on_residual(module, inp, out):
        # fallback: operate on residual hidden states as before
        if isinstance(out, tuple):
            h = out[0]
            tail = out[1:]
        else:
            h = out
            tail = None
        batch, seq, embed = h.shape
        head_dim = embed // num_heads
        h_view = h.view(batch, seq, num_heads, head_dim)
        for head_idx, (theta, sigma) in layer_dirs.items():
            if head_idx >= num_heads:
                continue
            add = (alpha * sigma) * theta.view(1, 1, head_dim).to(h_view.device)
            h_view[:, -1, head_idx, :] = h_view[:, -1, head_idx, :] + add
        new_h = h_view.view(batch, seq, embed)
        if tail is not None:
            return (new_h, *tail)
        return new_h

    return hook_on_qkv, hook_on_residual


def top_p_sample_next_token(probs, temperature=1.0, top_p=0.9):
    probs = probs.astype('float64')
    if temperature != 1.0:
        probs = np.log(probs + 1e-12) / float(temperature)
        probs = np.exp(probs)
    probs = probs / probs.sum()
    # nucleus
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, top_p) + 1
    keep = sorted_idx[:cutoff]
    new_probs = np.zeros_like(probs)
    new_probs[keep] = probs[keep]
    new_probs = new_probs / new_probs.sum()
    return np.random.choice(len(probs), p=new_probs)


def generate_with_iti(model, tokenizer, directions, layer, num_heads, alpha, max_new_tokens, device, temperature, top_p, safe_apply=False, target_perturb_norm=1.0):
    model.to(device)
    model.eval()
    # If safe_apply is requested, compute a conservative alpha cap based on directions' sigmas.
    if safe_apply:
        sigmas = [float(v.get('sigma', 0.0)) for v in directions.values()]
        max_sigma = max(sigmas) if len(sigmas) > 0 else 0.0
        if max_sigma <= 0:
            alpha_cap = alpha
        else:
            alpha_cap = float(target_perturb_norm) / float(max_sigma)
        if alpha > alpha_cap:
            print(f"[safe_apply] Capping alpha from {alpha} to {alpha_cap:.4f} to bound per-head perturbation to {target_perturb_norm}")
            alpha = alpha_cap

    # try to find a fused qkv linear for the requested layer; if found,
    # register a hook on it that modifies the value slice. Otherwise fall
    # back to registering a hook on the block output (residual) like before.
    hook_on_qkv, hook_on_residual = make_hook(directions, num_heads, device, alpha)
    qkv_lin = find_qkv_linear(model, layer)
    if qkv_lin is not None:
        handle = qkv_lin.register_forward_hook(hook_on_qkv)
    else:
        target = model.transformer.h[layer]
        handle = target.register_forward_hook(hook_on_residual)

    items = load_truthfulqa()
    results = []
    for ex in tqdm(items, desc="gen"):
        q = ex['question']
        prompt = f"Q: {q}\nA:"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        cur_ids = input_ids
        generated = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = model(input_ids=cur_ids)
                logits = out.logits
            next_logits = logits[:, -1, :].cpu().numpy().squeeze(0)
            probs = np.exp(next_logits - next_logits.max())
            probs = probs / probs.sum()
            nxt = top_p_sample_next_token(probs, temperature, top_p)
            generated.append(int(nxt))
            cur_ids = torch.cat([cur_ids, torch.tensor([[nxt]], device=device)], dim=1)
            # simple end condition
            if nxt == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append({'question': q, 'generation': text})

    handle.remove()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='EleutherAI/gpt-neo-2.7B')
    parser.add_argument('--directions', default='ITI/directions')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=15.0)
    parser.add_argument('--max_new_tokens', type=int, default=80)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    directions = load_directions(args.directions)
    results = generate_with_iti(model, tokenizer, directions, args.layer, args.num_heads, args.alpha, args.max_new_tokens, args.device, args.temperature, args.top_p)
    print(json.dumps(results[:10], indent=2))


if __name__ == '__main__':
    main()
