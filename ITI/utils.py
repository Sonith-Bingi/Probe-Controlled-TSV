"""Small utility helpers for ITI prototype.

Note: This is a prototype implementing a head-wise approximation by
splitting residual hidden vectors into equal-sized chunks (heads).
This is a simplification that works reasonably for many GPT-style
models where embed_dim is divisible by num_heads.
"""
from typing import List, Dict, Tuple
import json
import numpy as np
from datasets import load_dataset
import torch
from typing import Optional


def load_truthfulqa():
    """Load TruthfulQA 'generation' validation split and return list of dicts.
    Each item: { 'question': str, 'best_answer': str, 'correct_answers': [..], 'incorrect_answers': [..] }
    """
    ds = load_dataset("truthful_qa", "generation", split="validation")
    items = []
    for ex in ds:
        q = ex.get("question") or ex.get("prompt") or ex.get("Q")
        best = ex.get("answer") or ex.get("best_answer") or ex.get("gold_answer")
        # best may be None for some versions, fall back to first of answers
        if best is None:
            answers = ex.get("answers") or ex.get("answers_list") or []
            best = answers[0] if len(answers) else ""
        correct = ex.get("correct_answers") or ex.get("correct_answer") or []
        incorrect = ex.get("incorrect_answers") or ex.get("incorrect_answer") or []
        items.append({
            "question": q, 
            "best_answer": best, 
            "correct_answers": correct,
            "incorrect_answers": incorrect
        })
    return items


def save_directions(path: str, directions: Dict):
    # directions contains numpy arrays; save as npz + json metadata
    meta = {}
    npz_dict = {}
    for k, v in directions.items():
        # v is dict with 'theta' (np.array) and 'sigma' (float) and other metadata
        theta = np.array(v["theta"], dtype=np.float32)
        theta_norm = float(np.linalg.norm(theta))
        # normalize theta before saving so apply-time magnitude = alpha * sigma
        if theta_norm > 0:
            theta_unit = (theta / (theta_norm + 1e-12)).astype(np.float32)
        else:
            theta_unit = theta

        # store metadata (exclude 'theta' vector itself)
        meta[k] = {kk: vv for kk, vv in v.items() if kk != "theta"}
        # include original norm for diagnostics
        meta[k]["theta_norm"] = theta_norm
        npz_dict[f"{k}_theta"] = theta_unit

    np.savez_compressed(path + ".npz", **npz_dict)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_directions(path: str) -> Dict:
    meta = json.load(open(path + ".json", "r"))
    arrs = np.load(path + ".npz")
    directions = {}
    for k, info in meta.items():
        theta = arrs[f"{k}_theta"]
        # if the saved metadata doesn't have theta_norm, compute it and
        # treat stored theta as possibly unnormalized: normalize to unit vector
        theta = np.array(theta, dtype=np.float32)
        theta_norm = float(info.get("theta_norm", np.linalg.norm(theta)))
        if theta_norm > 0:
            # ensure unit vector during load (defensive)
            theta_unit = (theta / (theta_norm + 1e-12)).astype(np.float32)
        else:
            theta_unit = theta
        directions[k] = {**info, "theta": theta_unit}
    return directions


def split_into_heads(vecs: np.ndarray, num_heads: int) -> np.ndarray:
    """Split last-dim vectors into head slices.

    vecs: shape (..., embed_dim)
    returns: shape (..., num_heads, head_dim)
    """
    embed = vecs.shape[-1]
    if embed % num_heads != 0:
        raise ValueError(f"embed_dim ({embed}) not divisible by num_heads ({num_heads})")
    head_dim = embed // num_heads
    newshape = vecs.shape[:-1] + (num_heads, head_dim)
    return vecs.reshape(newshape)


def find_qkv_linear(model: torch.nn.Module, layer_id: int) -> Optional[torch.nn.Module]:
    """Try to locate the linear projection module that produces the concatenated
    query/key/value (qkv) tensor for a given transformer block.

    Returns the nn.Linear module if found, otherwise None.
    """
    # common HF causallm layout: model.transformer.h[layer].attn.query_key_value
    try:
        block = model.transformer.h[layer_id]
    except Exception:
        return None

    attn = getattr(block, 'attn', None) or getattr(block, 'attention', None)
    if attn is None:
        # give up
        return None

    # search for a Linear with out_features == 3 * embed_dim
    for name, module in attn.named_modules():
        if isinstance(module, torch.nn.Linear):
            outf = getattr(module, 'out_features', None)
            if outf is None:
                continue
            # if out_features is divisible by 3, it's likely the combined qkv proj
            if outf % 3 == 0:
                return module

    return None


def find_value_linear(model: torch.nn.Module, layer_id: int) -> Optional[torch.nn.Module]:
    """Try to locate a Linear module that specifically projects the VALUE
    (not fused qkv) for a given transformer block. Returns the nn.Linear
    module if found, otherwise None.
    """
    try:
        block = model.transformer.h[layer_id]
    except Exception:
        return None

    attn = getattr(block, 'attn', None) or getattr(block, 'attention', None)
    if attn is None:
        return None

    # try to find an obvious value linear by name first
    for name, module in attn.named_modules():
        if isinstance(module, torch.nn.Linear):
            lname = name.lower()
            if 'v_proj' in lname or 'value' in lname or '.v' in lname or 'v_' in lname or 'proj_v' in lname:
                return module

    # fall back to finding any Linear whose out_features equals embed dim
    # (likely the value projection when not fused)
    for name, module in attn.named_modules():
        if isinstance(module, torch.nn.Linear):
            outf = getattr(module, 'out_features', None)
            if outf is None:
                continue
            # if out_features seems equal to embed dim (not 3*embed), we may have found v
            if outf % 3 != 0:
                return module

    return None


def extract_value_from_qkv(qkv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Given a qkv tensor (batch, seq, 3*embed), return the value tensor
    shaped (batch, seq, num_heads, head_dim).

    Expects torch.Tensor input.
    """
    if not isinstance(qkv, torch.Tensor):
        raise ValueError("qkv must be a torch.Tensor")
    total = qkv.shape[-1]
    if total % 3 != 0:
        raise ValueError("qkv last-dim must be divisible by 3")
    embed = total // 3
    v = qkv[..., 2 * embed: 3 * embed]
    head_dim = embed // num_heads
    if embed % num_heads != 0:
        raise ValueError(f"embed ({embed}) not divisible by num_heads ({num_heads})")
    return v.view(*v.shape[:-1], num_heads, head_dim)