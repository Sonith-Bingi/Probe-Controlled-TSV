#!/usr/bin/env python3
"""Smoke test for EleutherAI/gpt-neo-2.7B architecture compatibility.

This script instantiates the model configuration (no pretrained weights),
loads the tokenizer, runs a tiny forward pass to get hidden states, checks for
presence/shape of `lm_head.weight`, computes a dummy TSV logit-shift matmul,
and runs a forward pass through the project's probe classes to ensure shapes
and calling conventions match.

Safe to run locally — it does NOT download model weights. It will download
tokenizer and config files from Hugging Face though.
"""
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from models.probe import MLPProbe, LinearProbe

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def main():
    model_name = "EleutherAI/gpt-neo-2.7B"
    layer_id = 9

    try:
        print("Loading config and tokenizer (no weights)...")
        cfg = AutoConfig.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)

        print("Instantiating model architecture from config (no pretrained weights)...")
        model = AutoModelForCausalLM.from_config(cfg)
        model.eval()

        print(f"Model hidden_size={model.config.hidden_size}, n_layers={getattr(model.config,'n_layer', getattr(model.config,'num_hidden_layers', 'N/A'))}")

        # Tokenize a tiny prompt
        prompt = "Q: What is 2+2? A:"
        inputs = tok(prompt, return_tensors="pt")

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        hs = out.hidden_states
        print(f"Got {len(hs)} hidden state layers")
        if len(hs) <= layer_id:
            print(f"Warning: model has only {len(hs)} layers, requested layer {layer_id}")
        else:
            print(f"Layer {layer_id} shape: {hs[layer_id].shape}")

        # Check lm_head
        if hasattr(model, "lm_head"):
            w = model.lm_head.weight
            print("lm_head.weight shape:", tuple(w.shape))
        else:
            raise RuntimeError("Model missing lm_head attribute — incompatible architecture for TSV logit-shift")

        # Dummy TSV vector and matmul
        hidden_size = model.config.hidden_size
        tsv_vec = torch.randn(hidden_size)
        try:
            logit_shift = torch.matmul(tsv_vec.unsqueeze(0).float(), w.float().T)
            print("TSV logit-shift shape:", tuple(logit_shift.shape))
        except Exception as e:
            print("Error computing TSV logit-shift:", e)
            raise

        # Probe shape check
        print("Instantiating MLPProbe and running forward pass...")
        probe = MLPProbe(hidden_size)
        probe.eval()
        x = torch.randn(1, hidden_size)
        with torch.no_grad():
            p_out = probe(x)
        print("Probe output shape:", tuple(p_out.shape))

        print('\nSmoke test PASSED: architectural operations succeeded (no pretrained weights loaded).')
    except Exception:
        print('\nSmoke test FAILED — see traceback below:\n')
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
