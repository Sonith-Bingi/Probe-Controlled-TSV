# ITI (Inference-Time Intervention) prototype

This folder contains a prototype implementation of the ITI method from:
"Inference-Time Intervention: Eliciting Truthful Answers from a Language Model".

Files
- `train_directions.py` - probe-training and direction-finding script. It:
  - extracts layer hidden states per (question, answer) pair,
  - approximates attention-head activations by slicing the residual vector,
  - trains a logistic probe per head,
  - computes mass-mean shift directions and per-head sigma,
  - saves top-K directions to `directions.npz` + `directions.json`.
- `apply_iti.py` - applies saved directions at inference time by registering a
  forward hook that injects `alpha * sigma * theta` into the head slices of the
  specified layer while generating answers.
- `utils.py` - small helpers for dataset loading and direction serialization.

Notes and assumptions
- This is a practical prototype that approximates attention-head outputs by
  splitting the residual vector (last-token hidden state) into head-sized chunks.
  It is not identical to extracting the `value` outputs from each head, but it is
  simple and works across many Hugging Face GPT-style models with embed_dim
  divisible by num_heads.
- For production-grade experiments you may want to extract per-head value
  vectors directly from attention modules (different models expose these
  tensors differently). This prototype is designed to be a clear starting point.

Quick start
1. Train directions (uses TruthfulQA validation split):

```bash
python ITI/train_directions.py --model_name EleutherAI/gpt-neo-2.7B --layer_id 10 --num_heads 64 --top_k 48 --out_path ITI/directions
```

2. Run inference with ITI:

```bash
python ITI/apply_iti.py --model_name EleutherAI/gpt-neo-2.7B --directions ITI/directions --layer 10 --num_heads 64 --alpha 15
```

You can adjust `alpha`, `top_k`, `layer`, and sampling hyperparameters.

Security / Safety
- The generated outputs may still be incorrect or biased. Evaluate carefully
  and don't rely on this prototype as a final solution for truthfulness-sensitive
  applications.

License: follow repo license.
