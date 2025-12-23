# Probe Controlled TSV

Brief: A collection of scripts and code to run and reproduce experiments for probe-controlled targeted steering verification (TSV) using GPT-Neo models.

This project contains bash scripts to run all experiments from `CS762-Final.ipynb`. All experiment scripts are located in the `scripts/` directory.

## Scripts Overview

### Individual Experiment Scripts

0. **`run_experiment_0_generate_samples.sh`**
   - Generates TruthfulQA model answers and BLEURT scores **if they do not already exist**
   - Wraps `01_generate_samples.sh` (root-level, configured for 2.7B) or falls back to `experiments/probe_controlled_tsv/scripts/01_generate_samples.sh`
   - Produces:
     - `save_for_eval/tqa_hal_det/answers/*.npy`
     - `ml_tqa_bleurt_score.npy`

1. **`run_experiment_1_train_tsv_logreg.sh`**
   - Trains TSV logistic regression model on full dataset
   - Corresponds to Cell 4 in the notebook
   - Output: `artifacts/gpt-neo-2.7B_logreg_tsv_817.pt`

2. **`run_experiment_2_train_probe.sh`**
   - Trains hallucination probe on full dataset
   - Corresponds to Cell 5 in the notebook
   - Output: `artifacts/gpt-neo-2.7B_probe_817.pt`

3. **`run_experiment_3_run_full_experiment.sh`**
   - Runs full comparative experiment with all methods
   - Corresponds to Cells 6, 7, and 8 in the notebook
   - Output: `results/full_experiment_comparison/`

4. **`run_experiment_4_iti_pipeline.sh`**
   - Runs ITI (Inverse Truthfulness Intervention) pipeline
   - Corresponds to Cell 9 in the notebook
   - Output: `results/full_experiment_comparison/`

### Master Script

**`run_all_experiments.sh`**
- Runs data generation (Experiment 0) and all experiments in sequence
- Can skip data generation and individual experiments using flags
- Provides summary and timing information

## Usage

**Important**: All scripts must be run from the `scripts/` directory. Navigate there first:

```bash
cd scripts
```

### Run All Experiments

From the `scripts/` directory:

```bash
./run_all_experiments.sh
```

### Run Individual Experiments

From the `scripts/` directory:

```bash
# Experiment 0: Generate Samples (if needed)
./run_experiment_0_generate_samples.sh

# Experiment 1: Train TSV
./run_experiment_1_train_tsv_logreg.sh

# Experiment 2: Train Probe
./run_experiment_2_train_probe.sh

# Experiment 3: Full Experiment
./run_experiment_3_run_full_experiment.sh

# Experiment 4: ITI Pipeline
./run_experiment_4_iti_pipeline.sh
```

### Skip Specific Experiments

From the `scripts/` directory:

```bash
# Skip experiment 0 (if data already exists)
./run_all_experiments.sh --skip-0

# Skip experiment 1 (if TSV model already exists)
./run_all_experiments.sh --skip-1

# Skip multiple experiments
./run_all_experiments.sh --skip-1 --skip-2
```

## Prerequisites

- Python 3 with required packages (see `requirements.txt`)
- CUDA-capable GPU (for `--device cuda`)
- All data files and artifacts in place:
  - `artifacts/gpt-neo-2.7B_split.npz`
  - `save_for_eval/tqa_hal_det/` directory with answer files
  - `ml_tqa_bleurt_score.npy`

## Experiment Parameters

All scripts use the exact parameters from `CS762-Final.ipynb`:

### Experiment 1
- Model: `EleutherAI/gpt-neo-2.7B`
- Layer: 17
- C: 0.1
- Threshold: 0.5
- Test size: 0.2
- Batch size: 16

### Experiment 2
- Model: `EleutherAI/gpt-neo-2.7B`
- Layer: 17
- Probe type: mlp
- Epochs: 100
- Learning rate: 0.001
- Batch size: 16
- Threshold: 0.5

### Experiment 3
- Max new tokens: 50
- Layer ID: 17
- Alpha max: 1.5

### Experiment 4
- Model: `EleutherAI/gpt-neo-2.7B`
- Split: `artifacts/gpt-neo-2.7B_split.npz`
- Layer: 17
- Num heads: 20
- Top k: 10
- Alpha: 7
- Directions: `ITI/directions`
- Max new tokens: 50

## Output Structure

```
CS762-Final/
├── artifacts/
│   ├── gpt-neo-2.7B_logreg_tsv_817.pt      # From Experiment 1
│   ├── gpt-neo-2.7B_probe_817.pt            # From Experiment 2
│   └── gpt-neo-2.7B_split.npz
└── results/
    └── full_experiment_comparison/          # From Experiments 3 & 4
        ├── baseline_generations.json
        ├── fixed_logistic_generations.json
        ├── adaptive_logistic_generations.json
        ├── original_tsv_generations.json
        ├── iti_generations.json
        ├── full_experiment_results.json
        └── summary.json
```

## Experimental Results

The following results were obtained by running all experiments on the TruthfulQA dataset (817 samples) using the `EleutherAI/gpt-neo-2.7B` model at layer 17.

### Experiment 1: TSV Logistic Regression Training

**Training Performance:**
- Train Accuracy: **1.0000** (perfect fit)
- Train AUC: **1.0000**
- Test Accuracy: **0.7195**
- Test AUC: **0.7735**
- TSV Vector Norm: **2.4344**
- Max Logit Change: **1.1427**

**Dataset Split:**
- Total Samples: 817
- Train: 653 (80%)
- Test: 164 (20%)
- Truthful (BLEURT > 0.5): 247 samples
- Hallucinated (BLEURT ≤ 0.5): 570 samples

### Experiment 2: Probe Training

**Training Performance:**
- Probe Type: MLP (Multi-Layer Perceptron)
- Train Accuracy: **0.8775**
- Train AUC: **0.9317**
- Test Accuracy: **0.8049**
- Test AUC: **0.7956**

The probe successfully learns to distinguish between truthful and hallucinated responses from hidden states.

### Experiment 3: Full Method Comparison

Results on held-out test set (164 questions):

| Method               | Accuracy | Hal Rate | BLEURT  | Style Sim | Steer Rate |
|----------------------|----------|----------|---------|-----------|------------|
| **Baseline**         | 0.3537   | 0.6463   | 0.3162  | 0.8474    | 0.0000     |
| **Fixed Logistic**   | 0.4024   | 0.5976   | 0.3220  | 0.8410    | 1.0000     |
| **Adaptive Logistic**| **0.4146**| **0.5854**| **0.3369**| 0.8381    | 0.7498     |
| **Original TSV**     | 0.3720   | 0.6280   | 0.3248  | 0.8435    | 1.0000     |

**Key Findings:**
- **Adaptive Logistic** achieves the best performance:
  - Highest accuracy (41.46%)
  - Lowest hallucination rate (58.54%)
  - Highest BLEURT score (0.3369)
  - Selective steering (74.98% of tokens steered, vs. 100% for fixed methods)
- **Fixed Logistic** shows strong performance:
  - Second highest accuracy (40.24%)
  - Second lowest hallucination rate (59.76%)
  - Good BLEURT score (0.3220)
- **Original TSV** performs better than baseline but below the logistic methods
- **Baseline** (no steering) has the highest hallucination rate (64.63%)

### Experiment 4: ITI Pipeline

**ITI Evaluation Results:**

| Method | Accuracy | Hal Rate | BLEURT  | Style Sim | Risk   |
|--------|----------|----------|---------|-----------|--------|
| **ITI** | 0.3841   | 0.6159   | **0.3393** | 0.8463    | 0.5024 |

**ITI Performance:**
- Accuracy: 0.3841 (competitive with Fixed Logistic)
- Hallucination Rate: 0.6159 (better than baseline)
- BLEURT: **0.3393** (highest among all methods)
- Style Similarity: 0.8463 (highest among all methods)
- Mean Risk: 0.5024 (moderate risk level)

**ITI Training Details:**
- Top head scores: [0.714, 0.691, 0.691, 0.668, 0.664]
- Training examples: 1306 (653 × 2, balanced True/False pairs)
- Test examples: 164

### Summary

1. **Best Overall Method**: Adaptive Logistic TSV with probe-controlled steering
   - Achieves highest accuracy (41.46%) and lowest hallucination rate (58.54%)
   - Highest BLEURT score among TSV methods (0.3369)
   - Uses selective steering (74.98% of tokens), making it more efficient than fixed methods

2. **Fixed Logistic Performance**: Strong alternative to adaptive method
   - Second highest accuracy (40.24%)
   - Second lowest hallucination rate (59.76%)
   - Simpler implementation (fixed alpha, no probe needed)

3. **ITI Performance**: Best semantic quality
   - Highest BLEURT score among all methods (0.3393)
   - Highest style similarity (0.8463)
   - Competitive accuracy (38.41%)

4. **Steering Effectiveness**: All steering methods outperform baseline
   - Fixed Logistic shows ~5% improvement in accuracy over baseline
   - Adaptive Logistic shows ~6% improvement over baseline
   - Original TSV shows ~2% improvement over baseline

5. **Training Quality**:
   - TSV logistic regression achieves perfect training fit (AUC=1.0) with good generalization (Test AUC=0.77)
   - Probe achieves strong performance (Test AUC=0.80, Test Accuracy=0.80) for risk detection

## Notes

- **All scripts must be run from the `scripts/` directory** - navigate there first with `cd scripts`
- All scripts set `PYTHONUNBUFFERED=1` for real-time output
- Scripts use absolute paths to ensure they work correctly from the scripts directory
- Each script includes error checking and will exit on failure
- The master script tracks execution time and provides a summary
- Typical execution time: ~37 minutes for all experiments on GPU

