# Neural Training Optimization & Stability

A controlled PyTorch study of **optimizer mechanics, randomized search, regularization, convergence speed, and seed sensitivity** for an imbalanced seven-class classification problem.

Rather than asking only which training recipe produced the lowest loss, this project tests a more deployment-relevant question:

> **Which optimization choices improve class-balanced generalization reliably, and which merely improve a surrogate objective or overfitting indicator?**

The experiments hold the data split, preprocessing, architecture, loss, batch size, update budgets, and hardware class constant. Model selection is based on **seed-aggregated Macro-F1, balanced accuracy, per-class behavior, runtime, and evaluation cost** rather than accuracy or validation loss alone.

[Read the full technical report](reports/optimization-and-stability-report.pdf)

---

## Executive summary

The study compared randomized hill climbing, simulated annealing, a genetic algorithm, seven gradient-optimizer configurations, five regularization strategies, and an integrated training recipe.

The main result is deliberately non-obvious:

> **A lower validation loss did not reliably produce a better classifier.**

Randomized hill climbing reduced validation loss after SGD pretraining, but median locked-test Macro-F1 declined from **0.5424 to 0.5219**. The best median Macro-F1 came from **Adam without bias correction at 0.5824**, but its seed variability was substantially higher than standard Adam. Standard Adam remained the safer default when stability mattered.

### Results at a glance

| Decision objective | Best observed choice | Result | Engineering interpretation |
|---|---|---:|---|
| Highest median test Macro-F1 | Adam without bias correction | **0.5824** | Strongest class-balanced F1, but high seed sensitivity |
| Highest test accuracy | AdamW | **0.6827** | Best aggregate correctness |
| Highest balanced accuracy | AdamW | **0.7533** | Strongest average class recall |
| Fastest threshold crossing | Adam without bias correction | **320 updates / 0.824 s** | Fast convergence, less reliable across seeds |
| Most stable Adam-family baseline | Standard Adam | Macro-F1 std. **0.0010** | Recommended default when reproducibility matters |
| Best individual regularizer | L2 weight decay | Macro-F1 **0.5697** | Small gain; not uniformly stable |
| Best randomized-search objective | RHC | Validation loss **0.6505** | Lower loss did not improve test Macro-F1 |
| Integrated recipe | L2 + dropout + RHC | Macro-F1 **0.5638** | More complexity did not beat the optimizer ablation |

---

## Why this project matters

Optimizer comparisons are often invalid because architecture, preprocessing, training duration, or search effort changes between experiments. This repository treats optimization as a controlled engineering investigation:

- The external test set remained locked during development.
- Every primary comparison used the same compact MLP backbone.
- Gradient-based methods received the same update budget.
- Randomized optimizers received the same function-evaluation budget.
- Results were aggregated across three fixed seeds.
- Minority-class precision and recall were reviewed alongside aggregate metrics.
- Runtime and evaluation counts were treated as first-class outcomes.

This prevents a faster optimizer, lower loss, or smaller train-validation gap from being mistaken for proven generalization.

<p align="center">
  <img src="assets/controlled-experiment-design.png" alt="Controlled neural training experiment design" width="100%">
</p>

---

## Experimental protocol

### Data

The project uses the UCI Forest Covertype dataset:

- **581,012** total records
- **54** input features
- **7** target classes
- Strong class imbalance
- No overlap between development and final test records

| Partition | Rows | Purpose |
|---|---:|---|
| Training | 16,000 | Parameter learning |
| Validation | 4,000 | Hyperparameter and recipe selection |
| Locked test | 561,012 | Final external evaluation only |

An exact row-multiset audit verifies that the development sample and external remainder reconstruct the complete dataset. Standardization is fitted only on the training partition and then applied to validation and test data.

### Fixed model

```text
Input: 54 features
  ↓
Linear(54, 64) + ReLU
  ↓
Linear(64, 64) + ReLU
  ↓
Linear(64, 7)
```

- **8,135 trainable parameters**
- Batch size: **512**
- Loss: class-weighted cross-entropy
- Hardware: CPU, four PyTorch threads
- Seeds: `42`, `202`, `7641`

### Primary metrics

- Macro-F1
- Balanced accuracy
- Per-class precision, recall, and F1
- Validation loss
- Accuracy
- Wall-clock time
- Gradient evaluations
- Validation-objective function evaluations

Macro-F1 is the principal metric because accuracy over-rewards the dominant classes.

---

## Experiment modules

### 1. Randomized final-layer optimization

The pretrained feature extractor was frozen while three derivative-free methods optimized only the 455-parameter output layer:

- Randomized hill climbing
- Simulated annealing
- Genetic algorithm

Each method received **300 validation-objective calls** using the same fixed stratified validation subset.

<p align="center">
  <img src="assets/randomized-optimization-outcome.png" alt="Randomized optimization validation loss versus locked test Macro-F1" width="82%">
</p>

**Finding:** RHC was the most compute-efficient validation-loss search, but every randomized method reduced test Macro-F1 relative to SGD pretraining. The weighted loss rewarded higher rare-class recall even when precision deteriorated enough to reduce class F1.

### 2. Adam-family and SGD ablations

Seven optimizer conditions were compared under **1,600 optimizer updates**:

- Plain SGD
- SGD with momentum
- Nesterov momentum
- Standard Adam
- Adam without bias correction
- Adam with `beta1 = 0`
- AdamW

<p align="center">
  <img src="assets/optimizer-scorecard.png" alt="Optimizer accuracy Macro-F1 and balanced accuracy scorecard" width="100%">
</p>

**Finding:** adaptive scaling and momentum substantially improved convergence speed. Adam without bias correction crossed the fixed validation-loss threshold fastest and produced the best median Macro-F1, but its results varied far more across seeds.

<p align="center">
  <img src="assets/speed-stability-tradeoff.png" alt="Adam convergence speed and seed stability tradeoff" width="82%">
</p>

The no-bias variant is therefore a **high-upside, higher-risk condition**, not a replacement for standard Adam by default.

### 3. Regularization study

Standard Adam was held fixed while the following interventions were tested:

- L2 weight decay
- Hidden-layer dropout
- Early stopping
- Label smoothing
- Training-only Gaussian input noise

<p align="center">
  <img src="assets/regularization-effects.png" alt="Regularization effects on Macro-F1 and balanced accuracy" width="92%">
</p>

**Finding:** L2 weight decay was the only individual regularizer with a positive median Macro-F1 movement. Dropout and early stopping reduced overfitting indicators, but those reductions did not translate into stronger test Macro-F1. Label smoothing performed especially poorly under class-weighted cross-entropy.

### 4. Integrated recipe

The final experiment combined validation-selected elements:

- Adam
- Weight decay `1e-4`
- Dropout `0.15`
- Final 30-call RHC pass

The integrated recipe did **not** outperform the strongest optimizer condition. The final RHC pass changed paired median Macro-F1 by approximately zero and added no reliable predictive benefit.

---

## Detailed comparison

Median outcomes across the three evaluation seeds:

| Method | Validation loss | Accuracy | Macro-F1 | Balanced accuracy | Time (s) | Gradient evals | Function evals |
|---|---:|---:|---:|---:|---:|---:|---:|
| SGD pretraining | 0.7203 | 0.6401 | 0.5424 | 0.6931 | 1.620 | 1,152 | 0 |
| SGD + RHC | 0.6505 | 0.6243 | 0.5219 | 0.7190 | 1.909 | 1,152 | 300 |
| SGD momentum | 0.5884 | 0.6642 | 0.5666 | 0.7416 | 3.651 | 1,600 | 0 |
| Standard Adam | 0.5990 | 0.6564 | 0.5659 | 0.7464 | 4.320 | 1,600 | 0 |
| **Adam, no bias correction** | **0.5712** | 0.6765 | **0.5824** | 0.7501 | 3.970 | 1,600 | 0 |
| Adam, `beta1 = 0` | 0.6028 | 0.6657 | 0.5640 | 0.7477 | 4.244 | 1,600 | 0 |
| **AdamW** | 0.5927 | **0.6827** | 0.5661 | **0.7533** | 4.349 | 1,600 | 0 |
| Adam + L2 | 0.5935 | 0.6599 | 0.5697 | 0.7453 | 4.342 | 1,600 | 0 |
| Adam + L2 + dropout | 0.5833 | 0.6411 | 0.5480 | 0.7514 | 4.866 | 1,600 | 0 |
| Integrated before RHC | 0.5780 | 0.6534 | 0.5626 | 0.7512 | 5.319 | 1,600 | 0 |
| Integrated + RHC | 0.5784 | 0.6468 | 0.5638 | 0.7489 | 5.349 | 1,600 | 30 |

### Class-level behavior

The most important errors were not visible in aggregate accuracy:

- RHC improved rare-class recall while sharply reducing rare-class precision.
- Adam without bias correction improved F1 for classes 1, 2, 5, 6, and 7 in the representative run.
- L2 improved classes 4 and 7 but reduced F1 for the dominant class 2.
- Dropout combinations shifted probability toward minority labels without achieving a net Macro-F1 gain.

This is why every recipe was evaluated using per-class metrics rather than a single leaderboard score.

---

## Decision rules derived from the study

| Deployment priority | Recommended starting point |
|---|---|
| Stability and reproducibility | Standard Adam: `lr=0.003`, `beta1=0.9`, `beta2=0.99` |
| Highest observed Macro-F1 | Adam without bias correction, only after repeated-seed validation |
| Accuracy and average class recall | AdamW |
| Small minority-class improvement | Standard Adam + `1e-4` L2, contingent on seed aggregation |
| Lower train-validation gap | Do not select dropout or early stopping without confirming test Macro-F1 |
| Final-layer randomized search | Reject unless gains survive full validation, multi-seed testing, and per-class review |

---

## Repository structure

```text
neural-training-optimization-and-stability/
├── run_all.sh
├── environment.yml			              # preferred Conda specification
├── requirements.txt			            # optional pip fallback
├── 01_eda_dataset_original.ipynb		  # Needed because it outputs stratified data sample
│   └── dataset_covertype/
│       ├── covtype.data			        # ! PLACE covtype dataset HERE
├── part1_randomized_optimization.ipynb
├── part2_adam_family_optimizer.ipynb
├── part3_regularization_study.ipynb
├── part4_integrated_best_combination.ipynb
├── output_data/
│   └── eda_dataset_original/
│       ├── dataset_stratified.csv
│       └── dataset_remainder.csv
├── output_figures/
└── output_results/

```

The notebook filenames can remain mapped to the original experiment parts; the numbered names above show the recommended portfolio presentation.

---

## Reproducing the experiments

### Conda

```bash
conda env create -f environment.yml
conda run -n neural-training-optimization bash run_all.sh
```

### Pip fallback

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash run_all.sh
```

### Expected data location

```text
dataset_covertype/
└── covtype.data
```

Raw data and generated split files should not be committed. The pipeline should rebuild the stratified development sample and locked remainder from the source dataset using fixed seeds and validate the final row counts before model execution.

---

## Skills demonstrated

- Controlled ML experimentation and ablation design
- PyTorch model and training-loop implementation
- Adam, AdamW, SGD, momentum, and bias-correction analysis
- Randomized hill climbing, simulated annealing, and genetic search
- Class-imbalance handling with weighted loss
- Multi-seed stability analysis
- Leakage-aware train/validation/test design
- Per-class error analysis
- Convergence and compute-budget measurement
- Evidence-based rejection of unnecessary model complexity

---

## Limitations

- Three seeds expose major instability but do not provide a high-confidence distribution estimate.
- Conclusions are specific to the fixed compact MLP, CPU environment, update budgets, and Covertype representation.
- Learning-rate selection and optimizer comparisons remain budget-dependent.
- Class-weighted cross-entropy changes the relationship between validation loss, calibration, precision, and recall.
- The no-bias Adam result should be treated as an empirical finding for this protocol, not a general recommendation.

## Next engineering steps

1. Expand evaluation to at least 10 seeds and report bootstrap confidence intervals.
2. Add probability calibration metrics such as expected calibration error and classwise Brier score.
3. Compare fixed-budget results with scheduler-aware and convergence-based stopping protocols.
4. Profile CPU and GPU throughput separately from end-to-end wall time.
5. Test whether the speed-stability result persists across larger MLPs and additional imbalanced datasets.

---

## Core takeaway

> **Optimization metrics describe training behavior; they do not prove predictive improvement.**

For an imbalanced multiclass neural network, a defensible training recipe must survive seed aggregation, locked-test evaluation, per-class precision and recall analysis, and compute-cost review. In this study, the simplest successful conclusion was also the strongest: standard Adam was the stable default, no-bias Adam was the highest-upside ablation, L2 offered only a small conditional gain, and additional randomized search did not justify its complexity.
