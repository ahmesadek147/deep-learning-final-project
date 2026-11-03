# P4 report — scratch-CNN architecture variants and Grid Search

Fill all metrics from actual `outputs/p4_scoped/` files. Do not invent scores.

## Models and rationale

| Candidate | Conv layout | Pooling | Residual shortcuts | Parameters |
|---|---|---|---|---:|
| Original SimpleCNN | Conv–BN–ReLU–Pool ×3, final Conv | Early | No | [fill] |
| Delayed-pooling CNN | Two Conv–BN–ReLU per stage | After stages 2+ | No | [fill] |
| Residual scratch CNN | Two Conv–BN–ReLU per stage | After stages 2+ | Yes | [fill] |

All models are randomly initialized. State why delayed spatial downsampling
may help preserve fine bird features; note that it increases compute cost.

## Fair evaluation protocol

- P2-provided duplicate-safe stratified group split of official CUB train;
  fixed seed 42; manifest SHA-256: [fill].
- Image size 224, 200 classes, P3 transform configuration, AdamW optimizer,
  shared P1 trainer, learning rate and weight decay held fixed in pilot.
- Validation macro-F1 is primary; accuracy secondary. The official test is
  untouched until model selection is frozen.

## Architecture pilot results

| Model | Best epoch | Val accuracy | Val macro-F1 | Params | Runtime | Peak T4 VRAM |
|---|---:|---:|---:|---:|---:|---:|
| SimpleCNN | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| Delayed pooling | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| Residual CNN | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |

Plot validation accuracy/macro-F1 and train/validation loss per model.
Describe convergence, overfitting and resource tradeoffs.

## Grid Search

Verified grid: 2 depths [4,5] × 2 widths [24,32] = **4 trials** of the
**residual CNN**, each trained for 40 epochs. Learning rate was held at 0.001
and weight decay at 0.0001; neither was tuned in this experiment. Reproduce
all four rows from `p4_residual_dw40_trials.csv`, including depth, width,
parameter count, best val accuracy, macro-F1, and epoch. The selected trial
was 04 (depth 5, width 32), at epoch 40: validation accuracy **31.9167%**,
macro-F1 **30.1291%**. Explain the 3.61M vs 2.83M parameter cost of choosing
width 32 rather than 24 at depth 5. Do not mix in the separate cosine run.

## Final training and handoff

Grid winner: `p4_residual_dw40-trial-04`; selected epoch: **40**;
val accuracy: **31.92%**; val macro-F1: **30.13%**;
grid-winner training time: **33.4 minutes**; checkpoint saved in its
`runs/p4_residual_dw40-trial-04/best_model.pth` directory in the original
Kaggle backup. Separate final-training/testing results: **pending**.
Use final history plots, report ReduceLROnPlateau and patience-20 stopping.
Give checkpoint and experiment tables to the leader for merging report sections.
P3 owns Grad-CAM; P5 owns scratch-vs-transfer comparison. One seed and one
split are limitations; the 10-epoch pilot ranking may not reflect 80 epochs.
