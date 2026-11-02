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

Grid: 2 depths [4,5] × 2 widths [24,32] × 2 LRs [.001,.0003] = 8 trials.
Weight decay fixed 0.0001. Reproduce all eight rows from `p4_grid_trials.csv`
with depth, width, LR, best val accuracy, macro-F1, and epoch. State chosen
hyperparameters and why they won; mention the limited tuning budget.

## Final training and handoff

Winner: [fill]; selected best epoch: [fill]; val accuracy: [fill];
val macro-F1: [fill]; total training time: [fill]; checkpoint path: [fill].
Use final history plots, report ReduceLROnPlateau and patience-20 stopping.
Give checkpoint and experiment tables to the leader for merging report sections.
P3 owns Grad-CAM; P5 owns scratch-vs-transfer comparison. One seed and one
split are limitations; the 10-epoch pilot ranking may not reflect 80 epochs.
