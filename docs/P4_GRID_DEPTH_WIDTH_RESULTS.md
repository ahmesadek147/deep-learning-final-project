# P4 — Scratch Residual CNN: 40-Epoch Depth × Width Grid Search

## Verified outcome

**Selected run:** `p4_residual_dw40-trial-04` (**depth 5, width 32**).

- Best validation accuracy: **31.9167%** (383/1,200 images; rounded to **31.92%**).
- Best validation macro-F1: **30.1291%** (rounded to **30.13%**).
- Selected epoch: **40/40**.
- Best validation cross-entropy loss: **3.0525**.
- Trainable parameters: **3,606,728**.
- Training time: **2,004.3 seconds** (approximately **33.4 minutes**).
- Recorded peak GPU memory: **1,493.4 MB**.
- The official CUB test partition was **not used** in these four trials.

The supplied `P4_Grid_DepthWidth40_COMPLETE.zip` passed the ZIP integrity test. The four `epochs.csv` files were checked against `search/p4_residual_dw40_trials.csv` and each run's `result.json`; all **160 epoch records** are present and the best-epoch/metric values agree. The search selection file records **4 completed trials, no completed-trial failures**, and selection by **validation macro-F1**. Trial 4's epoch 32 exists in its saved CSV, although it was absent from a previously copied console log.

## Experimental setup

| Item | Configuration |
|---|---|
| Task | CUB-200-2011 bird-species classification; 200 classes |
| Model family | Residual CNN trained from scratch; **no pretrained weights** |
| Data source | `Multimodal-Fatima/CUB_train` |
| Development split | 4,794 train / 1,200 validation images from 5,994 CUB training images |
| Split protocol | P2 `stratified_group`, version `v2`, seed `42` |
| Image size / batch size | 224 × 224 / 16 |
| Optimizer / loss | AdamW / CrossEntropyLoss |
| Learning rate / weight decay | **0.001 / 0.0001**, fixed across all four trials |
| Learning-rate schedule | `ReduceLROnPlateau`, monitoring validation macro-F1 (factor 0.5, patience 5, min LR `1e-6`) as implemented in `src/training/p4_experiments.py` |
| Epoch budget | 40 for every trial; early stopping disabled |
| Mixed precision | Enabled |
| Selection criterion | Highest best validation macro-F1; accuracy as a secondary criterion |
| Grid | Depth **{4, 5}** × base width **{24, 32}** = **4 trials** |
| Model settings held fixed | `max_channels=384`, `dropout=0.3`, `num_classes=200` |

Training augmentation specified in the saved configurations: random resized crop (scale 0.8–1.0), horizontal flip (probability 0.5), rotation (10°), brightness/contrast (0.15), saturation (0.1), hue (0.01), and random erasing (probability 0.1).

**Split integrity:** The separate, previously verified original P2 manifest has file SHA-256 `8f761ce564db71a32dfa8549210a66fc4cbfec282018609fa5709707298713eb` and internally stored manifest hash `6b003471157d8245087172e0792843b2413d0e8e5c5529f2ce534d82bb1925b2`. The manifest itself is **not inside this results ZIP**, so that prior verification is reported from the Colab verification record rather than re-performed from this ZIP.

## Complete grid-search results

Best metrics are measured at each trial's selected best **macro-F1** epoch; they are not independently chosen per metric.

| Trial | Depth | Width | Best epoch | Val accuracy | Val macro-F1 | Val loss | Parameters | Train time | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `trial-01` | 4 | 24 | 40 | 13.75% | 12.07% | 4.2005 | 724,040 | 25.5 min | 1,074.7 MB |
| `trial-02` | 4 | 32 | 39 | 15.67% | 13.53% | 4.1970 | 1,268,680 | 31.0 min | 1,424.5 MB |
| `trial-03` | 5 | 24 | 40 | 31.33% | 29.16% | 3.0628 | 2,829,128 | 28.4 min | 1,152.0 MB |
| **`trial-04` (winner)** | **5** | **32** | **40** | **31.92%** | **30.13%** | **3.0525** | **3,606,728** | **33.4 min** | **1,493.4 MB** |

Combined measured training time was approximately **118.3 minutes** (excluding dataset setup, download and other overhead).

## Analysis and architecture choice

1. **Depth drove the largest gains under this budget.** At width 24, increasing depth from 4 to 5 raised validation macro-F1 by **17.09 percentage points** (12.07% → 29.16%). At width 32, the increase was **16.60 points** (13.53% → 30.13%).
2. **Greater width helped, but more modestly.** At depth 5, moving from width 24 to 32 added **0.97 macro-F1 points** and **0.58 accuracy points**. It also added **777,600 parameters** (~27.5%) and ~**5 minutes** of training time (~17.6%).
3. **Trial 4 wins on the declared selection metric.** Its validation macro-F1 of 30.13% exceeds Trial 3's 29.16%. Trial 3 remains a useful lower-cost alternative (2.83M vs 3.61M parameters; ~1.15GB vs ~1.49GB recorded peak GPU memory).
4. **The best scores arrived near the end.** Trials 1, 3 and 4 peaked at epoch 40; Trial 2 at epoch 39. This suggests the 40-epoch budget may have truncated improvement. Additional training is a hypothesis to test, not a guaranteed gain.
5. **Generalization deserves monitoring.** Trial 4's final training accuracy was ~48.75%, compared with ~31.92% validation accuracy, an observed gap of ~16.83 percentage points. Validation loss and macro-F1 over longer training should be monitored for overfitting.

For context, an **earlier, separate 100-epoch depth-4/width-32 residual experiment** reached 28.83% validation accuracy and 27.45% macro-F1. It used a different learning-rate setup and **100 vs 40 epochs**, so it is **not** a controlled head-to-head comparison with this grid. A further **Colab cosine-schedule experiment** (AdamW, initial LR `3e-4`, WD `1e-4`, 100 epochs) is also **separate** and its final results are not incorporated here.

## Evidence and reproducibility

The compact GitHub evidence archive contains the following **for each trial**:

- `config.json`, `result.json`, and **all 40 epochs** in `epochs.csv`;
- `accuracy.png`, `macro_f1.png`, `loss.png`, and `learning_rate.png`;
- Search-level `p4_residual_dw40_state.json`, `p4_residual_dw40_selection.json`, and `p4_residual_dw40_trials.csv`.

The project code required to reproduce the training is on the P4 branch (`src/training/p4_experiments.py`, `src/models/p4_variants.py`, relevant YAML configs, and the P2 manifest/data pipeline). The original full ZIP also includes trial checkpoints; **model weights are intentionally excluded from the GitHub evidence archive** to keep the pull request small. Preserve the original ZIP or winning `best_model.pth` privately for restoration or final evaluation.

**Limitations and next step:** This is **four architecture combinations** with fixed LR/WD, **not an eight-run depth × width × LR search or a learning-rate/weight-decay grid**. It uses one split and one seed; variability across seeds is unknown. Before any official test reporting, finish the selected final-training plan, freeze the model/hyperparameters using validation only, then evaluate the untouched official test **once**. The saved evidence does not contain the original `outputs/p4_mlflow.db`; separately preserve that database if the team requires a portable MLflow history.

## Source and audit note

- Original user-supplied archive: `P4_Grid_DepthWidth40_COMPLETE.zip`.
- Archive SHA-256: `0daef393bae3ac1a6e20d38b6c25ed95cedfcc69d34573382097d4e30a9e9f24`.
- Source result files: `search/p4_residual_dw40_trials.csv`, `search/p4_residual_dw40_selection.json`, and each run's `result.json`/`epochs.csv`.
- The source includes a `failed_setups/p4_residual_dw40-trial-01_mlflow_missing` preflight folder, but the completed 4-trial search reports **no failed completed trials**. It is not counted as a fifth trial.
