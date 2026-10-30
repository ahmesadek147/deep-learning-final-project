# Final Submission Audit

Audit-only review of the CUB-200-2011 project on branch `ahmed-p4-p5`.
No models were retrained, searches rerun, checkpoints changed, source data
modified, or Git history changed.

## Executive summary

Milestone 1 classification work is substantially implemented and has verified
artifacts for EDA, data auditing, augmentation, P4/P5 searches, final test
evaluation, error analysis, Grad-CAM, MLflow, and training curves. The most
important data-quality finding is an exact duplicate image pair split across
derived training and validation subsets. This is a possible validation leakage
and should be corrected in a future training run; existing results are not
retroactively changed here.

An independent benchmark is not present. Colorization and image-captioning
models are not claimed complete because they belong to Milestone 2. The
repository also contains substantial uncommitted work, so the current working
tree—not the last commit—is required to reproduce the reported workflow.

## Milestone 1 compliance checklist

| Requirement | Evidence | Status | Notes |
|---|---|---|---|
| CUB dataset loading and deterministic validation split | `src/dataset.py`, `configs/experiment.yaml` | PASS | Real HF data; seed 42; 4,796 train / 1,198 validation derived from 5,994 images |
| Dataset EDA | `outputs/eda/` and `src/data/run_dataset_audit.py` | PASS | Summary, class distribution, dimensions, samples, caption statistics, and charts persisted |
| Data analysis and cleaning audit | `outputs/data_audit/` | PASS | Broken-image, duplicate, channel, and derived caption-cleaning artifacts persisted; source data was not modified |
| Augmentation documentation | `outputs/eda/augmentation_audit.md`, `outputs/eda/augmentation_examples.png` | PASS | Training-only random transforms and deterministic validation/test preprocessing verified |
| Scratch CNN | `src/models/simple_cnn.py`, P4 checkpoints | PASS | 200-class `SimpleCNN`; metadata-bearing best checkpoint exists |
| P4 Grid Search | `src/training/search.py`, `outputs/search_summaries/p4_18850aea_trials.csv` | PASS | 8/8 completed; cardinality 8; exhaustive grid validation is implemented |
| Transfer learning | `src/models/transfer_learning.py`, P5 checkpoint | PASS | ResNet18 builder; frozen-backbone head remains trainable |
| P5 Random Search | `src/training/search.py`, `outputs/search_summaries/p5_618382b5_trials.csv` | PASS | 8/8 completed; cardinality 36; seeded unique sampling |
| MLflow tracking | `outputs/mlflow.db` | PASS | Training/search parameters and epoch metrics exist; final-test runs are separate |
| Final untouched-test evaluation | `outputs/evaluation/p4/`, `outputs/evaluation/p5/`, `comparison.json` | PASS | Test loss, accuracy, macro precision/recall/F1, reports, and confusion matrices exist |
| Error analysis | `outputs/evaluation/error_analysis.md`, F1/confusion figures | PASS | P5 per-class and confusion analysis persisted |
| Grad-CAM | `src/visualization/gradcam.py`, `outputs/gradcam_improved/` | PASS | Final ResNet convolution target and representative correct/incorrect panels exist; interpretation remains qualitative |
| Training/validation curves | `outputs/final_submission/training_curves_analysis.md`, `figures/training_curves/` | PASS | Curves use actual MLflow epoch histories |
| CLI workflow | `src/cli.py`, `README.md` | PASS | Train, search, predict, and evaluate commands are implemented |
| Independent benchmark comparison | No benchmark implementation or measured baseline found | MISSING | `outputs/evaluation/comparison.csv` compares P4/P5 only, not an independent baseline |
| Formal final report | Existing README/audits only | PARTIAL | This audit is not a full narrative report with all rubric sections and figures |
| Colorization model | No verified Milestone 2 implementation | OUT OF SCOPE | Do not claim complete |
| Image-captioning model | Caption EDA/cleaning exists, but no verified caption model/training result | OUT OF SCOPE | Do not claim complete |

## Duplicate-data leakage investigation

The known duplicate pair was checked against the exact split logic in
`src/dataset.py` (`torch.randperm(5994, generator=torch.Generator().manual_seed(42))`,
with the first 4,796 indices assigned to training and the remaining 1,198 to
validation):

| Dataset index | Label | File name | Derived subset |
|---:|---:|---|---|
| 1712 | 57 | `Pigeon_Guillemot_0018_40195.jpg` | train |
| 1731 | 57 | `Pigeon_Guillemot_0081_40339.jpg` | validation |

The two records have the same normalized pixel SHA-256 in
`outputs/data_audit/duplicate_images.json`. They therefore cross the train/
validation boundary and represent possible leakage, even though their file
names differ. Existing metrics are retained unchanged. For future training,
deduplicate before splitting or use grouped splitting so duplicate image
content cannot cross the model-selection boundary, then rerun searches and
evaluation.

## Checkpoint and metrics consistency

Verified files:

- P4: `outputs/checkpoints/p4_grid/grid-18850aea-trial-003/best_model.pth`
- P5: `outputs/checkpoints/p5_random/random-618382b5-trial-005/best_model.pth`

The search CSVs select exactly those trials as the highest validation macro-F1
configurations. The evaluation JSON files point to those same checkpoint paths
and their embedded metadata reconstructs `simple_cnn` and `resnet18`, both with
200 outputs. The saved test metrics are:

- P4: test loss 4.589488, accuracy 0.059372, macro F1 0.045710.
- P5: test loss 2.181421, accuracy 0.453918, macro F1 0.442733.

The training-curve JSON was extracted from MLflow histories for run IDs
`d8bebc0669234e97ae94a3b4ed581a97` (P4) and
`a54f1b97a45e402fa7570045567dfcb9` (P5), with 15 and 10 logged epochs,
respectively. No missing curve values were interpolated. Required audited
outputs were present when checked.

## Reproducibility assessment

### Strengths

- Shared configuration is `configs/experiment.yaml`; only that YAML exists in
  `configs/`.
- README includes search, smoke-training, prediction, evaluation, and MLflow
  commands.
- Seeds, validation split, image size, batch sizes, AMP, checkpoint namespaces,
  and MLflow URI are configured.
- Search summaries and per-trial checkpoint paths are persisted.
- Tests cover models, device selection, config validation, searches, prediction,
  and Grad-CAM.

### Risks and limitations

- `requirements.txt` is unpinned and README says Python 3.12, while the verified
  environment used Python 3.14 at `C:\Python314\python.exe`; exact package
  versions and CUDA wheel selection are not captured in a lock file.
- Hugging Face dataset access depends on network/cache state and the README does
  not document the dataset identifiers as a standalone data setup section.
- The README's historical “no completed training metric is committed” wording
  is stale relative to the generated local artifacts.
- The working tree has many uncommitted changes and `configs/cls.yaml` is
  deleted in the working tree. The last commit (`bfee57d`, merge commit) does
  not contain the current completed P4/P5 workflow and output artifacts.

## Submission structure

The current structure is clear and should be preserved:

- `src/`: dataset, augmentation, models, training/search, evaluation, metrics,
  prediction, and visualization code.
- `configs/experiment.yaml`: shared and P4/P5 configuration.
- `outputs/checkpoints/`: per-trial metadata-bearing checkpoints.
- `outputs/search_summaries/`: persistent P4/P5 trial CSVs.
- `outputs/evaluation/`: test metrics, reports, confusion matrices, comparison,
  and error analysis.
- `outputs/eda/` and `outputs/data_audit/`: EDA, cleaning, and augmentation
  evidence.
- `outputs/final_submission/`: search comparison, curves, figures, and this
  audit.
- `outputs/mlflow.db`: local experiment tracking.
- `README.md`, `requirements.txt`, and `tests/`: usage, dependencies, and tests.

## Reproduction commands

Use the verified interpreter on this machine:

```powershell
C:\Python314\python.exe -m src.data.run_dataset_audit
C:\Python314\python.exe -m src.data.run_augmentation_audit
C:\Python314\python.exe -m src.training.search_comparison
C:\Python314\python.exe -m src.training.training_curves
C:\Python314\python.exe -m src.cli train --part p4 --epochs 1 --max-train-samples 16 --max-val-samples 8
C:\Python314\python.exe -m src.cli train --part p5 --epochs 1 --max-train-samples 16 --max-val-samples 8
C:\Python314\python.exe -m src.cli search --part p4 --config configs/experiment.yaml
C:\Python314\python.exe -m src.cli search --part p5 --config configs/experiment.yaml
C:\Python314\python.exe -m src.cli evaluate --part both --config configs/experiment.yaml `
  --checkpoint-p4 outputs/checkpoints/p4_grid/grid-18850aea-trial-003/best_model.pth `
  --checkpoint-p5 outputs/checkpoints/p5_random/random-618382b5-trial-005/best_model.pth
C:\Python314\python.exe -m pytest -q
```

The smoke commands are safe validation commands; the search commands are long
and should not be rerun for submission unless intentionally reproducing results.
The final evaluation command must be run only after the selected checkpoints
exist. Existing outputs should be preserved rather than overwritten.

## Prioritized action plan

### Critical before submission

1. Resolve the duplicate train/validation leakage in a future controlled rerun,
   or explicitly disclose it as a limitation if rerunning is not possible.
2. Add and measure the independent benchmark required by the official rubric,
   if it is mandatory for Milestone 1.
3. Commit the intended source/config/test changes and preserve the generated
   evidence in the submission package; do not commit large checkpoints unless
   required by the course.
4. Assemble a formal report that cites the persisted metrics, curves, EDA,
   augmentation, error-analysis, and Grad-CAM figures.

### Important improvements

1. Pin Python/PyTorch/torchvision/MLflow and other dependency versions, and
   document the CUDA-capable environment separately from CPU fallback.
2. Refresh README status text and add explicit Hugging Face dataset setup/cache
   instructions.
3. State clearly that P4/P5 scores compare different architectures and cannot
   isolate search strategy.

### Optional improvements

1. Add a machine-readable manifest of artifact paths, run IDs, and checksums.
2. Export MLflow metadata or screenshots for environments where `outputs/mlflow.db`
   is not submitted.
3. Add a grouped duplicate check to the data-loading validation path for future
   experiments.
