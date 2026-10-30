# Duplicate-Leakage Correction and P5 Validation

## Scope and preservation

This implementation preserves the historical P4/P5 checkpoints, metrics, MLflow runs, and source dataset. The legacy random split remains available as `legacy_random_v1`; new experiments can opt into the versioned duplicate-safe manifest `v2`.

## Corrected split

Manifest: `outputs/splits/cub_train_val_v2_seed42.json`

Summary: `outputs/splits/cub_train_val_v2_seed42_summary.json`

- Strategy: deterministic stratified group split (`StratifiedGroupKFold`, seed 42)
- Dataset: `Multimodal-Fatima/CUB_train`
- Samples: 5,994 total; 4,794 train; 1,200 validation
- Validation fraction: 0.2002002 (target 0.20)
- Classes: all 200 classes present in both splits
- Duplicate groups: one two-image group; cross-split duplicate groups: 0
- Manifest SHA-256: `6b003471157d8245087172e0792843b2413d0e8e5c5529f2ce534d82bb1925b2`
- Confirmed pair (indices 1712 and 1731, label 57): both assigned to train

## Corrected P5 training

Run ID: `8867f40406854e18b920fd181a8cdecf`

Checkpoint: `outputs/checkpoints/p5_corrected/resnet18-2e6872b0/best_model.pth`

Measured MLflow run duration: 612,944 ms (10 min 12.944 s), from `1793321384869` to `1793321997813`.

Effective parameters included batch size 8, ResNet18, pretrained=true, freeze_backbone=true, dropout=0.2, learning rate 0.001, weight decay 0.0001, AMP enabled, epoch limit 10, and the corrected split manifest hash above.

The best checkpoint is epoch 8 (lowest validation loss 2.2120219422). At epoch 10, validation accuracy was 0.4425 and macro F1 was 0.4095198830; the epoch-8 values were accuracy 0.4416666667 and macro F1 0.4049636307. These are validation metrics only and are not test results.

## Corrected test evaluation

The environment was restored from the Python 3.14 user site without changing the checkpoint. CUDA was available on the NVIDIA RTX 3050. Evaluation used the untouched `Multimodal-Fatima/CUB_test:test` split and wrote only to `outputs/evaluation/p5_corrected/`.

- Test samples: 5,794
- Test loss: 2.1891411581
- Accuracy: 0.4496030376
- Macro precision: 0.5287817196
- Macro recall: 0.4525818007
- Macro F1: 0.4415568058
- MLflow final-test run: `de57392f5f5e45a7861f39760775986e`

Outputs include `metrics.json`, classification report JSON/CSV, and confusion matrix CSV/NPY. Eight Grad-CAM panels were written to `outputs/gradcam_corrected/`. Historical test outputs were not reused or overwritten.

## Files changed for this phase

- `src/data/splits.py` and `src/data/build_split_manifest.py`: deterministic duplicate-safe manifest generation and validation.
- `src/dataset.py`: explicit legacy versus manifest-backed split selection.
- `src/training/run_training.py`: split provenance in MLflow parameters and checkpoint metadata.
- `src/training/search.py`: split provenance fields in future trial summaries.
- `src/cli.py`: split and isolated checkpoint-namespace options; isolated Grad-CAM evaluation output option.
- `src/evaluation.py`: safe single-part output layout, configurable Grad-CAM directory, and final-evaluation run ID persistence.
- `tests/test_split_manifest.py`: manifest consistency and strategy distinction checks.

No historical output or checkpoint was overwritten.
