#!/usr/bin/env bash
# Milestone 1 completion runs (Linux/macOS/WSL). PowerShell equivalent:
# scripts/run_milestone1_completion.ps1
#
#   ./scripts/run_milestone1_completion.sh            # everything, in order
#   ./scripts/run_milestone1_completion.sh p5         # just the transfer study
#   SMOKE=1 ./scripts/run_milestone1_completion.sh    # 1 epoch each, to check it runs
#
# Every stage is resumable and nothing overwrites an existing result.
set -euo pipefail

STAGE="${1:-all}"
PYTHON="${PYTHON:-python}"
SMOKE_ARGS=()
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_ARGS+=(--smoke)

step() { printf '\n=== %s ===\n' "$1"; }

if [[ ! -f outputs/splits/cub_train_val_v2_seed42.json ]]; then
    step "Building the duplicate-safe split manifest"
    "$PYTHON" -m src.data.build_split_manifest
fi

# Stage 1 - P5: two pretrained architectures, frozen vs fine-tuned at sane LRs.
if [[ "$STAGE" == "all" || "$STAGE" == "p5" ]]; then
    step "P5 transfer study (ResNet18 and EfficientNet-B0, frozen vs fine-tuned)"
    "$PYTHON" -m src.training.transfer_study --config configs/p5_transfer_study.yaml "${SMOKE_ARGS[@]}"
fi

# Stage 2 - P4: the scratch architecture comparison that was coded but never run.
if [[ "$STAGE" == "all" || "$STAGE" == "p4" ]]; then
    step "P4 architecture comparison (SimpleCNN vs delayed-pooling vs residual)"
    if [[ "${SMOKE:-0}" == "1" ]]; then
        "$PYTHON" -m src.training.p4_experiments train \
            --config configs/p4_architecture.yaml --run-id p4_smoke \
            --epochs 1 --max-train-samples 32 --max-val-samples 32
    else
        "$PYTHON" -m src.training.p4_experiments architectures \
            --config configs/p4_architecture.yaml --resume
        step "P4 capacity grid search (optional, 8 x 10 epochs)"
        "$PYTHON" -m src.training.p4_experiments grid \
            --config configs/p4_grid.yaml \
            --winner-from outputs/p4_scoped/search/p4_architectures_selection.json --resume
    fi
fi

# Stage 3 - evaluation only, no training.
if [[ ( "$STAGE" == "all" || "$STAGE" == "eval" ) && "${SMOKE:-0}" != "1" ]]; then
    step "Evaluation geometry ablation on the existing corrected P5 checkpoint"
    "$PYTHON" -m src.cli evaluate --part p5 --config configs/experiment.yaml \
        --checkpoint-p5 outputs/checkpoints/p5_corrected/resnet18-2e6872b0/best_model.pth \
        --eval-resize center_crop \
        --output-dir outputs/evaluation/p5_corrected_center_crop \
        --gradcam-dir outputs/gradcam_center_crop

    step "Final test evaluation of the transfer-study winner"
    cat <<'EOF'
Pick the winning checkpoint from outputs/transfer_study/study_results.csv, then run
(replacing <LABEL> with its label):

  python -m src.cli evaluate --part p5 --config configs/experiment.yaml \
    --checkpoint-p5 outputs/checkpoints/p5_transfer_study/<LABEL>/best_model.pth \
    --output-dir outputs/evaluation/p5_study_winner \
    --gradcam-dir outputs/gradcam_study_winner
EOF
fi

step "Refreshing derived report artifacts"
"$PYTHON" -m src.training.search_comparison
"$PYTHON" -m src.training.training_curves

printf '\nDone. Next: run "python -m src.reporting.update_report" to fold the new\n'
printf 'numbers into outputs/final_submission/milestone_1_report.md.\n'
