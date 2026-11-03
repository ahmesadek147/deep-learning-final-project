# Milestone 1 completion runs (Windows PowerShell).
#
# Closes the three gaps the Milestone 1 review found. Run from the repository
# root with the virtual environment active. Each stage is independent and
# resumable, so you can stop after any one of them.
#
#   .\scripts\run_milestone1_completion.ps1            # everything, in order
#   .\scripts\run_milestone1_completion.ps1 -Stage p5  # just the transfer study
#   .\scripts\run_milestone1_completion.ps1 -Smoke     # 1 epoch each, to check it runs
#
# Nothing here overwrites an existing result: the transfer study resumes from
# study_results.json, the P4 studies resume from their state files, and the
# evaluation stages write to new directories.

param(
    [ValidateSet("all", "p5", "p4", "eval")]
    [string]$Stage = "all",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$python = "python"
$smokeArgs = if ($Smoke) { @("--smoke") } else { @() }

function Step($message) {
    Write-Host ""
    Write-Host "=== $message ===" -ForegroundColor Cyan
}

# Every run below uses the duplicate-safe split manifest. Build it if missing.
if (-not (Test-Path "outputs/splits/cub_train_val_v2_seed42.json")) {
    Step "Building the duplicate-safe split manifest"
    & $python -m src.data.build_split_manifest
}

# ---------------------------------------------------------------------------
# Stage 1 - P5: two pretrained architectures, frozen vs fine-tuned at sane LRs
# Approximately 70-90 minutes on an RTX 3050 for the full 5 runs.
# ---------------------------------------------------------------------------
if ($Stage -eq "all" -or $Stage -eq "p5") {
    Step "P5 transfer study (5 runs: ResNet18 and EfficientNet-B0, frozen vs fine-tuned)"
    & $python -m src.training.transfer_study --config configs/p5_transfer_study.yaml @smokeArgs
}

# ---------------------------------------------------------------------------
# Stage 2 - P4: the scratch architecture comparison that was coded but never run
# Approximately 3 x 10 epochs. Add the capacity grid afterwards if time allows.
# ---------------------------------------------------------------------------
if ($Stage -eq "all" -or $Stage -eq "p4") {
    Step "P4 architecture comparison (SimpleCNN vs delayed-pooling vs residual)"
    if ($Smoke) {
        & $python -m src.training.p4_experiments train `
            --config configs/p4_architecture.yaml --run-id p4_smoke `
            --epochs 1 --max-train-samples 32 --max-val-samples 32
    } else {
        & $python -m src.training.p4_experiments architectures `
            --config configs/p4_architecture.yaml --resume

        Step "P4 capacity grid search (optional, 8 x 10 epochs)"
        & $python -m src.training.p4_experiments grid `
            --config configs/p4_grid.yaml `
            --winner-from outputs/p4_scoped/search/p4_architectures_selection.json --resume
    }
}

# ---------------------------------------------------------------------------
# Stage 3 - evaluation only, no training. Minutes, not hours.
# ---------------------------------------------------------------------------
if (($Stage -eq "all" -or $Stage -eq "eval") -and -not $Smoke) {
    Step "Evaluation geometry ablation on the existing corrected P5 checkpoint"
    # Same weights, same test split, only the resize geometry differs. Quantifies
    # the train/eval mismatch caused by squashing instead of center-cropping.
    & $python -m src.cli evaluate --part p5 --config configs/experiment.yaml `
        --checkpoint-p5 outputs/checkpoints/p5_corrected/resnet18-2e6872b0/best_model.pth `
        --eval-resize center_crop `
        --output-dir outputs/evaluation/p5_corrected_center_crop `
        --gradcam-dir outputs/gradcam_center_crop

    Step "Final test evaluation of the transfer-study winner"
    Write-Host "Pick the winning checkpoint from outputs/transfer_study/study_results.csv," -ForegroundColor Yellow
    Write-Host "then run (replacing <LABEL> with its label):" -ForegroundColor Yellow
    Write-Host "  $python -m src.cli evaluate --part p5 --config configs/experiment.yaml ``" -ForegroundColor Yellow
    Write-Host "    --checkpoint-p5 outputs/checkpoints/p5_transfer_study/<LABEL>/best_model.pth ``" -ForegroundColor Yellow
    Write-Host "    --output-dir outputs/evaluation/p5_study_winner ``" -ForegroundColor Yellow
    Write-Host "    --gradcam-dir outputs/gradcam_study_winner" -ForegroundColor Yellow
}

Step "Refreshing derived report artifacts"
& $python -m src.training.search_comparison
& $python -m src.training.training_curves

Write-Host ""
Write-Host "Done. Next: run 'python -m src.reporting.update_report' to fold the new" -ForegroundColor Green
Write-Host "numbers into outputs/final_submission/milestone_1_report.md." -ForegroundColor Green
