# P4-only continuation: epoch 80 to a maximum of 120

Adds only src/training/p4_continue.py and tests/test_p4_continuation.py.
Does not replace any existing P4 or teammate file. Requires the reviewed original
P4_Only_Colab_Overlay.zip already installed in the team repository.

Run from the repository root, with the existing Colab environment and GPU.

Prepare into a new folder without training:

```bash
python -m src.training.p4_continue --source-run /content/drive/MyDrive/CUB_P4_Experiments/runs/p4_residual_extended --output-root /content/drive/MyDrive/CUB_P4_Experiments --run-id p4_residual_continued_120 --epochs 120 --prepare-only
```

Then start (or resume after interruption):

```bash
python -m src.training.p4_continue --source-run /content/drive/MyDrive/CUB_P4_Experiments/runs/p4_residual_extended --output-root /content/drive/MyDrive/CUB_P4_Experiments --run-id p4_residual_continued_120 --epochs 120 --resume
```

The helper restores the complete epoch-80 last.pth state and preserves the epoch-79
best checkpoint as the initial best for the new lineage. It adjusts only the new
copy's configuration hash to accommodate the extended epoch budget. Original files
are never written. The existing P4 run_one and P1 epoch methods do the training.
Optimizer, scheduler, AMP scaler, RNG, best-key and patience counter are retained.
Existing early stopping remains enabled. Results include all 80 historical epochs
plus any additional epochs; accumulated runtime likewise includes the original run.
A continuation is not an independent trial or independent replication.

The helper records hashes of the original checkpoints/configuration, the currently
installed code, and the current split manifest. The old runner did not save original
source/manifest hashes: historical identity cannot be independently established by
this helper. It checks the two P4 source files against the uploaded reviewed overlay
and binds subsequent resumes to current source/manifest contents. P2's existing
manifest validation still runs. No official-test access is added.

Seven CPU fixture checks passed for preparation, state preservation, source-byte
preservation, overwrite prevention, missing state, early-stop handling, config
mismatch, path overlap and changed-manifest rejection. Full Colab/CUDA continuation
has not been executed here. No improved accuracy is claimed.

If preparation was interrupted, staging directories ending in .preparing-* can be
inspected; the source run is unaffected. A successfully prepared destination is
resumed explicitly. Do not rerun fresh preparation into an existing destination.

Final output location:
/content/drive/MyDrive/CUB_P4_Experiments/runs/p4_residual_continued_120
