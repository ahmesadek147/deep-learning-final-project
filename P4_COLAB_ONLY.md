# P4-only Colab guide — CUB-200 scratch CNNs

**Scope guarantee:** The overlay **adds** P4-owned files and does not modify
the existing `simple_cnn.py`, P1 shared trainer/CLI, P2 data/metrics/splits,
P3 augmentation/Grad-CAM, or P5 transfer code. `SimpleCNN` remains baseline.
The overlay is based on team-main commit `ddc2587`. Do not combine with the
earlier broad integrated ZIP.

1. In Colab, select **Runtime > Change runtime type > T4 GPU**.
2. Run Cell 1, upload the P4-only overlay in Cell 2, then test in Cell 3.
3. Request the P2 split manifest before training. P4 must use P2's split;
   this overlay doesn't create or change one.

## Cell 1 — fresh team clone

```python
%cd /content
!git clone https://github.com/ahmesadek147/deep-learning-final-project.git cub-team
%cd /content/cub-team
!git checkout ddc2587
!python -m pip install -q -r requirements.txt
!nvidia-smi
```

Start from a fresh runtime; if `cub-team` already exists, don't clone on top.

## Cell 2 — install only new P4 files

```python
from google.colab import files
from pathlib import Path
from zipfile import ZipFile

uploaded = files.upload()  # select P4_Only_Colab_Overlay.zip
overlay = Path("P4_Only_Colab_Overlay.zip")
assert overlay.name in uploaded
repo = Path.cwd().resolve()
with ZipFile(overlay) as z:
    for member in z.namelist():
        target = (repo / member).resolve()
        if not target.is_relative_to(repo) or target.exists():
            raise RuntimeError(f"Unsafe or existing path: {member}")
    z.extractall(repo)
print("P4-only new files installed")
```

## Cell 3 — offline P4 tests

```python
!python -m pytest -q tests/test_p4_scoped.py
!python -m src.training.p4_experiments --help
```

## Cell 4 — P2's shared split manifest

P2 must provide `outputs/splits/cub_train_val_v2_seed42.json` from the team's
existing `src.data.build_split_manifest` script, placed at that relative path.
It is generated using only `Multimodal-Fatima/CUB_train`. If P2 has not
created it, coordinate with P2 instead of silently switching splitting methods.

```python
from pathlib import Path
manifest = Path("outputs/splits/cub_train_val_v2_seed42.json")
assert manifest.is_file(), "Get the P2 split manifest first"
print("Found", manifest)
```

## Cell 5 — safe one-epoch smoke test

```python
!python -m src.training.p4_experiments train \
  --config configs/p4_architecture.yaml --run-id p4_smoke \
  --epochs 1 --max-train-samples 32 --max-val-samples 32
```

Do not cite smoke-test accuracy in the report: 32 samples cannot cover 200
classes. If this fails, stop and send the error before long training.

## Cell 6 — P4 architecture comparison (3 × 10 epochs)

```python
!python -m src.training.p4_experiments architectures \
  --config configs/p4_architecture.yaml
```

This fairly compares baseline SimpleCNN, delayed-pooling CNN and residual CNN
on the same data, seed, augmentation, LR, weight decay and epoch budget.

## Cell 7 — P4 Grid Search (8 × 10 epochs)

```python
!python -m src.training.p4_experiments grid \
  --config configs/p4_grid.yaml \
  --winner-from outputs/p4_scoped/search/p4_architectures_selection.json
```

Searches *depth* [4, 5], *width* [24, 32], *learning rate* [.001, .0003].
If baseline SimpleCNN wins, the capacity Grid Search intentionally stops;
discuss whether to explore the best advanced P4 variant with your leader.

## Cell 8 — final scratch training (up to 80 epochs)

```python
!python -m src.training.p4_experiments final \
  --config configs/p4_final.yaml \
  --winner-from outputs/p4_scoped/search/p4_grid_selection.json \
  --run-id p4_final
```

For interrupted architecture/grid studies, rerun their **same** command with
`--resume`. Final training uses `ReduceLROnPlateau` and patience-20 early
stopping. Selection is by best validation macro-F1, then validation accuracy.
No official test data is touched for selection.

## Evidence to save

Download `outputs/p4_scoped/` to Drive after each stage: best/last checkpoints,
`epochs.csv`, training curves, `result.json`, and the Grid Search summary CSVs
and selection JSON. MLflow is configured as `P4_Scratch` in the local
`outputs/p4_mlflow.db`. Also retain the P2 split manifest and its SHA-256.
Write your P4 report with `P4_REPORT_TEMPLATE.md`, and **share results** with
P1/P2/P3/P5 without editing any of their code.

Time note: 3×10 + 8×10 + 80 = **190 total training epochs**, potentially many
hours on a T4. Pilot ranking after 10 epochs may differ from ranking after 80;
report that limitation. No real accuracy numbers exist until runs finish.
