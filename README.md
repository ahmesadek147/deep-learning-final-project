
# Deep Learning Final Project

## From Pixels to Language

This project is developed using Python and PyTorch.

The system is designed to support:
- Image Classification
- Image Colorization
- Image Captioning

## P1 - Project Setup and Training Pipeline

P1 provides the shared infrastructure for the project:

- Modular project structure
- Training and validation loops
- Early stopping
- Model checkpoints
- MLflow experiment tracking
- YAML configuration files
- Command Line Interface (CLI)

## Requirements

- Python 3.12
- Git
- PyTorch
- MLflow

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd deep-learning-final-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Start MLflow

Run:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Open:

http://127.0.0.1:5000

## Part 4 and Part 5 training

Both experiment paths use the real Hugging Face `Multimodal-Fatima/CUB_train`
training split (5,994 images), with a deterministic train/validation split.
Validation metrics are accuracy, macro precision, macro recall, and macro F1.
No test data is used in these commands.

`configs/experiment.yaml` holds shared data, device/AMP, training, MLflow, and
checkpoint settings under `common`; `p4` and `p5` contain only their model
settings and search spaces. Both parts therefore use the same seeded validation
split. It is the only classification experiment configuration file.

To start the configured searches explicitly (they are intentionally not
automatic):

```bash
# Part 4: complete grid search (8 configurations)
python -m src.cli search --part p4 --config configs/experiment.yaml

# Part 5: reproducible random search (8 trials with the shipped config)
python -m src.cli search --part p5 --config configs/experiment.yaml

# Run P4 followed by P5 with the same common configuration
python -m src.cli search --part both --config configs/experiment.yaml
```

`p5` uses a pretrained ResNet18. When its backbone is
frozen, its replacement classification head remains trainable. Edit `search`
in either experiment section to change the search space or trial count. P4
requires its trial count to equal its complete grid; P5 samples unique discrete
configurations without replacement. The search result
selects the highest final validation macro F1 and reports its validation
accuracy, macro precision, macro recall, and macro F1. Each trial receives its
own checkpoint directory under `outputs/checkpoints` and an MLflow run.
The shipped P4/P5 configs log locally to `outputs/mlflow.db`; view them with:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

For a small direct smoke run (not a search), the CLI supports explicit epoch
and sample limits:

```bash
python -m src.cli train --part p4 --epochs 1 --max-train-samples 16 --max-val-samples 8
python -m src.cli train --part p5 --epochs 1 --max-train-samples 16 --max-val-samples 8
```

Search trials are summarized under `outputs/search_summaries/`. Prediction uses
a selected trial checkpoint, for example:

```bash
python -m src.cli predict --image path/to/bird.jpg --checkpoint outputs/checkpoints/p4_grid/<run-id>/best_model.pth
```

After model selection, evaluate the untouched CUB test split (this does not
retrain or tune either model):

```bash
python -m src.cli evaluate --part both --config configs/experiment.yaml \
  --checkpoint-p4 outputs/checkpoints/p4_grid/grid-18850aea-trial-003/best_model.pth \
  --checkpoint-p5 outputs/checkpoints/p5_random/random-618382b5-trial-005/best_model.pth
```

Metrics and reports are written to `outputs/evaluation/`; P5 Grad-CAM panels
(original, heatmap, and overlay) are written to `outputs/gradcam/`.

## Training Features

The Trainer supports:

1. Training for multiple epochs
2. Validation after each epoch
3. Loss and accuracy tracking
4. Saving the best model checkpoint
5. Early stopping
6. MLflow experiment logging

## Configuration

Training settings are stored in:

`configs/experiment.yaml`.

The configuration includes:

- Epochs
- Batch size
- Learning rate
- Optimizer
- Early stopping
- Checkpoint directory
- MLflow settings and search space

## Project Structure

- configs/: Training configurations
- notebooks/: Experiments
- src/data/: Dataset handling
- src/models/: Model architectures
- src/training/: Training pipeline
- src/evaluation/: Evaluation metrics
- src/visualization/: Visualization tools
- tests/: Testing code
- outputs/: Generated results and checkpoints

## Experiment Tracking

MLflow records:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Training hyperparameters

## Current Status

P4/P5 use the real CUB dataset and must be run explicitly; no completed
training metric is committed to the repository.

## Team Contributions

P1: Project setup, training loop, MLflow, CLI

P2: Dataset, metrics, benchmarking

P3: Captions, augmentation, Grad-CAM

P4: CNN from scratch and Grid Search

P5: Transfer learning and Random Search
