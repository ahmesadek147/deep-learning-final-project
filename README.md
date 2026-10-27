
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

## Training CLI

Run the classification pipeline:

```bash
python -m src.cli train --task classification --config configs/cls.yaml
```

The current CLI training pipeline uses synthetic
data and a temporary linear classifier for testing.

Integration with the real CUB-200-2011 dataset
and team models is still required.

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

configs/cls.yaml

The configuration includes:

- Epochs
- Batch size
- Learning rate
- Optimizer
- Early stopping
- Checkpoint directory
- MLflow settings

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

P1 infrastructure has been tested using
synthetic data.

Real dataset integration, model integration,
and full prediction support remain pending.

## Team Contributions

P1: Project setup, training loop, MLflow, CLI

P2: Dataset, metrics, benchmarking

P3: Captions, augmentation, Grad-CAM

P4: CNN from scratch and Grid Search

P5: Transfer learning and Random Search