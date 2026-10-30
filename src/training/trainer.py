"""Training and validation loops for CUB image classifiers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm


class Trainer:
    """Train a classifier and report validation metrics without owning data setup."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module, device: torch.device | str,
                 use_amp: bool = False) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = total = 0
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        if not total:
            raise ValueError("Training loader is empty.")
        return {"loss": total_loss / total, "accuracy": correct / total}

    @torch.inference_mode()
    def validate(self, val_loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = total = 0
        targets: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with self._autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            targets.append(labels.cpu())
            predictions.append(predicted.cpu())
        if not total:
            raise ValueError("Validation loader is empty.")
        precision, recall, f1, _ = precision_recall_fscore_support(
            torch.cat(targets).numpy(), torch.cat(predictions).numpy(), average="macro", zero_division=0
        )
        return {"loss": total_loss / total, "accuracy": correct / total,
                "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1)}

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
            epochs: int, early_stopping: Any = None, checkpoint: Any = None,
            mlflow_config: dict[str, Any] | None = None, run_name: str | None = None,
            params: dict[str, Any] | None = None) -> dict[str, list[float]]:
        """Fit the model and log every epoch in a single MLflow run."""
        import mlflow

        mlflow_config = mlflow_config or {}
        if tracking_uri := mlflow_config.get("tracking_uri"):
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_config.get("experiment_name", "Bird_Classification"))
        history: dict[str, list[float]] = {key: [] for key in (
            "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_precision",
            "val_macro_recall", "val_macro_f1")}
        run_params = {"epochs": epochs, "optimizer": type(self.optimizer).__name__,
                      "learning_rate": self.optimizer.param_groups[0]["lr"],
                      "model": type(self.model).__name__, "device": str(self.device),
                      "mixed_precision": self.use_amp, **(params or {})}
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({key: str(value) for key, value in run_params.items()})
            for epoch in range(1, epochs + 1):
                train_metrics = self.train_one_epoch(train_loader)
                val_metrics = self.validate(val_loader)
                epoch_metrics = {"train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                                 "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
                                 "val_macro_precision": val_metrics["macro_precision"],
                                 "val_macro_recall": val_metrics["macro_recall"], "val_macro_f1": val_metrics["macro_f1"]}
                mlflow.log_metrics(epoch_metrics, step=epoch)
                for key, value in epoch_metrics.items():
                    history[key].append(value)
                if checkpoint is not None:
                    checkpoint.save_if_best(self.model, self.optimizer, epoch, val_metrics["loss"])
                if early_stopping is not None and early_stopping(val_metrics["loss"]):
                    break
            if checkpoint is not None and checkpoint.path.exists():
                mlflow.log_artifact(str(checkpoint.path), artifact_path="checkpoints")
        return history
