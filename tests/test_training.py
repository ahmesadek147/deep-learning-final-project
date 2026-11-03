"""Trainer, early-stopping and checkpoint tests.

Everything here runs on tiny in-memory tensors and writes only inside pytest's
``tmp_path``. Nothing at module level trains a model or touches ``outputs/``:
module-level work runs during collection and would write MLflow databases and
checkpoints into the repository every time the suite is imported.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.checkpoints import ModelCheckpoint
from src.training.early_stopping import EarlyStopping
from src.training.run_training import summarize_history
from src.training.trainer import Trainer


def _loader(n_samples: int = 40, n_features: int = 4, n_classes: int = 5, batch_size: int = 8) -> DataLoader:
    x = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _model(n_features: int = 4, n_classes: int = 5) -> nn.Module:
    return nn.Linear(n_features, n_classes)


def _trainer(model: nn.Module) -> Trainer:
    return Trainer(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                   criterion=nn.CrossEntropyLoss(), device="cpu")


def test_pytorch_is_working():
    assert torch.tensor([1.0, 2.0, 3.0]).sum().item() == 6.0


def test_early_stopping_trips_after_patience_non_improving_epochs():
    early_stopping = EarlyStopping(patience=3)
    assert early_stopping(0.5) is False
    assert early_stopping(0.6) is False
    assert early_stopping(0.7) is False
    assert early_stopping(0.8) is True


def test_early_stopping_resets_its_counter_on_improvement():
    early_stopping = EarlyStopping(patience=2)
    early_stopping(0.5)
    early_stopping(0.9)
    early_stopping(0.4)  # improvement resets the counter
    assert early_stopping(0.9) is False
    assert early_stopping.counter == 1


def test_training_loop_reports_loss_and_accuracy_in_range():
    loader = _loader()
    trainer = _trainer(_model())

    train_result = trainer.train_one_epoch(loader)
    val_result = trainer.validate(loader)

    assert train_result["loss"] >= 0
    assert val_result["loss"] >= 0
    assert 0 <= train_result["accuracy"] <= 1
    assert 0 <= val_result["accuracy"] <= 1


def test_validation_reports_the_macro_metrics_used_for_model_selection():
    loader = _loader()
    metrics = _trainer(_model()).validate(loader)
    for key in ("macro_precision", "macro_recall", "macro_f1"):
        assert 0 <= metrics[key] <= 1


def test_checkpoint_saves_only_on_improvement(tmp_path):
    model = _model()
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = ModelCheckpoint(save_dir=tmp_path, metadata={"model_config": {"name": "linear"}})

    assert checkpoint.save_if_best(model, optimizer, epoch=1, val_loss=1.0) is True
    assert checkpoint.save_if_best(model, optimizer, epoch=2, val_loss=1.5) is False
    assert checkpoint.save_if_best(model, optimizer, epoch=3, val_loss=0.5) is True

    saved = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 3
    assert saved["val_loss"] == 0.5
    assert saved["metadata"]["model_config"]["name"] == "linear"


def test_macro_f1_checkpoint_keeps_the_highest_scoring_epoch(tmp_path):
    model = _model()
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = ModelCheckpoint(save_dir=tmp_path, monitor="val_macro_f1", mode="max")

    assert checkpoint.save_if_best(model, optimizer, epoch=1, val_loss=2.0, monitor_value=0.20)
    assert not checkpoint.save_if_best(model, optimizer, epoch=2, val_loss=1.5, monitor_value=0.15)
    assert checkpoint.save_if_best(model, optimizer, epoch=3, val_loss=2.2, monitor_value=0.35)

    saved = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 3
    assert saved["monitor"] == "val_macro_f1"
    assert saved["monitor_value"] == 0.35


def test_history_summary_uses_configured_macro_f1_selection_epoch():
    history = {
        "val_loss": [1.0, 0.8, 0.9],
        "val_accuracy": [0.4, 0.5, 0.6],
        "val_macro_precision": [0.3, 0.4, 0.5],
        "val_macro_recall": [0.2, 0.3, 0.4],
        "val_macro_f1": [0.25, 0.45, 0.35],
    }
    summary = summarize_history(history, selection_metric="val_macro_f1")
    assert summary["best_epoch"] == 2
    assert summary["best_val_loss"] == 0.8
    assert summary["best_val_macro_f1"] == 0.45


def test_fit_records_one_history_entry_per_epoch(tmp_path):
    loader = _loader()
    trainer = _trainer(_model())

    history = trainer.fit(
        train_loader=loader,
        val_loader=loader,
        epochs=3,
        early_stopping=EarlyStopping(patience=3),
        checkpoint=ModelCheckpoint(save_dir=tmp_path),
        mlflow_config={"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                       "experiment_name": "unit-test"},
        run_name="unit-test-run",
    )

    assert set(history) == {"train_loss", "train_accuracy", "val_loss", "val_accuracy",
                            "val_macro_precision", "val_macro_recall", "val_macro_f1"}
    assert all(len(values) == 3 for values in history.values())
    assert (tmp_path / "best_model.pth").is_file()
