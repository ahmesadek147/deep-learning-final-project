"""Data, model and trainer composition for CUB experiments."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import torch
import json

from src.dataset import get_dataloaders
from src.models.simple_cnn import SimpleCNN
from src.models.transfer_learning import build_transfer_model
from src.training.checkpoints import ModelCheckpoint
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer


def select_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA while retaining a CPU-only execution path."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cuda', or 'cpu'.")
    return torch.device(requested)


def build_model(model_config: dict) -> torch.nn.Module:
    name = model_config.get("name", "simple_cnn")
    if name == "simple_cnn":
        return SimpleCNN(num_classes=model_config.get("num_classes", 200),
                         dropout=model_config.get("dropout", 0.3))
    if name in {"resnet18", "efficientnet_b0"}:
        return build_transfer_model(model_name=name, num_classes=model_config.get("num_classes", 200),
                                    pretrained=model_config.get("pretrained", True),
                                    freeze_backbone=model_config.get("freeze_backbone", True),
                                    dropout=model_config.get("dropout", 0.2))
    raise ValueError(f"Unsupported model name: {name}")


def build_optimizer(model: torch.nn.Module, training_config: dict) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    common = {"lr": training_config["learning_rate"], "weight_decay": training_config.get("weight_decay", 0.0)}
    name = training_config.get("optimizer", "adam").lower()
    if name == "adam":
        return torch.optim.Adam(parameters, **common)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **common)
    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=training_config.get("momentum", 0.9), **common)
    raise ValueError(f"Unsupported optimizer: {name}")


def effective_mlflow_params(config: dict, amp_enabled: bool) -> dict:
    """Return the resolved run settings required to reproduce a trial."""
    training = config["training"]
    model = config["model"]
    data = config.get("data", {})
    split_manifest_hash = None
    split_strategy = data.get("split_strategy", "legacy_random")
    split_version = data.get("split_version", "legacy_random_v1")
    if data.get("split_manifest"):
        split_doc = json.loads(Path(data["split_manifest"]).read_text(encoding="utf-8"))
        split_manifest_hash = split_doc.get("manifest_sha256")
    params = {
        "model_name": model["name"],
        "pretrained": model.get("pretrained", False),
        "freeze_backbone": model.get("freeze_backbone", False),
        "dropout": model.get("dropout", 0.0),
        "learning_rate": training["learning_rate"],
        "weight_decay": training.get("weight_decay", 0.0),
        "optimizer": training.get("optimizer", "adam"),
        "loss": training.get("loss", "cross_entropy"),
        "batch_size": training["batch_size"],
        "epoch_limit": training["epochs"],
        "mixed_precision_requested": training.get("mixed_precision", False),
        "mixed_precision_enabled": amp_enabled,
        "seed": data.get("seed", 42),
        "validation_split": data.get("val_split", 0.2),
        "image_size": data.get("image_size", 224),
        "num_classes": model.get("num_classes", 200),
        "device": config.get("device", "auto"),
    }
    params.update({"split_strategy": split_strategy, "split_version": split_version})
    if split_manifest_hash:
        params["split_manifest_sha256"] = split_manifest_hash
        params["train_size"] = split_doc.get("train_count")
        params["validation_size"] = split_doc.get("validation_count")
    return params


def summarize_history(history: dict[str, list[float]]) -> dict[str, float | int]:
    """Select the epoch that produced the checkpoint (lowest validation loss)."""
    best_index = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
    return {
        "best_epoch": best_index + 1,
        "best_val_loss": history["val_loss"][best_index],
        "best_val_accuracy": history["val_accuracy"][best_index],
        "best_val_macro_precision": history["val_macro_precision"][best_index],
        "best_val_macro_recall": history["val_macro_recall"][best_index],
        "best_val_macro_f1": history["val_macro_f1"][best_index],
    }


def run_training(config: dict, run_name: str | None = None) -> dict:
    """Run one real CUB train/validation experiment and return its artifacts."""
    config = deepcopy(config)
    training_config, model_config = config["training"], config["model"]
    data_config = config.get("data", {})
    seed = data_config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_device(config.get("device", "auto"))
    train_loader, val_loader = get_dataloaders(
        dataset_name=data_config.get("dataset_name", "Multimodal-Fatima/CUB_train"),
        batch_size=training_config["batch_size"], val_split=data_config.get("val_split", 0.2), seed=seed,
        image_size=data_config.get("image_size", 224), num_workers=data_config.get("num_workers", 0),
        pin_memory=device.type == "cuda", augmentation_config=config.get("augmentation"),
        max_train_samples=data_config.get("max_train_samples"), max_val_samples=data_config.get("max_val_samples"),
        split_strategy=data_config.get("split_strategy", "legacy_random"),
        split_version=data_config.get("split_version", "legacy_random_v1"),
        split_manifest=data_config.get("split_manifest"),
    )
    model = build_model(model_config)
    optimizer = build_optimizer(model, training_config)
    early_config = config.get("early_stopping", {})
    early_stopping = EarlyStopping(patience=early_config.get("patience", 5),
                                   min_delta=early_config.get("min_delta", 0.0)) if early_config.get("enabled", True) else None
    experiment_id = run_name or f"{model_config.get('name', 'model')}-{uuid4().hex[:8]}"
    checkpoint_config = config.get("checkpoint", {})
    checkpoint_root = Path(checkpoint_config.get("save_dir", "outputs/checkpoints"))
    if namespace := checkpoint_config.get("namespace"):
        checkpoint_root /= namespace
    split_metadata = {key: data_config.get(key) for key in ("split_strategy", "split_version", "split_manifest")}
    if data_config.get("split_manifest"):
        split_metadata["split_manifest_sha256"] = json.loads(Path(data_config["split_manifest"]).read_text(encoding="utf-8")).get("manifest_sha256")
    checkpoint = ModelCheckpoint(
        save_dir=checkpoint_root / experiment_id,
        metadata={"model_config": model_config, "image_size": data_config.get("image_size", 224), "split": split_metadata},
    )
    trainer = Trainer(model, optimizer, torch.nn.CrossEntropyLoss(), device,
                      use_amp=training_config.get("mixed_precision", False))
    history = trainer.fit(
        train_loader, val_loader, epochs=training_config["epochs"], early_stopping=early_stopping,
        checkpoint=checkpoint, mlflow_config=config.get("mlflow"), run_name=experiment_id,
        params=effective_mlflow_params(config, trainer.use_amp),
    )
    return {"history": history, "metrics": summarize_history(history), "checkpoint": str(checkpoint.path),
            "device": str(device), "run_name": experiment_id}
