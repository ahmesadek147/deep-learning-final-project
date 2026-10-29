import torch
from torch import nn

from src.dataset import get_dataloaders
from src.models import ScratchCNN
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.training.checkpoints import ModelCheckpoint

def build_model(model_config):
    """Build a classification model from configuration."""

    model_name = model_config["name"]

    if model_name == "scratch_cnn":
        return ScratchCNN(
            num_classes=model_config["num_classes"],
            variant=model_config.get("variant", "basic"),
            dropout=model_config.get("dropout", 0.5),
        )

    if model_name == "simple_cnn":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 5),
        )

    raise ValueError(f"Unsupported model: {model_name}")

def validate_classification_batch(
    images,
    labels,
    num_classes,
    image_size,
):
    """Validate one batch before classification training."""

    if images.ndim != 4:
        raise ValueError(
            "Expected images with shape "
            "(batch_size, channels, height, width)."
        )

    if images.shape[1] != 3:
        raise ValueError(
            f"Expected RGB images with 3 channels, "
            f"but received {images.shape[1]}."
        )

    if images.shape[2:] != (image_size, image_size):
        raise ValueError(
            f"Expected image size {image_size}x{image_size}, "
            f"but received {tuple(images.shape[2:])}."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Expected labels with shape (batch_size,)."
        )

    if labels.dtype != torch.long:
        raise TypeError(
            f"Labels must use torch.long, but received "
            f"{labels.dtype}."
        )

    minimum_label = labels.min().item()
    maximum_label = labels.max().item()

    if minimum_label < 0 or maximum_label >= num_classes:
        raise ValueError(
            f"Labels must be between 0 and {num_classes - 1}, "
            f"but this batch contains values from "
            f"{minimum_label} to {maximum_label}."
        )

def run_training(config):
    print("Starting Training Pipeline...")

    # Read configuration
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]

    data_config = config.get("data", {})

    image_size = data_config.get("image_size", 224)
    val_split = data_config.get("val_split", 0.2)
    num_workers = data_config.get("num_workers", 0)
    seed = data_config.get("seed", 42)

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        val_split=val_split,
        image_size=image_size,
        num_workers=num_workers,
        seed=seed,
    )

    images, labels = next(iter(train_loader))

    validate_classification_batch(
        images=images,
        labels=labels,
        num_classes=config["model"]["num_classes"],
        image_size=image_size,
    )

    print(
        f"Verified training batch: images={tuple(images.shape)}, "
        f"labels={tuple(labels.shape)}, "
        f"label_dtype={labels.dtype}, "
        f"label_range=({labels.min().item()}, "
        f"{labels.max().item()})"
    )

    # Build the configured classification model
    model = build_model(config["model"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    early_stopping = (
        EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"],
        )
        if config["early_stopping"]["enabled"]
        else None
    )

    checkpoint = ModelCheckpoint(
        save_dir=config["checkpoint"]["save_dir"]
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
    )

    print("Training Pipeline Completed!")

    return history