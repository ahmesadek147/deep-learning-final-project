import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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

def run_training(config):
    print("Starting Training Pipeline...")

    # Read configuration
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]

    # Temporary dummy data for integration testing
    images = torch.randn(40, 3, 32, 32)
    labels = torch.randint(0, 5, (40,))

    dataset = TensorDataset(images, labels)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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