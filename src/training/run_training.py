
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.training.checkpoints import ModelCheckpoint


def run_training(config):
    print("Starting Training Pipeline...")

    # Read configuration
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]

    # Temporary dummy data for testing
    images = torch.randn(40, 3, 32, 32)
    labels = torch.randint(0, 5, (40,))

    dataset = TensorDataset(images, labels)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Temporary test model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 5)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )

    early_stopping = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"]
    ) if config["early_stopping"]["enabled"] else None

    checkpoint = ModelCheckpoint(
        save_dir=config["checkpoint"]["save_dir"]
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping=early_stopping,
        checkpoint=checkpoint
    )

    print("Training Pipeline Completed!")

    return history