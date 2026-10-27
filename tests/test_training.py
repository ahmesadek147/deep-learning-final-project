
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.training.checkpoints import ModelCheckpoint


# Create dummy image data
images = torch.randn(40, 3, 32, 32)
labels = torch.randint(0, 5, (40,))

dataset = TensorDataset(images, labels)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False
)


# Simple CNN model for testing
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 5)
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

early_stopping = EarlyStopping(patience=3)
checkpoint = ModelCheckpoint()

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=5,
    early_stopping=early_stopping,
    checkpoint=checkpoint
)

print("Training test completed successfully!")

def test_pytorch_is_working():
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])
    result = x.sum().item()

    assert result == 6.0


def test_early_stopping():
    from src.training.early_stopping import EarlyStopping

    early_stopping = EarlyStopping(patience=3)

    early_stopping(0.5)
    early_stopping(0.6)
    early_stopping(0.7)
    early_stopping(0.8)

    assert early_stopping.early_stop is True


def test_training_loop():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.training.trainer import Trainer

    # Small dummy dataset
    x = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=5
    )

    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu"
    )

    train_result = trainer.train_one_epoch(loader)
    val_result = trainer.validate(loader)

    assert train_result["loss"] >= 0
    assert val_result["loss"] >= 0

    assert 0 <= train_result["accuracy"] <= 1
    assert 0 <= val_result["accuracy"] <= 1

      

