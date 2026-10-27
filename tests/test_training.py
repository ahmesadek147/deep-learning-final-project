
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