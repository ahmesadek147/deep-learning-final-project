
import torch
from tqdm import tqdm
import mlflow


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, train_loader):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total
        }

    def validate(self, val_loader):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total
        }

    
    def fit(self, train_loader, val_loader, epochs=20,
            early_stopping=None, checkpoint=None):
        
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Bird_Classification")

        mlflow.start_run()

        mlflow.log_params({
            "epochs": epochs,
            "optimizer": type(self.optimizer).__name__,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "model": type(self.model).__name__
        })
        
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            mlflow.log_metrics({
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"]
            }, step=epoch)


            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Accuracy: {train_metrics['accuracy']:.4f}"
            )

            print(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}"
            )

            if checkpoint is not None:
                checkpoint.save_if_best(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics["loss"]
                )

            if early_stopping is not None:
                if early_stopping(val_metrics["loss"]):
                    print("Early stopping triggered!")
                    break
        mlflow.end_run()
        return history