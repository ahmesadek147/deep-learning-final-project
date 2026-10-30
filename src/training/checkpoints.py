
from pathlib import Path
import torch


class ModelCheckpoint:
    def __init__(self, save_dir="outputs/checkpoints", metadata=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.path = self.save_dir / "best_model.pth"
        self.metadata = dict(metadata or {})

    def save_if_best(
        self, model, optimizer, epoch, val_loss
    ):
        if val_loss < self.best_loss:
            self.best_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "metadata": self.metadata,
            }

            torch.save(checkpoint, self.path)

            print(
                f"Best model saved to {self.path} at epoch {epoch} "
                f"with validation loss {val_loss:.4f}"
            )

            return True

        return False
