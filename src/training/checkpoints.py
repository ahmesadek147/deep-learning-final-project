
from pathlib import Path
import torch


class ModelCheckpoint:
    def __init__(self, save_dir="outputs/checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

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
            }

            save_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, save_path)

            print(
                f"Best model saved at epoch {epoch} "
                f"with validation loss {val_loss:.4f}"
            )

            return True

        return False