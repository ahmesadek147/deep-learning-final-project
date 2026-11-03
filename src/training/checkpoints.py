
from pathlib import Path
import torch


class ModelCheckpoint:
    def __init__(self, save_dir="outputs/checkpoints", metadata=None,
                 monitor="val_loss", mode="min"):
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint mode must be 'min' or 'max'.")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.path = self.save_dir / "best_model.pth"
        self.metadata = dict(metadata or {})

    def save_if_best(
        self, model, optimizer, epoch, val_loss, monitor_value=None
    ):
        score = float(val_loss if monitor_value is None else monitor_value)
        improved = score < self.best_score if self.mode == "min" else score > self.best_score
        if improved:
            self.best_score = score

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "monitor": self.monitor,
                "monitor_value": score,
                "metadata": self.metadata,
            }

            torch.save(checkpoint, self.path)

            print(
                f"Best model saved to {self.path} at epoch {epoch} "
                f"with {self.monitor} {score:.4f}"
            )

            return True

        return False
