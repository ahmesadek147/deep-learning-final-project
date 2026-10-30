"""Final, test-only evaluation for the completed P4/P5 experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_class_names, get_test_dataloader
from src.metrics import calculate_metrics
from src.training.run_training import build_model, select_device
from src.visualization.gradcam import collect_examples, overlay_heatmap


def _load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    model_config = metadata.get("model_config")
    if not model_config:
        raise ValueError(f"Checkpoint {path} has no model_config metadata.")
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, metadata


def evaluate_checkpoint(checkpoint_path, loader, class_names, device, amp=False):
    """Evaluate a checkpoint without updating weights or selecting parameters."""
    model, metadata = _load_checkpoint(Path(checkpoint_path), device)
    criterion = torch.nn.CrossEntropyLoss()
    all_true, all_pred = [], []
    total_loss, total = 0.0, 0
    use_amp = bool(amp and device.type == "cuda")
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(logits.argmax(1).cpu().tolist())
    metrics = calculate_metrics(all_true, all_pred)
    labels = list(range(len(class_names)))
    report = classification_report(all_true, all_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics = {
        "test_loss": total_loss / max(total, 1),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["precision"],
        "macro_recall": metrics["recall"],
        "macro_f1": metrics["f1_score"],
        "num_samples": total,
        "checkpoint": str(checkpoint_path),
        "model_config": metadata["model_config"],
    }
    return metrics, report, np.asarray(all_true), np.asarray(all_pred), model


def _write_report(report, path: Path):
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    for name, values in report.items():
        if isinstance(values, dict):
            rows.append({"class": name, **values})
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        writer.writeheader(); writer.writerows(rows)


def _save_confusion(matrix, path: Path):
    np.savetxt(path.with_suffix(".csv"), matrix, fmt="%d", delimiter=",")
    np.save(path.with_suffix(".npy"), matrix)


def save_gradcam_examples(model, loader, class_names, device, out_dir: Path, n=4):
    examples = collect_examples(model, loader, device, n_correct=n, n_wrong=n)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, items in examples.items():
        for index, item in enumerate(items, 1):
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
            image, cam = item["image"], item["cam"]
            axes[0].imshow(image); axes[0].set_title("Original")
            axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1); axes[1].set_title("Grad-CAM heatmap")
            axes[2].imshow(overlay_heatmap(image, cam)); axes[2].set_title("Overlay")
            title = (f"True: {class_names[item['true']]} | Pred: {class_names[item['pred']]} "
                     f"| Confidence: {item['conf']:.1%}")
            fig.suptitle(title, color="green" if kind == "correct" else "red", fontsize=9)
            for axis in axes: axis.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{kind}_{index:02d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    return {kind: len(items) for kind, items in examples.items()}


def run_final_evaluation(config, checkpoints: dict[str, str], output_dir="outputs/evaluation", gradcam_dir="outputs/gradcam"):
    """Evaluate both selected checkpoints and log new, immutable MLflow runs."""
    data, training = config["data"], config["training"]
    device = select_device(config.get("device", "auto"))
    class_names = get_class_names(data["dataset_name"], "train")
    test_names = get_class_names(data["test_dataset_name"], data.get("test_split", "test"))
    if class_names != test_names:
        raise ValueError("Training and test ClassLabel mappings differ; refusing evaluation.")
    loader = get_test_dataloader(data["test_dataset_name"], data.get("test_split", "test"),
                                 batch_size=training.get("batch_size", 8), image_size=data["image_size"],
                                 num_workers=data.get("num_workers", 0), pin_memory=device.type == "cuda")
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    summary = []
    for part, checkpoint in checkpoints.items():
        metrics, report, true, pred, model = evaluate_checkpoint(checkpoint, loader, class_names, device,
                                                                  amp=training.get("mixed_precision", False))
        # A single-part evaluation writes directly to the requested directory;
        # multi-part evaluation retains the historical p4/p5 subdirectories.
        part_dir = root if len(checkpoints) == 1 else root / part
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_report(report, part_dir / "classification_report.json")
        _save_confusion(confusion_matrix(true, pred, labels=list(range(len(class_names)))), part_dir / "confusion_matrix")
        if part == "p5" and gradcam_dir:
            save_gradcam_examples(model, loader, class_names, device, Path(gradcam_dir))
        mlflow_cfg = config.get("_mlflow_by_part", {}).get(part, config["mlflow"])
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])
        with mlflow.start_run(run_name=f"final-test-{part}") as active_run:
            mlflow.log_params({"part": part, "checkpoint": str(checkpoint), "test_dataset": data["test_dataset_name"],
                               "test_split": data.get("test_split", "test"), "device": str(device),
                               "class_count": len(class_names)})
            mlflow.log_metrics({f"test_{key}": float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
            mlflow.log_artifacts(str(part_dir))
            metrics["mlflow_run_id"] = active_run.info.run_id
        (part_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary.append({"part": part, **{k: metrics[k] for k in ("test_loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")},
                        "checkpoint": str(checkpoint)})
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
