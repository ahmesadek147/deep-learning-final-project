"""P4-owned scratch-CNN experiments built on the team's existing interfaces.

Does not modify P1 Trainer/CLI, P2 dataset or metrics, P3 augmentation, or P5.
Run as ``python -m src.training.p4_experiments --help`` from repository root.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import yaml

from src.dataset import get_dataloaders
from src.models.simple_cnn import SimpleCNN
from src.models.p4_variants import P4ScratchCNN
from src.training.trainer import Trainer


P4_MODELS = {"simple_cnn", "delayed_pool_cnn", "residual_cnn"}


def build_p4_model(config: dict) -> torch.nn.Module:
    name = config.get("name")
    if name not in P4_MODELS or config.get("pretrained", False) is not False:
        raise ValueError("P4 requires a recognized from-scratch CNN; pretrained must be false")
    if name == "simple_cnn":
        return SimpleCNN(num_classes=config.get("num_classes", 200),
                         dropout=config.get("dropout", 0.3))
    return P4ScratchCNN(
        variant="delayed_pool" if name == "delayed_pool_cnn" else "residual",
        num_classes=config.get("num_classes", 200), depth=config.get("depth", 4),
        width=config.get("width", 32), max_channels=config.get("max_channels", 384),
        dropout=config.get("dropout", 0.3), pretrained=False,
    )


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("P4 configuration must contain a model mapping")
    build_p4_model(config["model"])  # validate without loading pretrained weights
    data = config["data"]
    if data.get("dataset_name") != "Multimodal-Fatima/CUB_train":
        raise ValueError("P4 searches must use the official CUB training partition")
    if data.get("split_strategy") != "stratified_group" or not data.get("split_manifest"):
        raise ValueError("P4 runs require the verified duplicate-safe split manifest")
    if config["model"].get("num_classes") != 200:
        raise ValueError("P4 CUB classification requires exactly 200 classes")
    if config["training"].get("monitor") != "val_macro_f1":
        raise ValueError("Use validation macro-F1 for P4 checkpoint selection")
    return config


def _digest(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(),
                  numpy_state[2], numpy_state[3], numpy_state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    arr = state["numpy"]
    np.random.set_state((arr[0], np.asarray(arr[1], dtype=np.uint32),
                         arr[2], arr[3], arr[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def _key(metrics: dict) -> tuple[float, float, float]:
    return (float(metrics["val_macro_f1"]), float(metrics["val_accuracy"]),
            -float(metrics["val_loss"]))


def _export_history(rows: list[dict], run_dir: Path) -> None:
    _write_json(run_dir / "history.json", rows)
    if rows:
        csv_path = run_dir / "epochs.csv"
        temp = csv_path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(csv_path)


def _export_plots(rows: list[dict], run_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, columns in (
        ("loss", ("train_loss", "val_loss")),
        ("accuracy", ("train_accuracy", "val_accuracy")),
        ("macro_f1", ("val_macro_f1",)),
        ("learning_rate", ("learning_rate",)),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for key in columns:
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], label=key)
        ax.legend()
        ax.set(xlabel="Epoch", ylabel=name)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(run_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def run_one(config: dict, output_root: Path, run_id: str, resume: bool = False) -> dict:
    """Run one P4 trial, reusing P1 epoch methods and P2/P3 loaders/transforms."""
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError("run_id must be a simple directory name")
    config = deepcopy(config)
    seed = config["data"].get("seed", 42)
    _set_seed(seed)
    run_dir = output_root / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Run already exists: {run_dir}; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _digest(config)
    if not resume:
        _write_json(run_dir / "config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("device") == "cuda" and device.type != "cuda":
        raise RuntimeError("T4/CUDA requested but unavailable")
    data, train_cfg = config["data"], config["training"]
    train_loader, val_loader = get_dataloaders(
        dataset_name=data["dataset_name"], batch_size=train_cfg["batch_size"],
        val_split=data.get("val_split", 0.2), seed=seed,
        image_size=data.get("image_size", 224), num_workers=data.get("num_workers", 0),
        pin_memory=device.type == "cuda", augmentation_config=config.get("augmentation"),
        split_strategy=data["split_strategy"], split_version=data["split_version"],
        split_manifest=data["split_manifest"],
        max_train_samples=data.get("max_train_samples"), max_val_samples=data.get("max_val_samples"),
    )
    model = build_p4_model(config["model"])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg.get("weight_decay", 0.0001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    trainer = Trainer(model, optimizer, torch.nn.CrossEntropyLoss(), device,
                      use_amp=train_cfg.get("mixed_precision", True))
    rows: list[dict] = []
    best_key: tuple | None = None
    best_epoch = 0
    start_epoch = 1
    wait = 0
    last_path = run_dir / "last.pth"
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["config_hash"] != config_hash:
            raise ValueError("Cannot resume: configuration has changed")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        trainer.scaler.load_state_dict(state["scaler_state_dict"])
        rows, best_epoch, wait = state["history"], state["best_epoch"], state["wait"]
        best_key = tuple(state["best_key"]) if state["best_key"] is not None else None
        start_epoch = state["epoch"] + 1
        _restore_rng(state["rng"])
    else:
        # Same loader shuffle/augmentation RNG despite differing model widths.
        _set_seed(seed)
    import mlflow
    mlflow_cfg = config.get("mlflow", {})
    if uri := mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "P4_Scratch"))
    elapsed = 0.0
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({"model": config["model"]["name"], "parameters": params,
                           "seed": seed, "config_sha256": config_hash,
                           "architecture_depth": config["model"].get("depth", ""),
                           "architecture_width": config["model"].get("width", ""),
                           "learning_rate": train_cfg["learning_rate"],
                           "weight_decay": train_cfg.get("weight_decay", 0),
                           "epochs": train_cfg["epochs"]})
        for epoch in range(start_epoch, train_cfg["epochs"] + 1):
            started = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_result = trainer.train_one_epoch(train_loader)
            val_result = trainer.validate(val_loader)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_time = time.perf_counter() - started
            elapsed += epoch_time
            record = {"epoch": epoch, "train_loss": train_result["loss"],
                      "train_accuracy": train_result["accuracy"],
                      "val_loss": val_result["loss"],
                      "val_accuracy": val_result["accuracy"],
                      "val_macro_precision": val_result["macro_precision"],
                      "val_macro_recall": val_result["macro_recall"],
                      "val_macro_f1": val_result["macro_f1"], "learning_rate": lr,
                      "epoch_seconds": epoch_time,
                      "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2
                                            if device.type == "cuda" else 0.0)}
            rows.append(record)
            scheduler.step(record["val_macro_f1"])
            key = _key(record)
            if best_key is None or key > best_key:
                best_key, best_epoch, wait = key, epoch, 0
                _save_checkpoint(run_dir / "best_model.pth", {
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "metadata": {"model_config": config["model"],
                                 "image_size": data.get("image_size", 224),
                                 "split_strategy": data["split_strategy"],
                                 "split_manifest": data["split_manifest"],
                                 "official_test_used": False},
                    "metrics": record, "monitor": "val_macro_f1"})
            else:
                wait += 1
            _save_checkpoint(last_path, {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": trainer.scaler.state_dict(),
                "rng": _rng_state(), "history": rows,
                "best_key": best_key, "best_epoch": best_epoch, "wait": wait,
                "config_hash": config_hash})
            _export_history(rows, run_dir)
            mlflow.log_metrics({k: float(v) for k, v in record.items() if k != "epoch"}, step=epoch)
            print(f"P4 {run_id} epoch {epoch}: val acc {record['val_accuracy']:.4f}, "
                  f"macro-F1 {record['val_macro_f1']:.4f}")
            if train_cfg.get("early_stopping", False) and wait >= train_cfg.get("patience", 20):
                break
    _export_plots(rows, run_dir)
    selected = rows[best_epoch - 1]
    result = {"run_id": run_id, "model": config["model"],
              "config": config, "best_epoch": best_epoch,
              "best_val_accuracy": selected["val_accuracy"],
              "best_val_macro_f1": selected["val_macro_f1"],
              "best_val_loss": selected["val_loss"],
              "parameter_count": params,
              "training_seconds": sum(r["epoch_seconds"] for r in rows),
              "peak_gpu_memory_mb": max(r["peak_gpu_memory_mb"] for r in rows),
              "checkpoint": str(run_dir / "best_model.pth"),
              "official_test_used": False}
    _write_json(run_dir / "result.json", result)
    return result


def run_study(config: dict, candidates: list[dict], output_root: Path,
              study_id: str, resume: bool = False) -> dict:
    """Run a fixed architecture/grid study with recoverable trial checkpoints."""
    if Path(study_id).name != study_id or study_id in {"", ".", ".."}:
        raise ValueError("study_id must be a simple name")
    state_path = output_root / "search" / f"{study_id}_state.json"
    fingerprint = _digest({"config": config, "candidates": candidates})
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"Study exists: {state_path}; pass --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config_hash"] != fingerprint:
            raise ValueError("Study configuration changed; refuse resume")
    else:
        if resume:
            raise FileNotFoundError(f"No saved study to resume: {state_path}")
        state = {"config_hash": fingerprint, "completed": {}, "failures": {}}
        _write_json(state_path, state)

    for index, overrides in enumerate(candidates, start=1):
        trial_id = f"{study_id}-trial-{index:02d}"
        if trial_id in state["completed"]:
            continue
        trial_config = deepcopy(config)
        for key, value in overrides.items():
            target = trial_config
            parts = key.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        run_dir = output_root / "runs" / trial_id
        try:
            if (run_dir / "result.json").exists():
                result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                if _digest(result["config"]) != _digest(trial_config):
                    raise ValueError("Saved trial has a different configuration")
            else:
                has_resume = (run_dir / "last.pth").exists()
                result = run_one(trial_config, output_root, trial_id, resume=has_resume)
            state["completed"][trial_id] = result
            state["failures"].pop(trial_id, None)
        except Exception as error:
            state["failures"][trial_id] = f"{type(error).__name__}: {error}"
            _write_json(state_path, state)
            raise  # preserve interrupted trial for explicit resume/diagnosis
        _write_json(state_path, state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed = list(state["completed"].values())
    if not completed:
        raise RuntimeError("No completed P4 trials")
    winner = max(completed, key=lambda r: (
        r["best_val_macro_f1"], r["best_val_accuracy"], -r["best_val_loss"]))
    result = {"study_id": study_id, "best_trial": winner,
              "completed_trials": len(completed), "selection_metric": "validation macro-F1",
              "results": completed, "failures": state["failures"]}
    _write_json(output_root / "search" / f"{study_id}_selection.json", result)
    summary_path = output_root / "search" / f"{study_id}_trials.csv"
    rows = [{key: r[key] for key in ("run_id", "best_epoch", "best_val_accuracy",
                                    "best_val_macro_f1", "best_val_loss", "parameter_count",
                                    "training_seconds", "peak_gpu_memory_mb")}
            for r in completed]
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _load_winner(path: Path, config: dict) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))["best_trial"]
    old = state["config"]["data"]
    current = config["data"]
    for name in ("dataset_name", "split_strategy", "split_version", "split_manifest",
                 "image_size", "seed"):
        if old.get(name) != current.get(name):
            raise ValueError(f"Winner uses a different data setting: {name}")
    config = deepcopy(config)
    config["model"] = deepcopy(state["config"]["model"])
    for name in ("learning_rate", "batch_size", "weight_decay"):
        config["training"][name] = state["config"]["training"][name]
    return config


def _grid_candidates(config: dict) -> list[dict]:
    space = config["search"]["space"]
    keys = list(space)
    if any(not values for values in space.values()):
        raise ValueError("Grid dimensions cannot be empty")
    return [dict(zip(keys, values)) for values in product(*(space[key] for key in keys))]


def predict(checkpoint_path: Path, image_path: Path) -> dict:
    """P4-only inference without modifying the team's shared CLI."""
    from PIL import Image
    from src.data.augmentation import build_eval_transform
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    model = build_p4_model(meta["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with Image.open(image_path) as img:
        x = build_eval_transform(image_size=meta["image_size"])(img.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probs = model(x).softmax(1)
    confidence, label = probs.max(dim=1)
    return {"class_id": int(label.item()), "confidence": float(confidence.item())}


def main() -> None:
    parser = argparse.ArgumentParser(description="P4-only scratch CNN experiments")
    parser.add_argument("command", choices=("architectures", "grid", "final", "train", "predict"))
    parser.add_argument("--config", default="configs/p4_architecture.yaml")
    parser.add_argument("--output-root", default="outputs/p4_scoped")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--winner-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.command == "predict":
        if not args.checkpoint or not args.image:
            parser.error("predict needs --checkpoint and --image")
        print(json.dumps(predict(args.checkpoint, args.image), indent=2))
        return
    config = load_config(args.config)
    root = Path(args.output_root)
    if args.command == "architectures":
        candidates = [{"model.name": name} for name in config["architecture_candidates"]]
        outcome = run_study(config, candidates, root, args.run_id or "p4_architectures", args.resume)
    elif args.command == "grid":
        if not args.winner_from:
            parser.error("Grid Search needs --winner-from architecture selection JSON")
        config = _load_winner(args.winner_from, config)
        if config["model"]["name"] == "simple_cnn":
            parser.error("SimpleCNN ignores depth/width; choose an advanced P4 model for capacity grid")
        outcome = run_study(config, _grid_candidates(config), root,
                            args.run_id or "p4_grid", args.resume)
    else:
        if args.command == "final":
            if not args.winner_from:
                parser.error("Final training needs --winner-from Grid Search selection JSON")
            config = _load_winner(args.winner_from, config)
        if args.epochs is not None:
            if args.epochs < 1:
                parser.error("--epochs must be positive")
            config["training"]["epochs"] = args.epochs
        for flag, name in ((args.max_train_samples, "max_train_samples"),
                           (args.max_val_samples, "max_val_samples")):
            if flag is not None:
                if flag < 1:
                    parser.error(f"{name} must be positive")
                config["data"][name] = flag
        outcome = run_one(config, root, args.run_id or "p4_final", args.resume)
    print(json.dumps({k: outcome[k] for k in ("run_id", "study_id", "best_epoch",
                                               "best_val_accuracy", "best_val_macro_f1",
                                               "completed_trials") if k in outcome}, indent=2))


if __name__ == "__main__":
    main()
