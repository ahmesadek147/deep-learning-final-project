"""Reproducible search execution and persistent trial summaries."""
from __future__ import annotations

import csv
from copy import deepcopy
from itertools import product
from pathlib import Path
from random import Random
from typing import Any, Iterator
from uuid import uuid4

import torch

from src.training.experiment_config import validate_search_config
from src.training.run_training import effective_mlflow_params, run_training


SUMMARY_FIELDS = [
    "search_id", "trial_id", "part", "status", "error_type", "error_message",
    "model_name", "pretrained", "freeze_backbone", "dropout", "learning_rate",
    "weight_decay", "optimizer", "loss", "batch_size", "epoch_limit", "mixed_precision_enabled",
    "mixed_precision_requested", "seed", "validation_split", "image_size",
    "num_classes", "device", "split_strategy", "split_version", "split_manifest_sha256", "train_size", "validation_size",
    "best_epoch", "best_val_loss", "best_val_accuracy", "best_val_macro_precision",
    "best_val_macro_recall", "best_val_macro_f1", "checkpoint_path",
]


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def grid_trials(search_space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    if not search_space or any(not values for values in search_space.values()):
        raise ValueError("Grid search space must contain non-empty value lists.")
    keys, values = zip(*search_space.items())
    yield from (dict(zip(keys, combination)) for combination in product(*values))


def random_trials(search_space: dict[str, list[Any]], n_trials: int, seed: int) -> Iterator[dict[str, Any]]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least one.")
    if not search_space or any(not values for values in search_space.values()):
        raise ValueError("Random search space must contain non-empty value lists.")
    keys, value_lists = zip(*search_space.items())
    candidates = list(product(*value_lists))
    if n_trials > len(candidates):
        raise ValueError(f"Random search requests {n_trials} unique configurations, but only {len(candidates)} exist.")
    for combination in Random(seed).sample(candidates, n_trials):
        yield dict(zip(keys, combination))


def summary_path(config: dict, part: str, search_id: str) -> Path:
    directory = Path(config["search"].get("summary_dir", "outputs/search_summaries"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{part}_{search_id}_trials.csv"


def write_trial_summary(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically refresh the CSV after each completed or failed trial."""
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def trial_record(search_id: str, trial_id: str, part: str, config: dict,
                 result: dict[str, Any] | None = None, error: Exception | None = None) -> dict[str, Any]:
    params = effective_mlflow_params(config, config["training"].get("mixed_precision", False))
    record = {"search_id": search_id, "trial_id": trial_id, "part": part, "status": "completed",
              "error_type": "", "error_message": "", **params,
              "best_epoch": "", "best_val_loss": "", "best_val_accuracy": "",
              "best_val_macro_precision": "", "best_val_macro_recall": "", "best_val_macro_f1": "",
              "checkpoint_path": ""}
    if error is not None:
        is_oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
        record.update({"status": "oom" if is_oom else "failed", "error_type": type(error).__name__,
                       "error_message": str(error).replace("\n", " ")})
        return record
    if result is None:
        raise ValueError("A completed trial requires a result.")
    record.update(result["metrics"])
    record["checkpoint_path"] = result["checkpoint"]
    return record


def run_search(config: dict, strategy: str) -> dict[str, Any]:
    """Run validated trials and leave a CSV record even when a trial fails."""
    if strategy not in {"grid", "random"}:
        raise ValueError(f"Unknown search strategy: {strategy}")
    search_config = config.get("search", {})
    space = search_config.get("space", {})
    seed = search_config.get("seed", config.get("data", {}).get("seed", 42))
    cardinality = validate_search_config(config, strategy)
    n_trials = search_config["n_trials"]
    trials = grid_trials(space) if strategy == "grid" else random_trials(space, n_trials, seed)
    part = config.get("experiment_part", "p4" if strategy == "grid" else "p5")
    search_id = uuid4().hex[:8]
    csv_path = summary_path(config, part, search_id)
    records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for index, parameters in enumerate(trials, start=1):
        trial_config = deepcopy(config)
        for key, value in parameters.items():
            _set_nested(trial_config, key, value)
        trial_id = f"{strategy}-{search_id}-trial-{index:03d}"
        try:
            result = run_training(trial_config, run_name=trial_id)
        except Exception as error:
            records.append(trial_record(search_id, trial_id, part, trial_config, error=error))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            result["parameters"] = parameters
            results.append(result)
            records.append(trial_record(search_id, trial_id, part, trial_config, result=result))
        write_trial_summary(csv_path, records)

    if not results:
        raise RuntimeError(f"All trials failed; see {csv_path} for failure details.")
    best = max(results, key=lambda result: result["history"]["val_macro_f1"][-1])
    return {"strategy": strategy, "part": part, "search_id": search_id, "summary_csv": str(csv_path),
            "search_cardinality": cardinality, "executed_trials": len(records), "completed_trials": len(results),
            "failed_trials": len(records) - len(results), "trials": results, "best_trial": best,
            "selection_metric": "validation macro F1"}
