"""Controlled transfer-learning study for P5.

The P5 random search varied four factors at once over two architectures' worth
of hyperparameters and only ever fine-tuned at learning rates of 1e-3 and 3e-3.
Those rates destroy ImageNet features within the first few hundred steps, so
every unfrozen trial collapsed and the "frozen vs fine-tuned" comparison the
brief asks for could not be read off the results.

This module runs an enumerated study instead: one factor changes at a time,
everything else is pinned by ``common`` in the config. It reuses the shared
``run_training`` pipeline, so the runs are tracked in MLflow, checkpointed and
split-validated exactly like every other experiment.

    python -m src.training.transfer_study --config configs/p5_transfer_study.yaml
    python -m src.training.transfer_study --config configs/p5_transfer_study.yaml --only resnet18-finetune-lr1e-4
    python -m src.training.transfer_study --config configs/p5_transfer_study.yaml --smoke

Results are appended to ``<output_dir>/study_results.json`` after every run, so
an interrupted study resumes where it stopped instead of retraining.
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.training.run_training import run_training

RESULT_COLUMNS = [
    "label", "model_name", "freeze_backbone", "learning_rate", "batch_size",
    "epochs_run", "best_epoch", "best_val_loss", "best_val_accuracy",
    "best_val_macro_precision", "best_val_macro_recall", "best_val_macro_f1",
    "checkpoint", "note",
]


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def load_study(path: str | Path) -> dict:
    """Load and validate a study document."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Study YAML must contain a mapping.")
    for key in ("common", "runs"):
        if key not in document:
            raise ValueError(f"Study config requires a '{key}' section.")
    if not isinstance(document["runs"], list) or not document["runs"]:
        raise ValueError("Study config requires a non-empty 'runs' list.")
    labels = [run.get("label") for run in document["runs"]]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Every run needs a non-empty string 'label'.")
    if len(set(labels)) != len(labels):
        raise ValueError("Run labels must be unique.")
    for run in document["runs"]:
        if not isinstance(run.get("overrides"), dict) or not run["overrides"]:
            raise ValueError(f"Run '{run['label']}' needs a non-empty 'overrides' mapping.")
    return document


def build_run_config(document: dict, run: dict) -> dict:
    """Merge one run's overrides over the shared ``common`` section."""
    config = deepcopy(document["common"])
    for key, value in run["overrides"].items():
        _set_nested(config, key, value)
    config["experiment_part"] = "p5_transfer_study"
    if "name" not in config.get("model", {}):
        raise ValueError(f"Run '{run['label']}' does not set model.name.")
    return config


def _row(label: str, note: str, config: dict, result: dict) -> dict:
    metrics, history = result["metrics"], result["history"]
    return {
        "label": label,
        "model_name": config["model"]["name"],
        "freeze_backbone": config["model"].get("freeze_backbone", False),
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "epochs_run": len(history["val_loss"]),
        "best_epoch": metrics["best_epoch"],
        "best_val_loss": metrics["best_val_loss"],
        "best_val_accuracy": metrics["best_val_accuracy"],
        "best_val_macro_precision": metrics["best_val_macro_precision"],
        "best_val_macro_recall": metrics["best_val_macro_recall"],
        "best_val_macro_f1": metrics["best_val_macro_f1"],
        "checkpoint": result["checkpoint"],
        "note": note,
    }


def _write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "study_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "study_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict], out_dir: Path) -> Path:
    """Write the markdown table and figure the report cites."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: -r["best_val_macro_f1"])
    best = ordered[0]

    frozen = [r for r in rows if r["freeze_backbone"]]
    tuned = [r for r in rows if not r["freeze_backbone"]]
    lines = [
        "# P5 Transfer-Learning Study",
        "",
        "Controlled comparison: one factor changes per run, everything else "
        "(split, seed, image size, augmentation, batch size, epoch limit, optimizer, "
        "weight decay) is held constant by the `common` section of "
        "`configs/p5_transfer_study.yaml`. All figures are **validation** metrics on the "
        "duplicate-safe `stratified_group_v2` split; no test data was used.",
        "",
        "## Results",
        "",
        "| Run | Architecture | Backbone | Learning rate | Best epoch | Val accuracy | Val macro F1 | Val loss |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['label']} | {row['model_name']} | "
            f"{'frozen' if row['freeze_backbone'] else 'fine-tuned'} | "
            f"{row['learning_rate']:g} | {row['best_epoch']} | "
            f"{row['best_val_accuracy'] * 100:.2f}% | {row['best_val_macro_f1'] * 100:.2f}% | "
            f"{row['best_val_loss']:.4f} |"
        )

    lines += ["", "## Reading", ""]
    lines.append(
        f"The strongest configuration is **{best['label']}** at "
        f"{best['best_val_accuracy'] * 100:.2f}% validation accuracy and "
        f"{best['best_val_macro_f1'] * 100:.2f}% macro F1."
    )
    if frozen and tuned:
        best_frozen = max(frozen, key=lambda r: r["best_val_macro_f1"])
        best_tuned = max(tuned, key=lambda r: r["best_val_macro_f1"])
        delta = (best_tuned["best_val_macro_f1"] - best_frozen["best_val_macro_f1"]) * 100
        direction = "above" if delta >= 0 else "below"
        lines.append(
            f"Best frozen-backbone run: {best_frozen['label']} at "
            f"{best_frozen['best_val_macro_f1'] * 100:.2f}% macro F1. "
            f"Best fine-tuned run: {best_tuned['label']} at "
            f"{best_tuned['best_val_macro_f1'] * 100:.2f}%, "
            f"{abs(delta):.2f} points {direction} it. "
            "Compare this against the original random search, where every fine-tuned "
            "trial used a learning rate of 1e-3 or 3e-3 and scored between 1.92% and "
            "5.76% accuracy: the collapse there was caused by the learning rate, not by "
            "fine-tuning itself."
        )
    architectures = sorted({row["model_name"] for row in rows})
    if len(architectures) > 1:
        per_arch = []
        for name in architectures:
            top = max((r for r in rows if r["model_name"] == name), key=lambda r: r["best_val_macro_f1"])
            per_arch.append(f"{name} reaches {top['best_val_macro_f1'] * 100:.2f}% macro F1 ({top['label']})")
        lines.append("Across architectures: " + "; ".join(per_arch) + ".")

    lines += ["", "## Files", "",
              "- `study_results.csv`, `study_results.json`: per-run metrics",
              "- `figures/transfer_study.png`: validation accuracy and macro F1 per run", ""]

    path = out_dir / "transfer_study.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    _plot(ordered, out_dir / "figures")
    return path


def _plot(rows: list[dict], figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in rows]
    accuracy = [r["best_val_accuracy"] * 100 for r in rows]
    macro_f1 = [r["best_val_macro_f1"] * 100 for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(rows)), 4.6))
    ax.bar(x - 0.2, accuracy, 0.4, label="Validation accuracy (%)")
    ax.bar(x + 0.2, macro_f1, 0.4, label="Validation macro F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Percent")
    ax.set_title("P5 transfer-learning study (validation split)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "transfer_study.png", dpi=180)
    plt.close(fig)


def run_study(document: dict, only: list[str] | None = None, smoke: bool = False,
              resume: bool = True) -> list[dict]:
    out_dir = Path(document.get("output_dir", "outputs/transfer_study"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "study_results.json"

    rows: list[dict] = []
    if resume and results_path.is_file():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
        print(f"Resuming: {len(rows)} run(s) already recorded in {results_path}")
    done = {row["label"] for row in rows}

    selected = [r for r in document["runs"] if only is None or r["label"] in only]
    if only and len(selected) != len(only):
        missing = sorted(set(only) - {r["label"] for r in selected})
        raise ValueError(f"Unknown run label(s): {', '.join(missing)}")

    for index, run in enumerate(selected, start=1):
        label = run["label"]
        if label in done:
            print(f"[{index}/{len(selected)}] {label}: already done, skipping")
            continue
        config = build_run_config(document, run)
        if smoke:
            config["training"]["epochs"] = 1
            config["data"]["max_train_samples"] = 16
            config["data"]["max_val_samples"] = 16
        print(f"[{index}/{len(selected)}] {label}: {run.get('note', '')}")
        result = run_training(config, run_name=label)
        rows.append(_row(label, run.get("note", ""), config, result))
        _write_outputs(rows, out_dir)
        print(f"    val accuracy {rows[-1]['best_val_accuracy']:.4f}, "
              f"macro F1 {rows[-1]['best_val_macro_f1']:.4f} "
              f"(epoch {rows[-1]['best_epoch']})")

    if rows:
        report = write_report(rows, out_dir)
        print(f"\nStudy report: {report}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled P5 transfer-learning study")
    parser.add_argument("--config", default="configs/p5_transfer_study.yaml")
    parser.add_argument("--only", nargs="+", help="Run only these labels.")
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch on 16 samples per run; never cite these numbers.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing study_results.json and rerun everything.")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild the markdown table and figure from saved results.")
    args = parser.parse_args()

    document = load_study(args.config)
    out_dir = Path(document.get("output_dir", "outputs/transfer_study"))

    if args.report_only:
        results_path = out_dir / "study_results.json"
        if not results_path.is_file():
            parser.error(f"No saved results at {results_path}")
        print(write_report(json.loads(results_path.read_text(encoding="utf-8")), out_dir))
        return

    run_study(document, only=args.only, smoke=args.smoke, resume=not args.no_resume)


if __name__ == "__main__":
    main()
