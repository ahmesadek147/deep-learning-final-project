"""Export MLflow tracking evidence for submission.

The brief requires "screenshots or exports from your tracking tool" in the
report. ``outputs/mlflow.db`` is a SQLite file that a marker may not be able to
open, so this module renders the same information as committed artifacts:

- ``runs.csv`` / ``runs.md``: every run with its parameters and final metrics
- ``<experiment>_metrics.png``: per-epoch curves straight from the logged history
- ``summary.md``: experiment counts, run IDs and the parameters recorded

    python -m src.reporting.export_mlflow
    python -m src.reporting.export_mlflow --tracking-uri sqlite:///outputs/mlflow.db

Nothing here writes to the tracking store; it is read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

DEFAULT_URI = "sqlite:///outputs/mlflow.db"
DEFAULT_OUT = Path("outputs/final_submission/figures/mlflow")

#: Parameters worth showing in the exported table, in reading order.
PARAM_COLUMNS = [
    "model_name", "pretrained", "freeze_backbone", "dropout", "learning_rate",
    "weight_decay", "optimizer", "batch_size", "epoch_limit",
    "mixed_precision_enabled", "seed", "image_size", "eval_resize",
    "split_strategy", "split_version", "split_manifest_sha256",
]
METRIC_COLUMNS = [
    "val_accuracy", "val_macro_f1", "val_loss", "train_accuracy", "train_loss",
]
CURVE_METRICS = [
    ("train_loss", "val_loss", "Loss"),
    ("train_accuracy", "val_accuracy", "Accuracy"),
]


def _experiments(client) -> list:
    return [e for e in client.search_experiments() if e.name != "Default"]


def export(tracking_uri: str = DEFAULT_URI, out_dir: Path = DEFAULT_OUT) -> Path:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = _experiments(client)
    if not experiments:
        raise RuntimeError(f"No MLflow experiments found at {tracking_uri}. "
                           "Run a training command first.")

    frames, summary = [], ["# MLflow tracking export", "",
                           f"Source: `{tracking_uri}`", ""]
    for experiment in experiments:
        runs = mlflow.search_runs([experiment.experiment_id])
        if runs.empty:
            continue
        runs = runs.copy()
        runs.insert(0, "experiment", experiment.name)
        frames.append(runs)
        summary.append(f"- **{experiment.name}**: {len(runs)} run(s)")
        _plot_experiment(client, experiment, runs, out_dir)

    if not frames:
        raise RuntimeError("MLflow experiments exist but contain no runs.")

    all_runs = pd.concat(frames, ignore_index=True)
    keep = ["experiment", "tags.mlflow.runName", "run_id", "status", "start_time", "end_time"]
    keep += [f"params.{name}" for name in PARAM_COLUMNS if f"params.{name}" in all_runs.columns]
    keep += [f"metrics.{name}" for name in METRIC_COLUMNS if f"metrics.{name}" in all_runs.columns]
    table = all_runs[[column for column in keep if column in all_runs.columns]].copy()
    table.columns = [column.split(".", 1)[-1] if "." in column else column for column in table.columns]
    table = table.rename(columns={"mlflow.runName": "run_name"})
    table.to_csv(out_dir / "runs.csv", index=False)
    (out_dir / "runs.md").write_text(table.to_markdown(index=False), encoding="utf-8")

    summary += ["", f"Total runs exported: {len(table)}", "",
                "## Parameters recorded per run", "",
                ", ".join(f"`{name}`" for name in PARAM_COLUMNS), "",
                "## Files", "",
                "- `runs.csv`, `runs.md`: all runs with parameters and final metrics",
                "- `<experiment>_metrics.png`: per-epoch curves from the logged history", ""]
    path = out_dir / "summary.md"
    path.write_text("\n".join(summary), encoding="utf-8")
    print(f"Exported {len(table)} run(s) from {len(frames)} experiment(s) to {out_dir}")
    return path


def _plot_experiment(client, experiment, runs: pd.DataFrame, out_dir: Path) -> None:
    """Plot per-epoch curves for every run in one experiment."""
    fig, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(6 * len(CURVE_METRICS), 4.2))
    axes = axes if len(CURVE_METRICS) > 1 else [axes]
    drew = False

    for axis, (train_key, val_key, title) in zip(axes, CURVE_METRICS):
        for _, row in runs.iterrows():
            name = row.get("tags.mlflow.runName") or row["run_id"][:8]
            for key, style in ((train_key, "--"), (val_key, "-")):
                try:
                    history = client.get_metric_history(row["run_id"], key)
                except Exception:
                    continue
                if not history:
                    continue
                history = sorted(history, key=lambda m: m.step)
                axis.plot([m.step for m in history], [m.value for m in history],
                          style, linewidth=1.2, label=f"{name} {key.split('_')[0]}")
                drew = True
        axis.set(xlabel="Epoch", ylabel=title, title=f"{experiment.name}: {title}")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles and len(handles) <= 12:
            axis.legend(fontsize=6, ncol=2)

    if drew:
        fig.tight_layout()
        safe = experiment.name.replace("/", "_")
        fig.savefig(out_dir / f"{safe}_metrics.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLflow runs as committed artifacts")
    parser.add_argument("--tracking-uri", default=DEFAULT_URI)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    export(args.tracking_uri, args.out_dir)


if __name__ == "__main__":
    main()
