"""Build a reproducible, artifact-only Grid-vs-Random search comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


def build_report(output_root="outputs"):
    root = Path(output_root)
    out = root / "final_submission"; figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    csvs = {"P4 Grid Search": root / "search_summaries/p4_18850aea_trials.csv",
            "P5 Random Search": root / "search_summaries/p5_618382b5_trials.csv"}
    frames = {name: pd.read_csv(path) for name, path in csvs.items()}
    run_rows = {}
    mlflow.set_tracking_uri("sqlite:///outputs/mlflow.db")
    for name, experiment in [("P4 Grid Search", "CUB_P4_GridSearch"), ("P5 Random Search", "CUB_P5_RandomSearch")]:
        exp = mlflow.get_experiment_by_name(experiment)
        runs = mlflow.search_runs([exp.experiment_id])
        runs = runs[runs["tags.mlflow.runName"].fillna("").str.contains("trial")]
        rows = {}
        for _, row in runs.iterrows():
            start = pd.to_datetime(row["start_time"], unit="ms", utc=True)
            end = pd.to_datetime(row["end_time"], unit="ms", utc=True) if pd.notna(row["end_time"]) else None
            rows[row["tags.mlflow.runName"]] = {"start": start, "end": end,
                "duration": (end - start).total_seconds() if end is not None else None,
                "status": row.get("status", "UNKNOWN")}
        run_rows[name] = rows

    summaries = {}
    for name, frame in frames.items():
        rows = run_rows[name]
        frame["run_name"] = frame["trial_id"].map(lambda x: x)
        frame["duration_seconds"] = frame["trial_id"].map(lambda x: rows.get(x, {}).get("duration"))
        completed = frame[frame["status"] == "completed"].copy()
        first = min(rows[x]["start"] for x in rows)
        last = max(rows[x]["end"] for x in rows if rows[x]["end"] is not None)
        summaries[name] = {"frame": frame, "completed": completed,
                           "first": first, "last": last,
                           "wall_seconds": (last - first).total_seconds(),
                           "sum_seconds": float(completed["duration_seconds"].sum()),
                           "mean_trial_seconds": float(completed["duration_seconds"].mean()),
                           "std_trial_seconds": float(completed["duration_seconds"].std(ddof=0))}

    metric_specs = [("best_val_macro_f1", "Validation Macro F1", "higher"),
                    ("best_val_accuracy", "Validation Accuracy", "higher"),
                    ("best_val_loss", "Validation Loss", "lower")]
    labels = list(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, title, _) in zip(axes, metric_specs):
        means = [summaries[x]["completed"][column].mean() for x in labels]
        bests = [summaries[x]["completed"][column].min() if column == "best_val_loss" else summaries[x]["completed"][column].max() for x in labels]
        x = np.arange(2); ax.bar(x - .18, means, .36, label="Mean trial"); ax.bar(x + .18, bests, .36, label="Best trial")
        ax.set_xticks(x, ["P4 Grid", "P5 Random"]); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "metric_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, title, _) in zip(axes, metric_specs):
        ax.boxplot([summaries[x]["completed"][column] for x in labels], tick_labels=["P4 Grid", "P5 Random"])
        ax.set_title(f"Trial {title}"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "trial_score_distributions.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, name in enumerate(labels):
        frame = summaries[name]["completed"]
        ax.bar(np.arange(len(frame)) + i * .4, frame["duration_seconds"] / 60, .4, label=name)
    ax.set(xlabel="Trial order", ylabel="Minutes", title="Recorded MLflow duration per trial"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "trial_durations.png", dpi=180); plt.close(fig)

    lines = ["# Grid Search vs Random Search (Milestone 1)", "", "This comparison uses the persisted search CSV summaries and MLflow run timestamps. No search was rerun, and no checkpoint was changed.", "", "## Experimental context", "", "P4 is an 8-configuration exhaustive Grid Search over the Scratch CNN (`simple_cnn`). P5 is an 8-trial seeded Random Search over a 36-configuration space using ResNet18 Transfer Learning (pretrained, with the sampled `freeze_backbone` setting). Because model family and training setup differ, these results cannot isolate Grid Search versus Random Search.", "", "## Search-level comparison", "", "| Quantity | P4 Grid Search | P5 Random Search |", "|---|---:|---:|"]
    for key, label in [("trials", "Completed trials"), ("cardinality", "Search-space cardinality"), ("failed", "Failed trials"), ("wall", "Wall-clock span (s)"), ("sum", "Sum of trial runtimes (s)"), ("avg", "Average trial runtime (s)"), ("std", "Trial runtime population SD (s)")]:
        vals=[]
        for name in labels:
            s=summaries[name]; f=s["completed"]; vals.append({"trials":len(f),"cardinality":8 if name.startswith("P4") else 36,"failed":int((s["frame"]["status"] != "completed").sum()),"wall":s["wall_seconds"],"sum":s["sum_seconds"],"avg":s["mean_trial_seconds"],"std":s["std_trial_seconds"]}[key])
        lines.append(f"| {label} | {vals[0]:.3f} | {vals[1]:.3f} |" if isinstance(vals[0],float) else f"| {label} | {vals[0]} | {vals[1]} |")
    lines += ["", "## Validation trial statistics", "", "| Metric | P4 mean | P4 SD | P4 best | P5 mean | P5 SD | P5 best |", "|---|---:|---:|---:|---:|---:|---:|"]
    for column, title, direction in metric_specs:
        vals=[]
        for name in labels:
            values=summaries[name]["completed"][column]; vals += [values.mean(), values.std(ddof=0), values.min() if direction=="lower" else values.max()]
        lines.append(f"| {title} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {vals[3]:.6f} | {vals[4]:.6f} | {vals[5]:.6f} |")
    lines += ["", "## Exact timing evidence", "", "Times below are UTC values recorded in MLflow. The wall-clock span is first trial start to last trial end; the runtime sum excludes inter-trial gaps. No timing estimates were used.", ""]
    for name in labels:
        s=summaries[name]; lines += [f"### {name}", f"- First start: `{s['first'].isoformat()}`", f"- Last end: `{s['last'].isoformat()}`", f"- Wall-clock span: `{s['wall_seconds']:.3f}` seconds", f"- Sum of trial runtimes: `{s['sum_seconds']:.3f}` seconds", ""]
        lines += ["| Trial | Start (UTC) | End (UTC) | Duration (s) |", "|---|---|---|---:|"]
        for trial_id in sorted(s["frame"]["trial_id"]):
            timing = run_rows[name].get(trial_id, {})
            if timing.get("duration") is None:
                lines.append(f"| {trial_id} | unavailable | unavailable | unavailable |")
            else:
                lines.append(f"| {trial_id} | {timing['start'].isoformat()} | {timing['end'].isoformat()} | {timing['duration']:.3f} |")
        lines.append("")
    lines += ["## Best hyperparameters", "", "- **P4 Grid, trial 003:** `learning_rate=0.001`, `weight_decay=0.00001`, `dropout=0.2`; best validation accuracy `0.0584307`, macro F1 `0.0436706`, loss `4.6153766`.", "- **P5 Random, trial 005:** `learning_rate=0.001`, `weight_decay=0.0001`, `dropout=0.2`, `freeze_backbone=True`; best validation accuracy `0.4532554`, macro F1 `0.4330889`, loss `2.1626992`.", "", "## Strengths and limitations", "", "Grid Search is exhaustive over its declared discrete space and provides complete coverage, but its cost grows multiplicatively with each dimension. Random Search explores more distinct combinations per fixed trial budget when the space is larger and can be reproducible with a seed, but it does not guarantee coverage or identify the global best configuration.", "", "Random Search should not be called superior from these results: P5 uses a different model family (ResNet18 Transfer Learning), different effective batch size, and a different search space. The large score gap primarily demonstrates the model/training difference, not an isolated search-strategy effect.", "", "## Conclusion", "", "Within the recorded experiments, P4 exhaustively evaluated all 8 Scratch CNN configurations and P5 sampled 8 of 36 ResNet18 Transfer Learning configurations. P5 achieved the stronger validation results, but the comparison is descriptive rather than causal. A fair search-strategy comparison would hold architecture, data, budget, and search dimensions constant, then compare exhaustive and random selection under matched compute.", "", "## Charts", "", "- `figures/metric_comparison.png`", "- `figures/trial_score_distributions.png`", "- `figures/trial_durations.png`", ""]
    path = out / "search_comparison.md"; path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_report())
