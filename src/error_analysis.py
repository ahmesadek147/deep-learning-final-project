"""Post-hoc error analysis for saved held-out evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze_p5(evaluation_dir="outputs/evaluation/p5", output_dir="outputs/evaluation"):
    evaluation_dir, output_dir = Path(evaluation_dir), Path(output_dir)
    report = json.loads((evaluation_dir / "classification_report.json").read_text(encoding="utf-8"))
    # The first 200 report entries are the CUB classes; aggregate rows follow.
    classes = [(name, values) for name, values in report.items()
               if isinstance(values, dict) and "f1-score" in values and values.get("support", 0) < 100]
    classes = classes[:200]
    matrix = np.load(evaluation_dir / "confusion_matrix.npy")
    top = sorted(classes, key=lambda item: (-item[1]["f1-score"], item[0]))[:10]
    bottom = sorted(classes, key=lambda item: (item[1]["f1-score"], item[0]))[:10]
    pairs = []
    for i, (_, _) in enumerate(classes):
        for j, (_, _) in enumerate(classes):
            if i != j and matrix[i, j] > 0:
                pairs.append((int(matrix[i, j]), classes[i][0], classes[j][0]))
    pairs = sorted(pairs, reverse=True)[:15]

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(classes, key=lambda item: item[1]["f1-score"])
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.barh([x[0] for x in ordered], [x[1]["f1-score"] for x in ordered], color="steelblue")
    ax.set(xlabel="F1 score", title="P5 per-class F1 on held-out CUB test set", xlim=(0, 1))
    fig.tight_layout(); fig.savefig(output_dir / "p5_per_class_f1.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [f"{true} → {pred}" for count, true, pred in pairs][::-1]
    counts = [count for count, _, _ in pairs][::-1]
    ax.barh(labels, counts, color="indianred")
    ax.set(xlabel="Number of test images", title="P5 most frequent off-diagonal confusions")
    fig.tight_layout(); fig.savefig(output_dir / "p5_top_confusions.png", dpi=180); plt.close(fig)

    lines = ["# P5 Error Analysis", "", "## Measured evaluation", "",
             "Computed from `outputs/evaluation/p5/classification_report.json` and `confusion_matrix.npy` "
             "on the 5,794-image held-out CUB test split. No training or validation images were used.", "",
             "### Top 10 classes by F1", "", "| Class | F1 | Support |", "|---|---:|---:|"]
    lines += [f"| {name} | {v['f1-score']:.4f} | {int(v['support'])} |" for name, v in top]
    lines += ["", "### Bottom 10 classes by F1", "", "| Class | F1 | Support |", "|---|---:|---:|"]
    lines += [f"| {name} | {v['f1-score']:.4f} | {int(v['support'])} |" for name, v in bottom]
    lines += ["", "### Most frequent confusions", "", "| True class | Predicted class | Count |", "|---|---|---:|"]
    lines += [f"| {true} | {pred} | {count} |" for count, true, pred in pairs]
    lines += ["", "The dominant measured error groups are visually related seabirds: terns (especially Forster's Tern), gulls (especially California Gull), cormorants, and hummingbirds. Other repeated pairs include tanagers, vireos, sparrows, and albatross/gull images.", "",
              "## Grad-CAM quality review", "",
              "All eight panels were inspected. Each has three labeled columns—Original, Grad-CAM heatmap, and Overlay—and a title containing true class, predicted class, and confidence. Correct panels use green titles; wrong panels use red titles.", "",
              "Visual interpretation: most reviewed heatmaps are nearly uniform dark blue, so they provide weak localization evidence. One wrong example shows a concentrated hotspot in the lower-left background rather than on the bird, suggesting a background-driven decision. These are qualitative observations, not additional metrics.", "",
              "## Limitations", "",
              "Grad-CAM is a coarse attribution map from the final ResNet convolutional block, not a segmentation mask. The panel sample is four correct and four incorrect predictions, not a statistically representative summary of all 5,794 images. Class-name similarity judgments are visual/domain interpretations of the measured confusion pairs.", ""]
    (output_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return {"top": top, "bottom": bottom, "pairs": pairs}


if __name__ == "__main__":
    analyze_p5()
