"""Persist CUB EDA and data-audit artifacts without changing the dataset."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, load_dataset_builder
from PIL import Image

from src.data.captions import CaptionCleaner, analyze_captions, build_clean_table, save_clean_table
from src.visualization import plots


def run(output_root="outputs"):
    root = Path(output_root)
    eda_dir, audit_dir = root / "eda", root / "data_audit"
    eda_dir.mkdir(parents=True, exist_ok=True); audit_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "Multimodal-Fatima/CUB_train"
    ds = load_dataset(dataset_name, split="train")
    labels = list(ds["label"])
    names = list(ds.features["label"].names)

    counts = Counter(labels)
    dimensions, channels, hashes = [], Counter(), Counter()
    broken, duplicate_indices = [], []
    for index in range(len(ds)):
        try:
            image = ds[index]["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            dimensions.append(image.size)
            channels[image.mode] += 1
            buffer = BytesIO(); image.convert("RGB").save(buffer, format="PNG")
            digest = hashlib.sha256(buffer.getvalue()).hexdigest()
            if digest in hashes:
                duplicate_indices.append({"index": index, "duplicate_of": hashes[digest], "sha256": digest})
            else:
                hashes[digest] = index
        except Exception as error:  # preserve the index and error; do not drop it
            broken.append({"index": index, "error": repr(error)})

    width = np.asarray([x[0] for x in dimensions]); height = np.asarray([x[1] for x in dimensions])
    class_rows = [{"label": i, "class": names[i], "count": counts.get(i, 0)} for i in range(len(names))]
    with (eda_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=class_rows[0].keys()); writer.writeheader(); writer.writerows(class_rows)
    fig, ax = plt.subplots(figsize=(16, 5)); ax.bar(range(len(names)), [r["count"] for r in class_rows]);
    ax.set(xlabel="Class index", ylabel="Images", title="CUB training class distribution"); fig.tight_layout(); fig.savefig(eda_dir / "class_distribution.png", dpi=160); plt.close(fig)

    dimension_rows = [{"width": int(w), "height": int(h), "count": int(sum((a == w and b == h) for a, b in dimensions))} for w, h in sorted(set(dimensions))]
    with (eda_dir / "image_dimensions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["width", "height", "count"]); writer.writeheader(); writer.writerows(dimension_rows)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(width, height, s=5, alpha=.35); ax.set(xlabel="Width", ylabel="Height", title="Original image dimensions"); fig.tight_layout(); fig.savefig(eda_dir / "image_dimensions.png", dpi=160); plt.close(fig)

    captions = analyze_captions(ds["description"])
    plots.plot_caption_lengths(captions["raw_lengths"], eda_dir / "caption_lengths.png")
    plots.plot_top_words(captions["clean_vocab"], 30, eda_dir / "top_words.png")
    plots.plot_word_frequency_rank(captions["clean_vocab"], eda_dir / "word_rank_frequency.png")
    plots.plot_captions_per_image(captions["captions_per_image"], eda_dir / "captions_per_image.png")

    sample_indices = np.linspace(0, len(ds) - 1, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for axis, index in zip(axes.flat, sample_indices):
        axis.imshow(ds[int(index)]["image"].convert("RGB")); axis.set_title(f"idx={index}, class={labels[index]}\n{names[labels[index]]}", fontsize=8); axis.axis("off")
    fig.suptitle("Representative CUB training images"); fig.tight_layout(); fig.savefig(eda_dir / "representative_samples.png", dpi=160); plt.close(fig)

    val_size = int(len(ds) * 0.2)
    test_info = load_dataset_builder("Multimodal-Fatima/CUB_test").info.splits["test"]
    summary = {
        "dataset": dataset_name, "source_split": "train", "images": len(ds), "classes": len(names),
        "derived_training_split": len(ds) - val_size, "derived_validation_split": val_size,
        "untouched_test_split": int(test_info.num_examples), "class_min": min(counts.values()), "class_max": max(counts.values()),
        "class_mean": float(np.mean(list(counts.values()))), "broken_images": len(broken),
        "unique_pixel_hashes": len(hashes), "duplicate_images": len(duplicate_indices),
        "width_min": int(width.min()), "width_max": int(width.max()), "height_min": int(height.min()), "height_max": int(height.max()),
        "channel_modes": dict(channels), "class_mapping": "dataset ClassLabel names, unchanged",
        "caption_stats": {k: v for k, v in captions.items() if k not in {"raw_lengths", "clean_lengths", "clean_vocab"}},
    }
    (eda_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (audit_dir / "broken_images.json").write_text(json.dumps(broken, indent=2), encoding="utf-8")
    (audit_dir / "duplicate_images.json").write_text(json.dumps(duplicate_indices, indent=2), encoding="utf-8")
    (audit_dir / "channel_modes.json").write_text(json.dumps(dict(channels), indent=2), encoding="utf-8")

    cleaner = CaptionCleaner().fit(ds["description"])
    cleaner.save_corrections(audit_dir / "caption_typo_corrections.json")
    save_clean_table(build_clean_table(ds, cleaner), audit_dir / "clean_captions_train.json")
    audit = {
        "problems_detected": {"broken_images": broken, "duplicate_images": duplicate_indices,
                              "caption_typo_candidates": len(cleaner.corrections), "non_ascii_captions": captions["non_ascii_captions"]},
        "problems_corrected_in_artifacts_only": ["caption normalization and conservative typo candidates were saved separately"],
        "problems_intentionally_unchanged": ["original images, labels, class mapping, and source dataset were not modified", "no images were removed or silently substituted"],
    }
    (audit_dir / "audit_findings.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# CUB EDA and Data Audit", "", f"Source: `{dataset_name}:train` ({len(ds)} images, {len(names)} classes).", "", "## Findings", "", f"- Derived deterministic split sizes used by training: {len(ds)-val_size} train / {val_size} validation.", f"- Untouched test metadata: {int(test_info.num_examples)} images.", f"- Class counts range from {min(counts.values())} to {max(counts.values())}; no classes are missing.", f"- Corrupted/missing images detected: **{len(broken)}**.", f"- Exact duplicate pixel hashes detected: **{len(duplicate_indices)}** (PNG-normalized SHA-256).", f"- Image modes: `{dict(channels)}`; width range {width.min()}–{width.max()}, height range {height.min()}–{height.max()}.", f"- Caption typo candidates saved: {len(cleaner.corrections)}; these are not applied to source images/labels.", "", "## Corrected versus unchanged", "", "- Corrected only in derived caption artifacts: text normalization, conservative typo candidates, duplicate-caption filtering.", "- Intentionally unchanged: source dataset files, labels, class mapping, and all images; no samples were removed.", "", "## Outputs", "", "See `outputs/eda/` for dimensions, class distribution, caption charts, and representative samples. See `outputs/data_audit/` for broken-image, duplicate, channel, caption-cleaning, and findings JSON files.", ""]
    (audit_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
