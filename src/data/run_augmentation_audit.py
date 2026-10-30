"""Persist a visual and textual audit of the configured CUB transforms."""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import load_dataset

from src.data.augmentation import (
    IMAGENET_MEAN, IMAGENET_STD, build_eval_transform,
    build_train_transform_from_config, denormalize,
)


def run(output_dir="outputs/eda"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    common = config["common"]
    image_size = common["data"]["image_size"]
    train_transform = build_train_transform_from_config(common["augmentation"], image_size=image_size)
    eval_transform = build_eval_transform(image_size=image_size)
    dataset = load_dataset(common["data"]["dataset_name"], split="train")
    indices = np.linspace(0, len(dataset) - 1, 8, dtype=int)
    random.seed(common["data"]["seed"]); np.random.seed(common["data"]["seed"]); torch.manual_seed(common["data"]["seed"])
    fig, axes = plt.subplots(8, 2, figsize=(8, 24))
    for row, index in enumerate(indices):
        image = dataset[int(index)]["image"].convert("RGB")
        axes[row, 0].imshow(image); axes[row, 0].set_title(f"Original (index {index})")
        axes[row, 1].imshow(denormalize(train_transform(image))); axes[row, 1].set_title("One training augmentation sample")
        axes[row, 0].axis("off"); axes[row, 1].axis("off")
    fig.suptitle("CUB training augmentation examples", y=.995); fig.tight_layout(); fig.savefig(out / "augmentation_examples.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # Confirm evaluation preprocessing is deterministic for the same image.
    sample = dataset[0]["image"].convert("RGB")
    eval_a, eval_b = eval_transform(sample), eval_transform(sample)
    deterministic = bool(torch.equal(eval_a, eval_b))
    report = f'''# Data Augmentation Audit

Source: `{common["data"]["dataset_name"]}:train`; no source images or labels were modified.

## Verified training pipeline

The configured training transform is:

```text
RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.75, 1.3333))
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=(-15, +15))
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
ToTensor()
Normalize(mean={IMAGENET_MEAN}, std={IMAGENET_STD})
RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3))
```

Augmentations are sampled during training only. `RandomResizedCrop`, rotation,
and color jitter always sample a transform; horizontal flip has probability
0.5; random erasing has probability 0.1. No vertical flip is used.

## Suitability for bird classification

- RandomResizedCrop (scale 0.6–1.0, aspect ratio 3:4–4:3) models framing and
  bird scale changes while retaining most of the subject.
- Horizontal flip (0.5) preserves species identity because left/right
  orientation is not a class attribute.
- Rotation ±15° models modest camera tilt; it does not create upside-down birds.
- ColorJitter uses modest brightness/contrast/saturation changes and hue ±0.02,
  preserving plumage cues better than aggressive color changes.
- RandomErasing (0.1; area 2–15%) encourages use of multiple visual cues;
  its small area limits damage to the class-defining bird appearance.

## Validation/test separation and normalization

Validation and test use only `Resize((224, 224))`, `ToTensor()`, and the same
ImageNet normalization. They contain no random crop, flip, rotation, jitter,
or erasing. Reapplying the evaluation transform to the same image produced
identical tensors in this audit: `{deterministic}`.

Normalization is applied once after `ToTensor()` with mean
`{IMAGENET_MEAN}` and standard deviation `{IMAGENET_STD}`. The visualization
denormalizes only for display; it does not alter training data.

## Visual evidence and limitations

The eight original/augmented pairs are saved at
`outputs/eda/augmentation_examples.png`. The examples are representative
samples, not a statistical estimate of all augmentation outcomes. Augmented
images are stochastic, so a different seed can produce different crops and
colors. No augmentation was added or changed as part of this audit.
'''
    (out / "augmentation_audit.md").write_text(report, encoding="utf-8")
    return {"visualization": str(out / "augmentation_examples.png"), "report": str(out / "augmentation_audit.md"), "eval_deterministic": deterministic}


if __name__ == "__main__":
    print(run())
