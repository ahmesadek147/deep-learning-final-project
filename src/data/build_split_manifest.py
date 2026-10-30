"""Generate the immutable duplicate-safe CUB train/validation manifest."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from src.data.splits import build_split_manifest, build_summary, save_manifest


def main():
    dataset_name = "Multimodal-Fatima/CUB_train"
    dataset = load_dataset(dataset_name, split="train")
    manifest = build_split_manifest(dataset, dataset_name, seed=42, val_split=0.2, version="v2")
    path = Path("outputs/splits/cub_train_val_v2_seed42.json")
    save_manifest(manifest, path)
    summary = build_summary(manifest)
    Path("outputs/splits/cub_train_val_v2_seed42_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
