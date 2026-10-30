"""Deterministic, duplicate-aware train/validation split manifests."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def image_sha256(image: Image.Image) -> str:
    """Hash normalized RGB pixels, independent of source JPEG encoding."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _dataset_fingerprint(dataset) -> str | None:
    return getattr(dataset, "_fingerprint", None)


def _canonical_payload(manifest: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def manifest_sha256(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(manifest)).hexdigest()


def build_split_manifest(dataset, dataset_name: str, seed: int = 42, val_split: float = 0.2,
                         version: str = "v2", n_splits: int = 5) -> dict[str, Any]:
    """Build a stratified group manifest from every image in ``dataset``.

    Every exact normalized-pixel hash is a group. A valid fold must contain all
    classes in both subsets; among valid folds, the closest validation ratio is
    selected deterministically by (distance, fold index).
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    labels = list(dataset["label"])
    names = list(dataset.features["label"].names)
    stable = []
    hashes = []
    for index in range(len(dataset)):
        item = dataset[index]
        image_hash = image_sha256(item["image"])
        stable_id = str(item.get("id", index))
        file_name = str(item.get("file_name", f"index-{index}"))
        stable.append((stable_id, file_name, int(item["label"])))
        hashes.append(image_hash)
    groups = np.asarray(hashes, dtype=object)
    y = np.asarray(labels)
    valid = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        if len(set(y[train_idx])) != len(names) or len(set(y[val_idx])) != len(names):
            continue
        ratio = len(val_idx) / len(y)
        valid.append((abs(ratio - val_split), fold, train_idx.tolist(), val_idx.tolist()))
    if not valid:
        raise ValueError("No stratified group fold contains all classes in both splits.")
    _, selected_fold, train_idx, val_idx = sorted(valid, key=lambda row: (row[0], row[1]))[0]
    assignment = {int(index): "train" for index in train_idx}
    assignment.update({int(index): "validation" for index in val_idx})
    samples = []
    for index, (stable_id, file_name, label) in enumerate(stable):
        samples.append({"index": index, "id": stable_id, "file_name": file_name, "label": label,
                        "image_hash": hashes[index], "duplicate_group_id": hashes[index],
                        "split": assignment[index]})
    counts = {"train": Counter(), "validation": Counter()}
    for row in samples:
        counts[row["split"]][str(row["label"])] += 1
    manifest = {
        "split_strategy": "stratified_group",
        "split_version": version,
        "seed": seed,
        "validation_ratio": val_split,
        "dataset_name": dataset_name,
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "selected_fold": selected_fold,
        "total_samples": len(samples),
        "train_count": len(train_idx),
        "validation_count": len(val_idx),
        "num_classes": len(names),
        "class_names": names,
        "per_class_counts": {"train": dict(counts["train"]), "validation": dict(counts["validation"])},
        "samples": samples,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    validate_manifest(manifest, dataset, verify_hashes=False)
    return manifest


def validate_manifest(manifest: dict[str, Any], dataset, verify_hashes: bool = False) -> None:
    """Validate identifiers, labels, coverage, hash groups, and manifest hash."""
    if manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("Manifest SHA-256 does not match its canonical contents.")
    samples = manifest.get("samples", [])
    if len(samples) != len(dataset) or manifest.get("total_samples") != len(dataset):
        raise ValueError("Manifest sample count does not match the dataset.")
    ids = [str(row.get("id")) for row in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate stable IDs.")
    expected = {}
    for index in range(len(dataset)):
        item = dataset[index]
        expected[str(item.get("id", index))] = (index, str(item.get("file_name", f"index-{index}")), int(item["label"]))
    if set(ids) != set(expected):
        raise ValueError("Manifest IDs do not resolve to the current dataset.")
    for row in samples:
        index, file_name, label = expected[row["id"]]
        if row["index"] != index or row["file_name"] != file_name or int(row["label"]) != label:
            raise ValueError(f"Manifest identifier mismatch for {row['id']}.")
        if verify_hashes and image_sha256(dataset[index]["image"]) != row["image_hash"]:
            raise ValueError(f"Image hash mismatch for {row['id']}.")
    groups = {}
    for row in samples:
        groups.setdefault(row["duplicate_group_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("A duplicate group crosses train and validation.")
    labels = set(range(manifest["num_classes"]))
    for split in ("train", "validation"):
        if {int(row["label"]) for row in samples if row["split"] == split} != labels:
            raise ValueError(f"{split} does not contain every class.")


def save_manifest(manifest: dict[str, Any], path: str | Path) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest["manifest_sha256"]


def build_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    groups = Counter(row["duplicate_group_id"] for row in manifest["samples"])
    duplicate_groups = {group: count for group, count in groups.items() if count > 1}
    return {
        "split_strategy": manifest["split_strategy"], "split_version": manifest["split_version"],
        "seed": manifest["seed"], "validation_ratio_target": manifest["validation_ratio"],
        "actual_validation_ratio": manifest["validation_count"] / manifest["total_samples"],
        "total_samples": manifest["total_samples"], "train_count": manifest["train_count"],
        "validation_count": manifest["validation_count"], "num_classes": manifest["num_classes"],
        "class_coverage": {split: len([label for label, count in manifest["per_class_counts"][split].items() if count > 0])
                           for split in ("train", "validation")},
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_sample_count": sum(duplicate_groups.values()),
        "cross_split_duplicate_count": 0,
        "manifest_sha256": manifest["manifest_sha256"],
    }
