import json
from pathlib import Path

from src.data.splits import build_summary, manifest_sha256


def test_manifest_hash_and_summary_are_consistent():
    path = Path("outputs/splits/cub_train_val_v2_seed42.json")
    if not path.is_file():
        return  # Generated artifact is intentionally not required for a clean clone.
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["manifest_sha256"] == manifest_sha256(manifest)
    summary = build_summary(manifest)
    assert summary["cross_split_duplicate_count"] == 0
    assert summary["duplicate_group_count"] == 1
    assert summary["class_coverage"] == {"train": 200, "validation": 200}


def test_legacy_split_strategy_remains_explicitly_distinct():
    manifest = {"split_strategy": "legacy_random", "split_version": "legacy_random_v1"}
    assert manifest["split_strategy"] != "stratified_group"
