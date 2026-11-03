"""Dataset-loading tests.

The CUB splits are downloaded from Hugging Face, so these tests are skipped
automatically when the dataset is not reachable (offline CI, no cache). The
module must never download anything at import time: pytest imports every test
file during collection, so a module-level download makes the *entire* suite
fail when the network is unavailable.
"""
from __future__ import annotations

import pytest

TRAIN_DATASET = "Multimodal-Fatima/CUB_train"
TEST_DATASET = "Multimodal-Fatima/CUB_test"

EXPECTED_TRAIN_IMAGES = 5994
EXPECTED_TEST_IMAGES = 5794
EXPECTED_CLASSES = 200
EXPECTED_CAPTIONS_PER_IMAGE = 10


def _load(dataset_name: str, split: str):
    """Load a split, or skip the test when the dataset cannot be reached."""
    datasets = pytest.importorskip("datasets")
    try:
        return datasets.load_dataset(dataset_name, split=split)
    except Exception as error:  # offline, no cache, auth or hub failure
        pytest.skip(f"{dataset_name}:{split} unavailable ({type(error).__name__}: {error})")


@pytest.fixture(scope="module")
def train_split():
    return _load(TRAIN_DATASET, "train")


def test_train_split_has_the_documented_size_and_classes(train_split):
    assert len(train_split) == EXPECTED_TRAIN_IMAGES
    assert len(train_split.features["label"].names) == EXPECTED_CLASSES


def test_train_records_expose_every_field_the_project_uses(train_split):
    record = train_split[0]
    assert {"image", "label", "description", "file_name"} <= set(record)
    assert record["image"].size[0] > 0 and record["image"].size[1] > 0
    assert isinstance(record["label"], int)


def test_description_column_holds_ten_newline_separated_captions(train_split):
    from src.data.captions import split_captions

    for index in (0, 1, 2):
        assert len(split_captions(train_split[index]["description"])) == EXPECTED_CAPTIONS_PER_IMAGE


def test_test_split_shares_the_training_class_mapping():
    train_names = _load(TRAIN_DATASET, "train").features["label"].names
    test_names = _load(TEST_DATASET, "test").features["label"].names
    assert list(train_names) == list(test_names)


def test_test_split_has_the_documented_size():
    assert len(_load(TEST_DATASET, "test")) == EXPECTED_TEST_IMAGES


def test_dataloaders_yield_normalized_batches_of_the_configured_size():
    import torch

    from src.dataset import get_dataloaders

    _load(TRAIN_DATASET, "train")  # skips here when the dataset is unavailable
    train_loader, val_loader = get_dataloaders(batch_size=4, max_train_samples=8, max_val_samples=8)
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32
    assert len(val_loader.dataset) == 8
