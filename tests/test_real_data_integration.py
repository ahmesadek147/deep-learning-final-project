from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import ScratchCNN
from src.training.run_training import (
    build_model,
    validate_classification_batch,
)


def test_real_classification_configuration_builds_200_class_model():
    config = {
        "name": "scratch_cnn",
        "num_classes": 200,
        "variant": "basic",
        "dropout": 0.5,
    }

    model = build_model(config)
    images = torch.randn(2, 3, 224, 224)
    logits = model(images)

    assert isinstance(model, ScratchCNN)
    assert logits.shape == (2, 200)


def test_valid_real_classification_batch():
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor(
        [0, 45, 100, 199],
        dtype=torch.long,
    )

    validate_classification_batch(
        images=images,
        labels=labels,
        num_classes=200,
        image_size=224,
    )


def test_invalid_label_range_is_rejected():
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor(
        [0, 200],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="between 0 and 199"):
        validate_classification_batch(
            images=images,
            labels=labels,
            num_classes=200,
            image_size=224,
        )


def test_non_long_labels_are_rejected():
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float32,
    )

    with pytest.raises(TypeError, match="torch.long"):
        validate_classification_batch(
            images=images,
            labels=labels,
            num_classes=200,
            image_size=224,
        )


def test_cross_entropy_forward_and_backward_with_200_classes():
    model = ScratchCNN(
        num_classes=200,
        variant="basic",
        dropout=0.5,
    )

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor(
        [0, 199],
        dtype=torch.long,
    )

    logits = model(images)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert logits.shape == (2, 200)
    assert torch.isfinite(loss)