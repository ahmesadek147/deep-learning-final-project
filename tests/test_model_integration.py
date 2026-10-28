import pytest
import torch
from torch import nn

from src.models import ScratchCNN
from src.training.run_training import build_model


def test_build_scratch_cnn_from_config():
    config = {
        "name": "scratch_cnn",
        "num_classes": 5,
        "variant": "basic",
        "dropout": 0.5,
    }

    model = build_model(config)
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)

    assert isinstance(model, ScratchCNN)
    assert model.variant == "basic"
    assert model.num_classes == 5
    assert logits.shape == (2, 5)


def test_build_batchnorm_variant_from_config():
    config = {
        "name": "scratch_cnn",
        "num_classes": 200,
        "variant": "batchnorm",
        "dropout": 0.3,
    }

    model = build_model(config)

    assert isinstance(model, ScratchCNN)
    assert model.variant == "batchnorm"
    assert model.num_classes == 200
    assert model.dropout_rate == 0.3


def test_build_existing_simple_model():
    config = {
        "name": "simple_cnn",
    }

    model = build_model(config)
    images = torch.randn(2, 3, 32, 32)
    logits = model(images)

    assert isinstance(model, nn.Sequential)
    assert logits.shape == (2, 5)


def test_unsupported_model_raises_error():
    config = {
        "name": "unsupported_model",
    }

    with pytest.raises(
        ValueError,
        match="Unsupported model: unsupported_model",
    ):
        build_model(config)