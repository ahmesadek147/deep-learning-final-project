import pytest
import torch
from torch import nn

from src.models import ScratchCNN


@pytest.mark.parametrize(
    ("variant", "expected_feature_size"),
    [
        ("basic", 256),
        ("batchnorm", 256),
        ("deep", 512),
    ],
)
def test_output_and_feature_shapes(variant, expected_feature_size):
    model = ScratchCNN(
        num_classes=200,
        variant=variant,
        dropout=0.5,
    )

    images = torch.randn(2, 3, 64, 64)

    logits = model(images)
    features = model.forward_features(images)

    assert logits.shape == (2, 200)
    assert features.shape == (2, expected_feature_size)


def test_backward_pass():
    model = ScratchCNN(
        num_classes=10,
        variant="basic",
        dropout=0.3,
    )

    images = torch.randn(2, 3, 64, 64)
    labels = torch.randint(0, 10, (2,))

    criterion = nn.CrossEntropyLoss()

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    parameters_with_gradients = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    assert torch.isfinite(loss)
    assert len(parameters_with_gradients) > 0


def test_gradcam_target_is_convolution():
    model = ScratchCNN(
        num_classes=200,
        variant="batchnorm",
    )

    target_layer = model.get_gradcam_target_layer()

    assert isinstance(target_layer, nn.Conv2d)


def test_different_image_sizes():
    model = ScratchCNN(
        num_classes=200,
        variant="basic",
    )

    small_images = torch.randn(2, 3, 64, 64)
    large_images = torch.randn(2, 3, 224, 224)

    small_output = model(small_images)
    large_output = model(large_images)

    assert small_output.shape == (2, 200)
    assert large_output.shape == (2, 200)


@pytest.mark.parametrize(
    "invalid_variant",
    ["invalid", "resnet", "", None],
)
def test_invalid_variant_raises_error(invalid_variant):
    with pytest.raises(ValueError):
        ScratchCNN(variant=invalid_variant)


@pytest.mark.parametrize(
    "invalid_dropout",
    [-0.1, 1.0, 1.5],
)
def test_invalid_dropout_raises_error(invalid_dropout):
    with pytest.raises(ValueError):
        ScratchCNN(dropout=invalid_dropout)


@pytest.mark.parametrize(
    "invalid_num_classes",
    [0, -1],
)
def test_invalid_num_classes_raises_error(invalid_num_classes):
    with pytest.raises(ValueError):
        ScratchCNN(num_classes=invalid_num_classes)


def test_parameter_count_is_positive():
    model = ScratchCNN()

    assert model.count_trainable_parameters() > 0