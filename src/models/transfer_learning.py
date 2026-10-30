
import torch.nn as nn
from torchvision import models


def build_transfer_model(
    model_name="resnet18",
    num_classes=200,
    pretrained=True,
    freeze_backbone=True,
    dropout=0.2,
):
    """Create an ImageNet-pretrained model for CUB-200 classification."""

    if model_name == "resnet18":
        weights = (
            models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        # Replacement layers are new objects, but make this invariant explicit.
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "efficientnet_b0":
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        for param in model.classifier[1].parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
