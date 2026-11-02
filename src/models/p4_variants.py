"""P4-only custom CNN variants. No pretrained weights or shared-module edits."""
from __future__ import annotations

import torch
from torch import nn


def _conv_norm(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class _DelayedStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool):
        super().__init__()
        self.conv1 = _conv_norm(in_channels, out_channels)
        self.conv2 = _conv_norm(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.conv2(self.relu(self.conv1(x)))))


class _ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool):
        super().__init__()
        self.conv1 = _conv_norm(in_channels, out_channels)
        self.conv2 = _conv_norm(out_channels, out_channels)
        self.skip = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels))
                     if in_channels != out_channels else nn.Identity())
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        return self.pool(self.relu(self.conv2(self.relu(self.conv1(x))) + identity))


class P4ScratchCNN(nn.Module):
    """Delayed-pooling or residual CNN, trained completely from scratch.

    Depth is a count of two-convolution stages. Channel counts double from
    ``width`` until ``max_channels``. The first stage preserves resolution;
    successive stages downsample after their two convolutions.
    """

    def __init__(self, variant: str, num_classes: int = 200, depth: int = 4,
                 width: int = 32, max_channels: int = 384,
                 dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        if variant not in {"delayed_pool", "residual"}:
            raise ValueError("variant must be delayed_pool or residual")
        if pretrained is not False:
            raise ValueError("P4 scratch models cannot use pretrained weights")
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if type(depth) is not int or not 3 <= depth <= 5:
            raise ValueError("depth must be between 3 and 5 stages")
        if type(width) is not int or width < 8:
            raise ValueError("width must be an integer >= 8")
        if type(max_channels) is not int or max_channels < width:
            raise ValueError("max_channels must be >= width")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.variant, self.depth, self.width = variant, depth, width
        self.num_classes = num_classes
        self.channels = [min(width * 2 ** i, max_channels) for i in range(depth)]
        self.feature_dim = self.channels[-1]
        stage_type = _ResidualStage if variant == "residual" else _DelayedStage
        in_channels = 3
        stages = []
        for i, out_channels in enumerate(self.channels):
            stages.append(stage_type(in_channels, out_channels, pool=i > 0))
            in_channels = out_channels
        self.features = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[2:]) < 2 ** (self.depth - 1):
            raise ValueError("Expected RGB BCHW input of sufficient size")
        return self.features(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_pool(self.forward_feature_maps(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def get_gradcam_target_layer(self) -> nn.Conv2d:
        return self.features[-1].conv2[0]

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
