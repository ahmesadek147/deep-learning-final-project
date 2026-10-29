import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolution block used by the custom CNN."""

    def __init__(
        self,
        in_channels,
        out_channels,
        use_batch_norm=False
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batch_norm
            )
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.extend([
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ScratchCNN(nn.Module):
    """
    Custom CNN trained completely from scratch.

    Supported variants:
    - basic: four convolution blocks without batch normalization
    - batchnorm: four convolution blocks with batch normalization
    - deep: five convolution blocks with batch normalization
    """

    VALID_VARIANTS = {"basic", "batchnorm", "deep"}

    def __init__(
        self,
        num_classes=200,
        variant="basic",
        dropout=0.5
    ):
        super().__init__()

        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Expected one of {sorted(self.VALID_VARIANTS)}."
            )

        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be greater than or equal to 0 "
                "and less than 1."
            )

        self.num_classes = num_classes
        self.variant = variant
        self.dropout_rate = dropout

        use_batch_norm = variant in {"batchnorm", "deep"}

        if variant == "deep":
            channels = [32, 64, 128, 256, 512]
        else:
            channels = [32, 64, 128, 256]

        blocks = []
        in_channels = 3

        for out_channels in channels:
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_batch_norm=use_batch_norm
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all model weights without pretrained parameters."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """
        Extract one feature vector for each input image.

        This method allows the model to be reused later as an encoder.
        """

        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        return x

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)

        return logits

    def get_gradcam_target_layer(self):
        """Return the final convolution layer for Grad-CAM."""

        final_block = self.features[-1].block

        for layer in final_block:
            if isinstance(layer, nn.Conv2d):
                return layer

        raise RuntimeError("No convolution layer found.")

    def count_trainable_parameters(self):
        """Return the number of trainable model parameters."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )