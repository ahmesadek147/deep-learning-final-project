"""Augmentation strategy for CUB-200-2011 (Task 1.1).

Design choices (explain these in the report):
- RandomResizedCrop: birds appear at different scales and positions; scale
  is kept >= 0.6 so the bird is rarely cropped out of the frame.
- RandomHorizontalFlip: a bird facing left or right is the same species.
- NO vertical flip: birds are almost never upside down in the test data.
- Small RandomRotation (<= 15 deg): mimics tilted cameras / perched birds.
- ColorJitter with brightness/contrast/saturation only and a very small hue
  shift: plumage colour identifies the species, so large hue changes would
  corrupt the label.
- RandomErasing (optional): forces the network to use several body parts
  instead of one discriminative patch.

Validation / test data are never augmented (only resize + normalise).
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int = 224,
    crop_scale: tuple[float, float] = (0.6, 1.0),
    flip_p: float = 0.5,
    rotation: float = 15.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.02,
    erasing_p: float = 0.0,
    normalize: bool = True,
) -> transforms.Compose:
    ops = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale, ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.RandomRotation(rotation),
        transforms.ColorJitter(brightness, contrast, saturation, hue),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    if erasing_p > 0:
        ops.append(transforms.RandomErasing(p=erasing_p, scale=(0.02, 0.15)))
    return transforms.Compose(ops)


#: Fraction of the shorter side kept by the center crop, i.e. resize to
#: ``image_size / CENTER_CROP_RATIO`` then crop. 224/256 is the ImageNet default.
CENTER_CROP_RATIO = 0.875


def build_eval_transform(image_size: int = 224, normalize: bool = True,
                         resize_mode: str = "squash") -> transforms.Compose:
    """Deterministic validation/test preprocessing - never augmented.

    ``resize_mode`` selects the geometry:

    - ``"squash"`` (default): ``Resize((image_size, image_size))``. Distorts
      aspect ratio - a 500x375 photo becomes square - which does NOT match the
      aspect-preserving ``RandomResizedCrop`` used in training. Kept as the
      default because every committed metric was measured with it; changing it
      silently would make the reported numbers unreproducible.
    - ``"center_crop"``: resize the shorter side to ``image_size / 0.875`` and
      center-crop, preserving aspect ratio. This is the standard pairing for
      ``RandomResizedCrop`` training and is the recommended setting for new
      runs. See ``scripts/run_eval_geometry_ablation.sh`` for the measured
      comparison on an existing checkpoint.
    """
    if resize_mode == "squash":
        ops = [transforms.Resize((image_size, image_size))]
    elif resize_mode == "center_crop":
        ops = [transforms.Resize(int(round(image_size / CENTER_CROP_RATIO))),
               transforms.CenterCrop(image_size)]
    else:
        raise ValueError(f"resize_mode must be 'squash' or 'center_crop', got {resize_mode!r}")
    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def build_train_transform_from_config(cfg: dict | None, image_size: int = 224) -> transforms.Compose:
    """Build the train transform from the ``augmentation`` section of a YAML config.

    Example YAML:
        augmentation:
          crop_scale: [0.6, 1.0]
          flip_p: 0.5
          rotation: 15
          hue: 0.02
          erasing_p: 0.1
    """
    cfg = dict(cfg or {})
    if "crop_scale" in cfg:
        cfg["crop_scale"] = tuple(cfg["crop_scale"])
    return build_train_transform(image_size=image_size, **cfg)


def denormalize(t: torch.Tensor) -> np.ndarray:
    """Normalised CxHxW tensor -> HxWx3 float array in [0, 1] for plotting."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (t.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
