"""Grad-CAM (Selvaraju et al., 2017) for any CNN classifier in PyTorch.

Works with the from-scratch CNN and with pretrained models (ResNet, etc.):
pass the last convolutional layer as ``target_layer`` or let
``find_target_layer`` pick it.

Algorithm:
1. Forward pass; keep the feature maps A^k of the target layer.
2. Backward pass from the score of the chosen class; keep dScore/dA^k.
3. Channel weights alpha_k = global average of the gradients over H and W.
4. CAM = ReLU( sum_k alpha_k * A^k ), upsampled to the input size, scaled to [0, 1].
"""
from __future__ import annotations

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.augmentation import denormalize


def find_target_layer(model: nn.Module) -> nn.Module:
    """Pick a sensible default target layer: ``layer4[-1]`` for ResNets,
    otherwise the last ``nn.Conv2d`` in the model."""
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No Conv2d layer found; pass target_layer explicitly.")
    return last


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or find_target_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        # Tensor hook: robust to in-place ReLUs and frozen backbones
        output.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def remove_hooks(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove_hooks()

    def generate(self, x: torch.Tensor, class_idx=None):
        """Compute Grad-CAM for a batch.

        Args:
            x: (B, C, H, W) normalised input batch.
            class_idx: None (use the predicted class), an int, or a (B,) tensor.

        Returns:
            cams:  (B, H, W) numpy array in [0, 1]
            preds: (B,) predicted class ids
            probs: (B,) softmax probability of the predicted class
        """
        was_training = self.model.training
        self.model.eval()
        # requires_grad on the input guarantees a graph even if the backbone is frozen
        x = x.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            if class_idx is None:
                target = preds
            elif isinstance(class_idx, int):
                target = torch.full_like(preds, class_idx)
            else:
                target = class_idx.to(logits.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, target[:, None]).sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # alpha_k
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        mn, mx = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
        cam = (cam - mn) / (mx - mn + 1e-8)

        self.model.train(was_training)
        return (
            cam.cpu().numpy(),
            preds.detach().cpu(),
            probs.gather(1, preds[:, None]).squeeze(1).detach().cpu(),
        )


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a jet heatmap over an HxWx3 float image in [0, 1]."""
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def collect_examples(
    model: nn.Module,
    loader,
    device,
    n_correct: int = 4,
    n_wrong: int = 4,
    target_layer: nn.Module | None = None,
) -> dict:
    """Run Grad-CAM on validation batches until enough correct and wrong
    predictions are collected. Returns {"correct": [...], "wrong": [...]}."""
    model.to(device)
    out = {"correct": [], "wrong": []}
    with GradCAM(model, target_layer) as cam_fn:
        for images, labels in loader:
            images = images.to(device)
            cams, preds, confs = cam_fn.generate(images)
            for i in range(images.size(0)):
                key = "correct" if preds[i].item() == labels[i].item() else "wrong"
                limit = n_correct if key == "correct" else n_wrong
                if len(out[key]) < limit:
                    out[key].append({
                        "image": denormalize(images[i]),
                        "cam": cams[i],
                        "true": int(labels[i]),
                        "pred": int(preds[i]),
                        "conf": float(confs[i]),
                    })
            if len(out["correct"]) >= n_correct and len(out["wrong"]) >= n_wrong:
                break
    return out
