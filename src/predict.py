"""Inference from a metadata-bearing experiment checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.augmentation import build_eval_transform
from src.training.run_training import build_model, select_device


def predict_image(image_path, task="classification", checkpoint_path=None, device="auto", model_config=None):
    if task != "classification":
        raise NotImplementedError(f"Prediction for {task} is not implemented yet.")
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required; choose a trial checkpoint explicitly.")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    resolved_model_config = metadata.get("model_config") or model_config
    if not resolved_model_config:
        raise ValueError("Checkpoint has no model metadata; provide model_config explicitly.")
    model = build_model(resolved_model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_device = select_device(device)
    model.to(target_device).eval()
    transform = build_eval_transform(image_size=metadata.get("image_size", 224))
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(target_device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)
        confidence, predicted_class = probabilities.max(dim=1)
    return {"class_id": predicted_class.item(), "confidence": confidence.item(),
            "checkpoint": str(checkpoint_path), "device": str(target_device)}
