
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms


def predict_image(image_path, task):
    if task != "classification":
        raise NotImplementedError(
            f"Prediction for {task} is not implemented yet."
        )

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Recreate the same temporary model used in training
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 5)
    )

    checkpoint_path = Path("outputs/checkpoints/best_model.pth")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load the trained weights
    device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Prepare the input image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

    result = {
        "class_id": predicted_class.item(),
        "confidence": confidence.item()
    }

    return result