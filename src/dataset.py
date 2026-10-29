import torch
from src.data.augmentation import build_train_transform, build_eval_transform
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split

class CUBDataset(Dataset):
    """
    Custom PyTorch Dataset wrapper for CUB-200-2011 from Hugging Face.
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(
    batch_size=32,
    val_split=0.2,
    image_size=224,
    num_workers=0,
    seed=42,
):
    """
    Load the CUB training dataset and create reproducible train/validation
    DataLoaders.

    The official test set is not loaded by this function.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")

    if image_size <= 0:
        raise ValueError("image_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    print("Loading CUB-200-2011 training dataset...")

    raw_dataset = load_dataset(
        "Multimodal-Fatima/CUB_train",
        split="train",
    )

    train_transform = build_train_transform(
        image_size=image_size
    )
    val_transform = build_eval_transform(
        image_size=image_size
    )

    total_size = len(raw_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    if train_size == 0 or val_size == 0:
        raise ValueError(
            "The selected validation split creates an empty subset."
        )

    print(
        f"Splitting dataset: {train_size} training samples, "
        f"{val_size} validation samples."
    )

    train_data_raw, val_data_raw = random_split(
        raw_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Separate wrappers preserve separate transforms while retaining
    # non-overlapping indices from the reproducible random split.
    train_dataset = CUBDataset(
        train_data_raw,
        transform=train_transform,
    )
    val_dataset = CUBDataset(
        val_data_raw,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader

    """
    Loads dataset, creates train/validation split, and returns PyTorch DataLoaders.
    """
    print("Loading CUB-200-2011 dataset...")
    raw_dataset = load_dataset("Multimodal-Fatima/CUB_train", split="train")
    
    # Augmentation for training; deterministic resize + normalize for validation
    train_transform = build_train_transform()
    val_transform = build_eval_transform()
    
    # Calculate split sizes
    total_size = len(raw_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"Splitting dataset: {train_size} training samples, {val_size} validation samples.")
    
    # Split the raw dataset indices with a fixed seed
    train_data_raw, val_data_raw = random_split(
        raw_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Wrap the split subsets in the CUBDataset class
    train_dataset = CUBDataset(train_data_raw, transform=train_transform)
    val_dataset = CUBDataset(val_data_raw, transform=val_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Sanity check to test the loaders
    train_loader, val_loader = get_dataloaders(batch_size=16)
    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch label tensor shape: {labels.shape}")