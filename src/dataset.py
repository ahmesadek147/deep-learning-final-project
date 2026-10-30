import json
from pathlib import Path

import torch
from src.data.augmentation import build_eval_transform, build_train_transform_from_config
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from src.data.splits import validate_manifest

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
    dataset_name="Multimodal-Fatima/CUB_train", batch_size=32, val_split=0.2, seed=42, image_size=224, num_workers=0,
    pin_memory=False, augmentation_config=None, max_train_samples=None, max_val_samples=None,
    split_strategy="legacy_random", split_version="legacy_random_v1", split_manifest=None,
):
    """
    Loads dataset, creates train/validation split, and returns PyTorch DataLoaders.
    """
    print("Loading CUB-200-2011 dataset...")
    raw_dataset = load_dataset(dataset_name, split="train")
    
    # Augmentation for training; deterministic resize + normalize for validation
    train_transform = build_train_transform_from_config(augmentation_config, image_size=image_size)
    val_transform = build_eval_transform(image_size=image_size)
    
    # Calculate split sizes
    total_size = len(raw_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    print(f"Splitting dataset: {train_size} training samples, {val_size} validation samples.")
    
    if split_strategy in {"legacy_random", "random"}:
        # Preserve the historical behavior exactly for old experiments.
        train_data_raw, val_data_raw = random_split(
            raw_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
    elif split_strategy == "stratified_group":
        if not split_manifest:
            raise ValueError("split_manifest is required for stratified_group splitting.")
        manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        validate_manifest(manifest, raw_dataset, verify_hashes=False)
        if manifest.get("split_version") != split_version:
            raise ValueError("Configured split_version does not match the manifest.")
        train_indices = [row["index"] for row in manifest["samples"] if row["split"] == "train"]
        val_indices = [row["index"] for row in manifest["samples"] if row["split"] == "validation"]
        train_data_raw = torch.utils.data.Subset(raw_dataset, train_indices)
        val_data_raw = torch.utils.data.Subset(raw_dataset, val_indices)
        print(f"Using validated split manifest {split_manifest}: {len(train_indices)} training, {len(val_indices)} validation.")
    else:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    # Limits are intended for explicit smoke runs; they are never a test split.
    if max_train_samples is not None:
        train_data_raw = torch.utils.data.Subset(train_data_raw, range(min(max_train_samples, len(train_data_raw))))
    if max_val_samples is not None:
        val_data_raw = torch.utils.data.Subset(val_data_raw, range(min(max_val_samples, len(val_data_raw))))
    
    # Wrap the split subsets in the CUBDataset class
    train_dataset = CUBDataset(train_data_raw, transform=train_transform)
    val_dataset = CUBDataset(val_data_raw, transform=val_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader


def get_test_dataloader(
    dataset_name="Multimodal-Fatima/CUB_test",
    split="test",
    batch_size=32,
    image_size=224,
    num_workers=0,
    pin_memory=False,
):
    """Build a loader for the untouched test split.

    This intentionally does not call ``get_dataloaders`` and never applies the
    training/validation split.  The caller can therefore use it only after
    model selection is complete.
    """
    raw_dataset = load_dataset(dataset_name, split=split)
    dataset = CUBDataset(raw_dataset, transform=build_eval_transform(image_size=image_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


def get_class_names(dataset_name, split):
    """Return the dataset's canonical ClassLabel names without changing data."""
    raw_dataset = load_dataset(dataset_name, split=split)
    feature = raw_dataset.features.get("label")
    names = getattr(feature, "names", None)
    if not names:
        raise ValueError(f"Dataset {dataset_name}:{split} has no ClassLabel names.")
    return list(names)

if __name__ == "__main__":
    # Sanity check to test the loaders
    train_loader, val_loader = get_dataloaders(batch_size=16)
    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch label tensor shape: {labels.shape}")
