import torch
from src.data.augmentation import build_train_transform, build_eval_transform
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

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

def get_dataloaders(batch_size=32, val_split=0.2):
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