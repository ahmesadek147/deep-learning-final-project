from collections import Counter
import hashlib
import io
from datasets import load_dataset
from PIL import Image

def audit_cub_dataset():
    print("Loading CUB-200-2011 dataset from Hugging Face...")
    dataset = load_dataset("Multimodal-Fatima/CUB_train", split="train")
    
    total_samples = len(dataset)
    print(f"Total images loaded: {total_samples}\n")
    
    # 1. Check Class Distribution
    labels = dataset["label"]
    class_counts = Counter(labels)
    print(f"Total unique classes: {len(class_counts)}")
    print(f"Min images in a class: {min(class_counts.values())}")
    print(f"Max images in a class: {max(class_counts.values())}")
    
    # 2. Check Image Sizes, Broken Files, and Duplicates
    print("\nAuditing image dimensions, scanning for errors, and checking for duplicates...")
    widths = []
    heights = []
    broken_images = 0
    image_hashes = set()
    duplicates = 0
    
    for i in range(total_samples):
        try:
            img = dataset[i]["image"]
            if isinstance(img, str):
                img = Image.open(img)
            
            # Track dimensions
            w, h = img.size
            widths.append(w)
            heights.append(h)
            
            # Check for exact duplicates using an MD5 hash of the image bytes
            buffer = io.BytesIO()
            # Default to JPEG if format isn't specified
            img_format = img.format if img.format else "JPEG"
            img.save(buffer, format=img_format)
            img_bytes = buffer.getvalue()
            
            file_hash = hashlib.md5(img_bytes).hexdigest()
            if file_hash in image_hashes:
                duplicates += 1
            else:
                image_hashes.add(file_hash)
                
        except Exception as e:
            broken_images += 1
            print(f"Error loading image at index {i}: {e}")
            
    print(f"\n--- Audit Summary ---")
    print(f"Scan complete!")
    print(f"Broken or corrupted images found: {broken_images}")
    print(f"Duplicate images found: {duplicates}")
    print(f"Average Image Width: {sum(widths) / len(widths):.2f} pixels")
    print(f"Average Image Height: {sum(heights) / len(heights):.2f} pixels")

if __name__ == "__main__":
    audit_cub_dataset()