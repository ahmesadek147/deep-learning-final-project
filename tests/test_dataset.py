from datasets import load_dataset

print("Loading CUB-200 dataset from Hugging Face...")
# Loading the training split of your dataset
dataset = load_dataset("Multimodal-Fatima/CUB_train", split="train")

print(f"Success! Dataset loaded with {len(dataset)} images.")
print("Sample item keys:", dataset[0].keys())