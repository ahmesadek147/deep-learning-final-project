"""Run the P3 caption EDA + augmentation examples and save figures.

    python -m src.data.run_caption_eda
"""
from datasets import load_dataset

from src.data.augmentation import build_train_transform
from src.data.captions import (
    CaptionCleaner, analyze_captions, build_clean_table, save_clean_table, summarize,
)
from src.visualization import plots

OUT = "outputs/figures"


def main():
    train = load_dataset("Multimodal-Fatima/CUB_train", split="train")
    descriptions = train["description"]

    # 1. Analyse the raw captions
    stats = analyze_captions(descriptions)
    print(summarize(stats))
    plots.plot_caption_lengths(stats["raw_lengths"], f"{OUT}/caption_lengths.png")
    plots.plot_top_words(stats["clean_vocab"], 30, f"{OUT}/top_words.png")
    plots.plot_word_frequency_rank(stats["clean_vocab"], f"{OUT}/word_rank_frequency.png")
    plots.plot_captions_per_image(stats["captions_per_image"], f"{OUT}/captions_per_image.png")

    # 2. Clean (fit on train only) and save for the captioning task
    cleaner = CaptionCleaner().fit(descriptions)
    print(f"Typo corrections found: {len(cleaner.corrections)}")
    for wrong, right in list(cleaner.corrections.items())[:20]:
        print(f"  {wrong} -> {right}")
    cleaner.save_corrections("outputs/caption_typo_corrections.json")
    save_clean_table(build_clean_table(train, cleaner), "outputs/clean_captions_train.json")

    # 3. Augmentation examples on a few images
    tf = build_train_transform(erasing_p=0.0)
    for i in (0, 100, 2000):
        plots.plot_augmentation_grid(
            train[i]["image"].convert("RGB"), tf, n=7, save_path=f"{OUT}/augmentation_{i}.png"
        )
    print(f"Figures saved to {OUT}/")


if __name__ == "__main__":
    main()
