"""Matplotlib figures for caption EDA, augmentation and Grad-CAM."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works on servers / Colab without a display
import matplotlib.pyplot as plt
import numpy as np

from src.data.augmentation import denormalize
from src.visualization.gradcam import overlay_heatmap


def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_caption_lengths(lengths, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=range(0, max(lengths) + 2), color="steelblue", edgecolor="white")
    ax.axvline(np.mean(lengths), color="red", ls="--", label=f"mean = {np.mean(lengths):.1f}")
    ax.set(xlabel="Words per caption", ylabel="Number of captions", title="Caption length distribution")
    ax.legend()
    return _save(fig, save_path)


def plot_top_words(vocab, k=30, save_path=None):
    words, counts = zip(*vocab.most_common(k))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(words[::-1], counts[::-1], color="seagreen")
    ax.set(xlabel="Frequency", title=f"Top {k} words (after cleaning)")
    return _save(fig, save_path)


def plot_word_frequency_rank(vocab, save_path=None):
    """Rank-frequency plot (log-log) showing the long tail of rare words / typos."""
    counts = sorted(vocab.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(range(1, len(counts) + 1), counts)
    ax.set(xlabel="Rank", ylabel="Frequency", title="Word frequency vs rank")
    return _save(fig, save_path)


def plot_captions_per_image(captions_per_image: dict, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([str(k) for k in captions_per_image], list(captions_per_image.values()), color="darkorange")
    ax.set(xlabel="Captions in description", ylabel="Images", title="Captions per image")
    return _save(fig, save_path)


def plot_augmentation_grid(pil_image, transform, n=7, save_path=None):
    """Original image followed by ``n`` random augmentations of it.

    ``transform`` should be built with ``normalize=True`` (default); the
    images are de-normalised before display.
    """
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
    axes[0].imshow(pil_image)
    axes[0].set_title("Original")
    for ax in axes[1:]:
        ax.imshow(denormalize(transform(pil_image)))
        ax.set_title("Augmented")
    for ax in axes:
        ax.axis("off")
    return _save(fig, save_path)


def plot_gradcam_grid(examples: dict, class_names=None, save_path=None):
    """Two blocks (correct / wrong); each column shows image and heatmap overlay."""
    name = (lambda i: class_names[i]) if class_names else str
    rows = [(k, examples[k]) for k in ("correct", "wrong") if examples[k]]
    ncols = max(len(v) for _, v in rows)
    fig, axes = plt.subplots(2 * len(rows), ncols, figsize=(3 * ncols, 6 * len(rows) / 1.2), squeeze=False)
    for r, (kind, items) in enumerate(rows):
        for c in range(ncols):
            for a in (axes[2 * r][c], axes[2 * r + 1][c]):
                a.axis("off")
            if c >= len(items):
                continue
            it = items[c]
            axes[2 * r][c].imshow(it["image"])
            axes[2 * r + 1][c].imshow(overlay_heatmap(it["image"], it["cam"]))
            color = "green" if kind == "correct" else "red"
            axes[2 * r][c].set_title(
                f"True: {name(it['true'])}\nPred: {name(it['pred'])} ({it['conf']:.0%})",
                fontsize=8, color=color,
            )
    fig.suptitle("Grad-CAM: correct predictions (top) vs wrong predictions (bottom)")
    fig.tight_layout()
    return _save(fig, save_path)
