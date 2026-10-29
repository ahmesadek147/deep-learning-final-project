import numpy as np
import torch
from PIL import Image
from torch import nn

from src.data.augmentation import build_eval_transform, build_train_transform
from src.data.captions import CaptionCleaner, analyze_captions, clean_caption, split_captions
from src.visualization.gradcam import GradCAM, collect_examples


def test_split_and_clean():
    desc = "This bird is Brown, with a LIGHTER crest.\n\n  a black-and-white bird! \n"
    caps = split_captions(desc)
    assert len(caps) == 2
    assert clean_caption(caps[0]) == "this bird is brown with a lighter crest"
    assert clean_caption(caps[1]) == "a black and white bird"


def test_cleaner_typo_and_dedup():
    good = "this bird has a brown belly and a short beak.\n" * 25
    descs = [good + "this bird has a browm belly and a short beak.\nshort"]
    cleaner = CaptionCleaner().fit(descs * 2)
    cleaned = cleaner.clean_description(descs[0])
    assert cleaned == ["this bird has a brown belly and a short beak"] or "browm" not in " ".join(cleaned)
    assert all(len(c.split()) >= 3 for c in cleaned)      # "short" dropped
    assert len(cleaned) == len(set(cleaned))              # duplicates dropped


def test_analyze():
    s = analyze_captions(["a b c.\nd e f g.", "x y z."])
    assert s["n_images"] == 2 and s["n_captions"] == 3


def test_transforms_shape():
    img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    assert build_train_transform()(img).shape == (3, 224, 224)
    assert build_eval_transform()(img).shape == (3, 224, 224)


def _tiny_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 5),
    )


def test_gradcam_shapes_and_range():
    model = _tiny_cnn()
    with GradCAM(model) as g:
        cams, preds, probs = g.generate(torch.randn(3, 3, 32, 32))
    assert cams.shape == (3, 32, 32)
    assert cams.min() >= 0 and cams.max() <= 1.0 + 1e-6
    assert preds.shape == (3,) and probs.shape == (3,)


def test_gradcam_frozen_backbone():
    model = _tiny_cnn()
    for p in model[:-1].parameters():
        p.requires_grad = False
    with GradCAM(model) as g:
        cams, _, _ = g.generate(torch.randn(2, 3, 32, 32), class_idx=1)
    assert cams.shape == (2, 32, 32)


def test_collect_examples():
    model = _tiny_cnn()
    data = [(torch.randn(3, 32, 32), i % 5) for i in range(40)]
    loader = torch.utils.data.DataLoader(data, batch_size=8)
    ex = collect_examples(model, loader, "cpu", n_correct=1, n_wrong=2)
    assert len(ex["wrong"]) >= 1
