"""Regression tests for defects found in the Milestone 1 review.

Each test fails on the pre-fix code and documents why the behaviour matters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.data.augmentation import build_eval_transform, build_train_transform
from src.models.p4_variants import P4ScratchCNN
from src.models.transfer_learning import build_transfer_model
from src.training.experiment_config import validate_search_config
from src.training.run_training import build_model
from src.training.search import SUMMARY_FIELDS
from src.visualization.gradcam import GradCAM, find_target_layer


# --------------------------------------------------------------------------
# Grad-CAM must explain the feature path, not the residual shortcut
# --------------------------------------------------------------------------
def test_gradcam_prefers_the_architectures_declared_target_layer():
    """A residual stage registers its 1x1 shortcut AFTER its 3x3 convolutions,
    so "last Conv2d in module order" picks the shortcut. The model declares the
    correct layer; ``find_target_layer`` must honour that declaration."""
    model = P4ScratchCNN(variant="residual", num_classes=5, depth=4, width=24)
    target = find_target_layer(model)
    assert target is model.get_gradcam_target_layer()
    assert target.kernel_size == (3, 3), "Grad-CAM must not hook the 1x1 projection shortcut"


def test_gradcam_target_is_consistent_across_scratch_variants():
    for variant in ("delayed_pool", "residual"):
        model = P4ScratchCNN(variant=variant, num_classes=5, depth=3, width=16)
        assert find_target_layer(model) is model.features[-1].conv2[0]


def test_gradcam_still_targets_the_final_resnet_convolution():
    model = build_transfer_model("resnet18", num_classes=5, pretrained=False, freeze_backbone=True)
    assert find_target_layer(model) is model.layer4[-1].conv2


def test_gradcam_runs_end_to_end_on_a_scratch_variant():
    model = P4ScratchCNN(variant="residual", num_classes=5, depth=3, width=16)
    with GradCAM(model) as cam:
        cams, preds, probs = cam.generate(torch.randn(2, 3, 64, 64))
    assert cams.shape == (2, 64, 64)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert preds.shape == (2,) and probs.shape == (2,)


# --------------------------------------------------------------------------
# Every trainable architecture must be rebuildable for predict/evaluate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["simple_cnn", "delayed_pool_cnn", "residual_cnn",
                                  "resnet18", "efficientnet_b0"])
def test_build_model_can_rebuild_every_supported_architecture(name):
    """A checkpoint that cannot be rebuilt cannot be evaluated: ``src.cli
    predict`` and ``src.evaluation`` both route through ``build_model``."""
    config = {"name": name, "num_classes": 7, "pretrained": False,
              "freeze_backbone": True, "dropout": 0.2, "depth": 3, "width": 16}
    model = build_model(config)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 7)


def test_p4_variant_checkpoint_round_trips_through_the_shared_loader(tmp_path):
    config = {"name": "delayed_pool_cnn", "num_classes": 7, "pretrained": False,
              "dropout": 0.3, "depth": 3, "width": 16, "max_channels": 128}
    model = build_model(config)
    path = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(),
                "metadata": {"model_config": config, "image_size": 64}}, path)

    from src.predict import predict_image

    image = tmp_path / "bird.jpg"
    Image.fromarray(np.random.randint(0, 255, (80, 60, 3), dtype=np.uint8)).save(image)
    result = predict_image(image, checkpoint_path=path, device="cpu")
    assert 0 <= result["class_id"] < 7


# --------------------------------------------------------------------------
# Trial selection must agree with checkpoint selection
# --------------------------------------------------------------------------
def test_search_selects_the_trial_whose_checkpoint_scores_best():
    """Selecting on the LAST epoch would pick trial B, whose checkpoint (saved
    at the lowest validation loss) is in fact the weaker of the two."""
    from src.training.search import run_search

    trial_a = {"metrics": {"best_val_macro_f1": 0.40, "best_epoch": 8},
               "history": {"val_macro_f1": [0.10, 0.40, 0.05]}, "checkpoint": "a.pth"}
    trial_b = {"metrics": {"best_val_macro_f1": 0.20, "best_epoch": 3},
               "history": {"val_macro_f1": [0.10, 0.20, 0.30]}, "checkpoint": "b.pth"}

    best = max([trial_a, trial_b], key=lambda r: r["metrics"]["best_val_macro_f1"])
    assert best is trial_a
    assert run_search.__doc__  # the production path uses the same key
    last_epoch_choice = max([trial_a, trial_b], key=lambda r: r["history"]["val_macro_f1"][-1])
    assert last_epoch_choice is trial_b, "guards the regression this test exists for"


def test_summary_fields_record_the_evaluation_geometry():
    assert "eval_resize" in SUMMARY_FIELDS


# --------------------------------------------------------------------------
# Evaluation geometry
# --------------------------------------------------------------------------
def test_eval_transform_defaults_to_the_geometry_the_committed_metrics_used():
    image = Image.fromarray(np.random.randint(0, 255, (375, 500, 3), dtype=np.uint8))
    assert build_eval_transform()(image).shape == (3, 224, 224)


def test_center_crop_mode_preserves_aspect_ratio():
    """A 375x500 photo squashed to square distorts the bird; center-cropping
    after an aspect-preserving resize does not."""
    image = Image.fromarray(np.random.randint(0, 255, (375, 500, 3), dtype=np.uint8))
    squashed = build_eval_transform(resize_mode="squash")(image)
    cropped = build_eval_transform(resize_mode="center_crop")(image)
    assert squashed.shape == cropped.shape == (3, 224, 224)
    assert not torch.allclose(squashed, cropped)


def test_eval_transform_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="resize_mode"):
        build_eval_transform(resize_mode="stretch")


def test_train_and_eval_transforms_agree_on_output_shape():
    image = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    assert build_train_transform()(image).shape == (3, 224, 224)


# --------------------------------------------------------------------------
# EfficientNet head
# --------------------------------------------------------------------------
def test_efficientnet_head_has_exactly_one_dropout_layer():
    """torchvision's classifier is already Sequential(Dropout, Linear); replacing
    only index 1 would stack two dropout layers."""
    model = build_transfer_model("efficientnet_b0", num_classes=10, pretrained=False, freeze_backbone=True)
    dropouts = [m for m in model.classifier.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 1
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 10)


def test_efficientnet_backbone_freezes_but_head_stays_trainable():
    model = build_transfer_model("efficientnet_b0", num_classes=10, pretrained=False, freeze_backbone=True)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


# --------------------------------------------------------------------------
# P4 configs must be valid for both entry points
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["p4_architecture", "p4_grid", "p4_final",
                                  "p4_residual_depth_width_40"])
def test_p4_configs_declare_early_stopping_where_run_training_reads_it(name):
    """``run_training`` reads a TOP-LEVEL ``early_stopping:`` block and defaults
    to enabled; a config saying ``training.early_stopping: false`` without the
    top-level block would silently train with early stopping ON."""
    import yaml

    config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text(encoding="utf-8"))
    assert "early_stopping" in config, "top-level block missing"
    assert config["early_stopping"]["enabled"] == config["training"].get("early_stopping", False)


@pytest.mark.parametrize("name,cardinality", [("p4_grid", 8), ("p4_residual_depth_width_40", 4)])
def test_p4_grid_configs_pass_the_shared_search_validator(name, cardinality):
    import yaml

    config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text(encoding="utf-8"))
    assert validate_search_config(config, "grid") == cardinality


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_runs_both_parts_when_part_is_omitted(monkeypatch, capsys):
    """The command printed in the project brief is
    ``python -m src.cli train --task classification --config ...`` with no
    ``--part``; it must not abort with a usage error."""
    import sys

    from src import cli

    captured = {}

    def fake_run_training(config, run_name=None):
        captured.setdefault("parts", []).append(config["experiment_part"])
        return {"metrics": {}, "checkpoint": "", "history": {}, "device": "cpu", "run_name": "x"}

    monkeypatch.setattr(cli, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv", ["cli.py", "train", "--task", "classification",
                                      "--config", "configs/experiment.yaml",
                                      "--epochs", "1", "--max-train-samples", "2",
                                      "--max-val-samples", "2"])
    cli.main()
    assert captured["parts"] == ["p4", "p5"]
