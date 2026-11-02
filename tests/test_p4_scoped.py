"""CPU-only checks for P4's additions; no dataset downloads or shared edits."""
from pathlib import Path

import pytest
import torch

from src.models.p4_variants import P4ScratchCNN
from src.training.p4_experiments import (
    _grid_candidates,
    _load_winner,
    build_p4_model,
    load_config,
    run_one,
)


@pytest.mark.parametrize("variant", ["delayed_pool", "residual"])
def test_custom_scratch_variants_support_forward_backward_and_gradcam(variant):
    torch.manual_seed(123)
    model = P4ScratchCNN(variant, num_classes=7, depth=3, width=8, max_channels=32)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 7)
    assert model.forward_feature_maps(x).shape == (2, 32, 8, 8)
    assert model.forward_features(x).shape == (2, 32)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert isinstance(model.get_gradcam_target_layer(), torch.nn.Conv2d)
    assert model.count_trainable_parameters() > 0


@pytest.mark.parametrize("variant", ["delayed_pool", "residual"])
def test_first_stage_has_no_pooling(variant):
    model = P4ScratchCNN(variant, depth=3, width=8, max_channels=32)
    assert isinstance(model.features[0].pool, torch.nn.Identity)
    assert isinstance(model.features[1].pool, torch.nn.MaxPool2d)
    assert isinstance(model.features[2].pool, torch.nn.MaxPool2d)


def test_pretrained_weights_are_forbidden():
    with pytest.raises(ValueError, match="pretrained"):
        P4ScratchCNN("residual", pretrained=True)
    with pytest.raises(ValueError, match="pretrained"):
        build_p4_model({"name": "simple_cnn", "pretrained": True})


def test_p4_configs_and_grid_are_valid():
    architecture = load_config(Path("configs/p4_architecture.yaml"))
    assert architecture["architecture_candidates"] == [
        "simple_cnn", "delayed_pool_cnn", "residual_cnn"]
    assert all(build_p4_model({**architecture["model"], "name": name}) is not None
               for name in architecture["architecture_candidates"])
    grid = load_config(Path("configs/p4_grid.yaml"))
    final = load_config(Path("configs/p4_final.yaml"))
    assert len(_grid_candidates(grid)) == 8
    assert all(config["data"] == architecture["data"] for config in (grid, final))
    assert final["training"]["epochs"] > grid["training"]["epochs"]
    assert final["training"]["early_stopping"]


def test_winner_must_use_the_same_data_definition(tmp_path):
    import json
    config = load_config(Path("configs/p4_grid.yaml"))
    prior = {"config": config.copy()}
    prior["config"] = {**config, "data": {**config["data"], "seed": 99}}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"best_trial": prior}), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        _load_winner(path, config)


def test_one_epoch_training_and_resume_with_mocked_p2_loader(monkeypatch, tmp_path):
    """Exercise P1 trainer reuse and P4 checkpoints without downloading CUB."""
    from torch.utils.data import DataLoader, TensorDataset
    config = load_config("configs/p4_architecture.yaml")
    config["device"] = "auto"
    config["model"].update({"name": "delayed_pool_cnn", "num_classes": 200,
                            "depth": 3, "width": 8, "max_channels": 32})
    config["data"]["image_size"] = 32
    config["training"].update({"epochs": 1, "batch_size": 2, "mixed_precision": False})
    config["mlflow"] = {"tracking_uri": f"sqlite:///{tmp_path / 'p4_test_mlflow.db'}",
                        "experiment_name": "p4_cpu_smoke"}
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 1, 0])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    monkeypatch.setattr("src.training.p4_experiments.get_dataloaders",
                        lambda **kwargs: (loader, loader))
    result = run_one(config, tmp_path, "tiny")
    assert result["best_epoch"] == 1
    assert result["official_test_used"] is False
    assert (tmp_path / "runs/tiny/best_model.pth").is_file()
    assert (tmp_path / "runs/tiny/last.pth").is_file()
    assert (tmp_path / "runs/tiny/epochs.csv").is_file()
    assert (tmp_path / "runs/tiny/macro_f1.png").is_file()
    resumed = run_one(config, tmp_path, "tiny", resume=True)
    assert resumed["best_val_macro_f1"] == result["best_val_macro_f1"]
