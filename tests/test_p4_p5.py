import torch
from pathlib import Path
from PIL import Image

import pytest

from src.models.simple_cnn import SimpleCNN
from src.models.transfer_learning import build_transfer_model
from src.training.run_training import effective_mlflow_params, select_device
from src.training.search import grid_trials, random_trials, trial_record, write_trial_summary, SUMMARY_FIELDS
from src.training.experiment_config import load_experiment_config, validate_search_config
from src.cli import load_part_configs
from src.predict import predict_image
from src.evaluation import evaluate_checkpoint


def test_cnn_output_shape():
    assert SimpleCNN(num_classes=200)(torch.randn(2, 3, 64, 64)).shape == (2, 200)


def test_frozen_transfer_head_is_trainable():
    model = build_transfer_model("resnet18", num_classes=200, pretrained=False, freeze_backbone=True)
    assert all(not parameter.requires_grad for parameter in model.conv1.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 200)


def test_device_selection(monkeypatch):
    monkeypatch.setattr("src.training.run_training.torch.cuda.is_available", lambda: False)
    assert select_device().type == "cpu"


def test_search_trial_configuration_is_reproducible():
    space = {"training.learning_rate": [0.1, 0.01], "model.dropout": [0.2, 0.4]}
    assert len(list(grid_trials(space))) == 4
    random = list(random_trials(space, 3, 7))
    assert random == list(random_trials(space, 3, 7))
    assert len({tuple(sorted(trial.items())) for trial in random}) == 3


def test_shared_config_merges_common_settings_and_validates_trial_counts():
    path = Path("configs/experiment.yaml")
    p4 = load_experiment_config(path, "p4")
    p5 = load_experiment_config(path, "p5")
    assert p4["data"] == p5["data"]
    assert p4["training"]["batch_size"] == 16
    assert p5["training"]["batch_size"] == 8
    assert p4["checkpoint"]["namespace"] == "p4_grid"
    assert p5["checkpoint"]["namespace"] == "p5_random"
    assert validate_search_config(p4, "grid") == 8
    assert validate_search_config(p5, "random") == 36


def test_grid_search_cannot_silently_truncate():
    config = load_experiment_config(Path("configs/experiment.yaml"), "p4")
    config["search"]["n_trials"] = 2
    with pytest.raises(ValueError, match="Grid search has 8 configurations"):
        validate_search_config(config, "grid")


def test_cli_loads_both_parts_from_the_single_shared_config():
    parts = load_part_configs(Path("configs/experiment.yaml"), "both")
    assert [name for name, _ in parts] == ["p4", "p5"]


def test_effective_trial_parameters_include_search_values():
    config = load_experiment_config(Path("configs/experiment.yaml"), "p5")
    config["model"].update({"dropout": 0.4, "freeze_backbone": False})
    config["training"].update({"batch_size": 8, "learning_rate": 0.0003, "weight_decay": 0.00001})
    params = effective_mlflow_params(config, amp_enabled=True)
    assert params["learning_rate"] == 0.0003
    assert params["dropout"] == 0.4
    assert params["weight_decay"] == 0.00001
    assert params["batch_size"] == 8
    assert params["freeze_backbone"] is False
    assert params["mixed_precision_enabled"] is True


def test_trial_summary_records_failures_without_metrics(tmp_path):
    config = load_experiment_config(Path("configs/experiment.yaml"), "p5")
    record = trial_record("search1", "trial1", "p5", config, error=torch.OutOfMemoryError("CUDA out of memory"))
    assert record["status"] == "oom"
    assert record["best_val_loss"] == ""
    path = tmp_path / "trials.csv"
    write_trial_summary(path, [record])
    assert path.is_file()
    assert set(record).issubset(set(SUMMARY_FIELDS))


def test_prediction_loads_new_checkpoint_metadata(tmp_path):
    model_config = {"name": "simple_cnn", "num_classes": 3, "pretrained": False, "dropout": 0.1}
    from src.training.run_training import build_model
    model = build_model(model_config)
    checkpoint = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "metadata": {"model_config": model_config, "image_size": 32}}, checkpoint)
    image_path = tmp_path / "bird.jpg"
    Image.new("RGB", (40, 40), color=(120, 90, 60)).save(image_path)
    result = predict_image(image_path, checkpoint_path=checkpoint, device="cpu")
    assert 0 <= result["class_id"] < 3
    assert result["checkpoint"] == str(checkpoint)


def test_evaluation_returns_200_class_metrics_without_training(tmp_path):
    model_config = {"name": "simple_cnn", "num_classes": 3, "pretrained": False, "dropout": 0.1}
    from src.training.run_training import build_model
    model = build_model(model_config)
    checkpoint = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "metadata": {"model_config": model_config, "image_size": 32}}, checkpoint)
    loader = [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))]
    metrics, report, true, pred, _ = evaluate_checkpoint(checkpoint, loader, ["a", "b", "c"], torch.device("cpu"))
    assert metrics["num_samples"] == 2
    assert len(report) == 6  # three classes plus accuracy/macro/weighted averages
    assert true.shape == pred.shape == (2,)
