"""Tests for the transfer study and the report regeneration pipeline.

These run entirely on synthetic result files: no training, no dataset, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.reporting import update_report
from src.training.run_training import build_model
from src.training.transfer_study import build_run_config, load_study, write_report

STUDY_CONFIG = Path("configs/p5_transfer_study.yaml")
REPORT = Path("outputs/final_submission/milestone_1_report.md")


# ---------------------------------------------------------------------------
# Transfer study configuration
# ---------------------------------------------------------------------------
def test_study_config_loads_and_every_run_is_buildable():
    document = load_study(STUDY_CONFIG)
    assert len(document["runs"]) >= 4
    for run in document["runs"]:
        config = build_run_config(document, run)
        model = build_model({**config["model"], "pretrained": False})
        assert model is not None


def test_study_covers_two_pretrained_architectures():
    """The brief asks for 2-3 pretrained models; one is not enough."""
    document = load_study(STUDY_CONFIG)
    names = {build_run_config(document, run)["model"]["name"] for run in document["runs"]}
    assert len(names) >= 2, f"only one architecture in the study: {names}"


def test_study_compares_frozen_against_fine_tuned():
    document = load_study(STUDY_CONFIG)
    frozen = {build_run_config(document, r)["model"]["freeze_backbone"] for r in document["runs"]}
    assert frozen == {True, False}


def test_study_fine_tunes_below_the_head_only_learning_rate():
    """Fine-tuning a pretrained backbone at the head's learning rate destroys
    its features - that is the failure this study exists to correct."""
    document = load_study(STUDY_CONFIG)
    tuned, frozen = [], []
    for run in document["runs"]:
        config = build_run_config(document, run)
        target = frozen if config["model"]["freeze_backbone"] else tuned
        target.append(config["training"]["learning_rate"])
    assert tuned and frozen
    assert max(tuned) < min(frozen), (
        f"fine-tuning rates {sorted(tuned)} must sit below head-only rates {sorted(frozen)}")


def test_study_uses_the_duplicate_safe_split():
    document = load_study(STUDY_CONFIG)
    data = document["common"]["data"]
    assert data["split_strategy"] == "stratified_group"
    assert data["split_manifest"].endswith("cub_train_val_v2_seed42.json")


def test_study_holds_every_other_factor_constant():
    """A controlled study changes one thing at a time; anything a run does not
    override must be identical across runs."""
    document = load_study(STUDY_CONFIG)
    configs = [build_run_config(document, run) for run in document["runs"]]
    for key in ("seed", "val_split", "image_size", "split_strategy"):
        assert len({c["data"][key] for c in configs}) == 1
    for key in ("epochs", "batch_size", "optimizer", "weight_decay"):
        assert len({c["training"][key] for c in configs}) == 1


@pytest.mark.parametrize("broken,message", [
    ({"common": {}}, "runs"),
    ({"runs": [{"label": "a", "overrides": {"model.name": "resnet18"}}]}, "common"),
])
def test_study_config_validation_rejects_incomplete_documents(tmp_path, broken, message):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_study(path)


def test_study_config_rejects_duplicate_labels(tmp_path):
    document = {"common": {"model": {}, "training": {}, "data": {}},
                "runs": [{"label": "x", "overrides": {"model.name": "resnet18"}},
                         {"label": "x", "overrides": {"model.name": "efficientnet_b0"}}]}
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_study(path)


def _row(label, model, frozen, lr, f1, acc):
    return {"label": label, "model_name": model, "freeze_backbone": frozen,
            "learning_rate": lr, "batch_size": 8, "epochs_run": 10, "best_epoch": 8,
            "best_val_loss": 2.0, "best_val_accuracy": acc, "best_val_macro_precision": 0.5,
            "best_val_macro_recall": 0.45, "best_val_macro_f1": f1,
            "checkpoint": "c.pth", "note": ""}


def test_study_report_ranks_runs_and_names_the_winner(tmp_path):
    rows = [_row("frozen", "resnet18", True, 1e-3, 0.40, 0.42),
            _row("tuned", "resnet18", False, 1e-4, 0.52, 0.54),
            _row("effnet", "efficientnet_b0", True, 1e-3, 0.46, 0.48)]
    text = write_report(rows, tmp_path).read_text(encoding="utf-8")
    assert "**tuned**" in text
    assert text.index("| tuned") < text.index("| effnet") < text.index("| frozen")
    assert "points above" in text
    assert (tmp_path / "figures" / "transfer_study.png").is_file()


def test_study_report_states_direction_honestly_when_fine_tuning_loses(tmp_path):
    rows = [_row("frozen", "resnet18", True, 1e-3, 0.50, 0.52),
            _row("tuned", "resnet18", False, 1e-4, 0.30, 0.32)]
    text = write_report(rows, tmp_path).read_text(encoding="utf-8")
    assert "points below" in text


# ---------------------------------------------------------------------------
# Report regeneration
# ---------------------------------------------------------------------------
def test_report_contains_every_auto_block():
    text = REPORT.read_text(encoding="utf-8")
    for name in update_report.SECTIONS:
        assert f"<!-- AUTO:{name} -->" in text
        assert f"<!-- /AUTO:{name} -->" in text


def test_updater_fills_measured_sections_from_committed_artifacts(tmp_path):
    target = tmp_path / "report.md"
    target.write_text(REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    update_report.update(target)
    text = target.read_text(encoding="utf-8")
    # The three committed evaluations must appear with their real numbers.
    assert "44.96%" in text and "5.94%" in text and "45.39%" in text


def test_updater_marks_missing_results_instead_of_inventing_numbers(tmp_path, monkeypatch):
    template = REPORT.read_text(encoding="utf-8")  # read before changing directory
    target = tmp_path / "report.md"
    target.write_text(template, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # No outputs/ at all in this cwd: every section must report itself unrun.
    _, pending = update_report.update(target)
    assert set(pending) == set(update_report.SECTIONS)
    assert "Not run yet" in target.read_text(encoding="utf-8")


def test_updater_is_idempotent(tmp_path):
    target = tmp_path / "report.md"
    target.write_text(REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    update_report.update(target)
    first = target.read_text(encoding="utf-8")
    update_report.update(target)
    assert target.read_text(encoding="utf-8") == first


def test_updater_never_touches_prose_outside_the_markers(tmp_path):
    target = tmp_path / "report.md"
    target.write_text(REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    update_report.update(target)
    text = target.read_text(encoding="utf-8")
    # Hand-written prose from outside every AUTO block must survive verbatim.
    for sentence in ("## 9. Evaluation Metrics",
                     "Macro F1 below accuracy is therefore the signature",
                     "the search space was wrong",
                     "vertical flipping is excluded because birds are essentially never"):
        assert sentence in text, f"updater clobbered prose: {sentence!r}"


def test_updater_raises_when_a_marker_is_missing(tmp_path):
    target = tmp_path / "report.md"
    target.write_text("# Report\n\nno markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AUTO:"):
        update_report.update(target)


def test_report_has_no_unfilled_name_placeholder_left_unflagged():
    """`[NAME]` is intentional until the team fills it in, but it must be
    visible, not hidden in a sentence."""
    text = REPORT.read_text(encoding="utf-8")
    if "[NAME]" in text:
        assert "Replace each `[NAME]`" in text


# ---------------------------------------------------------------------------
# PDF build
# ---------------------------------------------------------------------------
def test_pdf_builder_strips_the_title_headings_so_sections_are_not_nested(tmp_path):
    """Leaving the markdown H1 in place nests every section under it and
    produces headings like "1.2 1. Introduction" in the PDF."""
    from src.reporting.build_pdf import _body_without_title

    source = tmp_path / "r.md"
    source.write_text("# Title\n## Subtitle\n\n**Project:** x\n\n## 1. Introduction\n\nbody\n",
                      encoding="utf-8")
    body = _body_without_title(source)
    assert not body.startswith("# Title")
    assert "## Subtitle" not in body
    assert "## 1. Introduction" in body
    assert "**Project:** x" in body


def test_pdf_builder_reports_a_usable_error_without_pandoc(tmp_path, monkeypatch):
    from src.reporting import build_pdf

    monkeypatch.setattr(build_pdf.shutil, "which", lambda name: None)
    source = tmp_path / "r.md"
    source.write_text("# T\n\nbody\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pandoc"):
        build_pdf.build(source, tmp_path / "r.pdf")
