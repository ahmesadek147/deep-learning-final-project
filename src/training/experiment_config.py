"""Loading, merging, and validation for the shared P4/P5 experiment YAML."""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml


EXPERIMENTS = ("p4", "p5")
STRATEGIES = {"p4": "grid", "p5": "random"}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Search key '{dotted_key}' does not exist in the resolved experiment config.")
        value = value[part]
    return value


def validate_search_config(config: dict[str, Any], strategy: str) -> int:
    """Validate a discrete search space and return its exact cardinality."""
    if strategy not in {"grid", "random"}:
        raise ValueError(f"Unknown search strategy: {strategy}")
    search = config.get("search")
    if not isinstance(search, dict) or not isinstance(search.get("space"), dict) or not search["space"]:
        raise ValueError("Search config requires a non-empty 'space' mapping.")
    space = search["space"]
    if any(not isinstance(key, str) or not isinstance(values, list) or not values for key, values in space.items()):
        raise ValueError("Every search-space entry must use a dotted string key and a non-empty list of values.")
    for key in space:
        _get_nested(config, key)
    cardinality = 1
    for values in space.values():
        cardinality *= len(values)
    n_trials = search.get("n_trials")
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("search.n_trials must be a positive integer.")
    if strategy == "grid" and n_trials != cardinality:
        raise ValueError(f"Grid search has {cardinality} configurations, but search.n_trials is {n_trials}.")
    if strategy == "random" and n_trials > cardinality:
        raise ValueError(f"Random search requests {n_trials} unique configurations, but only {cardinality} exist.")
    return cardinality


def resolve_experiment_config(document: dict[str, Any], part: str) -> dict[str, Any]:
    """Merge ``common`` and one experiment section into a runnable config."""
    if part not in EXPERIMENTS:
        raise ValueError(f"part must be one of {EXPERIMENTS}, got '{part}'.")
    common = document.get("common")
    experiment = document.get(part)
    if not isinstance(common, dict) or not isinstance(experiment, dict):
        raise ValueError("Shared experiment config requires 'common', 'p4', and 'p5' mappings.")
    config = _deep_merge(common, experiment)
    config["experiment_part"] = part
    expected_strategy = STRATEGIES[part]
    strategy = config.get("search", {}).get("strategy")
    if strategy != expected_strategy:
        raise ValueError(f"{part}.search.strategy must be '{expected_strategy}'.")
    validate_search_config(config, strategy)
    return config


def load_experiment_config(path: str | Path, part: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if not isinstance(document, dict):
        raise ValueError("Experiment YAML must contain a mapping.")
    return resolve_experiment_config(document, part)


def search_cardinality(search_space: dict[str, list[Any]]) -> int:
    """Return the count of discrete parameter combinations after validation."""
    return len(list(product(*search_space.values())))
