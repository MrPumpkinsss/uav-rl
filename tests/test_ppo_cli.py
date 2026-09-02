"""Regression tests for the small, current PPO command-line surface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script(filename: str, module_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ppo" / filename
    specification = importlib.util.spec_from_file_location(module_name, script_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_train_script():
    return _load_script("train.py", "ppo_train_cli")


def _load_true_validation_script():
    return _load_script("validate_true_policy.py", "surrogate_ppo_true_validation_cli")


def test_run_directory_layout_and_overwrite_protection(tmp_path: Path) -> None:
    train = _load_train_script()
    run_directory = tmp_path / "round-01"
    paths = train.run_paths(run_directory)
    args = SimpleNamespace(run_dir=run_directory, resume=False)

    train.prepare_run_directory(args, paths)
    assert run_directory.is_dir()
    assert paths["best_model"].name == "best_policy.pth"
    assert paths["state"].name == "training_state.pth"
    assert paths["ppl_cache"].name == "ppl_cache.jsonl"
    assert paths["launch_config"].name == "run_config.json"
    assert paths["result"].name == "evaluation.json"

    train.write_json(paths["launch_config"], {"format_version": 1})
    with pytest.raises(FileExistsError, match="--resume"):
        train.prepare_run_directory(args, paths)


def test_resume_requires_the_canonical_state_file(tmp_path: Path) -> None:
    train = _load_train_script()
    run_directory = tmp_path / "round-01"
    paths = train.run_paths(run_directory)
    args = SimpleNamespace(run_dir=run_directory, resume=True)

    with pytest.raises(FileNotFoundError, match="resume state"):
        train.prepare_run_directory(args, paths)

    run_directory.mkdir()
    paths["state"].write_bytes(b"state")
    train.prepare_run_directory(args, paths)



def test_true_validation_stages_have_isolated_evidence_and_cache_paths(tmp_path: Path) -> None:
    validate = _load_true_validation_script()
    output, selected, cache = validate._stage_paths(tmp_path, "screening_validation")

    assert output.name == "screening_validation.json"
    assert selected.name == "screening_validation_best_policy.pth"
    assert cache.name == "screening_validation_cache.jsonl"
    with pytest.raises(ValueError, match="stage name"):
        validate._stage_paths(tmp_path, "Screening-validation")


def test_true_validation_can_limit_evaluation_to_frozen_candidates(tmp_path: Path) -> None:
    validate = _load_true_validation_script()
    candidate_directory = tmp_path / "candidate_policies"
    candidate_directory.mkdir()
    first = candidate_directory / "episode_000200.pth"
    second = candidate_directory / "episode_000400.pth"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert validate._requested_candidates(candidate_directory, None) == [
        first.resolve(),
        second.resolve(),
    ]
    assert validate._requested_candidates(candidate_directory, [second]) == [second.resolve()]
    with pytest.raises(ValueError, match="not frozen"):
        validate._requested_candidates(candidate_directory, [tmp_path / "outside.pth"])
