"""Tests for Runner.load()'s corrupt-run-directory guard (T048).

A run directory is resumable by design: ``Runner.learn()`` iterates
``range(learning_ongoing_epoch, max_epoch)``, trusting ``run_info.json``'s epoch counters to
reflect the model weights actually persisted next to it. ``RLModel.load()`` tolerates a missing
``model.json`` silently -- that tolerance is required for a fresh run to work -- so a directory
whose ``run_info.json`` reports a non-zero epoch with no ``model.json`` on disk is corrupt and
must be rejected loudly instead of silently resuming training from a randomly initialised model.
"""

import json
from pathlib import Path

import pytest

from hercule.models import model_file_name
from hercule.run import Runner, run_info_file_name


def _write_run_info(directory: Path, learning_ongoing_epoch: int) -> None:
    run_info = {
        "learning_ongoing_epoch": learning_ongoing_epoch,
        "testing_ongoing_epoch": 0,
        "learning_metrics": [],
        "testing_metrics": [],
        "model_hyperparameters": {},
    }
    with open(directory / run_info_file_name, "w", encoding="utf-8") as f:
        json.dump(run_info, f)


def _write_model_file(directory: Path) -> None:
    with open(directory / model_file_name, "w", encoding="utf-8") as f:
        json.dump({"model_name": "dummy"}, f)


class TestResumeGuard:
    """Runner.load() must refuse a run directory whose counters and weights disagree."""

    def test_nonzero_epoch_without_model_file_raises(self, tmp_path):
        """run_info.json at a non-zero epoch with no model.json is corrupt and must raise."""
        _write_run_info(tmp_path, learning_ongoing_epoch=5000)

        with pytest.raises(ValueError) as exc_info:
            Runner.load(tmp_path)

        message = str(exc_info.value)
        assert model_file_name in message
        assert "5000" in message

    def test_fresh_directory_with_neither_file_does_not_raise(self, tmp_path):
        """A genuinely fresh run directory (no run_info.json, no model.json) is not corrupt."""
        runner = Runner.load(tmp_path)

        assert runner.learning_ongoing_epoch == 0

    def test_zero_epoch_without_model_file_does_not_raise(self, tmp_path):
        """run_info.json at epoch 0 with no model.json describes a fresh run, not a corrupt one."""
        _write_run_info(tmp_path, learning_ongoing_epoch=0)

        runner = Runner.load(tmp_path)

        assert runner.learning_ongoing_epoch == 0

    def test_nonzero_epoch_with_model_file_does_not_raise(self, tmp_path):
        """Both files present describes a consistent, resumable run directory."""
        _write_run_info(tmp_path, learning_ongoing_epoch=5000)
        _write_model_file(tmp_path)

        runner = Runner.load(tmp_path)

        assert runner.learning_ongoing_epoch == 5000
