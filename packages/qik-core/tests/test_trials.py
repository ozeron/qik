from __future__ import annotations

from pathlib import Path

import pytest

from qik.trials import TRIAL_LOG_ENV, count_trials, log_trial, trial_log_path


def test_counts_calls_and_unique_params(tmp_path: Path) -> None:
    log = tmp_path / "t.jsonl"
    log_trial({"a": 1, "b": 2}, {"sharpe": 1.0}, project="P001", stage="optimize", path=log)
    log_trial({"b": 2, "a": 1}, {"sharpe": 1.0}, project="P001", stage="optimize", path=log)
    log_trial({"a": 3}, {"sharpe": 0.5}, project="P001", stage="debug", path=log)
    log_trial({"a": 4}, {"sharpe": 0.1}, project="P002", stage="optimize", path=log)
    assert count_trials(log).unique == 3
    assert count_trials(log).calls == 4
    assert count_trials(log, project="P001").unique == 2
    assert count_trials(log, project="P001", stage="optimize").calls == 2


def test_missing_log_is_zero(tmp_path: Path) -> None:
    assert count_trials(tmp_path / "none.jsonl").calls == 0


def test_env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRIAL_LOG_ENV, str(tmp_path / "env.jsonl"))
    assert trial_log_path() == tmp_path / "env.jsonl"
    log_trial({"x": 1}, {}, project="P", stage="s")
    assert count_trials().unique == 1
