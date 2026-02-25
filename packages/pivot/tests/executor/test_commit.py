from __future__ import annotations

from types import SimpleNamespace

import pytest

from pivot import exceptions
from pivot.executor import commit


def test_split_identity_with_key() -> None:
    producer, key = commit._split_identity("train:metrics")
    assert producer == "train"
    assert key == "metrics"


def test_split_identity_with_empty_key_becomes_none() -> None:
    producer, key = commit._split_identity("train:")
    assert producer == "train"
    assert key is None


def test_split_identity_without_separator() -> None:
    producer, key = commit._split_identity("train")
    assert producer == "train"
    assert key is None


def test_commit_stages_rejects_unknown_stage_names(mocker) -> None:
    fake_registry = SimpleNamespace(list_stages=lambda: ["known"])

    mocker.patch("pivot.cli.helpers.get_registry", autospec=True, return_value=fake_registry)
    mocker.patch(
        "pivot.cli.helpers._get_pipeline",
        autospec=True,
        return_value=SimpleNamespace(name="pipeline", input_bindings={}),
    )

    with pytest.raises(exceptions.StageNotFoundError):
        commit.commit_stages(stage_names=["missing"])
