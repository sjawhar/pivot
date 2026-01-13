# pyright: reportPrivateUsage=false
"""Tests for StageDef pickling - critical for multiprocessing execution.

These tests verify that StageDef instances can be pickled and unpickled,
which is required for multiprocessing workers to receive stage params.
"""

from __future__ import annotations

import pickle
from typing import Any

import pandas  # noqa: TC002 - needed at runtime for type annotation

from pivot import loaders, stage_def  # noqa: TC001 - needed at runtime for Dep/Out descriptors

# ==============================================================================
# Module-level StageDef classes (required for pickling)
# ==============================================================================


class SimpleParams(stage_def.StageDef):
    """StageDef with only params, no deps/outs."""

    threshold: float = 0.5
    name: str = "test"


class ParamsWithDeps(stage_def.StageDef):
    """StageDef with deps."""

    threshold: float = 0.5
    data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("input.csv", loaders.CSV())


class ParamsWithOuts(stage_def.StageDef):
    """StageDef with outs."""

    threshold: float = 0.5
    result: stage_def.Out[dict[str, Any]] = stage_def.Out("output.json", loaders.JSON())


class ParamsWithDepsAndOuts(stage_def.StageDef):
    """StageDef with both deps and outs."""

    threshold: float = 0.5
    data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("input.csv", loaders.CSV())
    result: stage_def.Out[dict[str, Any]] = stage_def.Out("output.json", loaders.JSON())


# ==============================================================================
# Pickle tests
# ==============================================================================


def test_simple_params_is_picklable() -> None:
    """StageDef with only params should be picklable."""
    params = SimpleParams(threshold=0.7)
    pickled = pickle.dumps(params)
    restored = pickle.loads(pickled)

    assert restored.threshold == 0.7
    assert restored.name == "test"


def test_params_with_deps_is_picklable() -> None:
    """StageDef with deps should be picklable."""
    params = ParamsWithDeps(threshold=0.7)
    pickled = pickle.dumps(params)
    restored = pickle.loads(pickled)

    assert restored.threshold == 0.7
    # The deps specs should still be accessible after unpickling
    assert "data" in restored._deps_specs
    assert restored._deps_specs["data"].path == "input.csv"


def test_params_with_outs_is_picklable() -> None:
    """StageDef with outs should be picklable."""
    params = ParamsWithOuts(threshold=0.7)
    pickled = pickle.dumps(params)
    restored = pickle.loads(pickled)

    assert restored.threshold == 0.7
    # The outs specs should still be accessible after unpickling
    assert "result" in restored._outs_specs
    assert restored._outs_specs["result"].path == "output.json"


def test_params_with_deps_and_outs_is_picklable() -> None:
    """StageDef with both deps and outs should be picklable."""
    params = ParamsWithDepsAndOuts(threshold=0.7)
    pickled = pickle.dumps(params)
    restored = pickle.loads(pickled)

    assert restored.threshold == 0.7
    # Both deps and outs specs should be accessible after unpickling
    assert "data" in restored._deps_specs
    assert "result" in restored._outs_specs
    assert restored._deps_specs["data"].path == "input.csv"
    assert restored._outs_specs["result"].path == "output.json"


def test_pickle_roundtrip_preserves_class_identity() -> None:
    """Unpickled instance should be same class type."""
    params = ParamsWithDeps(threshold=0.7)
    pickled = pickle.dumps(params)
    restored = pickle.loads(pickled)

    assert type(restored) is ParamsWithDeps
    assert isinstance(restored, stage_def.StageDef)
