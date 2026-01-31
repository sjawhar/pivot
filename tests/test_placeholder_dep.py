# pyright: reportUnusedFunction=false
"""Integration tests for PlaceholderDep functionality."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, TypedDict

import pandas
import pytest

from helpers import register_test_stage
from pivot import loaders, outputs
from pivot.engine import sources

if TYPE_CHECKING:
    import pathlib

    from pivot.pipeline.pipeline import Pipeline


class _CompareOutputs(TypedDict):
    diff: Annotated[dict[str, float], outputs.Out("diff.json", loaders.JSON[dict[str, float]]())]


def _compare_datasets(
    baseline: Annotated[pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())],
    experiment: Annotated[
        pandas.DataFrame, outputs.PlaceholderDep(loaders.CSV[pandas.DataFrame]())
    ],
) -> _CompareOutputs:
    """Compare two datasets and compute difference in means."""
    baseline_mean = float(baseline["value"].mean())
    experiment_mean = float(experiment["value"].mean())
    return _CompareOutputs(diff={"delta": experiment_mean - baseline_mean})


@pytest.fixture
def comparison_data(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Create baseline and experiment CSV files."""
    baseline = tmp_path / "model_a" / "results.csv"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("value\n10\n20\n30\n")

    experiment = tmp_path / "model_b" / "results.csv"
    experiment.parent.mkdir(parents=True)
    experiment.write_text("value\n15\n25\n35\n")

    return baseline, experiment


def test_placeholder_dep_e2e_execution(
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
    tmp_path: pathlib.Path,
    comparison_data: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """PlaceholderDep stage should execute correctly with overridden paths."""
    from pivot.engine.engine import Engine

    baseline_path, experiment_path = comparison_data

    # Register with overrides
    register_test_stage(
        _compare_datasets,
        name="compare_ab",
        dep_path_overrides={
            "baseline": str(baseline_path.relative_to(tmp_path)),
            "experiment": str(experiment_path.relative_to(tmp_path)),
        },
    )

    # Execute via Engine
    with Engine(pipeline=test_pipeline) as engine:
        engine.add_source(sources.OneShotSource(stages=["compare_ab"], force=True, reason="test"))
        engine.run(exit_on_completion=True)

    # Verify output
    output = tmp_path / "diff.json"
    assert output.exists()
    result = json.loads(output.read_text())
    assert result["delta"] == 5.0  # (15+25+35)/3 - (10+20+30)/3 = 25 - 20 = 5


def test_placeholder_dep_reuse_function_different_overrides(
    test_pipeline: Pipeline,
    tmp_path: pathlib.Path,
    comparison_data: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Same function can be registered multiple times with different overrides."""
    baseline_path, experiment_path = comparison_data

    # Create a third dataset
    third = tmp_path / "model_c" / "results.csv"
    third.parent.mkdir(parents=True)
    third.write_text("value\n100\n200\n300\n")

    # Register same function twice with different overrides
    register_test_stage(
        _compare_datasets,
        name="compare_ab_v2",
        dep_path_overrides={
            "baseline": str(baseline_path.relative_to(tmp_path)),
            "experiment": str(experiment_path.relative_to(tmp_path)),
        },
        out_path_overrides={"diff": "diff_ab.json"},
    )

    register_test_stage(
        _compare_datasets,
        name="compare_ac",
        dep_path_overrides={
            "baseline": str(baseline_path.relative_to(tmp_path)),
            "experiment": str(third.relative_to(tmp_path)),
        },
        out_path_overrides={"diff": "diff_ac.json"},
    )

    # Both should be registered
    assert test_pipeline.get("compare_ab_v2") is not None
    assert test_pipeline.get("compare_ac") is not None

    # Dependencies should be different
    ab_info = test_pipeline.get("compare_ab_v2")
    ac_info = test_pipeline.get("compare_ac")

    assert ab_info["deps"]["experiment"] != ac_info["deps"]["experiment"]
