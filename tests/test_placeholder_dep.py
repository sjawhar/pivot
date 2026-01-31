# pyright: reportUnusedFunction=false
"""Integration tests for PlaceholderDep functionality."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, TypedDict

import pandas
import pytest

from helpers import register_test_stage
from pivot import exceptions, loaders, outputs
from pivot.engine import sources
from pivot.engine.engine import Engine
from pivot.engine.sinks import ResultCollectorSink

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


@pytest.mark.anyio
async def test_placeholder_dep_e2e_execution(
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
    tmp_path: pathlib.Path,
    comparison_data: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """PlaceholderDep stage should execute correctly with overridden paths."""
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
    async with Engine(pipeline=test_pipeline) as engine:
        engine.add_source(sources.OneShotSource(stages=["compare_ab"], force=True, reason="test"))
        await engine.run(exit_on_completion=True)

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


@pytest.mark.anyio
async def test_placeholder_dep_missing_file_fails_execution(
    test_pipeline: Pipeline,
    mock_discovery: Pipeline,
    tmp_path: pathlib.Path,
) -> None:
    """PlaceholderDep with nonexistent file path fails during execution."""
    # Register stage with override to nonexistent path
    nonexistent = tmp_path / "does_not_exist.csv"

    register_test_stage(
        _compare_datasets,
        name="compare_missing",
        dep_path_overrides={
            "baseline": str(nonexistent.relative_to(tmp_path)),
            "experiment": str(nonexistent.relative_to(tmp_path)),
        },
    )

    # Execute via Engine - should fail with DependencyNotFoundError
    async with Engine(pipeline=test_pipeline) as engine:
        collector = ResultCollectorSink()
        engine.add_sink(collector)
        engine.add_source(
            sources.OneShotSource(stages=["compare_missing"], force=True, reason="test")
        )

        # Should raise ExceptionGroup wrapping DependencyNotFoundError
        with pytest.raises(ExceptionGroup) as exc_info:
            await engine.run(exit_on_completion=True)

        # Verify the error is about missing dependency
        assert len(exc_info.value.exceptions) == 1
        inner = exc_info.value.exceptions[0]
        assert isinstance(inner, exceptions.DependencyNotFoundError)
        assert "does_not_exist.csv" in str(inner)


def test_placeholder_dep_override_validation(
    test_pipeline: Pipeline,
    tmp_path: pathlib.Path,
) -> None:
    """PlaceholderDep stage validates override paths are provided at registration."""
    # Register stage without providing required override
    # This should raise ValidationError at registration time
    with pytest.raises(
        exceptions.ValidationError, match="Placeholder dependencies missing overrides"
    ):
        register_test_stage(
            _compare_datasets,
            name="compare_no_override",
            dep_path_overrides={
                # Only provide one of two required overrides
                "baseline": "model_a/results.csv",
                # "experiment" is missing - validation catches this
            },
        )


def test_placeholder_dep_multiple_stages_independent_overrides(
    test_pipeline: Pipeline,
    tmp_path: pathlib.Path,
    comparison_data: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """Multiple stages using same function maintain independent override namespaces."""
    baseline_path, experiment_path = comparison_data

    # Create additional files
    alt_baseline = tmp_path / "alt_baseline.csv"
    alt_baseline.write_text("value\n100\n")

    alt_experiment = tmp_path / "alt_experiment.csv"
    alt_experiment.write_text("value\n200\n")

    # Register same function twice with different overrides
    register_test_stage(
        _compare_datasets,
        name="compare_original",
        dep_path_overrides={
            "baseline": str(baseline_path.relative_to(tmp_path)),
            "experiment": str(experiment_path.relative_to(tmp_path)),
        },
        out_path_overrides={"diff": "diff_original.json"},
    )

    register_test_stage(
        _compare_datasets,
        name="compare_alt",
        dep_path_overrides={
            "baseline": str(alt_baseline.relative_to(tmp_path)),
            "experiment": str(alt_experiment.relative_to(tmp_path)),
        },
        out_path_overrides={"diff": "diff_alt.json"},
    )

    # Verify stages are independent
    original_info = test_pipeline.get("compare_original")
    alt_info = test_pipeline.get("compare_alt")

    # deps is a dict
    assert original_info["deps"]["baseline"] != alt_info["deps"]["baseline"]
    assert original_info["deps"]["experiment"] != alt_info["deps"]["experiment"]

    # outs is a list of Out objects
    original_out_paths = [str(out.path) for out in original_info["outs"]]
    alt_out_paths = [str(out.path) for out in alt_info["outs"]]
    assert any("diff_original.json" in p for p in original_out_paths)
    assert any("diff_alt.json" in p for p in alt_out_paths)
