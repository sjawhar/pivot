"""Tests for pivot.cli.targets — pipeline file target resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.cli import targets as cli_targets
from pivot.compose import Pipeline

if TYPE_CHECKING:
    import pathlib


# =============================================================================
# resolve_pipeline_file_targets Tests
# =============================================================================


def test_resolve_pipeline_file_targets_pipeline_py(tmp_path: pathlib.Path) -> None:
    """Pipeline .py file target resolves to its stage names."""
    sub = tmp_path / "sub"
    sub.mkdir()
    pipeline_code = """\
import pathlib

from pivot.compose import Pipeline, stage


@stage
def stage_a() -> dict:
    return {}


@stage
def stage_b() -> dict:
    return {}


with Pipeline("test_pipe", root=pathlib.Path(__file__).parent) as pipeline:
    stage_a()
    stage_b()
"""
    (sub / "pipeline.py").write_text(pipeline_code)

    target = str(sub / "pipeline.py")
    resolved, remaining, pipelines = cli_targets.resolve_pipeline_file_targets([target])

    assert resolved == {"test_pipe/stage_a", "test_pipe/stage_b"}, (
        "Should resolve both stages from pipeline file"
    )
    assert remaining == [], "No targets should remain unresolved"
    assert pipelines, "Should return loaded pipeline"
    assert isinstance(pipelines[0], Pipeline), "Loaded pipeline should be returned"


def test_resolve_pipeline_file_targets_nonexistent_path() -> None:
    """Non-existent pipeline path falls through (returns empty)."""
    resolved, remaining, pipelines = cli_targets.resolve_pipeline_file_targets(
        ["nonexistent/pipeline.py"]
    )

    assert resolved == set(), "Non-existent path should not resolve any stages"
    assert remaining == ["nonexistent/pipeline.py"], "Non-existent path should remain unresolved"
    assert pipelines == [], "Non-existent path should not load pipelines"


def test_resolve_pipeline_file_targets_not_pipeline_filename(tmp_path: pathlib.Path) -> None:
    """File exists but filename doesn't match pipeline patterns — falls through."""
    other_file = tmp_path / "my_script.py"
    other_file.write_text("x = 1\n")

    target = str(other_file)
    resolved, remaining, pipelines = cli_targets.resolve_pipeline_file_targets([target])

    assert resolved == set(), "Non-pipeline filename should not resolve any stages"
    assert remaining == [target], "Non-pipeline filename should remain unresolved"
    assert pipelines == [], "Non-pipeline filename should not load pipelines"


def test_resolve_pipeline_file_targets_mixed_targets(tmp_path: pathlib.Path) -> None:
    """Mix of pipeline file and non-pipeline targets — pipeline resolved, others pass through."""
    sub = tmp_path / "sub"
    sub.mkdir()
    pipeline_code = """\
import pathlib

from pivot.compose import Pipeline, stage


@stage
def my_stage() -> dict:
    return {}


with Pipeline("mixed_test", root=pathlib.Path(__file__).parent) as pipeline:
    my_stage()
"""
    (sub / "pipeline.py").write_text(pipeline_code)

    targets = [str(sub / "pipeline.py"), "some_other_target"]
    resolved, remaining, pipelines = cli_targets.resolve_pipeline_file_targets(targets)

    assert resolved == {"mixed_test/my_stage"}, "Should resolve stages from the pipeline file"
    assert remaining == ["some_other_target"], "Non-pipeline targets pass through"
    assert pipelines, "Should return loaded pipeline"
