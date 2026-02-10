"""Tests for pivot.cli.targets — pipeline file target resolution."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from pivot.cli import targets as cli_targets
from pivot.pipeline import pipeline as pipeline_mod

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
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("test_pipe")

def _stage_a():
    pass

def _stage_b():
    pass

pipeline.register(_stage_a, name="stage_a")
pipeline.register(_stage_b, name="stage_b")
"""
    (sub / "pipeline.py").write_text(pipeline_code)

    target = str(sub / "pipeline.py")
    resolved, remaining, pipelines = cli_targets.resolve_pipeline_file_targets([target])

    assert resolved == {"stage_a", "stage_b"}, "Should resolve both stages from pipeline file"
    assert remaining == [], "No targets should remain unresolved"
    assert pipelines, "Should return loaded pipeline"
    assert isinstance(pipelines[0], pipeline_mod.Pipeline), "Loaded pipeline should be returned"


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


def test_resolve_pipeline_file_targets_pivot_yaml(tmp_path: pathlib.Path) -> None:
    """pivot.yaml file target resolves to its stage names."""
    # Create a real stage module for the YAML to reference
    stages_code = """\
def preprocess():
    pass

def train():
    pass
"""
    (tmp_path / "stages.py").write_text(stages_code)

    yaml_content = """\
stages:
  preprocess:
    python: stages.preprocess
  train:
    python: stages.train
"""
    (tmp_path / "pivot.yaml").write_text(yaml_content)

    sys.path.insert(0, str(tmp_path))
    try:
        target = str(tmp_path / "pivot.yaml")
        resolved, remaining, pipelines = cli_targets.resolve_pipeline_file_targets([target])

        assert "preprocess" in resolved, "Should resolve preprocess stage from pivot.yaml"
        assert "train" in resolved, "Should resolve train stage from pivot.yaml"
        assert remaining == [], "No targets should remain unresolved"
        assert pipelines, "Should return loaded pipeline"
        assert isinstance(pipelines[0], pipeline_mod.Pipeline), "Loaded pipeline should be returned"
    finally:
        sys.path.remove(str(tmp_path))
        if "stages" in sys.modules:
            del sys.modules["stages"]


def test_resolve_pipeline_file_targets_mixed_targets(tmp_path: pathlib.Path) -> None:
    """Mix of pipeline file and non-pipeline targets — pipeline resolved, others pass through."""
    sub = tmp_path / "sub"
    sub.mkdir()
    pipeline_code = """\
from pivot.pipeline.pipeline import Pipeline

pipeline = Pipeline("mixed_test")

def _my_stage():
    pass

pipeline.register(_my_stage, name="my_stage")
"""
    (sub / "pipeline.py").write_text(pipeline_code)

    targets = [str(sub / "pipeline.py"), "some_other_target"]
    resolved, remaining, pipelines = cli_targets.resolve_pipeline_file_targets(targets)

    assert resolved == {"my_stage"}, "Should resolve stages from the pipeline file"
    assert remaining == ["some_other_target"], "Non-pipeline targets pass through"
    assert pipelines, "Should return loaded pipeline"
