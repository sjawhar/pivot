"""`pivot repro` reports a mapped stage's expansion (``Pipeline.map``) on the console and in JSONL."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

from pivot import cli, compose, loaders

if TYPE_CHECKING:
    from click.testing import CliRunner


@compose.stage
def plan_files(
    raw: str,
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    return {"a.yaml": {"value": 1}, "b.yaml": {"value": 2}}


@compose.stage
def cut(entry: dict[str, int]) -> dict[str, int]:
    return {"cut": entry["value"] * 10}


def _map_pipeline(pipeline: compose.Pipeline) -> None:
    raw_dir = pipeline.root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text("data")
    with pipeline:
        cut(pipeline.map(plan_files(pipeline.input("input.txt", t=str))))


def test_repro_prints_one_line_when_a_mapped_stage_expands(
    mock_discovery: compose.Pipeline, runner: CliRunner
) -> None:
    _map_pipeline(mock_discovery)

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    expansion = [line.strip() for line in lines if "expanded" in line]
    assert expansion == ["test/cut: expanded into 2 stages"], result.output
    for instance in ("test/cut@a", "test/cut@b"):
        assert any(instance in line and "done" in line for line in lines), result.output


def test_repro_jsonl_reports_the_expansion_before_its_instances_start(
    mock_discovery: compose.Pipeline, runner: CliRunner
) -> None:
    _map_pipeline(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--jsonl"])

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    expanded = [event for event in events if event["type"] == "stages_expanded"]
    assert expanded == [
        {"type": "stages_expanded", "template": "test/cut", "stages": ["test/cut@a", "test/cut@b"]}
    ], result.output
    instance_starts = [
        position
        for position, event in enumerate(events)
        if event["type"] == "stage_start" and event["stage"].startswith("test/cut@")
    ]
    assert len(instance_starts) == 2, result.output
    assert events.index(expanded[0]) < min(instance_starts), "the expansion precedes its instances"
