"""Tests for pivot repro CLI command."""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir, send_rpc
from pivot import cli, compose, executor
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing

# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def stage_a() -> str:
    return "a"


@compose.stage
def stage_b(dep: str) -> str:
    return "b"


@compose.stage
def stage_c(dep: str) -> str:
    return "c"


@compose.stage
def process(input_file: str) -> str:
    return "done"


@compose.stage
def printer() -> str:
    # This will be captured and should appear in output
    sys.stdout.write("Processing data...\n")
    sys.stdout.flush()
    return "done"


@compose.stage
def warner() -> str:
    sys.stderr.write("Warning: something happened\n")
    sys.stderr.flush()
    return "done"


@compose.stage
def quiet() -> str:
    sys.stdout.write("This should not appear\n")
    sys.stdout.flush()
    return "done"


_EMPTY_PIPELINE = (
    "from pivot.compose import Pipeline\n\nwith Pipeline('test') as pipeline:\n    pass\n"
)


def _output(pipeline: compose.Pipeline, stage_name: str) -> pathlib.Path:
    return pipeline.root / "data" / "test" / f"{stage_name}.txt"


def _register_a(pipeline: compose.Pipeline) -> None:
    with pipeline:
        stage_a()


def _register_a_b_c(pipeline: compose.Pipeline) -> None:
    with pipeline:
        stage_c(stage_b(stage_a()))


def _write_input(pipeline: compose.Pipeline) -> pathlib.Path:
    path = pipeline.root / "data" / "raw" / "input.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")
    return path


def _register_process(pipeline: compose.Pipeline) -> None:
    with pipeline:
        process(pipeline.input("input.txt", t=str))


def _run_process_then_track_and_delete_input(pipeline: compose.Pipeline) -> None:
    """Run `process`, write a .pvt for its input, then delete the input (CI scenario)."""
    input_path = _write_input(pipeline)
    _register_process(pipeline)
    results = executor.run(pipeline=pipeline)
    assert results["test/process"]["status"] == "ran", results

    input_hash, _ = cache.hash_file(input_path)
    pvt_data = track.PvtData(path="input.txt", hash=input_hash, size=4)
    track.write_pvt_file(input_path.parent / "input.txt.pvt", pvt_data)

    input_path.unlink()


# =============================================================================
# Basic Command Tests
# =============================================================================


def test_repro_runs_entire_pipeline(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro without arguments runs entire pipeline."""
    _register_a_b_c(mock_discovery)

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").read_text() == "a"
    assert _output(mock_discovery, "stage_b").read_text() == "b"
    assert _output(mock_discovery, "stage_c").read_text() == "c"


def test_repro_runs_stage_with_dependencies(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro STAGE runs stage and its dependencies."""
    _register_a_b_c(mock_discovery)

    # Run only stage_b - should also run stage_a
    result = runner.invoke(cli.cli, ["repro", "test/stage_b"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").exists(), "Dependency should have been created"
    assert _output(mock_discovery, "stage_b").exists(), "Target stage should have run"
    assert not _output(mock_discovery, "stage_c").exists(), "Downstream stage should NOT run"


def test_repro_bare_stage_name_runs_stage_with_dependencies(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro resolves a bare stage name to its qualified form (names.resolve_stage_name)."""
    _register_a_b_c(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "stage_b"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").exists()
    assert _output(mock_discovery, "stage_b").exists()
    assert not _output(mock_discovery, "stage_c").exists()


# =============================================================================
# Dry Run Tests
# =============================================================================


def test_repro_dry_run_shows_what_would_run(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --dry-run shows what would run without executing."""
    with mock_discovery:
        stage_b(stage_a())

    result = runner.invoke(cli.cli, ["repro", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would run:" in result.output
    assert "test/stage_a" in result.output
    assert "test/stage_b" in result.output
    assert not _output(mock_discovery, "stage_a").exists(), "Should not create files"


def test_repro_dry_run_json_output(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --dry-run --json outputs JSON format."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "stages" in data
    assert "test/stage_a" in data["stages"]


def test_repro_dry_run_allow_missing_uses_pvt_hash(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --dry-run --allow-missing uses .pvt hash when dep file is missing."""
    _run_process_then_track_and_delete_input(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--dry-run", "--allow-missing"])

    # Should show "would skip" not "Missing deps"
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert "would skip" in result.output.lower(), f"Got: {result.output}"


def test_repro_reports_paths_for_multiple_missing_pipeline_inputs(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Plain repro formats every missing input path from its error group."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(
            """
from pivot.compose import Pipeline, stage


@stage
def combine(left: str, extra: str) -> str:
    return left + extra


with Pipeline("demo") as pipeline:
    left = pipeline.input("left.csv", path="missing/left.csv", t=str)
    extra = pipeline.input("extra.csv", path="missing/extra.csv", t=str)
    combine(left, extra)
"""
        )

        result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code != 0
    assert "2 missing dependency error(s):" in result.output
    assert "input 'left.csv' (path missing/left.csv)" in result.output
    assert "input 'extra.csv' (path missing/extra.csv)" in result.output
    assert "Tip: Ensure all dependency files exist or are produced by other stages" in result.output


# Pipeline "demo" stage "x" field "y" and pipeline "demo/x" stage "y" both map to
# data/demo/x/y.jsonl.
_COLLIDING_OUTPUTS_PIPELINE = """
from typing import TypedDict

import pandas as pd

from pivot import Pipeline, stage


class XResult(TypedDict):
    y: pd.DataFrame


@stage
def x(raw: pd.DataFrame) -> XResult:
    out = raw.copy()
    out["who"] = "x field y"
    return {"y": out}


@stage
def y(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["who"] = "stage y"
    return out


with Pipeline("demo/x") as inner:
    y(inner.input("t.csv"))

with Pipeline("demo") as pipeline:
    x(pipeline.input("t.csv"))

pipeline.include(inner)
"""


def test_repro_refuses_two_outputs_that_map_to_one_path(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Two artifacts resolving to one workspace file fail at load, naming both, before any write."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "t.csv").write_text("a\n1\n")
        pathlib.Path("pipeline.py").write_text(_COLLIDING_OUTPUTS_PIPELINE)

        result = runner.invoke(cli.cli, ["repro"])
        shared_file_written = (cwd / "data" / "demo" / "x" / "y.jsonl").exists()

    assert result.exit_code != 0, result.output
    assert "Output path conflict" in result.output, result.output
    assert "'demo/x:y'" in result.output, result.output
    assert "'demo/x/y'" in result.output, result.output
    assert "data/demo/x/y.jsonl" in result.output, result.output
    assert not shared_file_written, "Neither stage may run once their paths collide"


_ALIASED_INPUT_PIPELINE = """
import pandas as pd

from pivot import Pipeline, stage


@stage
def join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([left, right])


with Pipeline("demo") as pipeline:
    join(pipeline.input("t.csv"), pipeline.input("alias", path="data/raw/t.csv"))
"""


def test_list_refuses_two_input_names_bound_to_one_path(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Two input names on one file fail at load with a binding conflict, not mid-repro."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "t.csv").write_text("a\n1\n")
        pathlib.Path("pipeline.py").write_text(_ALIASED_INPUT_PIPELINE)

        result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code != 0, result.output
    assert (
        "Input binding conflict: inputs 't.csv' and 'alias' are both bound to 'data/raw/t.csv'"
        in result.output
    ), result.output
    assert "ExceptionGroup" not in result.output, result.output


# Pipeline "raw": stage "s" writes data/raw/s.jsonl, which is also the default binding of
# pipeline.input("s.jsonl"); an explicit path to another stage's output collides the same way.
_INPUT_ON_OUTPUT_PIPELINE = """
import pandas as pd

from pivot import Pipeline, stage


@stage
def s(seed: pd.DataFrame) -> pd.DataFrame:
    return seed


@stage
def t(data: pd.DataFrame) -> pd.DataFrame:
    return data


with Pipeline("raw") as pipeline:
    s(pipeline.input("seed.csv"))
    t(INPUT)
"""


@pytest.mark.parametrize(
    ("input_call", "input_name", "path", "output"),
    [
        pytest.param(
            'pipeline.input("s.jsonl")', "s.jsonl", "data/raw/s.jsonl", "raw/s", id="default-path"
        ),
        pytest.param(
            'pipeline.input("copy", path="data/raw/s.jsonl")',
            "copy",
            "data/raw/s.jsonl",
            "raw/s",
            id="explicit-path",
        ),
        pytest.param(
            'pipeline.input("dir", path="data/raw")',
            "dir",
            "data/raw",
            "raw/s",
            id="input-contains-output",
        ),
    ],
)
def test_list_refuses_an_input_bound_to_an_output_path(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    input_call: str,
    input_name: str,
    path: str,
    output: str,
) -> None:
    """An input on, inside, or around another stage's output fails at load, naming both."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "seed.csv").write_text("a\n1\n")
        pathlib.Path("pipeline.py").write_text(
            _INPUT_ON_OUTPUT_PIPELINE.replace("INPUT", input_call)
        )

        result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code != 0, result.output
    assert "Output path conflict" in result.output, result.output
    assert f"input '{input_name}'" in result.output, result.output
    assert f"'{output}'" in result.output, result.output
    assert path in result.output, result.output


# =============================================================================
# Explain Mode Tests
# =============================================================================


def test_repro_explain_shows_detailed_breakdown(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --explain shows detailed stage explanations."""
    with mock_discovery:
        stage_b(stage_a())

    result = runner.invoke(cli.cli, ["repro", "--explain"])

    assert result.exit_code == 0, result.output
    # Explain mode shows more detailed output than dry-run
    assert "Stage: test/stage_a" in result.output
    assert "Stage: test/stage_b" in result.output
    assert "No previous run" in result.output


def test_repro_explain_allow_missing_uses_pvt_hash(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --explain --allow-missing uses .pvt hash when dep file is missing."""
    _run_process_then_track_and_delete_input(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--explain", "--allow-missing"])

    # Should NOT show error about missing deps
    assert "Missing deps" not in result.output, f"Got: {result.output}"
    assert result.exit_code == 0, f"Expected success, got: {result.output}"


# =============================================================================
# Option Validation Tests
# =============================================================================


def test_repro_allow_missing_requires_dry_run_or_explain(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --allow-missing without --dry-run or --explain errors."""
    _write_input(mock_discovery)
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--allow-missing"])

    assert result.exit_code != 0
    assert "--allow-missing" in result.output
    assert "--dry-run" in result.output or "--explain" in result.output


def test_repro_serve_requires_watch(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --serve without --watch errors."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--serve"])

    assert result.exit_code != 0
    assert "--serve" in result.output
    assert "--watch" in result.output


def test_repro_serve_with_watch_uses_headless_mode(tmp_path: pathlib.Path) -> None:
    """repro --serve --watch starts an RPC-accessible headless daemon."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "input.txt").write_text("data")
    (tmp_path / "pipeline.py").write_text("""\
from __future__ import annotations

import pathlib

from pivot.compose import Pipeline, stage


@stage
def process(input_file: str) -> str:
    return input_file


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    process(pipeline.input("input.txt"))
""")

    socket_path = tmp_path / ".pivot" / "agent.sock"
    stdout_path = tmp_path / "serve.stdout"
    stderr_path = tmp_path / "serve.stderr"
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    # The child logs every RPC disconnect. Files avoid a PIPE filling and blocking it.
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(
            ["uv", "run", "pivot", "repro", "--watch", "--serve"],
            cwd=tmp_path,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    try:
        response: dict[str, object] | None = None
        for _ in range(100):
            if socket_path.exists():
                try:
                    candidate = send_rpc(socket_path, "status")
                    if "result" in candidate:
                        response = candidate
                        break
                except (ConnectionRefusedError, FileNotFoundError):
                    pass
            time.sleep(0.1)

        assert response is not None, (
            "Agent socket did not become RPC-ready within 10s.\n"
            + f"stdout: {stdout_path.read_text(errors='replace')}\n"
            + f"stderr: {stderr_path.read_text(errors='replace')}"
        )
        assert "result" in response
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_repro_debounce_requires_watch(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --debounce without --watch errors."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--debounce", "500"])

    assert result.exit_code != 0
    assert "--debounce" in result.output
    assert "--watch" in result.output


def test_repro_tui_log_cannot_use_with_json(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --tui-log cannot be used with --json."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--tui-log", "test.log", "--json"])

    assert result.exit_code != 0
    assert "--tui-log" in result.output
    assert "--jsonl" in result.output


def test_repro_tui_log_cannot_use_with_dry_run(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --tui-log cannot be used with --dry-run."""
    _register_a(mock_discovery)

    # Note: --tui is required for --tui-log, so we include it to test the --dry-run validation
    result = runner.invoke(cli.cli, ["repro", "--tui", "--tui-log", "test.log", "--dry-run"])

    assert result.exit_code != 0
    assert "--tui-log" in result.output
    assert "--dry-run" in result.output


def test_repro_unknown_stage_errors(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro with unknown stage name errors with helpful message."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "nonexistent_stage" in result.output


def test_repro_pipeline_file_target_registers_stages(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """repro with pipeline.py path registers and resolves stage targets."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        subdir = cwd / "sub"
        subdir.mkdir()
        pipeline_path = subdir / "pipeline.py"
        pipeline_path.write_text("""\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def stage_a() -> str:
    return "a"


with Pipeline("sub") as pipeline:
    stage_a()
""")

        result = runner.invoke(cli.cli, ["repro", "--dry-run", str(pipeline_path)])

        assert result.exit_code == 0, f"repro failed: {result.output}"
        assert "Would run:" in result.output, "Dry-run output should include plan summary"
        assert "sub/stage_a" in result.output, "Pipeline file stage should be resolved"


# =============================================================================
# File Path Resolution Tests
# =============================================================================


def test_repro_with_output_file_path(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro with output file path runs correct stage and dependencies."""
    with mock_discovery:
        stage_b(stage_a())

    # Run repro with stage_b's output file path
    result = runner.invoke(cli.cli, ["repro", "data/test/stage_b.txt"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    assert _output(mock_discovery, "stage_a").exists(), "Dependency stage_a should have run"
    assert _output(mock_discovery, "stage_b").exists(), "Target stage_b should have run"


def test_repro_unknown_file_path_errors(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro with unknown file path shows helpful error mentioning output paths."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "nonexistent.csv"])

    assert result.exit_code != 0, "Should error on unknown file path"
    assert "nonexistent.csv" in result.output, "Error should mention the unknown target"
    assert "output file paths" in result.output or "Targets can be" in result.output, (
        "Error should mention that targets can be file paths"
    )


# =============================================================================
# Force Mode Tests
# =============================================================================


def test_repro_force_reruns_cached_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --force re-runs stages even if cached."""
    _register_a(mock_discovery)

    # First run
    result = runner.invoke(cli.cli, ["repro"])
    assert result.exit_code == 0, result.output

    # Second run without force - should skip
    result = runner.invoke(cli.cli, ["repro", "--dry-run"])
    assert "would skip" in result.output.lower()

    # Run with force - should run
    result = runner.invoke(cli.cli, ["repro", "--dry-run", "--force"])
    assert "would run" in result.output.lower()


# =============================================================================
# Keep-Going Mode Tests
# =============================================================================


def test_repro_keep_going_option_accepted(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --keep-going option is accepted."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--keep-going"])

    assert result.exit_code == 0, result.output
    assert _output(mock_discovery, "stage_a").exists()


# =============================================================================
# No Single-Stage Option Tests
# =============================================================================


def test_repro_does_not_have_single_stage_option(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro does not have --single-stage option."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--single-stage"])

    # Should error because option doesn't exist
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


# =============================================================================
# Empty Pipeline Tests
# =============================================================================


def test_repro_empty_pipeline_shows_message(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro with no stages shows appropriate message."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--dry-run"])

        # Should show "no stages" message or succeed with empty output
        assert result.exit_code == 0, result.output
        assert "No stages to run" in result.output


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_repro_json_output_streams_jsonl(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --json streams JSONL output."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--json"])

    assert result.exit_code == 0, result.output
    # Check that output is valid JSONL (multiple JSON lines)
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)  # Should not raise


def test_repro_jsonl_flag_accepted(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--jsonl flag works and streams JSONL output."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--jsonl"])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.strip().split("\n") if ln]
    assert len(lines) > 0
    for line in lines:
        json.loads(line)  # Should not raise


def test_repro_help_shows_jsonl(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """repro help shows --jsonl flag with JSONL description."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["repro", "--help"])

    assert result.exit_code == 0
    assert "--jsonl" in result.output
    assert "JSONL" in result.output


# =============================================================================
# Cache Options Tests
# =============================================================================


def test_repro_no_commit_option_accepted(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """repro --no-commit option is accepted."""
    _register_a(mock_discovery)

    result = runner.invoke(cli.cli, ["repro", "--no-commit"])

    assert result.exit_code == 0, result.output


# =============================================================================
# Show Output Flag Tests
# =============================================================================


def test_repro_show_output_mutually_exclusive_with_tui(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--show-output and --tui are mutually exclusive."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--show-output", "--tui"])

        assert result.exit_code != 0
        assert "--show-output and --tui are mutually exclusive" in result.output


def test_repro_show_output_mutually_exclusive_with_json(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--show-output and --json are mutually exclusive."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)

        result = runner.invoke(cli.cli, ["repro", "--show-output", "--json"])

        assert result.exit_code != 0
        assert "--show-output and --jsonl/--json are mutually exclusive" in result.output


def test_repro_show_output_mutually_exclusive_with_quiet(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--show-output and --quiet are mutually exclusive."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)

        result = runner.invoke(cli.cli, ["--quiet", "repro", "--show-output"])

        assert result.exit_code != 0
        assert "--show-output and --quiet are mutually exclusive" in result.output


def test_repro_show_output_streams_stage_logs(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stage stdout to terminal."""
    with mock_discovery:
        printer()

    result = runner.invoke(cli.cli, ["repro", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should contain the log line with stage prefix
    assert "printer" in result.output
    assert "Processing data..." in result.output


def test_repro_show_output_streams_stderr_in_red(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """--show-output streams stderr with red formatting."""
    with mock_discovery:
        warner()

    result = runner.invoke(cli.cli, ["repro", "--show-output"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should contain stderr line
    assert "warner" in result.output
    assert "Warning: something happened" in result.output


def test_repro_without_show_output_hides_logs(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """Default behavior (no --show-output) doesn't show stage logs."""
    with mock_discovery:
        quiet()

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, f"Failed: {result.output}"
    # Should NOT contain the log line (only stage completion message)
    assert "This should not appear" not in result.output
    # But should show completion
    assert "quiet" in result.output
