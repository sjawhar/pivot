"""Tests for pivot repro CLI command."""

from __future__ import annotations

import decimal
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Annotated

import numpy as np
import pandas as pd
import pytest

from conftest import isolated_pivot_dir, send_rpc
from pivot import cli, compose, executor, loaders
from pivot.storage import cache, track

if TYPE_CHECKING:
    from collections.abc import Callable

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


@compose.stage
def frame_metric() -> Annotated[pd.DataFrame, compose.metric]:
    return pd.DataFrame({"value": [1.0]})


@compose.stage
def open_ended_dims() -> pd.DataFrame:
    valid_to = np.array(["9999-12-31", "2024-01-01"], dtype="datetime64[us]")
    return pd.DataFrame({"k": ["a", "b"], "valid_to": valid_to})


@compose.stage
def current_dims(dims: pd.DataFrame) -> list[str]:
    return [str(k) for k in dims.loc[dims["valid_to"] > pd.Timestamp("2025-01-01"), "k"]]


@compose.stage
def matrix() -> pd.DataFrame:
    return pd.DataFrame(np.array([[1.5, 2.5], [3.5, 4.5]]))


@compose.stage
def first_col(m: pd.DataFrame) -> list[float]:
    return [float(value) for value in m[0]]


@compose.stage
def chatty_lossy() -> pd.DataFrame:
    for row in range(60):
        logging.getLogger("etl").warning("skipping malformed row %d", row)
    return pd.DataFrame({"price": [decimal.Decimal("0.1")]})


_EMPTY_PIPELINE = (
    "from pivot.compose import Pipeline\n\nwith Pipeline('test') as pipeline:\n    pass\n"
)

_DEEP_OUTPUT_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def make() -> dict:
    return {"made": True}


with Pipeline("deep") as pipeline:
    make()
"""


def test_repro_skips_an_oversized_output_hash_cache_key(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """An output path above LMDB's key limit still produces a successful committed run."""
    root = tmp_path / ("a" * 200) / ("b" * 200) / ("c" * 80)
    root.mkdir(parents=True)
    with isolated_pivot_dir(runner, root):
        pathlib.Path("pipeline.py").write_text(_DEEP_OUTPUT_PIPELINE)

        first = runner.invoke(cli.cli, ["repro"])
        second = runner.invoke(cli.cli, ["repro"])

        assert first.exit_code == 0, first.output
        assert "1 ran" in first.output, first.output
        assert pathlib.Path(".pivot", "stages", "deep", "make.lock").exists()
        assert second.exit_code == 0, second.output
        assert "0 ran, 1 cached" in second.output, second.output


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


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            "@stage\ndef s(previous: dict) -> dict:\n    return previous\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    s(previous=pipeline.previous())\n',
            "uses p.previous() but the return value is not tagged incremental",
            id="previous-on-untagged-return",
        ),
        pytest.param(
            "from typing import Annotated\n\nfrom pivot import incremental\n\n\n"
            + "@stage\ndef s(a: dict, b: dict) -> Annotated[dict, incremental]:\n"
            + "    return a\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n'
            + "    s(a=pipeline.previous(), b=pipeline.previous())\n",
            "parameters 'a' and 'b' both bind p.previous() to the return value",
            id="previous-bound-twice",
        ),
        pytest.param(
            "@stage\ndef s(raw: pd.DataFrame) -> pd.DataFrame:\n    return raw\n\n\n"
            + "@stage\ndef t(rows: pd.DataFrame) -> pd.DataFrame:\n    return rows\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    t(s(pipeline.input("t.csv")).rows)\n',
            "Stage 's' has no output 'rows': it has a single output, the return value",
            id="attribute-on-single-output",
        ),
        pytest.param(
            "@stage\ndef s(raw: pd.DataFrame) -> pd.DataFrame:\n    return raw\n\n\n"
            + "@stage\ndef t(rows: pd.DataFrame) -> pd.DataFrame:\n    return rows\n\n\n"
            + 'with Pipeline("demo") as pipeline:\n    t(s(pipeline.input("t.csv"))["rows"])\n',
            "Stage 's' has no output 'rows': it has a single output, the return value",
            id="item-on-single-output",
        ),
    ],
)
def test_list_names_a_single_output_as_the_return_value(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, body: str, expected: str
) -> None:
    """Errors about a stage's single output never show the internal key '_single'."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "t.csv").write_text("a\n1\n")
        pathlib.Path("pipeline.py").write_text(
            "import pandas as pd\n\nfrom pivot import Pipeline, stage\n\n\n" + body
        )

        result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code != 0, result.output
    # Quoted: the temp directory name ("test_list_names_a_single_output...") contains the bare word.
    assert repr(compose.SINGLE_OUTPUT_KEY) not in result.output, result.output
    assert expected in result.output, result.output


_SUB_ROOTED_PIPELINE = """
import pathlib

import pandas as pd

from pivot.compose import Pipeline, stage


@stage
def double(x: pd.DataFrame) -> pd.DataFrame:
    return x * 2


with Pipeline("sub", root=pathlib.Path(__file__).parent) as pipeline:
    double(pipeline.input("x.csv"))
"""


@pytest.mark.parametrize(
    "raw_dir",
    [
        pytest.param("sub/data/raw", id="input-under-the-pipeline-root"),
        pytest.param("data/raw", id="input-under-the-project-root"),
    ],
)
def test_repro_refuses_a_pipeline_rooted_below_the_project(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_dir: str,
) -> None:
    """root= that is not the project root would split locks, StateDB and outputs: refuse it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / raw_dir).mkdir(parents=True)
        (cwd / raw_dir / "x.csv").write_text("a\n1\n")
        (cwd / "sub").mkdir(exist_ok=True)
        (cwd / "sub" / "pipeline.py").write_text(_SUB_ROOTED_PIPELINE)
        monkeypatch.chdir(cwd / "sub")

        result = runner.invoke(cli.cli, ["repro"])

        assert result.exit_code == 1, result.output
        assert "Pipeline 'sub' sets root=" in result.output, result.output
        assert "always uses the project's .pivot and data/" in result.output, result.output
        assert "remove root=" in result.output, result.output
        assert not list(cwd.rglob("*.lock")), "no stage may run, so no lock is written"
        assert not (cwd / "data" / "sub").exists(), "no output is written"
        assert not (cwd / "sub" / "data" / "sub").exists(), "no output is written"


def test_repro_of_a_pipeline_file_refuses_a_root_below_the_project(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`pivot repro sub/pipeline.py` loads the file directly; it is refused the same way."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "x.csv").write_text("a\n1\n")
        (cwd / "sub").mkdir()
        (cwd / "sub" / "pipeline.py").write_text(_SUB_ROOTED_PIPELINE)
        (cwd / "pipeline.py").write_text(
            "from pivot.compose import Pipeline\n\nwith Pipeline('top') as pipeline:\n    pass\n"
        )

        result = runner.invoke(cli.cli, ["repro", "sub/pipeline.py"])

        assert result.exit_code == 1, result.output
        assert "Pipeline 'sub' sets root=" in result.output, result.output
        assert not list(cwd.rglob("*.lock")), "no stage may run, so no lock is written"


_SYMLINKED_STATE_PIPELINE = """
from pivot.compose import Pipeline, stage


@stage
def copy(value: str) -> str:
    return value


with Pipeline("symlinked") as pipeline:
    copy(pipeline.input("value.txt", t=str))
"""


def test_repro_and_status_accept_a_symlinked_project_state_dir(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A `.pivot` symlink is a supported project-state layout."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "value.txt").write_text("value")
        (cwd / "pipeline.py").write_text(_SYMLINKED_STATE_PIPELINE)
        state_dir = cwd.parent / f"{cwd.name}-state"
        (cwd / ".pivot").rename(state_dir)
        (cwd / ".pivot").symlink_to(state_dir, target_is_directory=True)

        repro = runner.invoke(cli.cli, ["repro"])
        status = runner.invoke(cli.cli, ["status"])

    assert repro.exit_code == 0, repro.output
    assert status.exit_code == 0, status.output
    assert (cwd / "data" / "symlinked" / "copy.txt").read_text() == "value"


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


def test_repro_serve_runs_stale_stages_at_startup(tmp_path: pathlib.Path) -> None:
    """repro --watch --serve runs stale stages once at startup without --force."""
    # Logs live outside the watched project so writing them triggers no watch events.
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".pivot").mkdir()
    raw_dir = project / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "input.txt").write_text("data")
    (project / "pipeline.py").write_text("""\
from __future__ import annotations

import pathlib

from pivot.compose import Pipeline, stage


@stage
def process(input_file: str) -> str:
    return input_file


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    process(pipeline.input("input.txt"))
""")

    output_path = project / "data" / "test" / "process.txt"
    stdout_path = tmp_path / "serve.stdout"
    stderr_path = tmp_path / "serve.stderr"
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(
            ["uv", "run", "pivot", "repro", "--watch", "--serve"],
            cwd=project,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    try:
        for _ in range(150):
            if output_path.exists():
                break
            time.sleep(0.1)

        assert output_path.exists(), (
            "Serve mode did not run the stale stage within 15s.\n"
            + f"stdout: {stdout_path.read_text(errors='replace')}\n"
            + f"stderr: {stderr_path.read_text(errors='replace')}"
        )
        assert output_path.read_text() == "data"
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


def test_repro_shows_lossy_output_warning_without_show_output(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """A plain `pivot repro` prints a stage's warning in its report and still exits 0."""
    with mock_discovery:
        frame_metric()

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    # The console wraps long lines; compare the words.
    text = " ".join(result.output.split())
    warning = (
        "WARNING: Stage 'test/frame_metric' wrote its YAML output in a form Pivot cannot "
        + "read back: YAML output must contain plain data; got DataFrame at root."
    )
    assert warning in text, result.output
    assert text.index("✓ test/frame_metric done") < text.index(warning), "prints under its stage"
    assert "Summary: 1 ran, 0 cached, 0 blocked, 0 failed" in text


def test_repro_quiet_suppresses_lossy_output_warning(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    with mock_discovery:
        frame_metric()

    result = runner.invoke(cli.cli, ["--quiet", "repro"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "", "Quiet mode should suppress the warning"


def test_repro_never_writes_an_out_of_range_date_as_a_different_date(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """A 9999-12-31 sentinel in a datetime64[us] column must not reach the next stage as 1816."""
    with mock_discovery:
        current_dims(open_ended_dims())

    result = runner.invoke(cli.cli, ["repro"])

    text = " ".join(result.output.split())
    assert "WARNING: Stage 'test/open_ended_dims' wrote its DataFrameJSONL output" in text, (
        result.output
    )
    assert "column 'valid_to'" in text
    written = (mock_discovery.root / "data" / "test" / "open_ended_dims.jsonl").read_text()
    assert "1816" not in written
    assert written == pd.DataFrame(
        {
            "k": ["a", "b"],
            "valid_to": np.array(["9999-12-31", "2024-01-01"], dtype="datetime64[us]"),
        }
    ).to_json(lines=True, orient="records")


def test_repro_keeps_integer_column_labels_for_the_next_stage(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """A frame built from a numpy array has labels 0, 1; the next stage indexes m[0]."""
    with mock_discovery:
        first_col(matrix())

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    written = (mock_discovery.root / "data" / "test" / "first_col.yaml").read_text()
    assert written == "- 1.5\n- 3.5\n"


def test_repro_reports_lossy_output_after_many_stage_warnings(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """The stage's own warnings are capped with a count; Pivot's own warning always shows."""
    with mock_discovery:
        chatty_lossy()

    result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "WARNING: skipping malformed row 49" in text
    assert "WARNING: skipping malformed row 50" not in text
    assert (
        "WARNING: Stage 'test/chatty_lossy' wrote its DataFrameJSONL output in a form Pivot "
        + "cannot read back: DataFrameJSONL cannot write column 'price'"
    ) in text, result.output
    assert "… 10 more warnings (run with --show-output)" in text


def _same_named_pipeline(side: str) -> str:
    return (
        "import pandas as pd\n\nfrom pivot import Pipeline, stage\n\n\n"
        + "@stage\ndef make(x: pd.DataFrame) -> dict:\n"
        + f'    return {{"from": "{side}"}}\n\n\n'
        + f"@stage\ndef use_{side}(m: dict) -> dict:\n"
        + '    return {"saw": m["from"]}\n\n\n'
        + 'with Pipeline("rep") as pipeline:\n'
        + f'    use_{side}(make(pipeline.input("x.csv")))\n'
    )


def test_list_all_refuses_two_pipeline_files_with_one_name(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Two pipeline.py files declaring Pipeline("rep") would share stages, locks and outputs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "x.csv").write_text("a\n1\n")
        for side in ("left", "right"):
            (cwd / side).mkdir()
            (cwd / side / "pipeline.py").write_text(_same_named_pipeline(side))

        result = runner.invoke(cli.cli, ["list", "--all"])

    assert result.exit_code != 0, result.output
    assert "Pipeline name 'rep'" in result.output, result.output
    assert "left/pipeline.py" in result.output, result.output
    assert "right/pipeline.py" in result.output, result.output


def test_list_refuses_include_of_a_different_stage_under_a_registered_name(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """include() keeps one stage reached twice, but not two functions under one name."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "x.csv").write_text("a\n1\n")
        for side in ("left", "right"):
            (cwd / side).mkdir()
            (cwd / side / "pipeline.py").write_text(_same_named_pipeline(side))
        pathlib.Path("pipeline.py").write_text(
            "from pivot import Pipeline\n"
            + "from left.pipeline import pipeline as left\n"
            + "from right.pipeline import pipeline as right\n\n"
            + 'pipeline = Pipeline("all")\n'
            + "pipeline.include(left)\n"
            + "pipeline.include(left)\n"
            + "pipeline.include(right)\n"
        )

        result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code != 0, result.output
    assert "Stage 'rep/make' is defined twice" in result.output, result.output
    assert "left/pipeline.py" in result.output, result.output
    assert "right/pipeline.py" in result.output, result.output


_SHARED_STAGES = (
    "import pandas as pd\n\nfrom pivot import StageParams, stage\n\n\n"
    + 'class TagParams(StageParams):\n    tag: str = "none"\n\n\n'
    + "@stage\ndef make(x: pd.DataFrame, params: TagParams) -> dict:\n"
    + '    return {"a": int(x["a"].iloc[0]), "tag": params.tag}\n'
)


def _shared_stage_pipeline(side: str, input_name: str) -> str:
    return (
        "from pivot import Pipeline, stage\n"
        + "from shared.stages import TagParams, make\n\n\n"
        + f"@stage\ndef use_{side}(m: dict) -> dict:\n"
        + '    return {"saw": m}\n\n\n'
        + 'with Pipeline("rep") as pipeline:\n'
        + f'    m = make(pipeline.input("{input_name}"), params=TagParams(tag="{side}"))\n'
        + f"    use_{side}(m)\n"
    )


@pytest.mark.parametrize(
    ("right_input", "expected"),
    [
        ("y.csv", ["with different inputs (x=x.csv vs x=y.csv)"]),
        ("x.csv", ["with different params", "tag='left'", "tag='right'"]),
    ],
    ids=["inputs-differ", "only-params-differ"],
)
def test_list_refuses_two_same_named_pipelines_calling_one_shared_stage(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, right_input: str, expected: list[str]
) -> None:
    """One shared function called with other inputs or params is two stages, not one."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "x.csv").write_text("a\n1\n")
        (cwd / "data" / "raw" / "y.csv").write_text("a\n2\n")
        (cwd / "shared").mkdir()
        (cwd / "shared" / "stages.py").write_text(_SHARED_STAGES)
        for side, input_name in (("left", "x.csv"), ("right", right_input)):
            (cwd / side).mkdir()
            (cwd / side / "pipeline.py").write_text(_shared_stage_pipeline(side, input_name))
        (cwd / "pipeline.py").write_text(
            "from pivot import Pipeline\n"
            + "from left.pipeline import pipeline as left\n"
            + "from right.pipeline import pipeline as right\n\n"
            + 'with Pipeline("all") as pipeline:\n'
            + "    pipeline.include(left)\n"
            + "    pipeline.include(right)\n"
        )

        result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code != 0, result.output
    assert "Stage 'rep/make' is defined twice" in result.output, result.output
    for fragment in expected:
        assert fragment in result.output, result.output
    assert "shared/stages.py" in result.output, result.output
    assert "rename one of the pipelines" in result.output, result.output


def test_list_refuses_two_same_named_pipelines_reading_one_input_differently(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The caller picks an input's format: one shared stage parsing x.csv two ways is two."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "x.csv").write_text("a;b\n1;2\n")
        (cwd / "shared").mkdir()
        (cwd / "shared" / "stages.py").write_text(
            "import pandas as pd\n\nfrom pivot import stage\n\n\n"
            + '@stage\ndef make(x: pd.DataFrame) -> dict:\n    return {"cols": list(x.columns)}\n'
        )
        for side, fmt in (("left", ""), ("right", ', format=loaders.CSV(sep=";")')):
            (cwd / side).mkdir()
            (cwd / side / "pipeline.py").write_text(
                "from pivot import Pipeline, loaders, stage\nfrom shared.stages import make\n\n\n"
                + f"@stage\ndef use_{side}(m: dict) -> dict:\n    return m\n\n\n"
                + 'with Pipeline("rep") as pipeline:\n'
                + f'    use_{side}(make(pipeline.input("x.csv"{fmt})))\n'
            )
        (cwd / "pipeline.py").write_text(
            "from pivot import Pipeline\n"
            + "from left.pipeline import pipeline as left\n"
            + "from right.pipeline import pipeline as right\n\n"
            + 'with Pipeline("all") as pipeline:\n'
            + "    pipeline.include(left)\n"
            + "    pipeline.include(right)\n"
        )

        result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code != 0, result.output
    assert (
        "Stage 'rep/make' is defined twice with different input formats "
        + "(x=CSV(index_col=None, sep=',', dtype=None) vs x=CSV(index_col=None, sep=';', "
        + "dtype=None)), both calling make in shared/stages.py"
    ) in result.output, result.output


def test_repro_included_stage_whose_loader_holds_a_plain_object_runs_then_stays_cached(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """include() deep-copies `OptsYAML(opts=Opts())`; a default-repr object has no address key."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "rows.yaml").write_text("a: 1\n")
        (cwd / "prep").mkdir()
        (cwd / "prep" / "pipeline.py").write_text(
            "import dataclasses\n\nfrom pivot import Pipeline, loaders, stage\n\n\n"
            + "class Opts:\n    def __init__(self) -> None:\n        self.depth = 2\n\n\n"
            + "@dataclasses.dataclass(frozen=True)\nclass OptsYAML(loaders.YAML):\n"
            + "    opts: object = None\n\n\n"
            + '@stage\ndef keep(rows: dict) -> dict:\n    return {"n": len(rows)}\n\n\n'
            + 'with Pipeline("prep") as pipeline:\n'
            + '    kept = keep(pipeline.input("rows.yaml", format=OptsYAML(opts=Opts())))\n'
        )
        (cwd / "report").mkdir()
        (cwd / "report" / "pipeline.py").write_text(
            "from pivot import Pipeline, stage\nfrom prep.pipeline import kept\n\n\n"
            + "@stage\ndef count(k: dict) -> dict:\n    return k\n\n\n"
            + 'with Pipeline("report") as pipeline:\n    count(kept)\n'
        )
        (cwd / "pipeline.py").write_text(
            "from pivot import Pipeline\n"
            + "from prep.pipeline import pipeline as prep\n"
            + "from report.pipeline import pipeline as report\n\n"
            + 'with Pipeline("all") as pipeline:\n'
            + "    pipeline.include(prep)\n"
            + "    pipeline.include(report)\n"
        )

        first = runner.invoke(cli.cli, ["repro"])
        second = runner.invoke(cli.cli, ["repro"])

    assert first.exit_code == 0, first.output
    assert "2 ran" in first.output, first.output
    assert second.exit_code == 0, second.output
    assert "0 ran, 2 cached" in second.output, second.output


def test_repro_pipeline_file_target_whose_loader_holds_a_user_function_runs(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Importing prep again re-creates `JSON(empty_factory=make_empty)`: still one stage."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "rows.json").write_text('{"a": 1}\n')
        (cwd / "prep").mkdir()
        (cwd / "prep" / "pipeline.py").write_text(
            "from pivot import Pipeline, loaders, stage\n\n\n"
            + "def make_empty() -> dict:\n    return {}\n\n\n"
            + '@stage\ndef keep(rows: dict) -> dict:\n    return {"n": len(rows)}\n\n\n'
            + 'with Pipeline("prep") as pipeline:\n'
            + '    keep(pipeline.input("rows.json", format=loaders.JSON(empty_factory=make_empty)))\n'
        )
        (cwd / "pipeline.py").write_text(
            "from pivot import Pipeline\nfrom prep.pipeline import pipeline as prep\n\n"
            + 'with Pipeline("all") as pipeline:\n    pipeline.include(prep)\n'
        )

        result = runner.invoke(cli.cli, ["repro", "prep/pipeline.py"])

    assert result.exit_code == 0, result.output
    assert "1 ran" in result.output, result.output


def _scale_pipeline(name: str) -> str:
    return (
        "import pandas as pd\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
        + "class ScaleParams(StageParams):\n    factor: int = 10\n\n\n"
        + "@stage\ndef scale(numbers: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:\n"
        + "    return numbers * params.factor\n\n\n"
        + f'with Pipeline("{name}") as pipeline:\n'
        + '    scale(pipeline.input("numbers.csv"), params=ScaleParams())\n'
    )


@pytest.mark.parametrize("run_from", ["root", "sub"])
def test_repro_pipeline_file_target_already_discovered_runs(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    run_from: str,
) -> None:
    """`pivot repro pipeline.py` imports the discovered file again; its stages are the same."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
        (cwd / "pipeline.py").write_text(_scale_pipeline("root"))
        (cwd / "sub").mkdir()
        (cwd / "sub" / "pipeline.py").write_text(_scale_pipeline("sub"))
        if run_from == "sub":
            monkeypatch.chdir(cwd / "sub")

        result = runner.invoke(cli.cli, ["repro", "pipeline.py"])

        assert result.exit_code == 0, result.output
        assert "1 ran" in result.output, result.output
        assert (cwd / "data" / run_from / "scale.jsonl").exists()


def _prep_pipeline(params_fields: str = "") -> str:
    return (
        "import pandas as pd\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
        + "class DropLowParams(StageParams):\n    min_score: float = 0.5\n"
        + params_fields
        + "\n\n@stage\ndef drop_low(raw: pd.DataFrame, params: DropLowParams) -> pd.DataFrame:\n"
        + '    return raw[raw["score"] >= params.min_score]\n\n\n'
        + 'with Pipeline("prep") as pipeline:\n'
        + '    kept = drop_low(pipeline.input("scores.csv"), params=DropLowParams())\n'
    )


_PREP_PIPELINE = _prep_pipeline()

_REPORT_PIPELINE = (
    "from typing import Annotated\n\nimport pandas as pd\n\n"
    + "from pivot import Pipeline, metric, stage\nfrom prep.pipeline import kept\n\n\n"
    + "@stage\ndef count(rows: pd.DataFrame) -> Annotated[dict, metric]:\n"
    + '    return {"rows": len(rows)}\n\n\n'
    + 'with Pipeline("report") as pipeline:\n'
    + "    count(kept)\n"
)


def _write_prep_report_project(cwd: pathlib.Path, prep: str = _PREP_PIPELINE) -> None:
    """docs/guides/multi-pipeline.md: prep/, report/ and a root pipeline including both."""
    (cwd / "data" / "raw").mkdir(parents=True)
    (cwd / "data" / "raw" / "scores.csv").write_text("name,score\na,0.2\nb,0.7\nc,0.9\n")
    for name, text in (("prep", prep), ("report", _REPORT_PIPELINE)):
        (cwd / name).mkdir()
        (cwd / name / "pipeline.py").write_text(text)
    (cwd / "pipeline.py").write_text(
        "from pivot import Pipeline\n"
        + "from prep.pipeline import pipeline as prep\n"
        + "from report.pipeline import pipeline as report\n\n"
        + 'with Pipeline("all") as pipeline:\n'
        + "    pipeline.include(prep)\n"
        + "    pipeline.include(report)\n"
    )


@pytest.mark.parametrize(
    ("target", "summary"),
    [("prep/pipeline.py", "1 ran"), ("report/pipeline.py", "2 ran")],
    ids=["prep", "report"],
)
def test_repro_included_pipeline_file_target_runs(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, target: str, summary: str
) -> None:
    """A target file the root pipeline already includes is imported again and still runs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_prep_report_project(cwd)

        result = runner.invoke(cli.cli, ["repro", target])

    assert result.exit_code == 0, result.output
    assert summary in result.output, result.output


_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")


def _pivot_with_hash_seed(
    cwd: pathlib.Path, hash_seed: int, *args: str
) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    env["PYTHONHASHSEED"] = str(hash_seed)
    return subprocess.run(
        [_PIVOT_BIN, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize("hash_seed", [0, 1, 2, 3])
def test_params_holding_sets_and_nan_stay_one_stage_in_every_process(
    tmp_path: pathlib.Path, hash_seed: int
) -> None:
    """include() deep-copies a set, which may reorder it, and a re-import makes a new NaN."""
    _write_prep_report_project(
        tmp_path,
        _prep_pipeline(
            "    ids: set[int] = {0, 3, 11}\n"
            + '    tags: frozenset[str] = frozenset({"alpha", "beta", "gamma", "delta", "eps"})\n'
            + '    ceiling: float = float("nan")\n'
        ),
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".pivot").mkdir()

    listed = _pivot_with_hash_seed(tmp_path, hash_seed, "list")
    ran = _pivot_with_hash_seed(tmp_path, hash_seed, "repro", "prep/pipeline.py")

    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "prep/drop_low" in listed.stdout, listed.stdout
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "1 ran" in ran.stdout + ran.stderr, ran.stdout + ran.stderr


def test_list_all_skips_a_file_that_re_exports_another_files_pipeline(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A root pipeline.py exporting prep's pipeline object declares no second pipeline."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "scores.csv").write_text("name,score\na,0.2\n")
        (cwd / "prep").mkdir()
        (cwd / "prep" / "pipeline.py").write_text(_PREP_PIPELINE)
        (cwd / "pipeline.py").write_text("from prep.pipeline import pipeline\n")

        result = runner.invoke(cli.cli, ["--verbose", "list", "--all", "--json"])

    assert result.exit_code == 0, result.output
    assert [s["name"] for s in json.loads(result.stdout)["stages"]] == ["prep/drop_low"]
    assert (
        "--all: prep/pipeline.py exports the same pipeline object 'prep' as pipeline.py; skipping"
    ) in result.stderr, result.stderr


def test_repro_diamond_of_includes_runs_the_shared_stage_once(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A includes B and C; both include D and consume its handle. D's stage is one stage."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "x.csv").write_text("a\n1\n")
        (cwd / "d").mkdir()
        (cwd / "d" / "pipeline.py").write_text(
            "import pandas as pd\n\nfrom pivot import Pipeline, stage\n\n\n"
            + '@stage\ndef make(x: pd.DataFrame) -> dict:\n    return {"a": 1}\n\n\n'
            + 'with Pipeline("d") as pipeline:\n    made = make(pipeline.input("x.csv"))\n'
        )
        for side in ("b", "c"):
            (cwd / side).mkdir()
            (cwd / side / "pipeline.py").write_text(
                "from pivot import Pipeline, stage\n"
                + "from d.pipeline import made\nfrom d.pipeline import pipeline as d\n\n\n"
                + f"@stage\ndef use_{side}(m: dict) -> dict:\n    return m\n\n\n"
                + f'with Pipeline("{side}") as pipeline:\n'
                + f"    pipeline.include(d)\n    use_{side}(made)\n"
            )
        (cwd / "pipeline.py").write_text(
            "from pivot import Pipeline\n"
            + "from b.pipeline import pipeline as b\nfrom c.pipeline import pipeline as c\n\n"
            + 'with Pipeline("a") as pipeline:\n'
            + "    pipeline.include(b)\n    pipeline.include(c)\n"
        )

        result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code == 0, result.output
    assert "3 ran" in result.output, result.output


def test_list_keeps_one_pipeline_included_three_times(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Twice inside the construction block and once after it: still one stage."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "scores.csv").write_text("name,score\na,0.2\n")
        (cwd / "prep").mkdir()
        (cwd / "prep" / "pipeline.py").write_text(_PREP_PIPELINE)
        (cwd / "pipeline.py").write_text(
            "from pivot import Pipeline\nfrom prep.pipeline import pipeline as prep\n\n"
            + 'with Pipeline("all") as pipeline:\n'
            + "    pipeline.include(prep)\n    pipeline.include(prep)\n"
            + "pipeline.include(prep)\n"
        )

        result = runner.invoke(cli.cli, ["list", "--json"])

    assert result.exit_code == 0, result.output
    assert [s["name"] for s in json.loads(result.stdout)["stages"]] == ["prep/drop_low"]


_WATCH_SCALE_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, StageParams, stage


class ScaleParams(StageParams):
    factor: int = 10


@stage
def scale(numbers: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    return numbers * params.factor


with Pipeline("smoke") as pipeline:
    scale(pipeline.input("numbers.csv"), params=ScaleParams())
"""


def _wait_for_watch(
    proc: subprocess.Popen[bytes], log_path: pathlib.Path, done: Callable[[], bool]
) -> None:
    """Wait up to 60s for ``done``; fail at once, with the log, if watch exits."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if done():
            return
        assert proc.poll() is None, (
            f"watch exited with rc {proc.returncode}:\n{log_path.read_text(errors='replace')}"
        )
        time.sleep(0.1)
    raise AssertionError(f"timed out:\n{log_path.read_text(errors='replace')}")


def test_repro_watch_waits_for_a_fixed_params_yaml_then_reruns(tmp_path: pathlib.Path) -> None:
    """A params.yaml error during watch is an invalid pipeline: watch stays up until it is fixed."""
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".pivot").mkdir()
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
    (project / "pipeline.py").write_text(_WATCH_SCALE_PIPELINE)
    params_yaml = project / "params.yaml"
    params_yaml.write_text("smoke/scale:\n  factor: 2\n")
    output = project / "data" / "smoke" / "scale.jsonl"
    log_path = tmp_path / "watch.log"
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    def scaled() -> list[int] | None:
        if not output.exists():
            return None
        return [int(n) for n in loaders.DataFrameJSONL().load(output)["n"].tolist()]

    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [_PIVOT_BIN, "repro", "--watch", "--debounce", "100"],
            cwd=project,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_for_watch(proc, log_path, lambda: scaled() == [2, 4])

        params_yaml.write_text("smoke/scael:\n  factor: 3\n")
        _wait_for_watch(
            proc,
            log_path,
            lambda: "Pipeline invalid - waiting for fix" in log_path.read_text(errors="replace"),
        )
        log_text = log_path.read_text(errors="replace")
        assert (
            "params.yaml key 'smoke/scael' names no stage in pipeline 'smoke'; "
            + "did you mean 'smoke/scale'?"
        ) in log_text, log_text
        time.sleep(1)
        assert proc.poll() is None, log_path.read_text(errors="replace")
        assert scaled() == [2, 4], "nothing runs while params.yaml is invalid"

        params_yaml.write_text("smoke/scale:\n  factor: 5\n")
        _wait_for_watch(proc, log_path, lambda: scaled() == [5, 10])
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


_WATCH_CPU_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


@stage(resources={"cpu": 2})
def solo(numbers: pd.DataFrame) -> pd.DataFrame:
    return numbers * 2


@stage
def other(numbers: pd.DataFrame) -> pd.DataFrame:
    return numbers * 3


with Pipeline("w") as pipeline:
    solo(pipeline.input("solo.csv"))
    other(pipeline.input("other.csv"))
"""


def test_repro_watch_runs_a_lone_cpu_stage_and_survives_a_capacity_error(
    tmp_path: pathlib.Path,
) -> None:
    """A watch run of one `cpu: 2` stage fits the worker count; an over-capacity request is an
    invalid pipeline, not the end of the watch."""
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".pivot").mkdir()
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "solo.csv").write_text("n\n1\n")
    (project / "data" / "raw" / "other.csv").write_text("n\n1\n")
    (project / ".pivot" / "config.yaml").write_text("core:\n  max_workers: 2\n")
    pipeline_py = project / "pipeline.py"
    pipeline_py.write_text(_WATCH_CPU_PIPELINE)
    output = project / "data" / "w" / "solo.jsonl"
    log_path = tmp_path / "watch.log"
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    def doubled() -> list[int] | None:
        if not output.exists():
            return None
        return [int(value) for value in loaders.DataFrameJSONL().load(output)["n"].tolist()]

    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [_PIVOT_BIN, "repro", "--watch", "--debounce", "100"],
            cwd=project,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_for_watch(proc, log_path, lambda: doubled() == [2])

        # Only `solo` is scheduled now: one stage, but the cpu capacity is still 2 workers.
        (project / "data" / "raw" / "solo.csv").write_text("n\n5\n")
        _wait_for_watch(proc, log_path, lambda: doubled() == [10])

        pipeline_py.write_text(_WATCH_CPU_PIPELINE.replace('"cpu": 2', '"gpu": 2'))
        _wait_for_watch(
            proc,
            log_path,
            lambda: "Pipeline invalid - waiting for fix" in log_path.read_text(errors="replace"),
        )
        assert "requests 2 units of resource 'gpu', but its capacity is 1" in log_path.read_text(
            errors="replace"
        )
        time.sleep(1)
        assert proc.poll() is None, log_path.read_text(errors="replace")
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_repro_watch_ctrl_c_exits_while_a_reload_import_hangs(tmp_path: pathlib.Path) -> None:
    """Ctrl-C stops plain watch at once even while an edited pipeline.py's import never returns."""
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".pivot").mkdir()
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "input.txt").write_text("data")
    source = """\
from __future__ import annotations

import pathlib

from pivot.compose import Pipeline, stage


@stage
def process(input_file: str) -> str:
    return input_file


with Pipeline("test", root=pathlib.Path(__file__).parent) as pipeline:
    process(pipeline.input("input.txt"))
"""
    pipeline_py = project / "pipeline.py"
    pipeline_py.write_text(source)
    output_path = project / "data" / "test" / "process.txt"
    # Outside the project, so neither the marker nor the logs are watch events
    importing = tmp_path / "importing"
    stdout_path = tmp_path / "watch.stdout"
    stderr_path = tmp_path / "watch.stderr"
    env = os.environ.copy()
    env["PIVOT_CACHE_DIR"] = str(tmp_path / "cache")

    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(
            [_PIVOT_BIN, "repro", "--watch"],
            cwd=project,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    def _logs() -> str:
        return (
            f"stdout: {stdout_path.read_text(errors='replace')}\n"
            + f"stderr: {stderr_path.read_text(errors='replace')}"
        )

    try:
        for _ in range(300):
            if output_path.exists():
                break
            time.sleep(0.1)
        assert output_path.exists(), f"the startup run did not finish within 30s\n{_logs()}"

        pipeline_py.write_text(
            source
            + f"\npathlib.Path({str(importing)!r}).write_text('importing')\n"
            + "import time\ntime.sleep(600)\n"
        )
        for _ in range(300):
            if importing.exists():
                break
            time.sleep(0.1)
        assert importing.exists(), f"the reload did not import the edit within 30s\n{_logs()}"

        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pytest.fail(f"watch was still running 2 s after Ctrl-C\n{_logs()}")
        assert proc.returncode == 0, _logs()
        assert "Watch mode stopped." in stdout_path.read_text(), _logs()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
