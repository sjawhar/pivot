from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import pytest
import yaml

from conftest import isolated_pivot_dir
from pivot import cli, compose
from pivot import status as status_mod
from pivot.storage import cache, track
from pivot.types import RemoteSyncInfo

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture


# =============================================================================
# Module-level stage functions (required for pickling)
# =============================================================================


@compose.stage
def process(input_file: str) -> str:
    return "done"


# =============================================================================
# pipeline.py templates (for tests that go through real CLI discovery)
# =============================================================================

_EMPTY_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline

with Pipeline("test") as pipeline:
    pass
"""

# PROCESS_RESULT is substituted so tests can change the stage's code
_PROCESS_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def process(input_file: str) -> str:
    return PROCESS_RESULT


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""


_RESOURCE_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage(resources={"gpu": CAPACITY})
def process(input_file: str) -> str:
    return "done"


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""


_ALIASED_RESOURCE_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage as task


@task(resources={"gpu": CAPACITY})
def process(input_file: str) -> str:
    return "done"


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""


_NESTED_RESOURCE_DECORATOR_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage(resources={"gpu": 1})
def process(input_file: str) -> str:
    class Other:
        @staticmethod
        def stage(*, resources: dict[str, int]):
            def decorate(func):
                return func
            return decorate

    other = Other()

    @other.stage(resources={"delta": DELTA})
    def nested() -> str:
        return "done"

    return nested()


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""


_USER_DECORATOR_RESOURCE_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


def marker_one(func):
    return func


def marker_two(func):
    return func


@MARKER
@stage(resources={"gpu": 1})
def process(input_file: str) -> str:
    return "done"


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""


_HOSTILE_RESOURCE_DECORATOR_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


class Bomb:
    def __getattribute__(self, name):
        if name == "__dict__":
            raise RuntimeError("fingerprinting evaluated unrelated decorator")
        return object.__getattribute__(self, name)

    def stage(self, *, resources: dict[str, int]):
        def decorate(func):
            return func
        return decorate


bomb = Bomb()


@bomb.stage(resources={"delta": DELTA})
@stage(resources={"gpu": 1})
def process(input_file: str) -> str:
    return "done"


with Pipeline("test") as pipeline:
    process(pipeline.input("input.txt"))
"""

_STAGE_DECORATOR_FORMS = {
    "bare": ("from pivot.compose import Pipeline, stage\n", "@stage"),
    "called": ("from pivot.compose import Pipeline, stage\n", '@stage(resources={"gpu": 1})'),
    "module-bare": (
        "from pivot import compose\nfrom pivot.compose import Pipeline\n",
        "@compose.stage",
    ),
    "module-called": (
        "from pivot import compose\nfrom pivot.compose import Pipeline\n",
        '@compose.stage(resources={"gpu": 1})',
    ),
}


def _stage_decorator_pipeline(imports: str, decorator: str) -> str:
    return (
        f"from __future__ import annotations\n\n{imports}\n\n{decorator}\n"
        + 'def process(input_file: str) -> str:\n    return "done"\n\n\n'
        + 'with Pipeline("test") as pipeline:\n    process(pipeline.input("input.txt"))\n'
    )


_STAGES_A_B_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def stage_a(input_file: str) -> str:
    return "output a"


@stage
def stage_b(input_file: str) -> str:
    return "output b"


with Pipeline("test") as pipeline:
    raw = pipeline.input("input.txt")
    stage_a(raw)
    stage_b(raw)
"""

# stage_b consumes stage_a's output; STAGE_A_RESULT is substituted to change stage_a's code
_UPSTREAM_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage


@stage
def stage_a(input_file: str) -> str:
    return STAGE_A_RESULT


@stage
def stage_b(a_output: str) -> str:
    return "b"


with Pipeline("test") as pipeline:
    stage_b(stage_a(pipeline.input("input.txt")))
"""


def _write_input(root: pathlib.Path, content: str = "data") -> None:
    """Write data/raw/input.txt, where ``pipeline.input("input.txt")`` reads."""
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "input.txt").write_text(content)


def _write_process_pipeline(result: str = "done") -> None:
    pathlib.Path("pipeline.py").write_text(
        _PROCESS_PIPELINE.replace("PROCESS_RESULT", repr(result))
    )


def _write_upstream_pipeline(stage_a_result: str) -> None:
    pathlib.Path("pipeline.py").write_text(
        _UPSTREAM_PIPELINE.replace("STAGE_A_RESULT", repr(stage_a_result))
    )


def _register_process(pipeline: compose.Pipeline) -> None:
    _write_input(pipeline.root)
    with pipeline:
        process(pipeline.input("input.txt", t=str))


def _write_tracked_file(content: str) -> None:
    data_file = pathlib.Path("data.txt")
    data_file.write_text(content)
    file_hash, _ = cache.hash_file(data_file)
    pvt_data = track.PvtData(path="data.txt", hash=file_hash, size=len(content))
    track.write_pvt_file(pathlib.Path("data.txt.pvt"), pvt_data)


# =============================================================================
# Help and Basic Tests
# =============================================================================


def test_status_help(runner: click.testing.CliRunner) -> None:
    """Status command should show help."""
    result = runner.invoke(cli.cli, ["status", "--help"])

    assert result.exit_code == 0
    assert "--verbose" in result.output
    assert "--json" in result.output
    assert "--stages-only" in result.output
    assert "--tracked-only" in result.output
    assert "--remote-only" in result.output
    assert "--remote" in result.output


def test_status_no_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status with no stages shows appropriate message."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "No stages registered" in result.output


# =============================================================================
# Pipeline Status Tests
# =============================================================================


def test_status_shows_stale_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status shows stale stages."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_process_pipeline()

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "stale" in result.output
        assert "test/process" in result.output


def test_status_shows_cached_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status shows cached stages after run."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_process_pipeline()

        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "1 cached" in result.output
        assert "0 stale" in result.output


_NAN_PARAMS_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, StageParams, stage


class ScaleParams(StageParams):
    factor: int = 10
    ceiling: float = float("nan")


@stage
def scale(numbers: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:
    return numbers * params.factor


@stage
def total(scaled: pd.DataFrame, params: ScaleParams) -> dict:
    return {"sum": int(scaled.to_numpy().sum())}


with Pipeline("nan") as pipeline:
    total(scale(pipeline.input("numbers.csv"), params=ScaleParams()), params=ScaleParams())
"""


def test_status_keeps_a_stage_with_a_nan_param_cached(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The lock records NaN, and nan != nan; params compare by their canonical JSON text."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
        pathlib.Path("pipeline.py").write_text(_NAN_PARAMS_PIPELINE)
        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        status = runner.invoke(cli.cli, ["status", "--explain"])
        again = runner.invoke(cli.cli, ["--quiet", "repro", "--json"])

    assert status.exit_code == 0, status.output
    assert "2 unchanged" in status.output, status.output
    assert "Params changed" not in status.output, status.output
    assert again.exit_code == 0, again.output
    reasons = {
        event["stage"]: event["reason"]
        for event in map(json.loads, again.output.strip().splitlines())
        if event["type"] == "stage_complete"
    }
    assert reasons == {"nan/scale": "unchanged", "nan/total": "unchanged (generation)"}, reasons


def test_status_keeps_stage_cached_when_only_resources_change(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Scheduler resource declarations do not invalidate a stage fingerprint."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pipeline_path = pathlib.Path("pipeline.py")
        pipeline_path.write_text(_RESOURCE_PIPELINE.replace("CAPACITY", "1"))

        repro = runner.invoke(cli.cli, ["repro", "--resources", "gpu=2"])
        assert repro.exit_code == 0, repro.output

        pipeline_path.write_text(_RESOURCE_PIPELINE.replace("CAPACITY", "2"))
        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "1 cached" in result.output
        assert "0 stale" in result.output


def test_status_keeps_aliased_resource_stage_cached(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """An aliased Pivot stage decorator excludes resources from its fingerprint."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pipeline_path = pathlib.Path("pipeline.py")
        pipeline_path.write_text(_ALIASED_RESOURCE_PIPELINE.replace("CAPACITY", "1"))

        repro = runner.invoke(cli.cli, ["repro", "--resources", "gpu=2"])
        assert repro.exit_code == 0, repro.output

        pipeline_path.write_text(_ALIASED_RESOURCE_PIPELINE.replace("CAPACITY", "2"))
        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "1 cached" in result.output
        assert "0 stale" in result.output


def test_status_detects_nested_unrelated_resource_decorator_change(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A nested decorator that is not Pivot's stage decorator remains fingerprinted."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pipeline_path = pathlib.Path("pipeline.py")
        pipeline_path.write_text(_NESTED_RESOURCE_DECORATOR_PIPELINE.replace("DELTA", "1"))

        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        pipeline_path.write_text(_NESTED_RESOURCE_DECORATOR_PIPELINE.replace("DELTA", "2"))
        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "0 cached" in result.output
        assert "1 stale" in result.output


def test_status_detects_user_decorator_change_on_resource_stage(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A user decorator remains in a resource stage's fingerprint."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pipeline_path = pathlib.Path("pipeline.py")
        pipeline_path.write_text(_USER_DECORATOR_RESOURCE_PIPELINE.replace("MARKER", "marker_one"))

        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        pipeline_path.write_text(_USER_DECORATOR_RESOURCE_PIPELINE.replace("MARKER", "marker_two"))
        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "0 cached" in result.output
        assert "1 stale" in result.output


def test_status_keeps_hostile_decorator_unevaluated_and_hashed(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Fingerprinting does not inspect an unrelated decorator owner."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pipeline_path = pathlib.Path("pipeline.py")
        pipeline_path.write_text(_HOSTILE_RESOURCE_DECORATOR_PIPELINE.replace("DELTA", "1"))

        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        pipeline_path.write_text(_HOSTILE_RESOURCE_DECORATOR_PIPELINE.replace("DELTA", "2"))
        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "0 cached" in result.output
        assert "1 stale" in result.output


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("bare", "called", id="add-resources"),
        pytest.param("called", "bare", id="remove-resources"),
        pytest.param("module-bare", "module-called", id="module-add-resources"),
        pytest.param("bare", "module-called", id="import-style-change"),
    ],
)
def test_status_keeps_stage_cached_across_stage_decorator_forms(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, before: str, after: str
) -> None:
    """`@stage`, `@stage(resources=...)` and `@compose.stage` hash the same."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pipeline_path = pathlib.Path("pipeline.py")
        pipeline_path.write_text(_stage_decorator_pipeline(*_STAGE_DECORATOR_FORMS[before]))

        repro = runner.invoke(cli.cli, ["repro", "--resources", "gpu=2"])
        assert repro.exit_code == 0, repro.output

        pipeline_path.write_text(_stage_decorator_pipeline(*_STAGE_DECORATOR_FORMS[after]))
        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0, result.output
        assert "1 cached" in result.output
        assert "0 stale" in result.output


def test_status_verbose_shows_all_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Verbose status shows all stages including cached."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pathlib.Path("pipeline.py").write_text(_STAGES_A_B_PIPELINE)

        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        result = runner.invoke(cli.cli, ["status", "--verbose"])

        assert result.exit_code == 0
        assert "test/stage_a" in result.output
        assert "test/stage_b" in result.output


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param("producer", id="list-producer"),
        pytest.param("params", id="list-params"),
        pytest.param("out-hash", id="short-out-hash"),
        pytest.param("dep-hash", id="short-dep-hash"),
        pytest.param("delete", id="deleted-lock-control"),
    ],
)
def test_status_and_repro_treat_a_wrongly_shaped_lock_as_missing(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, corrupt: str
) -> None:
    """A hand-edited lock that fails validation behaves exactly like a deleted one.

    status reports the stage stale; repro does not skip it on the bad lock but processes it as
    a stage with no lock (unchanged inputs restore from the run cache) and writes a valid lock,
    after which status agrees the stage is cached.
    """
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_upstream_pipeline("a")
        first = runner.invoke(cli.cli, ["repro"])
        assert first.exit_code == 0, first.output

        lock_path = cwd / ".pivot" / "stages" / "test" / "stage_b.lock"
        document = yaml.safe_load(lock_path.read_text())
        if corrupt == "delete":
            lock_path.unlink()
        else:
            if corrupt == "producer":
                assert document["deps"][0]["producer"] == "test/stage_a"
                document["deps"][0]["producer"] = ["test/stage_a"]
            elif corrupt == "out-hash":
                document["outs"][0]["hash"] = "abc"
            elif corrupt == "dep-hash":
                document["deps"][0]["hash"] = "abc"
            else:
                document["params"] = []
            lock_path.write_text(yaml.safe_dump(document))

        status = runner.invoke(cli.cli, ["status"])
        repro = runner.invoke(cli.cli, ["repro", "--json"])
        repaired = yaml.safe_load(lock_path.read_text())
        status_after = runner.invoke(cli.cli, ["status"])

    assert status.exit_code == 0, status.output
    assert "1 cached, 1 stale" in status.output, status.output
    assert "test/stage_b" in status.output, status.output
    assert "No previous run" in status.output, status.output

    assert repro.exit_code == 0, repro.output
    events = [json.loads(line) for line in repro.output.splitlines() if line.startswith("{")]
    stage_b = next(
        e for e in events if e["type"] == "stage_complete" and e["stage"] == "test/stage_b"
    )
    assert stage_b["reason"] == "unchanged (run cache)", stage_b

    assert repaired["deps"][0]["producer"] == "test/stage_a"
    assert repaired["params"] == {}
    assert len(repaired["deps"][0]["hash"]) == cache.XXHASH64_HEX_LENGTH, repaired
    assert len(repaired["outs"][0]["hash"]) == cache.XXHASH64_HEX_LENGTH, repaired
    assert status_after.exit_code == 0, status_after.output
    assert "2 cached, 0 stale" in status_after.output, status_after.output


def test_status_specific_stages(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """Status with stage argument filters to specific stage."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pathlib.Path("pipeline.py").write_text(_STAGES_A_B_PIPELINE)

        result = runner.invoke(cli.cli, ["status", "test/stage_a"])

        assert result.exit_code == 0, result.output
        assert "test/stage_a" in result.output
        assert "test/stage_b" not in result.output


def test_status_bare_stage_name_resolves(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status accepts a bare stage name in a single-pipeline project (names.py design)."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        pathlib.Path("pipeline.py").write_text(_STAGES_A_B_PIPELINE)

        result = runner.invoke(cli.cli, ["status", "stage_a"])

        assert result.exit_code == 0, result.output
        assert "test/stage_a" in result.output
        assert "test/stage_b" not in result.output


def test_status_unknown_stage_errors(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status with unknown stage shows error."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)

        result = runner.invoke(cli.cli, ["status", "nonexistent"])

        assert result.exit_code != 0
        assert "nonexistent" in result.output.lower()


# =============================================================================
# Tracked Files Tests
# =============================================================================


def test_status_shows_tracked_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status shows tracked files section with verbose."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)
        _write_tracked_file("content")

        result = runner.invoke(cli.cli, ["status", "--verbose"])

        assert result.exit_code == 0
        assert "Tracked Files" in result.output
        assert "data.txt" in result.output
        assert "clean" in result.output


def test_status_shows_modified_tracked_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status shows modified tracked files."""
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_EMPTY_PIPELINE)
        _write_tracked_file("original")

        pathlib.Path("data.txt").write_text("modified content")

        result = runner.invoke(cli.cli, ["status"])

        assert result.exit_code == 0
        assert "modified" in result.output


# =============================================================================
# Filter Options Tests
# =============================================================================


def test_status_stages_only(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--stages-only shows only pipeline status."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_tracked_file("content")
        _write_process_pipeline()

        result = runner.invoke(cli.cli, ["status", "--stages-only"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "Tracked Files" not in result.output


def test_status_tracked_only(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """--tracked-only shows only tracked files."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_tracked_file("content")
        _write_process_pipeline()

        result = runner.invoke(cli.cli, ["status", "--tracked-only"])

        assert result.exit_code == 0
        assert "Tracked Files" in result.output
        assert "Pipeline Status" not in result.output


def test_status_remote_only_no_remotes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--remote-only without configured remotes shows error."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["status", "--remote-only"])

        assert result.exit_code != 0
        assert "No remotes configured" in result.output


# =============================================================================
# JSON Output Tests
# =============================================================================


def test_status_json_output(
    runner: click.testing.CliRunner, mock_discovery: compose.Pipeline
) -> None:
    """--json outputs valid JSON."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["status", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "stages" in data
    assert len(data["stages"]) == 1
    assert data["stages"][0]["name"] == "test/process"
    assert data["stages"][0]["status"] == "stale"


def test_status_json_includes_suggestions(
    runner: click.testing.CliRunner, mock_discovery: compose.Pipeline
) -> None:
    """--json includes suggestions when applicable."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["status", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "suggestions" in data
    assert any("pivot repro" in s for s in data["suggestions"])


# =============================================================================
# Suggestions Tests
# =============================================================================


def test_status_shows_suggestions(
    runner: click.testing.CliRunner, mock_discovery: compose.Pipeline
) -> None:
    """Status shows actionable suggestions."""
    _register_process(mock_discovery)

    result = runner.invoke(cli.cli, ["status"])

    assert result.exit_code == 0, result.output
    assert "Suggestions" in result.output
    assert "pivot repro" in result.output


# =============================================================================
# Empty Section Behavior Tests
# =============================================================================


def test_status_tracked_only_no_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--tracked-only with no tracked files shows explicit message."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["status", "--tracked-only"])

        assert result.exit_code == 0
        assert "Tracked Files" in result.output
        assert "No tracked files" in result.output


def test_status_stages_only_no_stages(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """--stages-only with no stages shows explicit message."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["status", "--stages-only"])

        assert result.exit_code == 0
        assert "Pipeline Status" in result.output
        assert "No stages registered" in result.output


def test_status_json_includes_empty_arrays(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """--json includes empty arrays for requested sections."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data, "Should include stages key even if empty"
        assert data["stages"] == []
        assert "tracked_files" in data, "Should include tracked_files key even if empty"
        assert data["tracked_files"] == []


def test_status_json_stages_only_empty(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """--json --stages-only includes empty stages array."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.cli, ["status", "--json", "--stages-only"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "stages" in data
        assert data["stages"] == []
        assert "tracked_files" not in data, "Should not include unrequested sections"


# =============================================================================
# Quiet Mode Tests
# =============================================================================


def test_status_quiet_no_output_when_clean(
    runner: click.testing.CliRunner,
    mock_discovery: compose.Pipeline,
) -> None:
    """pivot --quiet status produces no output when all stages are cached."""
    _register_process(mock_discovery)

    # Run once to cache via CLI
    repro = runner.invoke(cli.cli, ["repro"])
    assert repro.exit_code == 0, repro.output

    result = runner.invoke(cli.cli, ["--quiet", "status"])

    assert result.exit_code == 0
    assert result.output.strip() == "", "Quiet mode should suppress output when clean"


def test_status_quiet_exits_1_when_stale(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """pivot --quiet status exits 1 when stages are stale."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _register_process(mock_discovery)

        # Don't run - stage is stale
        result = runner.invoke(cli.cli, ["--quiet", "status"])

        assert result.exit_code == 1, "Should exit 1 when stages are stale"
        assert result.output.strip() == "", "Quiet mode should suppress output"


def test_status_quiet_exits_1_when_modified(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """pivot --quiet status exits 1 when files are modified."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _register_process(mock_discovery)

        # Run to cache via CLI
        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        # Modify input file
        _write_input(mock_discovery.root, "modified data")

        result = runner.invoke(cli.cli, ["--quiet", "status"])

        # Stage should now be stale due to modified input
        assert result.exit_code == 1, "Should exit 1 when files are modified"
        assert result.output.strip() == "", "Quiet mode should suppress output"


# =============================================================================
# Remote Status Tests
# =============================================================================


def test_status_remote_with_configured_remote(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    mock_discovery: compose.Pipeline,
) -> None:
    """--remote shows sync status when remote is configured."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Mock the remote status function
        mock_remote_status = RemoteSyncInfo(
            name="default",
            url="s3://mybucket/cache",
            push_count=5,
            pull_count=3,
        )
        mocker.patch.object(
            status_mod, "get_remote_status", autospec=True, return_value=mock_remote_status
        )

        result = runner.invoke(cli.cli, ["status", "--remote"])

        assert result.exit_code == 0
        assert "Remote Status" in result.output
        assert "5 to push" in result.output
        assert "3 to pull" in result.output


def test_status_remote_only_with_remote(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    mock_discovery: compose.Pipeline,
) -> None:
    """--remote-only shows only remote sync counts."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _register_process(mock_discovery)

        mock_remote_status = RemoteSyncInfo(
            name="myremote",
            url="s3://bucket/path",
            push_count=10,
            pull_count=2,
        )
        mocker.patch.object(
            status_mod, "get_remote_status", autospec=True, return_value=mock_remote_status
        )

        result = runner.invoke(cli.cli, ["status", "--remote-only"])

        assert result.exit_code == 0
        assert "Remote Status" in result.output
        # Should NOT show pipeline or tracked files sections
        assert "Pipeline Status" not in result.output
        assert "Tracked Files" not in result.output


def test_status_json_with_remote(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    mock_discovery: compose.Pipeline,
) -> None:
    """JSON output includes remote status when --remote is used."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        mock_remote_status = RemoteSyncInfo(
            name="default",
            url="s3://bucket/cache",
            push_count=7,
            pull_count=4,
        )
        mocker.patch.object(
            status_mod, "get_remote_status", autospec=True, return_value=mock_remote_status
        )

        result = runner.invoke(cli.cli, ["status", "--json", "--remote"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "remote" in data
        assert data["remote"]["name"] == "default"
        assert data["remote"]["push_count"] == 7
        assert data["remote"]["pull_count"] == 4


# =============================================================================
# --explain Flag Tests
# =============================================================================


def test_status_explain_flag_in_help(runner: click.testing.CliRunner) -> None:
    """--explain flag should appear in status help output."""
    result = runner.invoke(cli.cli, ["status", "--help"])

    assert result.exit_code == 0
    assert "--explain" in result.output


def test_status_explain_shows_detailed_output(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """--explain shows detailed change information."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _register_process(mock_discovery)

        result = runner.invoke(cli.cli, ["status", "--explain"])

        assert result.exit_code == 0, result.output
        assert "test/process" in result.output
        # Should show detailed info like reason
        assert "No previous run" in result.output
        assert "WILL RUN" in result.output


def test_status_explain_shows_code_changes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain shows code changes when code differs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_process_pipeline("v1")
        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        # Rewrite the stage body to simulate a code change
        _write_process_pipeline("v2_different")

        result = runner.invoke(cli.cli, ["status", "--explain"])

        assert result.exit_code == 0, result.output
        assert "Code Changes:" in result.output


def test_status_explain_short_flag(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """-e short flag works like --explain."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _register_process(mock_discovery)

        result = runner.invoke(cli.cli, ["status", "-e"])

        assert result.exit_code == 0, result.output
        assert "test/process" in result.output
        assert "WILL RUN" in result.output


def test_status_explain_json_format(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """--explain --json returns extended format with change arrays."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _register_process(mock_discovery)

        result = runner.invoke(cli.cli, ["status", "--explain", "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "stages" in data
        assert len(data["stages"]) == 1
        stage = data["stages"][0]
        assert "code_changes" in stage, "Explain JSON should include code_changes"
        assert "param_changes" in stage, "Explain JSON should include param_changes"
        assert "dep_changes" in stage, "Explain JSON should include dep_changes"


def test_status_json_without_explain_no_changes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, mock_discovery: compose.Pipeline
) -> None:
    """--json without --explain does NOT include change arrays."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _register_process(mock_discovery)

        result = runner.invoke(cli.cli, ["status", "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "stages" in data
        assert len(data["stages"]) == 1
        stage = data["stages"][0]
        # Regular status JSON should NOT include change arrays
        assert "code_changes" not in stage, "Regular status JSON should not include code_changes"


# =============================================================================
# Upstream Propagation Consistency Tests
# =============================================================================


def test_status_upstream_propagation(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status shows B as stale when upstream A is stale."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_upstream_pipeline("a_v1")
        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        # Modify stage_a's code
        _write_upstream_pipeline("a_v2_different_code")

        result = runner.invoke(cli.cli, ["status", "--verbose"])

        assert result.exit_code == 0, result.output
        # Both stages should show as stale
        assert "test/stage_a" in result.output
        assert "test/stage_b" in result.output
        assert "2 stale" in result.output
        # B should show upstream stale reason
        assert "Upstream stale" in result.output or "upstream" in result.output.lower()


def test_status_explain_upstream_propagation(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain shows B with upstream_stale when A is stale."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_upstream_pipeline("a_v1")
        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        _write_upstream_pipeline("a_v2_different_code")

        result = runner.invoke(cli.cli, ["status", "--explain"])

        assert result.exit_code == 0, result.output
        # B should show upstream stale reason
        assert "Upstream stale" in result.output or "upstream" in result.output.lower()


def test_status_explain_json_upstream_propagation(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain --json shows upstream_stale in stage info."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_input(cwd)
        _write_upstream_pipeline("a_v1")
        repro = runner.invoke(cli.cli, ["repro"])
        assert repro.exit_code == 0, repro.output

        _write_upstream_pipeline("a_v2_different_code")

        result = runner.invoke(cli.cli, ["--quiet", "status", "--explain", "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "stages" in data

        # Find stage_b in output
        stage_b = next((s for s in data["stages"] if s["name"] == "test/stage_b"), None)
        assert stage_b is not None, "stage_b should be in output"
        assert "upstream_stale" in stage_b, "stage_b should have upstream_stale field"
        assert "test/stage_a" in stage_b["upstream_stale"], (
            "stage_a should be in stage_b's upstream_stale"
        )


# =============================================================================
# Explain Command Removal Tests
# =============================================================================


def test_explain_command_removed(runner: click.testing.CliRunner, tmp_path: pathlib.Path) -> None:
    """pivot explain command should no longer exist."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["explain"])

        assert result.exit_code != 0
        assert "No such command" in result.output or "Error" in result.output


# =============================================================================
# No Pipeline Tests
# =============================================================================


def test_status_works_without_pipeline(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status should succeed in a Pivot project without a pipeline definition."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["status"])
        assert result.exit_code == 0, f"Status should succeed without pipeline: {result.output}"
        # Should show tracked files section (even if empty)
        assert "Tracked Files" in result.output or "No tracked files" in result.output, (
            "Should show tracked files section"
        )


def test_status_json_without_pipeline(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Status --json should work without a pipeline definition."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["status", "--json"])
        assert result.exit_code == 0, f"Status --json should succeed: {result.output}"
        data = json.loads(result.output)
        assert "stages" in data, "JSON output should include stages key"
        assert data["stages"] == [], "Stages should be empty without pipeline"
