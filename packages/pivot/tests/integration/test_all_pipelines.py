"""Integration tests for ``--all`` across compositional pipeline.py files."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

import pytest

from pivot import cli
from pivot.storage import lock, track
from tests.conftest import isolated_pivot_dir

if TYPE_CHECKING:
    import pathlib

    from click.testing import CliRunner


def _helper_pipeline_code(name: str, stage_name: str) -> str:
    """Return a complete compositional pipeline.py with one value-returning stage."""
    return f'''\
from pivot.compose import Pipeline, stage


@stage
def {stage_name}() -> str:
    return "{stage_name} output"


with Pipeline("{name}") as pipeline:
    {stage_name}()
'''


def _setup_multi_pipeline_project(root: pathlib.Path) -> None:
    """Create alpha and beta compositional pipelines in subdirectories of one project."""
    alpha = root / "alpha"
    alpha.mkdir()
    (alpha / "pipeline.py").write_text(_helper_pipeline_code("alpha", "stage_a"))

    beta = root / "beta"
    beta.mkdir()
    (beta / "pipeline.py").write_text(_helper_pipeline_code("beta", "stage_b"))


def test_repro_all_discovers_stages_from_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --all --dry-run discovers stages from all sub-pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--dry-run"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output, "alpha stage not found"
        assert "beta/stage_b" in result.output, "beta stage not found"


def test_repro_all_executes_stages_from_all_pipelines(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """repro --all runs stages from all discovered pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output
        assert "beta/stage_b" in result.output
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        assert (cwd / "data" / "beta" / "stage_b.txt").read_text() == "stage_b output"


def test_repro_all_writes_every_lock_to_the_project_state_dir(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """Lock files of every pipeline go to the project's .pivot/stages/ directory."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all"])
        assert result.exit_code == 0, f"Expected success, got: {result.output}"

        stages_dir = lock.get_stages_dir(cwd / ".pivot")
        assert lock.StageLock("alpha/stage_a", stages_dir).read() is not None
        assert lock.StageLock("beta/stage_b", stages_dir).read() is not None
        assert not (cwd / "beta" / ".pivot").exists(), "no sub-pipeline state directory"


def test_status_all_shows_all_pipelines(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """status --all shows stages from all pipelines."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        result = runner.invoke(cli.cli, ["status", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output
        assert "beta/stage_b" in result.output


def test_verify_all_reads_locks_of_every_pipeline(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """verify --all reads the lock files of every discovered pipeline."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        run_result = runner.invoke(cli.cli, ["repro", "--all"])
        assert run_result.exit_code == 0, f"repro failed: {run_result.output}"

        result = runner.invoke(cli.cli, ["verify", "--all"])

        assert result.exit_code == 0, f"Expected success, got: {result.output}"
        assert "alpha/stage_a" in result.output
        assert "beta/stage_b" in result.output


def test_commit_all_after_no_commit_run(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """commit --all promotes pending locks of every pipeline into the project state dir."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _setup_multi_pipeline_project(cwd)

        run_result = runner.invoke(cli.cli, ["repro", "--all", "--no-commit"])
        assert run_result.exit_code == 0, f"repro failed: {run_result.output}"

        commit_result = runner.invoke(cli.cli, ["commit", "--all"])
        assert commit_result.exit_code == 0, f"commit failed: {commit_result.output}"

        beta_lock = lock.StageLock("beta/stage_b", lock.get_stages_dir(cwd / ".pivot"))
        assert beta_lock.read() is not None, "Production lock not in the project state dir"


def test_all_flag_with_single_pipeline(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """--all works gracefully when the project has only one pipeline."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        alpha = cwd / "alpha"
        alpha.mkdir()
        (alpha / "pipeline.py").write_text(_helper_pipeline_code("alpha", "stage_a"))

        result = runner.invoke(cli.cli, ["status", "--all"])

        assert result.exit_code == 0, f"status --all failed: {result.output}"
        assert "alpha/stage_a" in result.output


def _helper_broken_pipeline_code(
    stages: int, *, name: str = "broken", input_name: str = "missing.txt", downstream: bool = False
) -> str:
    """Return pipeline ``name``'s pipeline.py whose ``stages`` stages ``read_<i>`` each read input
    ``input_name``, by default one that does not exist; with ``downstream``, stage ``after`` reads
    ``read_0``'s output."""
    definitions = "".join(
        f"\n\n@stage\ndef read_{i}(raw: str) -> str:\n    return raw\n" for i in range(stages)
    )
    calls = [f"    read_{i}(source)\n" for i in range(stages)]
    if downstream:
        definitions += '\n\n@stage\ndef after(text: str) -> str:\n    return text + "-after"\n'
        calls[0] = "    after(read_0(source))\n"
    return (
        f"from pivot.compose import Pipeline, stage\n{definitions}\n\n"
        + f'with Pipeline("{name}") as pipeline:\n'
        + f'    source = pipeline.input("{input_name}", t=str)\n'
        + "".join(calls)
    )


@pytest.mark.parametrize(
    ("broken_stages", "healthy", "keep_going", "skipped", "error"),
    [
        pytest.param(
            1, False, True, 1, "1 missing dependency error(s):", id="keep-going-none-left"
        ),
        pytest.param(
            2, False, True, 2, "2 missing dependency error(s):", id="keep-going-none-left-2"
        ),
        pytest.param(2, True, True, 2, None, id="keep-going-partial"),
        pytest.param(
            1,
            False,
            False,
            0,
            "Error: Stage 'broken/read_0' depends on input 'missing.txt'",
            id="fail-fast",
        ),
        pytest.param(2, True, False, 0, "2 missing dependency error(s):", id="fail-fast-2"),
    ],
)
def test_repro_all_reports_each_missing_dependency(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    broken_stages: int,
    healthy: bool,
    keep_going: bool,
    skipped: int,
    error: str | None,
) -> None:
    """Keep-going across all pipelines warns about every stage it skips for a missing input and
    reports every missing input when no stage is left; fail-fast reports them without skipping."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "broken").mkdir()
        (cwd / "broken" / "pipeline.py").write_text(_helper_broken_pipeline_code(broken_stages))
        if healthy:
            (cwd / "alpha").mkdir()
            (cwd / "alpha" / "pipeline.py").write_text(_helper_pipeline_code("alpha", "stage_a"))

        result = runner.invoke(
            cli.cli, ["repro", "--all", *(["--keep-going"] if keep_going else [])]
        )

        assert result.output.count("Skipping: Stage 'broken/read_") == skipped, result.output
        if error is None:
            assert result.exit_code == 0, result.output
            assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        else:
            assert result.exit_code == 1, result.output
            assert error in result.output, result.output
            assert ("missing dependency error(s)" in result.output) == ("error(s)" in error)


# =============================================================================
# A missing tracked file under --all --keep-going (leave-out mode)
# =============================================================================

# broken/read_0 reads data/raw/missing.txt, which does not exist
_READER_SKIPPED = (
    "Skipping: Stage 'broken/read_0' depends on input 'missing.txt' "
    + "(path data/raw/missing.txt), which does not exist"
)

# dirs/scan reads the directory data/raw/dir/
_DIRECTORY_READER_CODE = """\
import pathlib

from pivot.compose import Pipeline, stage


@stage
def scan(folder: pathlib.Path) -> str:
    return ",".join(sorted(entry.name for entry in folder.iterdir()))


with Pipeline("dirs") as pipeline:
    scan(pipeline.input("dir", t=pathlib.Path))
"""

# m/cut is mapped over m/plan's files (keys a and b), and its shared input is data/raw/missing.txt
_MAPPED_SHARED_INPUT_CODE = """\
from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def plan() -> Annotated[dict[str, str], directory]:
    return {"a.txt": "a", "b.txt": "b"}


@stage
def cut(entry: str, shared: str) -> str:
    return entry + shared


with Pipeline("m") as pipeline:
    cut(pipeline.map(plan()), pipeline.input("missing.txt", t=str))
"""


def _helper_track_without_data(data_path: pathlib.Path) -> None:
    """Write a .pvt for ``data_path`` alone, as ``tests/test_executor_pvt.py`` does: a tracked
    file with no data on disk and no object in the cache."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    track.write_pvt_file(
        track.get_pvt_path(data_path),
        track.PvtData(path=data_path.name, hash="0123456789abcdef", size=10),
    )


def _helper_write_pipeline(root: pathlib.Path, name: str, code: str) -> None:
    (root / name).mkdir()
    (root / name / "pipeline.py").write_text(code)


def _helper_reader_project(root: pathlib.Path, *, tracked: bool = True) -> None:
    """A1's tree: ``broken/read_0`` reads data/raw/missing.txt, which ``broken/after`` is
    downstream of, and ``alpha/stage_a`` is healthy. With ``tracked`` the input is a tracked file
    with no data and no cache object; without it, an untracked input that does not exist."""
    _helper_write_pipeline(root, "broken", _helper_broken_pipeline_code(1, downstream=True))
    _helper_write_pipeline(root, "alpha", _helper_pipeline_code("alpha", "stage_a"))
    if tracked:
        _helper_track_without_data(root / "data" / "raw" / "missing.txt")


def _helper_skipping_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if "Skipping:" in line]


def test_repro_all_tracked_keep_going_leaves_out_only_the_reader_of_a_missing_tracked_file(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A1 (R1, R2): the reader of a missing tracked file is left out with one line naming it
    and the file; the stage downstream of it is left out with no line; alpha runs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])

        assert result.exit_code == 0, result.output
        assert _helper_skipping_lines(result.output) == [_READER_SKIPPED], result.output
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        assert not (cwd / "data" / "broken" / "read_0.txt").exists(), "the reader must not run"
        assert not (cwd / "data" / "broken" / "after.txt").exists(), "nor its downstream stage"
        assert (cwd / "data" / "raw" / "missing.txt.pvt").is_file()


def test_repro_all_tracked_keep_going_leaves_out_a_stage_reading_a_directory_that_holds_it(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A2 (R2): a stage reading an existing directory that holds a missing tracked file is left
    out, with one line naming the directory and the file."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write_pipeline(cwd, "dirs", _DIRECTORY_READER_CODE)
        _helper_write_pipeline(cwd, "alpha", _helper_pipeline_code("alpha", "stage_a"))
        _helper_track_without_data(cwd / "data" / "raw" / "dir" / "f.txt")
        (cwd / "data" / "raw" / "dir" / "present.txt").write_text("present")

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])

        assert result.exit_code == 0, result.output
        assert _helper_skipping_lines(result.output) == [
            "Skipping: Stage 'dirs/scan' depends on directory 'dir', which holds tracked file "
            + "'data/raw/dir/f.txt' that does not exist"
        ], result.output
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        assert not (cwd / "data" / "dirs" / "scan.txt").exists(), "the stage must not run"


def test_repro_all_tracked_keep_going_checkout_missing_runs_the_reader_of_a_restored_file(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R1, R6: --checkout-missing restores the tracked file whose object the cache holds, and its
    reader runs; the reader of the tracked file with no object is left out."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)
        _helper_write_pipeline(
            cwd, "kept", _helper_broken_pipeline_code(1, name="kept", input_name="cached.txt")
        )
        cached = cwd / "data" / "raw" / "cached.txt"
        cached.write_text("cached bytes")
        tracked = runner.invoke(cli.cli, ["track", "data/raw/cached.txt"])
        assert tracked.exit_code == 0, tracked.output
        cached.unlink()

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing"])

        assert result.exit_code == 0, result.output
        assert cached.read_text() == "cached bytes"
        assert (cwd / "data" / "kept" / "read_0.txt").read_text() == "cached bytes"
        assert _helper_skipping_lines(result.output) == [_READER_SKIPPED], result.output
        assert not (cwd / "data" / "broken" / "read_0.txt").exists(), "the reader must not run"
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"


def test_repro_all_tracked_keep_going_leaves_out_a_mapped_stage_whose_shared_input_is_missing(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R1: a mapped stage whose shared input is a missing tracked file is left out, so none of
    its instances is added; its producer and alpha run."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write_pipeline(cwd, "m", _MAPPED_SHARED_INPUT_CODE)
        _helper_write_pipeline(cwd, "alpha", _helper_pipeline_code("alpha", "stage_a"))
        _helper_track_without_data(cwd / "data" / "raw" / "missing.txt")

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "--jsonl"])

        assert result.exit_code == 0, result.output
        assert _helper_skipping_lines(result.output) == [
            "Skipping: Stage 'm/cut' depends on input 'missing.txt' "
            + "(path data/raw/missing.txt), which does not exist"
        ], result.output
        events = [json.loads(line) for line in result.stdout.splitlines()]
        completed = {
            event["stage"]: event["status"] for event in events if event["type"] == "stage_complete"
        }
        assert completed == {"alpha/stage_a": "ran", "m/plan": "ran"}, result.output
        assert not [event for event in events if event["type"] == "stages_expanded"]
        assert (cwd / "data" / "m" / "plan" / "a.txt").read_text() == "a"


@pytest.mark.parametrize("tracked", [True, False], ids=["tracked", "untracked"])
def test_repro_all_keep_going_left_out_target_fails_the_run_after_the_other_target_runs(
    runner: CliRunner, tmp_path: pathlib.Path, tracked: bool
) -> None:
    """A3 (R4): of targets broken/after, downstream of the reader of a missing input, and the
    healthy alpha/stage_a, alpha/stage_a runs; the run then exits 1 naming broken/after and the
    missing file, and no target is an unknown stage. The tracked case has a second missing
    tracked file that no stage reads."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd, tracked=tracked)
        if tracked:
            _helper_track_without_data(cwd / "data" / "raw" / "unrelated.txt")

        result = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "broken/after", "alpha/stage_a"]
        )

        assert result.exit_code == 1, result.output
        assert "Unknown stage" not in result.output, result.output
        assert (
            "The run left out 1 target for a missing input:\n"
            + "  - broken/after: Stage 'broken/read_0' depends on input 'missing.txt' "
            + "(path data/raw/missing.txt), which does not exist"
        ) in result.output, result.output
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        assert not (cwd / "data" / "broken" / "after.txt").exists()


def test_repro_all_keep_going_left_out_target_alone_runs_no_other_stage(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R4: when every target is left out, the run runs nothing, not every stage, and exits 1
    naming the target."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "broken/after"])

        assert result.exit_code == 1, result.output
        assert (
            "  - broken/after: Stage 'broken/read_0' depends on input 'missing.txt' "
        ) in result.output, result.output
        assert not (cwd / "data" / "alpha" / "stage_a.txt").exists(), "no other stage runs"


def test_repro_all_keep_going_left_out_target_names_an_instance_of_a_left_out_mapped_stage(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R4: a target naming an instance of a mapped stage left out for its shared input is named
    as given; the other target runs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write_pipeline(cwd, "m", _MAPPED_SHARED_INPUT_CODE)
        _helper_write_pipeline(cwd, "alpha", _helper_pipeline_code("alpha", "stage_a"))
        _helper_track_without_data(cwd / "data" / "raw" / "missing.txt")

        result = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "m/cut@a", "alpha/stage_a"]
        )

        assert result.exit_code == 1, result.output
        assert (
            "  - m/cut@a: Stage 'm/cut' depends on input 'missing.txt' "
            + "(path data/raw/missing.txt), which does not exist"
        ) in result.output, result.output
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        assert not (cwd / "data" / "m" / "cut@a.txt").exists()


@pytest.mark.parametrize(
    "flags", [["--all"], ["--all", "--keep-going"]], ids=["fail-fast", "keep-going"]
)
def test_repro_all_left_out_target_rule_still_refuses_a_target_naming_no_stage(
    runner: CliRunner, tmp_path: pathlib.Path, flags: list[str]
) -> None:
    """R4: a target that names no stage is refused before the engine runs, as in 0.10.4."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(cli.cli, ["repro", *flags, "alpha/nosuch"])

        assert result.exit_code == 1, result.output
        assert "Unknown target(s): 'alpha/nosuch'" in result.output, result.output
        assert "Skipping:" not in result.output, "the engine must not run"
        assert "tracked files are missing" not in result.output, "the engine must not run"
        assert not (cwd / "data" / "alpha" / "stage_a.txt").exists()


def test_keep_going_without_all_still_raises(runner: CliRunner, tmp_path: pathlib.Path) -> None:
    """A6 (R6): `repro --keep-going` without --all, from broken/ where discovery finds the
    pipeline, still stops on the missing tracked file before any stage runs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        with contextlib.chdir(cwd / "broken"):
            listed = runner.invoke(cli.cli, ["list", "--json"])
            result = runner.invoke(cli.cli, ["repro", "--keep-going"])

        assert listed.exit_code == 0, listed.output
        assert sorted(stage["name"] for stage in json.loads(listed.stdout)["stages"]) == [
            "broken/after",
            "broken/read_0",
        ], "discovery from broken/ finds the broken pipeline alone"
        assert result.exit_code == 1, result.output
        assert "The following tracked files are missing" in result.output, result.output
        assert "data/raw/missing.txt" in result.output, result.output
        assert "Skipping:" not in result.output, result.output
        assert not (cwd / "data" / "broken" / "read_0.txt").exists()
