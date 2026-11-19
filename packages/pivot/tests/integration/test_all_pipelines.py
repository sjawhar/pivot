"""Integration tests for ``--all`` across compositional pipeline.py files."""

from __future__ import annotations

import contextlib
import json
import shutil
from typing import TYPE_CHECKING

import pytest

from pivot import cli
from pivot.storage import lock
from tests import helpers
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

# A run of target broken/after, downstream of broken/read_0, leaves it out
_LEFT_OUT_AFTER = (
    "The run left out 1 target for a missing input:\n"
    + "  - broken/after: Stage 'broken/read_0' depends on input 'missing.txt' "
    + "(path data/raw/missing.txt), which does not exist"
)


def _helper_directory_reader_code(input_name: str = "dir") -> str:
    """Pipeline dirs, whose stage dirs/scan reads the directory data/raw/<input_name>/ and lists
    its entries."""
    return f"""\
import pathlib

from pivot.compose import Pipeline, stage


@stage
def scan(folder: pathlib.Path) -> str:
    return ",".join(sorted(entry.name for entry in folder.iterdir()))


with Pipeline("dirs") as pipeline:
    scan(pipeline.input("{input_name}", t=pathlib.Path))
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
        helpers.track_without_data(root / "data" / "raw" / "missing.txt")


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


@pytest.mark.parametrize("through_symlink", [False, True], ids=["directory", "symlink"])
def test_repro_all_tracked_keep_going_leaves_out_a_stage_reading_a_directory_that_holds_it(
    runner: CliRunner, tmp_path: pathlib.Path, through_symlink: bool
) -> None:
    """A2 (R2): a stage reading an existing directory that holds a missing tracked file is left
    out, with one line naming the directory and the file; so is one reading a symlink
    data/raw/dir to the directory data/raw/real that holds it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write_pipeline(cwd, "dirs", _helper_directory_reader_code())
        _helper_write_pipeline(cwd, "alpha", _helper_pipeline_code("alpha", "stage_a"))
        holder = cwd / "data" / "raw" / ("real" if through_symlink else "dir")
        helpers.track_without_data(holder / "f.txt")
        (holder / "present.txt").write_text("present")
        if through_symlink:
            (cwd / "data" / "raw" / "dir").symlink_to("real")

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])

        assert result.exit_code == 0, result.output
        assert _helper_skipping_lines(result.output) == [
            "Skipping: Stage 'dirs/scan' depends on directory 'dir', which holds tracked file "
            + f"'data/raw/{holder.name}/f.txt' that does not exist"
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
        helpers.track_without_data(cwd / "data" / "raw" / "missing.txt")

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
            helpers.track_without_data(cwd / "data" / "raw" / "unrelated.txt")

        result = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "broken/after", "alpha/stage_a"]
        )

        assert result.exit_code == 1, result.output
        assert "Unknown stage" not in result.output, result.output
        assert _LEFT_OUT_AFTER in result.output, result.output
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
        helpers.track_without_data(cwd / "data" / "raw" / "missing.txt")

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


# =============================================================================
# The dry run and --explain under --all --keep-going (R5)
# =============================================================================

_DRY_RUN_KEEP_GOING = ["repro", "--all", "--keep-going", "--dry-run", "--jsonl"]


def _helper_completed(stdout: str) -> dict[str, dict[str, object]]:
    """The `stage_complete` event of each stage a `--jsonl` run reports, by stage."""
    events = [json.loads(line) for line in stdout.splitlines()]
    return {event["stage"]: event for event in events if event["type"] == "stage_complete"}


def test_repro_all_dry_run_keep_going_lists_the_stages_the_run_runs(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5 on A1: the dry run's stages are those the run on the same tree runs or skips; the
    reader of the missing tracked file and the stage downstream of it are absent, and one
    `Skipping:` line on stderr names the reader."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        dry_run = runner.invoke(cli.cli, _DRY_RUN_KEEP_GOING)
        ran = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "--jsonl"])

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": {"would_run": True, "reason": "No previous run"}}
        }, dry_run.output
        assert _helper_skipping_lines(dry_run.stderr) == [_READER_SKIPPED], dry_run.output
        assert ran.exit_code == 0, ran.output
        completed = _helper_completed(ran.stdout)
        assert {stage: event["status"] for stage, event in completed.items()} == {
            "alpha/stage_a": "ran"
        }, ran.output
        assert completed.keys() == json.loads(dry_run.stdout)["stages"].keys()


def test_repro_all_dry_run_keep_going_checkout_missing_counts_a_cached_file_as_present(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5, D7: with --checkout-missing, the reader of a missing tracked file whose object the
    cache holds is listed, with the reason the run gives after it restores the file (its .pvt
    names other bytes than the reader last read); without the flag it is left out. The reader
    of a tracked file with no object is left out either way, and the dry run restores nothing."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)
        _helper_write_pipeline(
            cwd, "kept", _helper_broken_pipeline_code(1, name="kept", input_name="cached.txt")
        )
        cached = cwd / "data" / "raw" / "cached.txt"
        cached.write_text("first bytes")
        tracked = runner.invoke(cli.cli, ["track", "data/raw/cached.txt"])
        assert tracked.exit_code == 0, tracked.output
        first = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])
        assert first.exit_code == 0, first.output
        cached.unlink()
        cached.write_text("cached bytes")
        retracked = runner.invoke(cli.cli, ["track", "--force", "data/raw/cached.txt"])
        assert retracked.exit_code == 0, retracked.output
        cached.unlink()

        dry_run = runner.invoke(cli.cli, [*_DRY_RUN_KEEP_GOING, "--checkout-missing"])
        restored = cached.exists()
        without = runner.invoke(cli.cli, _DRY_RUN_KEEP_GOING)
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert not restored, "a dry run restores nothing"
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": False, "reason": "unchanged"},
                "kept/read_0": {"would_run": True, "reason": "Input dependencies changed"},
            }
        }, dry_run.output
        assert _helper_skipping_lines(dry_run.stderr) == [_READER_SKIPPED], dry_run.output
        assert without.exit_code == 0, without.output
        assert json.loads(without.stdout)["stages"].keys() == {"alpha/stage_a"}, without.output
        assert ran.exit_code == 0, ran.output
        completed = _helper_completed(ran.stdout)
        assert {stage: event["status"] for stage, event in completed.items()} == {
            "alpha/stage_a": "cached",
            "kept/read_0": "ran",
        }, ran.output
        assert completed["kept/read_0"]["reason"] == "Input dependencies changed", ran.output
        assert cached.read_text() == "cached bytes"


def test_repro_all_dry_run_keep_going_checkout_missing_counts_a_file_reached_through_a_symlink(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5, D7: kept/read_0 reads data/raw/alias/x.txt, where alias links to inputs/sub and the
    cache can restore the missing tracked file inputs/sub/x.txt. With --checkout-missing the
    dry run lists kept/read_0, as the run restores the file and runs it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write_pipeline(
            cwd, "kept", _helper_broken_pipeline_code(1, name="kept", input_name="alias/x.txt")
        )
        _helper_write_pipeline(cwd, "alpha", _helper_pipeline_code("alpha", "stage_a"))
        tracked = cwd / "data" / "raw" / "inputs" / "sub" / "x.txt"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("x bytes")
        tracking = runner.invoke(cli.cli, ["track", "data/raw/inputs/sub/x.txt"])
        assert tracking.exit_code == 0, tracking.output
        tracked.unlink()
        (cwd / "data" / "raw" / "alias").symlink_to("inputs/sub")

        dry_run = runner.invoke(cli.cli, [*_DRY_RUN_KEEP_GOING, "--checkout-missing"])
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": True, "reason": "No previous run"},
                "kept/read_0": {"would_run": True, "reason": "No previous run"},
            }
        }, dry_run.output
        assert _helper_skipping_lines(dry_run.stderr) == [], dry_run.output
        assert ran.exit_code == 0, ran.output
        completed = _helper_completed(ran.stdout)
        assert {stage: event["status"] for stage, event in completed.items()} == {
            "alpha/stage_a": "ran",
            "kept/read_0": "ran",
        }, ran.output
        assert (cwd / "data" / "kept" / "read_0.txt").read_text() == "x bytes"


def test_repro_all_dry_run_keep_going_checkout_missing_follows_the_restored_directory_listing(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5, D7: data/raw/images is a missing tracked directory the cache can restore, whose
    listing holds a.png and sub/b.png. With --checkout-missing, dirs/scan reads its subdirectory
    images/sub, which the restore brings back: the dry run lists it, and the run restores and
    runs it. cat/read_0 reads images/cat.png, which the listing does not hold: the run still
    finds it missing after the restore and leaves cat/read_0 out, and so does the dry run, with
    the same `Skipping:` line."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write_pipeline(cwd, "dirs", _helper_directory_reader_code("images/sub"))
        _helper_write_pipeline(
            cwd, "cat", _helper_broken_pipeline_code(1, name="cat", input_name="images/cat.png")
        )
        _helper_write_pipeline(cwd, "alpha", _helper_pipeline_code("alpha", "stage_a"))
        images = cwd / "data" / "raw" / "images"
        (images / "sub").mkdir(parents=True)
        (images / "a.png").write_text("a")
        (images / "sub" / "b.png").write_text("b")
        tracking = runner.invoke(cli.cli, ["track", "data/raw/images"])
        assert tracking.exit_code == 0, tracking.output
        shutil.rmtree(images)

        dry_run = runner.invoke(cli.cli, [*_DRY_RUN_KEEP_GOING, "--checkout-missing"])
        restored = images.exists()
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
        )

        skipped = [
            "Skipping: Stage 'cat/read_0' depends on input 'images/cat.png' "
            + "(path data/raw/images/cat.png), which does not exist"
        ]
        assert dry_run.exit_code == 0, dry_run.output
        assert not restored, "a dry run restores nothing"
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": True, "reason": "No previous run"},
                "dirs/scan": {"would_run": True, "reason": "No previous run"},
            }
        }, dry_run.output
        assert _helper_skipping_lines(dry_run.stderr) == skipped, dry_run.output
        assert ran.exit_code == 0, ran.output
        completed = _helper_completed(ran.stdout)
        assert {stage: event["status"] for stage, event in completed.items()} == {
            "alpha/stage_a": "ran",
            "dirs/scan": "ran",
        }, ran.output
        assert _helper_skipping_lines(ran.stderr) == skipped, ran.output
        assert (cwd / "data" / "dirs" / "scan.txt").read_text() == "b.png"


def test_repro_all_dry_run_keep_going_leaves_out_a_restored_directory_holding_a_missing_file(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5, D7, R1: data/raw/images is a missing tracked directory the cache can restore, and the
    hand-edited data/raw/extra.pvt tracks images/extra.bin, which images' listing does not hold
    and whose object no cache holds. dirs/scan reads images. The run restores images, still
    finds extra.bin missing inside it and leaves dirs/scan out with the `Skipping:` line of the
    directory that holds it. The dry run with --checkout-missing counts images as present, finds
    the same file inside it and leaves dirs/scan out with the same line."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_write_pipeline(cwd, "dirs", _helper_directory_reader_code("images"))
        _helper_write_pipeline(cwd, "alpha", _helper_pipeline_code("alpha", "stage_a"))
        images = cwd / "data" / "raw" / "images"
        images.mkdir(parents=True)
        (images / "a.png").write_text("a")
        tracking = runner.invoke(cli.cli, ["track", "data/raw/images"])
        assert tracking.exit_code == 0, tracking.output
        shutil.rmtree(images)
        (cwd / "data" / "raw" / "extra.pvt").write_text(
            "path: images/extra.bin\nhash: 0123456789abcdef\nsize: 10\n"
        )

        dry_run = runner.invoke(cli.cli, [*_DRY_RUN_KEEP_GOING, "--checkout-missing"])
        restored = images.exists()
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
        )

        skipped = [
            "Skipping: Stage 'dirs/scan' depends on directory 'images', which holds tracked file "
            + "'data/raw/images/extra.bin' that does not exist"
        ]
        assert dry_run.exit_code == 0, dry_run.output
        assert not restored, "a dry run restores nothing"
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": {"would_run": True, "reason": "No previous run"}}
        }, dry_run.output
        assert _helper_skipping_lines(dry_run.stderr) == skipped, dry_run.output
        assert ran.exit_code == 0, ran.output
        completed = _helper_completed(ran.stdout)
        assert {stage: event["status"] for stage, event in completed.items()} == {
            "alpha/stage_a": "ran"
        }, ran.output
        assert completed.keys() == json.loads(dry_run.stdout)["stages"].keys()
        assert _helper_skipping_lines(ran.stderr) == skipped, ran.output
        assert (images / "a.png").read_text() == "a", "the run restores images"
        assert not (cwd / "data" / "dirs" / "scan.txt").exists()


def test_repro_all_dry_run_keep_going_target_naming_no_stage_lists_what_the_run_runs(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R4, R5: a target that is a pipeline file with no stages leaves no target at all, and
    none was left out, so the run runs every stage it keeps; the dry run lists the same stages,
    not none."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)
        _helper_write_pipeline(
            cwd,
            "sub",
            'from pivot.compose import Pipeline\n\nwith Pipeline("sub") as pipeline:\n    pass\n',
        )

        dry_run = runner.invoke(cli.cli, [*_DRY_RUN_KEEP_GOING, "sub/pipeline.py"])
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--jsonl", "sub/pipeline.py"]
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": {"would_run": True, "reason": "No previous run"}}
        }, dry_run.output
        assert _helper_skipping_lines(dry_run.stderr) == [_READER_SKIPPED], dry_run.output
        assert ran.exit_code == 0, ran.output
        completed = _helper_completed(ran.stdout)
        assert {stage: event["status"] for stage, event in completed.items()} == {
            "alpha/stage_a": "ran"
        }, ran.output


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "run"])
def test_repro_all_dry_run_keep_going_refuses_allow_missing(
    runner: CliRunner, tmp_path: pathlib.Path, dry_run: bool
) -> None:
    """R5, D8: --allow-missing with --all --keep-going is a usage error, with or without
    --dry-run, before anything runs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(
            cli.cli,
            [
                "repro",
                "--all",
                "--keep-going",
                "--allow-missing",
                *(["--dry-run"] if dry_run else []),
            ],
        )

        assert result.exit_code == 2, result.output
        assert (
            "--allow-missing cannot be used with --all --keep-going, which leaves out each stage "
            + "with a missing input"
        ) in result.stderr, result.output
        assert "Skipping:" not in result.output, result.output
        assert not (cwd / "data" / "alpha" / "stage_a.txt").exists()


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param(["--explain", "--fail-fast"], id="explain-fail-fast"),
        pytest.param(["--dry-run", "--jsonl", "--watch"], id="dry-run-watch"),
    ],
)
def test_repro_all_dry_run_keep_going_allow_missing_outside_leave_out_mode_works_as_before(
    runner: CliRunner, tmp_path: pathlib.Path, flags: list[str]
) -> None:
    """R6: with --fail-fast or --watch beside --all --keep-going, --explain and the dry run are
    not in leave-out mode, so --allow-missing works as in 0.10.4: the reader of the missing
    tracked file is listed with every other stage, and nothing is left out."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--allow-missing", *flags]
        )

        assert result.exit_code == 0, result.output
        assert "Skipping:" not in result.output, result.output
        if "--explain" in flags:
            listed = {line for line in result.output.splitlines() if line.startswith("Stage: ")}
            assert listed == {
                "Stage: alpha/stage_a",
                "Stage: broken/after",
                "Stage: broken/read_0",
            }, result.output
        else:
            assert json.loads(result.stdout)["stages"].keys() == {
                "alpha/stage_a",
                "broken/after",
                "broken/read_0",
            }, result.output


def test_repro_all_dry_run_keep_going_explain_leaves_out_the_reader(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R5: --explain on A1 explains alpha/stage_a alone; the reader and its downstream stage are
    not in its stage list, and one `Skipping:` line names the reader."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "--explain"])

        assert result.exit_code == 0, result.output
        assert [line for line in result.output.splitlines() if line.startswith("Stage: ")] == [
            "Stage: alpha/stage_a"
        ], result.output
        assert "Summary: 1 will run, 0 unchanged" in result.output, result.output
        assert _helper_skipping_lines(result.output) == [_READER_SKIPPED], result.output


@pytest.mark.parametrize(
    ("quiet", "targets", "document"),
    [
        pytest.param(
            False,
            ["broken/after", "alpha/stage_a"],
            {"stages": {"alpha/stage_a": {"would_run": True, "reason": "No previous run"}}},
            id="other-target",
        ),
        pytest.param(False, ["broken/after"], {"stages": {}}, id="every-target"),
        pytest.param(True, ["broken/after", "alpha/stage_a"], None, id="quiet"),
    ],
)
def test_repro_all_dry_run_keep_going_left_out_target_fails_after_the_output(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    quiet: bool,
    targets: list[str],
    document: dict[str, object] | None,
) -> None:
    """R4 in the dry run: the document for the targets left is printed on stdout (an empty one
    when none is left), then the command exits 1 naming the left-out target and the missing
    file, as the run does; under --quiet without --jsonl, which prints nothing, it still exits
    1."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)
        command = ["repro", "--all", "--keep-going", "--dry-run", *targets]

        result = runner.invoke(cli.cli, ["--quiet", *command] if quiet else [*command, "--jsonl"])

        assert result.exit_code == 1, result.output
        assert _LEFT_OUT_AFTER in result.stderr, result.output
        assert "Unknown stage" not in result.output, result.output
        if document is None:
            assert result.stdout == "", result.output
        else:
            assert json.loads(result.stdout) == document, result.output


def test_repro_all_dry_run_keep_going_stops_on_a_missing_file_tracked_inside_an_output(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R3 in the dry run: a missing tracked file inside a loaded stage's directory output stops
    it as it stops the run, naming that file alone; no document is printed and no stage is left
    out."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)
        _helper_write_pipeline(cwd, "m", _MAPPED_SHARED_INPUT_CODE)
        helpers.track_without_data(cwd / "data" / "m" / "plan" / "c.txt")

        result = runner.invoke(cli.cli, _DRY_RUN_KEEP_GOING)
        ran = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])

        overlap = "data/m/plan/c.txt: it is inside output 'data/m/plan' of stage 'm/plan'\n"
        assert result.exit_code == 1, result.output
        assert "The following tracked files are missing:\n" in result.stderr, result.output
        assert overlap in result.stderr, result.output
        assert "data/raw/missing.txt" not in result.output, result.output
        assert "Skipping:" not in result.output, result.output
        assert result.stdout == "", result.output
        assert ran.exit_code == 1, ran.output
        assert overlap in ran.stderr, ran.output
        assert not (cwd / "data" / "alpha" / "stage_a.txt").exists(), "the run runs no stage"


def test_dry_run_keep_going_negative_control_without_keep_going_still_raises(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """c4's negative control: the A1 dry run across all pipelines without --keep-going still
    raises DependencyNotFoundError for the reader, as in 0.10.4, and prints no document."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--dry-run", "--jsonl"])

        assert result.exit_code == 1, result.output
        assert (
            "Error: Stage 'broken/read_0' depends on input 'missing.txt' "
            + "(path data/raw/missing.txt), which does not exist"
        ) in result.stderr, result.output
        assert "Skipping:" not in result.output, result.output
        assert result.stdout == "", result.output


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
