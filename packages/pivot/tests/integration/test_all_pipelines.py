"""Integration tests for ``--all`` across compositional pipeline.py files."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import socket
import stat
import threading
from typing import TYPE_CHECKING

import pytest
import xxhash

import pivot_tui.run
from pivot import cli
from pivot.storage import cache, lock, naming, track
from tests import helpers
from tests.conftest import isolated_pivot_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from click.testing import CliRunner, Result
    from pytest_mock import MockerFixture


def _setup_multi_pipeline_project(root: pathlib.Path) -> None:
    """Create alpha and beta compositional pipelines in subdirectories of one project."""
    alpha = root / "alpha"
    alpha.mkdir()
    (alpha / "pipeline.py").write_text(helpers.one_stage_pipeline_code("alpha", "stage_a"))

    beta = root / "beta"
    beta.mkdir()
    (beta / "pipeline.py").write_text(helpers.one_stage_pipeline_code("beta", "stage_b"))


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
        (alpha / "pipeline.py").write_text(helpers.one_stage_pipeline_code("alpha", "stage_a"))

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
            (cwd / "alpha" / "pipeline.py").write_text(
                helpers.one_stage_pipeline_code("alpha", "stage_a")
            )

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


def _helper_reader_project(root: pathlib.Path, *, tracked: bool = True) -> None:
    """A1's tree: ``broken/read_0`` reads data/raw/missing.txt, which ``broken/after`` is
    downstream of, and ``alpha/stage_a`` is healthy. With ``tracked`` the input is a tracked file
    with no data and no cache object; without it, an untracked input that does not exist."""
    helpers.write_pipeline(root, "broken", _helper_broken_pipeline_code(1, downstream=True))
    helpers.write_pipeline(root, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    if tracked:
        helpers.track_without_data(root / "data" / "raw" / "missing.txt")


def test_repro_all_tracked_keep_going_leaves_out_only_the_reader_of_a_missing_tracked_file(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A1 (R1, R2): the reader of a missing tracked file is left out with one line naming it
    and the file; the stage downstream of it is left out with no line; alpha runs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])

        assert result.exit_code == 0, result.output
        assert helpers.skipping_lines(result.output) == [_READER_SKIPPED], result.output
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
        helpers.write_pipeline(cwd, "dirs", _helper_directory_reader_code())
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        holder = cwd / "data" / "raw" / ("real" if through_symlink else "dir")
        helpers.track_without_data(holder / "f.txt")
        (holder / "present.txt").write_text("present")
        if through_symlink:
            (cwd / "data" / "raw" / "dir").symlink_to("real")

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])

        assert result.exit_code == 0, result.output
        assert helpers.skipping_lines(result.output) == [
            "Skipping: Stage 'dirs/scan' depends on directory 'dir' (path data/raw/dir), which "
            + f"holds tracked file 'data/raw/{holder.name}/f.txt' that does not exist"
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
        helpers.write_pipeline(
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
        assert helpers.skipping_lines(result.output) == [_READER_SKIPPED], result.output
        assert not (cwd / "data" / "broken" / "read_0.txt").exists(), "the reader must not run"
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"


def test_repro_all_tracked_keep_going_leaves_out_a_mapped_stage_whose_shared_input_is_missing(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R1: a mapped stage whose shared input is a missing tracked file is left out, so none of
    its instances is added; its producer and alpha run."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        helpers.write_pipeline(cwd, "m", _MAPPED_SHARED_INPUT_CODE)
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        helpers.track_without_data(cwd / "data" / "raw" / "missing.txt")

        result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "--jsonl"])

        assert result.exit_code == 0, result.output
        assert helpers.skipping_lines(result.output) == [
            "Skipping: Stage 'm/cut' depends on input 'missing.txt' "
            + "(path data/raw/missing.txt), which does not exist"
        ], result.output
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert helpers.stage_statuses(result.stdout) == {
            "alpha/stage_a": "ran",
            "m/plan": "ran",
        }, result.output
        assert not [event for event in events if event["type"] == "stages_expanded"]
        assert (cwd / "data" / "m" / "plan" / "a.txt").read_text() == "a"


def test_repro_all_tracked_keep_going_leaves_out_the_reader_of_a_tracked_path_that_is_no_file(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R1, R2, R5: the pre-flight counts a tracked path that is neither a regular file nor a
    directory as missing (here a Unix socket where data/raw/missing.txt should be; a FIFO is the
    same case, and its reader would block for ever). The stage that reads that path is left out
    with one `Skipping:` line, as the reader of a tracked file with no data is, and alpha runs;
    the dry run on the same tree lists alpha alone. It must not run the reader over the socket."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)
        with socket.socket(socket.AF_UNIX) as node:
            # A relative path: an AF_UNIX address holds at most 108 bytes
            node.bind("data/raw/missing.txt")
            dry_run = runner.invoke(cli.cli, helpers.DRY_RUN_KEEP_GOING)
            result = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])

        assert dry_run.exit_code == 0, dry_run.output
        assert list(json.loads(dry_run.stdout)["stages"]) == ["alpha/stage_a"], dry_run.output
        assert result.exit_code == 0, result.output
        assert helpers.skipping_lines(result.output) == [_READER_SKIPPED], result.output
        assert (cwd / "data" / "alpha" / "stage_a.txt").read_text() == "stage_a output"
        assert not (cwd / "data" / "broken" / "read_0.txt").exists(), "the reader must not run"


def _helper_checkout_missing_dry_run_and_run(runner: CliRunner) -> tuple[Result, Result]:
    """`pivot repro --all --keep-going --checkout-missing` in the current project, as a dry run
    (``--dry-run --jsonl``) and then as a run (``--jsonl``), after checking that the dry run
    restored nothing: the workspace under data/ holds the same entries after it."""
    before = sorted(pathlib.Path("data").rglob("*"))
    dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])
    assert sorted(pathlib.Path("data").rglob("*")) == before, "a dry run restores nothing"
    ran = runner.invoke(
        cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
    )
    return dry_run, ran


def _helper_assert_only_alpha_runs(dry_run: Result, ran: Result, skipped: list[str]) -> None:
    """The dry run exits 0 and lists alpha/stage_a alone, the run on the same tree exits 0 and
    runs alpha/stage_a alone, and each gives exactly the `Skipping:` lines ``skipped``."""
    assert dry_run.exit_code == 0, dry_run.output
    assert json.loads(dry_run.stdout) == {
        "stages": {"alpha/stage_a": {"would_run": True, "reason": "No previous run"}}
    }, dry_run.output
    assert helpers.skipping_lines(dry_run.stderr) == skipped, dry_run.output
    assert ran.exit_code == 0, ran.output
    assert helpers.stage_statuses(ran.stdout) == {"alpha/stage_a": "ran"}, ran.output
    assert helpers.skipping_lines(ran.stderr) == skipped, ran.output


@contextlib.contextmanager
def _helper_locked(path: pathlib.Path, mode: int) -> Generator[pathlib.Path]:
    """``path`` at mode ``mode`` while the block runs, then at its own mode again; gives
    ``path``."""
    before = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode)
    try:
        yield path
    finally:
        path.chmod(before)


def _helper_assert_restore_warnings(
    dry_run: Result, ran: Result, dry_run_warning: str, run_prefix: str
) -> None:
    """The dry run gives exactly the `Cannot restore` warning ``dry_run_warning``, and the run on
    the same tree exactly one, starting with ``run_prefix``: the dry run names what it found,
    where the run's warning goes on with the error its restore met."""
    assert helpers.restore_warnings(dry_run.stderr) == [dry_run_warning], dry_run.output
    warnings = helpers.restore_warnings(ran.stderr)
    assert len(warnings) == 1, ran.output
    assert warnings[0].startswith(run_prefix), ran.output


def _helper_assert_both_refuse(dry_run: Result, ran: Result, link: str) -> None:
    """The dry run and the run on the same tree each exit 1 on the refusal to link ``link``
    (the start of what the words "Refusing to link" name), with no `Skipping:` line and no
    restore warning; the dry run prints no document and the run runs no stage."""
    for result in (dry_run, ran):
        assert result.exit_code == 1, result.output
        assert f"Error: Refusing to link {link}" in result.stderr, result.output
        assert helpers.skipping_lines(result.stderr) == [], result.output
        assert helpers.restore_warnings(result.stderr) == [], result.output
    assert dry_run.stdout == "", dry_run.output
    assert helpers.completed_stages(ran.stdout) == {}, ran.output


def _helper_locked_file_project(
    runner: CliRunner, root: pathlib.Path, input_name: str
) -> tuple[pathlib.Path, pathlib.Path]:
    """kept/read_0 reads data/raw/``input_name`` and alpha/stage_a is healthy. The file
    data/raw/locked/cached.txt is tracked with the real ``pivot track``, which puts its object in
    the cache, then removed. Returns data/raw/locked, for a test to lock, and cached.txt."""
    helpers.write_pipeline(
        root, "kept", _helper_broken_pipeline_code(1, name="kept", input_name=input_name)
    )
    helpers.write_pipeline(root, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    locked = root / "data" / "raw" / "locked"
    locked.mkdir(parents=True)
    cached = locked / "cached.txt"
    cached.write_text("cached bytes")
    tracking = runner.invoke(cli.cli, ["track", "data/raw/locked/cached.txt"])
    assert tracking.exit_code == 0, tracking.output
    cached.unlink()
    return locked, cached


# dirs/scan reads data/raw/images, a missing tracked directory
_IMAGES_SKIPPED = [
    "Skipping: Stage 'dirs/scan' depends on input 'images' (path data/raw/images), which does "
    + "not exist"
]


def _helper_tracked_images_project(runner: CliRunner, root: pathlib.Path) -> pathlib.Path:
    """dirs/scan reads data/raw/images, a directory holding a.png that `pivot track` tracked and
    that is then removed, and alpha/stage_a is healthy. Returns the path of the listing's object
    in the cache."""
    helpers.write_pipeline(root, "dirs", _helper_directory_reader_code("images"))
    helpers.write_pipeline(root, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    images = root / "data" / "raw" / "images"
    images.mkdir(parents=True)
    (images / "a.png").write_text("a")
    tracking = runner.invoke(cli.cli, ["track", "data/raw/images"])
    assert tracking.exit_code == 0, tracking.output
    shutil.rmtree(images)
    record = track.read_pvt_file(track.get_pvt_path(images))
    assert record is not None
    return cache.get_cache_path(root / ".pivot" / "cache" / "files", record["hash"])


def _helper_make_listing_stale(root: pathlib.Path) -> pathlib.Path:
    """Make data/raw/images' committed listing stale (it no longer hashes to the .pvt's record);
    return its path."""
    committed = root / "data" / "raw" / "images.pvt.listing"
    committed.write_bytes(committed.read_bytes().replace(b"a.png", b"b.png"))
    return committed


@pytest.mark.parametrize("committed", ["stale", "empty"])
def test_tracked_keep_going_and_dry_run_keep_going_leave_out_the_reader_of_a_lost_listing(
    runner: CliRunner, tmp_path: pathlib.Path, committed: str
) -> None:
    """R1, R5: dirs/scan reads data/raw/images, a missing tracked directory whose committed
    listing is stale or empty while the cache holds no copy of the listing (a fresh clone). The
    run's --checkout-missing restore cannot bring it back: the run warns once, as `pivot
    checkout` does, leaves dirs/scan out with a `Skipping:` line, runs alpha and exits 0. The
    dry run with --checkout-missing exits 0, lists alpha alone, and gives the same warning and
    the same line."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_tracked_images_project(runner, cwd).unlink()
        if committed == "stale":
            committed_listing = _helper_make_listing_stale(cwd)
        else:
            committed_listing = cwd / "data" / "raw" / "images.pvt.listing"
            committed_listing.write_bytes(b"")

        dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        _helper_assert_only_alpha_runs(dry_run, ran, _IMAGES_SKIPPED)
        images = cwd / "data" / "raw" / "images"
        warnings = helpers.restore_warnings(dry_run.stderr)
        assert len(warnings) == 1, dry_run.output
        assert warnings[0].startswith(
            f"Cannot restore {images}: {committed_listing}: its bytes hash to "
        ), dry_run.output
        assert helpers.restore_warnings(ran.stderr) == warnings, ran.output
        assert not images.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a cache object whatever its mode")
@pytest.mark.parametrize("committed", ["intact", "stale"])
@pytest.mark.parametrize("unreadable", ["object", "prefix_directory"])
def test_tracked_keep_going_and_dry_run_keep_going_leave_out_the_reader_of_an_unreadable_listing(
    runner: CliRunner, tmp_path: pathlib.Path, committed: str, unreadable: str
) -> None:
    """R1, R5: dirs/scan reads data/raw/images, a missing tracked directory whose listing object
    in the cache this user cannot read: the object's mode, or its prefix directory's, is 0o000.
    Its committed listing is intact (the restore still reads the cache's copy, to check it for a
    collision) or stale. The run's --checkout-missing restore cannot read the listing: it warns
    once, naming the path and the permission error with no listing remedy, leaves dirs/scan out
    with a `Skipping:` line, runs alpha and exits 0. The dry run with --checkout-missing exits 0,
    lists alpha alone, and gives the same warning and the same line."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        listing_object = _helper_tracked_images_project(runner, cwd)
        if committed == "stale":
            _helper_make_listing_stale(cwd)
        locked = listing_object if unreadable == "object" else listing_object.parent
        mode = stat.S_IMODE(locked.stat().st_mode)

        locked.chmod(0)
        try:
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)
        finally:
            locked.chmod(mode)

        _helper_assert_only_alpha_runs(dry_run, ran, _IMAGES_SKIPPED)
        images = cwd / "data" / "raw" / "images"
        denied = f"[Errno 13] Permission denied: '{listing_object}'"
        # Reading the object fails as 0.10.4 failed, with a CorruptListingError naming it; the
        # lookup through an unsearchable prefix directory fails with its raw PermissionError
        reason = (
            f"{listing_object}: cannot be read ({denied})" if unreadable == "object" else denied
        )
        warning = f"Cannot restore {images}: {reason}"
        assert helpers.restore_warnings(dry_run.stderr) == [warning], dry_run.output
        assert helpers.restore_warnings(ran.stderr) == [warning], ran.output
        assert not images.exists()


@pytest.mark.parametrize("committed", ["intact", "stale"])
def test_tracked_keep_going_and_dry_run_keep_going_stop_on_a_cached_listing_that_does_not_hash(
    runner: CliRunner, tmp_path: pathlib.Path, committed: str
) -> None:
    """R5, D1: data/raw/images is a missing tracked directory whose listing object in the cache
    holds bytes that do not hash to the .pvt's record, beside an intact or a stale committed
    listing. That is a corrupt listing, not a lost or unreadable one: the run's
    --checkout-missing restore stops on it before any stage runs, and so does the dry run, with
    the same error and no document."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        listing_object = _helper_tracked_images_project(runner, cwd)
        if committed == "stale":
            _helper_make_listing_stale(cwd)
        listing_object.unlink()
        listing_object.write_bytes(b"junk\n")
        record = track.read_pvt_file(cwd / "data" / "raw" / "images.pvt")
        assert record is not None

        dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        problem = (
            f"Error: {listing_object}: its bytes hash to {xxhash.xxh64(b'junk\n').hexdigest()}, "
            + f"not to the directory hash {record['hash']} its lock or .pvt records\n"
        )
        assert dry_run.exit_code == 1, dry_run.output
        assert problem in dry_run.stderr, dry_run.output
        assert dry_run.stdout == "", dry_run.output
        assert ran.exit_code == 1, ran.output
        assert problem in ran.stderr, ran.output
        assert helpers.restore_warnings(dry_run.stderr + ran.stderr) == []
        assert not (cwd / "data" / "alpha" / "stage_a.txt").exists(), "the run runs no stage"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
def test_tracked_keep_going_and_dry_run_keep_going_name_a_failed_restore_over_an_output(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R3, R5: data/alpha/stage_a.txt, alpha/stage_a's output, is a missing tracked file whose
    object the cache holds, in the read-only directory data/alpha. The run's --checkout-missing
    restore fails, so the file stays missing over an output and the run stops before any stage
    runs; the dry run stops with the same error. The error names the file as one whose restore
    failed, not one the local cache lacks, and suggests no pull."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        tracked = cwd / "data" / "alpha" / "stage_a.txt"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("tracked bytes")
        # Tracked before alpha declares the output: `pivot track` refuses a path an output holds
        tracking = runner.invoke(cli.cli, ["track", "data/alpha/stage_a.txt"])
        assert tracking.exit_code == 0, tracking.output
        tracked.unlink()
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))

        tracked.parent.chmod(0o555)
        try:
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)
        finally:
            tracked.parent.chmod(0o755)

        refusal = (
            "data/alpha/stage_a.txt: it is output 'data/alpha/stage_a.txt' of stage "
            + "'alpha/stage_a'; its restore failed\n\n"
            + "Their restore failed: a 'Cannot restore' warning above names each cause.\n"
        )
        tip = "Tip: Fix what each 'Cannot restore' warning above names, then run again"
        for result in (dry_run, ran):
            assert result.exit_code == 1, result.output
            assert refusal in result.stderr, result.output
            assert tip in result.stderr, result.output
            assert "not in local cache" not in result.output, result.output
            assert "pivot pull" not in result.output, result.output
            assert len(helpers.restore_warnings(result.stderr)) == 1, result.output
        assert dry_run.stdout == "", dry_run.output
        assert not tracked.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
def test_tracked_keep_going_and_dry_run_keep_going_leave_out_the_reader_of_a_read_only_path(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R1, R5: kept/read_0 reads data/raw/locked/cached.txt, whose object the cache holds, in the
    read-only directory data/raw/locked. The run's --checkout-missing restore fails with
    PermissionError: the run warns once, naming the path and the error, leaves kept/read_0 out
    with a `Skipping:` line, runs alpha and exits 0. The dry run with --checkout-missing does
    not count the file as present: it lists alpha alone and leaves kept/read_0 out with the same
    line, warning that it cannot write in data/raw/locked."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        locked, cached = _helper_locked_file_project(runner, cwd, "locked/cached.txt")

        with _helper_locked(locked, 0o555):
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        _helper_assert_only_alpha_runs(
            dry_run,
            ran,
            [
                "Skipping: Stage 'kept/read_0' depends on input 'locked/cached.txt' "
                + "(path data/raw/locked/cached.txt), which does not exist"
            ],
        )
        _helper_assert_restore_warnings(
            dry_run,
            ran,
            f"Cannot restore {cached}: this user cannot write in {locked}",
            f"Cannot restore {cached}: [Errno 13] Permission denied: '{locked}/",
        )
        assert not cached.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a cache object whatever its mode")
@pytest.mark.parametrize("unreadable", ["object", "prefix_directory"])
def test_tracked_keep_going_and_dry_run_keep_going_leave_out_the_reader_of_an_unreadable_object(
    runner: CliRunner, tmp_path: pathlib.Path, unreadable: str
) -> None:
    """R1, R5: kept/read_0 reads data/raw/cached.txt, which is missing and whose object the cache
    holds, but which this user cannot read: the object's mode, or its prefix directory's in the
    cache, is 0o000. The run's --checkout-missing restore fails with PermissionError: the run
    warns once, naming the path and the error, leaves kept/read_0 out with a `Skipping:` line,
    runs alpha and exits 0. The dry run with --checkout-missing reads the object as the restore
    does, so it does not count the file as present: it lists alpha alone and leaves kept/read_0
    out with the same line, warning that the object cannot be read."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        helpers.write_pipeline(
            cwd, "kept", _helper_broken_pipeline_code(1, name="kept", input_name="cached.txt")
        )
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        cached = cwd / "data" / "raw" / "cached.txt"
        cached.parent.mkdir(parents=True)
        cached.write_text("cached bytes")
        tracking = runner.invoke(cli.cli, ["track", "data/raw/cached.txt"])
        assert tracking.exit_code == 0, tracking.output
        cached.unlink()
        record = track.read_pvt_file(track.get_pvt_path(cached))
        assert record is not None
        cache_object = cache.get_cache_path(cwd / ".pivot" / "cache" / "files", record["hash"])
        locked = cache_object if unreadable == "object" else cache_object.parent
        mode = stat.S_IMODE(locked.stat().st_mode)

        locked.chmod(0)
        try:
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)
        finally:
            locked.chmod(mode)

        _helper_assert_only_alpha_runs(
            dry_run,
            ran,
            [
                "Skipping: Stage 'kept/read_0' depends on input 'cached.txt' "
                + "(path data/raw/cached.txt), which does not exist"
            ],
        )
        _helper_assert_restore_warnings(
            dry_run,
            ran,
            f"Cannot restore {cached}: [Errno 13] Permission denied: '{cache_object}'",
            f"Cannot restore {cached}: [Errno 13] Permission denied: ",
        )
        assert not cached.exists()


def _helper_images_tree_project(
    runner: CliRunner, root: pathlib.Path, modes: str
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """`_helper_tracked_images_project` with the checkout modes ``modes`` (the list
    `cache.checkout_mode` gives, SYMLINK first) and the whole directory tree of data/raw/images
    in the cache, complete and marked (``helpers.cache_whole_tree``). Returns data/raw/images,
    the tree and the object of its member a.png."""
    (root / ".pivot" / "config.yaml").write_text(f"cache:\n  checkout_mode: [{modes}]\n")
    _helper_tracked_images_project(runner, root)
    files = root / ".pivot" / "cache" / "files"
    images = root / "data" / "raw" / "images"
    record = track.read_pvt_file(track.get_pvt_path(images))
    assert record is not None
    member_hash = helpers.cache_whole_tree(root, images)["files"][0]["hash"]
    return images, cache.tree_path(files, record["hash"]), cache.get_cache_path(files, member_hash)


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches a directory whatever its mode")
def test_tracked_keep_going_and_dry_run_keep_going_read_a_symlink_tree_as_the_restore_does(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R1, R5 (SYMLINK first): data/raw/images is a missing tracked directory whose whole
    directory tree the cache holds, complete and marked, as a stage output with the same files
    leaves it; its member object is unreadable (0o000). With the tree searchable, the restore
    links the directory to the tree and reads no object, so dirs/scan runs, and the dry run
    lists it. With the tree's directory unsearchable too (0o000), the restore cannot stat the
    tree's files and restores file by file, which fails on the member object: the run leaves
    dirs/scan out, and so does the dry run, with the same warning."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        images, tree, member_object = _helper_images_tree_project(
            runner, cwd, "symlink, hardlink, copy"
        )
        tree_mode = stat.S_IMODE(tree.stat().st_mode)

        member_object.chmod(0)
        try:
            linked_dry_run, linked_run = _helper_checkout_missing_dry_run_and_run(runner)
            linked = images.is_symlink() and images.resolve() == tree.resolve()
            images.unlink()
            tree.chmod(0)
            try:
                dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)
            finally:
                tree.chmod(tree_mode)
        finally:
            member_object.chmod(0o444)

        assert linked_dry_run.exit_code == 0, linked_dry_run.output
        assert json.loads(linked_dry_run.stdout)["stages"].keys() == {
            "alpha/stage_a",
            "dirs/scan",
        }, linked_dry_run.output
        assert linked_run.exit_code == 0, linked_run.output
        assert helpers.stage_statuses(linked_run.stdout) == {
            "alpha/stage_a": "ran",
            "dirs/scan": "ran",
        }, linked_run.output
        assert linked, "the restore links the directory to the tree"
        assert (cwd / "data" / "dirs" / "scan.txt").read_text() == "a.png"
        assert helpers.restore_warnings(linked_dry_run.stderr + linked_run.stderr) == []
        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": {"would_run": False, "reason": "unchanged"}}
        }, dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == _IMAGES_SKIPPED, dry_run.output
        assert ran.exit_code == 0, ran.output
        assert helpers.stage_statuses(ran.stdout) == {"alpha/stage_a": "cached"}, ran.output
        assert helpers.skipping_lines(ran.stderr) == _IMAGES_SKIPPED, ran.output
        warning = f"Cannot restore {images}: [Errno 13] Permission denied: '{member_object}'"
        assert helpers.restore_warnings(dry_run.stderr) == [warning], dry_run.output
        assert helpers.restore_warnings(ran.stderr) == [warning], ran.output
        assert not images.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
def test_tracked_keep_going_and_dry_run_keep_going_find_a_member_absent_past_a_failed_tree_link(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R1, R5 (SYMLINK first, links allowed): data/raw/images is a missing tracked directory
    whose whole tree the cache holds, complete and marked, its member a.png's object gone, and
    data/raw is read-only (0o555). The restore links the tree as it is into data/raw, which
    fails, then looks a.png's object up before it restores file by file, and finds it absent:
    the run leaves dirs/scan out, runs alpha and exits 0, with no restore warning, and so does
    the dry run (it warned that it cannot write in data/raw before, where the run warns of
    nothing)."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        images, _, member_object = _helper_images_tree_project(runner, cwd, _COPY_AMONG_MODES)
        member_object.unlink()

        with _helper_locked(cwd / "data" / "raw", 0o555):
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        _helper_assert_only_alpha_runs(dry_run, ran, _IMAGES_SKIPPED)
        for result in (dry_run, ran):
            assert helpers.restore_warnings(result.stderr) == [], result.output
        assert not images.exists()


def _helper_unreadable_member(
    runner: CliRunner, root: pathlib.Path, modes: str
) -> contextlib.AbstractContextManager[pathlib.Path | None]:
    """`_helper_images_tree_project`'s tree, whose member object the lock makes unreadable."""
    _, _, member_object = _helper_images_tree_project(runner, root, modes)
    return _helper_locked(member_object, 0)


def _helper_read_only_parent(
    runner: CliRunner, root: pathlib.Path, modes: str
) -> contextlib.AbstractContextManager[pathlib.Path | None]:
    """`_helper_images_tree_project`'s tree, whose parent data/raw the lock makes read-only."""
    _helper_images_tree_project(runner, root, modes)
    return _helper_locked(root / "data" / "raw", 0o555)


def _helper_unstattable_tree_without_member(
    runner: CliRunner, root: pathlib.Path, modes: str
) -> contextlib.AbstractContextManager[pathlib.Path | None]:
    """`_helper_images_tree_project`'s tree, whose member object is gone from the cache and
    whose directory the lock makes unsearchable, so its files cannot be stat'd."""
    _, tree, member_object = _helper_images_tree_project(runner, root, modes)
    member_object.unlink()
    return _helper_locked(tree, 0)


def _helper_images_without_tree(
    runner: CliRunner, root: pathlib.Path, modes: str
) -> contextlib.AbstractContextManager[pathlib.Path | None]:
    """`_helper_tracked_images_project` with the checkout modes ``modes`` and no whole tree in
    the cache, so a restore goes file by file; nothing is locked."""
    (root / ".pivot" / "config.yaml").write_text(f"cache:\n  checkout_mode: [{modes}]\n")
    _helper_tracked_images_project(runner, root)
    return contextlib.nullcontext()


def _helper_read_only_parent_without_tree(
    runner: CliRunner, root: pathlib.Path, modes: str
) -> contextlib.AbstractContextManager[pathlib.Path | None]:
    """`_helper_images_without_tree`, whose parent data/raw the lock makes read-only."""
    _helper_images_without_tree(runner, root, modes)
    return _helper_locked(root / "data" / "raw", 0o555)


def _helper_empty_listing(
    runner: CliRunner, root: pathlib.Path, modes: str
) -> contextlib.AbstractContextManager[pathlib.Path | None]:
    """dirs/scan reads data/raw/images, a directory holding only the empty directory sub/, that
    `pivot track` tracked with the checkout modes ``modes`` and that is then removed, and
    alpha/stage_a is healthy. Its listing holds no file, and the cache no whole tree; nothing is
    locked."""
    (root / ".pivot" / "config.yaml").write_text(f"cache:\n  checkout_mode: [{modes}]\n")
    helpers.write_pipeline(root, "dirs", _helper_directory_reader_code("images"))
    helpers.write_pipeline(root, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    images = root / "data" / "raw" / "images"
    (images / "sub").mkdir(parents=True)
    tracking = runner.invoke(cli.cli, ["track", "data/raw/images"])
    assert tracking.exit_code == 0, tracking.output
    shutil.rmtree(images)
    return contextlib.nullcontext()


# A project builder of the read-only-register agreement tests: it builds the tree with the
# checkout modes it is given and returns the lock to hold while both run
type _LockedProject = Callable[
    [CliRunner, pathlib.Path, str], contextlib.AbstractContextManager[pathlib.Path | None]
]

# Checkout modes, SYMLINK first, with COPY among them and without
_COPY_AMONG_MODES = "symlink, hardlink, copy"
_NO_COPY_MODES = "symlink, hardlink"


def _helper_read_only_cache_dry_run_and_run(
    runner: CliRunner,
    root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    project: _LockedProject,
    modes: str,
) -> tuple[Result, Result, pathlib.Path | None]:
    """``project``'s tree with the checkout modes ``modes``, whose cache's register is on a
    read-only mount (``helpers.register_on_a_read_only_mount``): the run cannot register there,
    so it refuses links into the cache (`naming.naming_unit`), and the dry run reads the mount's
    flag. Returns the dry run and the run on that tree, each made while ``project``'s lock is
    held, and the path it locked (None: nothing)."""
    lock = project(runner, root, modes)
    helpers.register_on_a_read_only_mount(monkeypatch, root / ".pivot" / "cache")
    with lock as locked:
        dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)
    assert cache.links_refused(root / ".pivot" / "cache" / "files"), (
        "the run's registration met the read-only register"
    )
    return dry_run, ran, locked


# A warning that a read-only-register agreement test's dry run gives, and the start of the
# run's, for a restore that cannot read the locked member object
_UNREADABLE_MEMBER = (
    "Cannot restore {images}: [Errno 13] Permission denied: '{locked}'",
    "Cannot restore {images}: [Errno 13] Permission denied: '{locked}'",
)
# The same for a restore that cannot write in the locked directory
_READ_ONLY_PARENT = (
    "Cannot restore {images}: this user cannot write in {locked}",
    "Cannot restore {images}: [Errno 13] Permission denied: '{locked}/",
)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads and writes whatever the mode")
@pytest.mark.parametrize(
    ("project", "modes", "warnings"),
    [
        pytest.param(
            _helper_unreadable_member,
            _COPY_AMONG_MODES,
            _UNREADABLE_MEMBER,
            id="unreadable_member-copy_among_modes",
        ),
        pytest.param(
            _helper_read_only_parent,
            _COPY_AMONG_MODES,
            _READ_ONLY_PARENT,
            id="read_only_parent-copy_among_modes",
        ),
        pytest.param(
            _helper_read_only_parent_without_tree,
            _COPY_AMONG_MODES,
            _READ_ONLY_PARENT,
            id="read_only_parent_file_by_file-copy_among_modes",
        ),
        pytest.param(
            _helper_read_only_parent_without_tree,
            _NO_COPY_MODES,
            _READ_ONLY_PARENT,
            id="read_only_parent_file_by_file",
        ),
        pytest.param(
            _helper_unstattable_tree_without_member,
            _COPY_AMONG_MODES,
            None,
            id="unstattable_tree_without_member-copy_among_modes",
        ),
        pytest.param(
            _helper_unstattable_tree_without_member,
            _NO_COPY_MODES,
            None,
            id="unstattable_tree_without_member",
        ),
    ],
)
def test_tracked_keep_going_and_dry_run_keep_going_leave_out_what_a_read_only_cache_cannot_restore(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    project: _LockedProject,
    modes: str,
    warnings: tuple[str, str] | None,
) -> None:
    """R1, R5: data/raw/images is a missing tracked directory that dirs/scan reads, and the
    cache's register is on a read-only mount. With COPY among the checkout modes the restore
    copies file by file, opening each object; with no COPY it refuses where it reaches a link.
    On each tree the restore leaves images missing before any refusal: the run leaves dirs/scan
    out, runs alpha and exits 0, and so does the dry run, each with ``warnings``' warning (the
    dry run's, and the start of the run's) or with none.

    In `unreadable_member` the cache holds images' whole tree, searchable, and its member object
    is unreadable (0o000): the copy fails on it. In `read_only_parent` data/raw is read-only
    (0o555): the copy cannot stage the directory. In `read_only_parent_file_by_file` the cache
    holds no whole tree, so the restore stages the directory, and cannot, either way. In
    `unstattable_tree_without_member` the cache holds images' whole tree, complete and marked,
    whose directory this user cannot search (0o000), and a.png's object is gone: the restore
    cannot stat the tree's files, so it restores file by file, and finds the object absent
    before it reaches a link, with no warning."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        images = cwd / "data" / "raw" / "images"

        dry_run, ran, locked = _helper_read_only_cache_dry_run_and_run(
            runner, cwd, monkeypatch, project, modes
        )

        _helper_assert_only_alpha_runs(dry_run, ran, _IMAGES_SKIPPED)
        if warnings is None:
            for result in (dry_run, ran):
                assert helpers.restore_warnings(result.stderr) == [], result.output
        else:
            assert locked is not None
            dry_run_warning, run_prefix = (
                warning.format(images=images, locked=locked) for warning in warnings
            )
            _helper_assert_restore_warnings(dry_run, ran, dry_run_warning, run_prefix)
        assert not images.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads and writes whatever the mode")
@pytest.mark.parametrize(
    "project",
    [
        pytest.param(_helper_unreadable_member, id="unreadable_member"),
        pytest.param(_helper_read_only_parent, id="read_only_parent"),
        pytest.param(_helper_images_without_tree, id="file_by_file"),
    ],
)
def test_tracked_keep_going_and_dry_run_keep_going_refuse_a_link_into_a_read_only_cache_as_the_run_does(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    project: _LockedProject,
) -> None:
    """R1, R5: data/raw/images is a missing tracked directory that dirs/scan reads, the cache's
    register is on a read-only mount and COPY is not among the checkout modes (SYMLINK first).
    The restore refuses where it reaches a link, which stops the run before any stage runs, and
    the dry run refuses there alone, with no document.

    In `unreadable_member` the cache holds images' whole tree, searchable, and its member object
    is unreadable (0o000): the symlink restore refuses at the tree, opening no object. In
    `read_only_parent` data/raw is read-only (0o555): the symlink restore refuses at the tree
    before it writes there. In `file_by_file` the cache holds no whole tree, so the restore
    stages the directory and refuses at its first file, a.png. The dry run's refusal names
    images and the cache's object store, where the run's names the tree or the file it links,
    so only the words before those are compared."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        images = cwd / "data" / "raw" / "images"

        dry_run, ran, _ = _helper_read_only_cache_dry_run_and_run(
            runner, cwd, monkeypatch, project, _NO_COPY_MODES
        )

        _helper_assert_both_refuse(dry_run, ran, str(images))
        assert not images.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads and writes whatever the mode")
@pytest.mark.parametrize(
    "modes",
    [
        pytest.param(_COPY_AMONG_MODES, id="empty_listing-copy_among_modes"),
        pytest.param(_NO_COPY_MODES, id="empty_listing"),
    ],
)
def test_tracked_keep_going_and_dry_run_keep_going_restore_an_empty_listing_from_a_read_only_cache(
    runner: CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, modes: str
) -> None:
    """R1, R5: data/raw/images is a missing tracked directory that dirs/scan reads, whose
    listing holds only the empty directory sub/, and the cache's register is on a read-only
    mount. With COPY among the checkout modes or not, the restore links nothing and brings
    images back: dirs/scan and alpha run, and the dry run lists both."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        dry_run, ran, _ = _helper_read_only_cache_dry_run_and_run(
            runner, cwd, monkeypatch, _helper_empty_listing, modes
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": True, "reason": "No previous run"},
                "dirs/scan": {"would_run": True, "reason": "No previous run"},
            }
        }, dry_run.output
        assert ran.exit_code == 0, ran.output
        assert helpers.stage_statuses(ran.stdout) == {
            "alpha/stage_a": "ran",
            "dirs/scan": "ran",
        }, ran.output
        for result in (dry_run, ran):
            assert helpers.skipping_lines(result.stderr) == [], result.output
            assert helpers.restore_warnings(result.stderr) == [], result.output
        assert (cwd / "data" / "dirs" / "scan.txt").read_text() == "sub"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads and writes whatever the mode")
@pytest.mark.parametrize(
    ("read_only_register", "modes"),
    [
        pytest.param(False, _COPY_AMONG_MODES, id="links"),
        pytest.param(True, _NO_COPY_MODES, id="read_only_register"),
    ],
)
def test_tracked_keep_going_and_dry_run_keep_going_fail_a_tree_restore_that_cannot_make_its_directory(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    read_only_register: bool,
    modes: str,
) -> None:
    """R1, R5 (SYMLINK first): data/raw/images.pvt names sub/images, so dirs/scan reads the
    missing tracked directory data/raw/sub/images, whose directory sub/ does not exist, and
    data/raw is read-only (0o555): no restore can make sub/. The cache holds the directory's
    whole tree, complete and marked, whose directory this user cannot search (0o000), and
    a.png's object is gone. A restore that starts from the whole tree makes the directory that
    holds its path before it reads the tree, so the run's restore fails there, with a warning,
    before it would find the tree's files unstattable, restore file by file and find the object
    absent. The run leaves dirs/scan out and runs alpha, and so does the dry run, warning that
    it cannot write in data/raw. So it is with the cache's links allowed (`links`), and with its
    register on a read-only mount and no COPY among the checkout modes, where the restore would
    refuse its link (`read_only_register`)."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _, tree, member_object = _helper_images_tree_project(runner, cwd, modes)
        member_object.unlink()
        raw = cwd / "data" / "raw"
        pvt_path = raw / "images.pvt"
        record = track.read_pvt_file(pvt_path)
        assert record is not None and "nfiles" in record
        track.write_pvt_file(
            pvt_path,
            track.PvtData(
                path="sub/images", hash=record["hash"], size=record["size"], nfiles=record["nfiles"]
            ),
        )
        # No committed listing can lie beside a path whose directory does not exist
        (raw / "images.pvt.listing").unlink()
        (cwd / "dirs" / "pipeline.py").write_text(_helper_directory_reader_code("sub/images"))
        images = raw / "sub" / "images"
        if read_only_register:
            helpers.register_on_a_read_only_mount(monkeypatch, cwd / ".pivot" / "cache")

        with _helper_locked(tree, 0), _helper_locked(raw, 0o555):
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        assert cache.links_refused(cwd / ".pivot" / "cache" / "files") == read_only_register
        _helper_assert_only_alpha_runs(
            dry_run,
            ran,
            [
                "Skipping: Stage 'dirs/scan' depends on input 'sub/images' "
                + "(path data/raw/sub/images), which does not exist"
            ],
        )
        _helper_assert_restore_warnings(
            dry_run,
            ran,
            f"Cannot restore {images}: this user cannot write in {raw}",
            f"Cannot restore {images}: [Errno 13] Permission denied: '{raw}/",
        )
        assert not images.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
@pytest.mark.parametrize("nested", [False, True], ids=["in_it", "below_a_missing_directory"])
def test_tracked_keep_going_and_dry_run_keep_going_refuse_a_file_link_as_the_run_does(
    runner: CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    """R1, R5: kept/read_0 reads a missing tracked file whose object the cache holds, the
    directory data/raw/locked is read-only, the cache's register is on a read-only mount and
    copy is not a checkout mode. The run's restore of a file makes the directory that holds it,
    then refuses to link the file's object before it writes there. In `in_it` the file is
    data/raw/locked/cached.txt: the run stops before any stage runs with the refusal, and so
    does the dry run, with no document. In `below_a_missing_directory` its .pvt names
    sub/cached.txt, whose directory sub/ does not exist: the restore cannot make it in
    data/raw/locked, so the run warns, leaves kept/read_0 out and runs alpha, and so does the
    dry run, warning that it cannot write in data/raw/locked."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        input_name = "locked/sub/cached.txt" if nested else "locked/cached.txt"
        (cwd / ".pivot" / "config.yaml").write_text(
            "cache:\n  checkout_mode: [hardlink, symlink]\n"
        )
        locked, cached = _helper_locked_file_project(runner, cwd, input_name)
        if nested:
            pvt_path = track.get_pvt_path(cached)
            record = track.read_pvt_file(pvt_path)
            assert record is not None
            track.write_pvt_file(
                pvt_path,
                track.PvtData(path="sub/cached.txt", hash=record["hash"], size=record["size"]),
            )
            cached = locked / "sub" / "cached.txt"
        helpers.register_on_a_read_only_mount(monkeypatch, cwd / ".pivot" / "cache")

        with _helper_locked(locked, 0o555):
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        assert cache.links_refused(cwd / ".pivot" / "cache" / "files"), (
            "the run's registration met the read-only register"
        )
        if nested:
            _helper_assert_only_alpha_runs(
                dry_run,
                ran,
                [
                    f"Skipping: Stage 'kept/read_0' depends on input '{input_name}' "
                    + f"(path data/raw/{input_name}), which does not exist"
                ],
            )
            _helper_assert_restore_warnings(
                dry_run,
                ran,
                f"Cannot restore {cached}: this user cannot write in {locked}",
                f"Cannot restore {cached}: [Errno 13] Permission denied: '{locked}/",
            )
        else:
            _helper_assert_both_refuse(dry_run, ran, f"{cached} to ")
        assert not cached.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
def test_tracked_keep_going_and_dry_run_keep_going_limit_a_cache_register_this_user_cannot_write(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R5's limit, pinned: data/raw/images is a missing tracked directory whose objects the cache
    holds, but the cache's register (`projects/`) holds no entry for this project and is a
    read-only directory on a writable filesystem. The run's registration must write an entry
    there, cannot, and stops the run before any stage runs, restoring nothing. The dry run
    writes nothing, so it cannot tell: whether a registration writes in the register, and
    fails, only the write tells (a current entry, on a kernel that lets the removal of an absent
    file pass in such a directory, needs no write). It lists dirs/scan as one the restore lets
    run."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_tracked_images_project(runner, cwd)
        register = cwd / ".pivot" / "cache" / naming.REGISTER_DIR
        entries = [path for path in register.iterdir() if path.suffix in (".json", ".nonce")]
        assert entries, "pivot track registered the project"
        for entry in entries:
            entry.unlink()

        register.chmod(0o555)
        try:
            dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)
        finally:
            register.chmod(0o755)

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout)["stages"].keys() == {
            "alpha/stage_a",
            "dirs/scan",
        }, dry_run.output
        assert ran.exit_code == 1, ran.output
        assert "Error: Cannot record this project in the cache register at " in ran.stderr, (
            ran.output
        )
        assert helpers.completed_stages(ran.stdout) == {}, ran.output
        assert not (cwd / "data" / "raw" / "images").exists()


def test_tracked_keep_going_and_dry_run_keep_going_refuse_a_cache_whose_store_is_its_own_mount(
    runner: CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache's object store, files/, is the root of a mount of its own: a bind mount of the
    directory onto itself, which keeps its device, so only this process's mount table shows it
    (`/proc/self/mounts`, which this user cannot change, stands in a file of this test's own).
    No register can show every project of such a store (`naming.StoreLayout.unshared`), so the
    run refuses at its start, before any stage runs, in every mode. The dry run in leave-out
    mode refuses the same way, with the same error and no document, where it listed
    alpha/stage_a before."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        store = cwd / ".pivot" / "cache" / "files"
        store.mkdir(parents=True)
        table = tmp_path / "mounts"
        table.write_text(
            f"/dev/root / ext4 rw 0 0\n/dev/root {os.path.realpath(store)} ext4 rw 0 0\n"
        )
        monkeypatch.setattr(naming, "_MOUNTS_TABLE", table)
        monkeypatch.setattr(naming, "_SHARED_MOUNT_WATCH", naming._SharedMountWatch())

        dry_run = runner.invoke(cli.cli, helpers.DRY_RUN_KEEP_GOING)
        ran = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "--jsonl"])

        refusal = (
            f"Error: Refusing to name objects in {cwd / '.pivot' / 'cache'}: its files/ "
            + f"({os.path.realpath(store)}) is a mount of its own"
        )
        for result in (dry_run, ran):
            assert result.exit_code == 1, result.output
            assert refusal in result.stderr, result.output
        assert dry_run.stdout == "", dry_run.output
        assert helpers.completed_stages(ran.stdout) == {}, ran.output
        assert not (cwd / "data" / "alpha" / "stage_a.txt").exists(), "no stage runs"


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches a directory whatever its mode")
def test_tracked_keep_going_off_an_unreadable_listing_prefix_directory_stops_as_in_0_10_4(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """R6, D1: outside leave-out mode (`repro --all --checkout-missing`, no --keep-going), the
    restore of data/raw/images, a missing tracked directory whose listing object's prefix
    directory in the cache this user cannot search (0o000), stops the run before any stage runs
    with the raw PermissionError of the listing object's lookup, as 0.10.4 stops it, not a
    listing error."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        listing_object = _helper_tracked_images_project(runner, cwd)
        prefix = listing_object.parent
        mode = stat.S_IMODE(prefix.stat().st_mode)

        prefix.chmod(0)
        try:
            ran = runner.invoke(cli.cli, ["repro", "--all", "--checkout-missing", "--jsonl"])
        finally:
            prefix.chmod(mode)

        assert ran.exit_code == 1, ran.output
        assert f"Error: Permission denied: {listing_object}\n" in ran.stderr, ran.output
        assert "listing" not in ran.stderr, ran.output
        assert helpers.completed_stages(ran.stdout) == {}, ran.output
        assert not (cwd / "data" / "raw" / "images").exists()


class _FakeTuiApp:
    """Stands for ``pivot_tui.run.PivotApp`` in a one-shot ``repro --tui``: it starts no terminal
    UI, and its ``run`` returns once the engine thread posts ``TuiShutdown`` at its end, as the
    real app exits then (``_run_common.run_tui_with_engine``)."""

    shut_down: threading.Event
    return_code: int | None

    def __init__(self, **_options: object) -> None:
        self.shut_down = threading.Event()
        self.return_code = None

    def post_message(self, message: object) -> bool:
        if isinstance(message, pivot_tui.run.TuiShutdown):
            self.shut_down.set()
        return True

    def set_event_poller(self, _poller: object) -> None:
        return None

    def run(self) -> None:
        assert self.shut_down.wait(timeout=300), "the engine thread never ended"
        self.return_code = 0


@pytest.mark.parametrize(
    ("tracked", "tui"),
    [
        pytest.param(True, False, id="tracked"),
        pytest.param(False, False, id="untracked"),
        pytest.param(True, True, id="tui"),
    ],
)
def test_repro_all_keep_going_left_out_target_fails_the_run_after_the_other_target_runs(
    runner: CliRunner, tmp_path: pathlib.Path, mocker: MockerFixture, tracked: bool, tui: bool
) -> None:
    """A3 (R4): of targets broken/after, downstream of the reader of a missing input, and the
    healthy alpha/stage_a, alpha/stage_a runs; the run then exits 1 naming broken/after and the
    missing file, and no target is an unknown stage. The tracked cases have a second missing
    tracked file that no stage reads. The tui case runs through the one-shot TUI
    (``cli/repro.py`` ``_run_oneshot_mode``'s TUI branch), whose app is a stand-in that shows
    nothing; the engine, its RPC socket and the event poller are the real ones."""
    tui_app = mocker.patch.object(pivot_tui.run, "PivotApp", autospec=True, side_effect=_FakeTuiApp)
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd, tracked=tracked)
        if tracked:
            helpers.track_without_data(cwd / "data" / "raw" / "unrelated.txt")

        result = runner.invoke(
            cli.cli,
            [
                "repro",
                "--all",
                "--keep-going",
                *(["--tui"] if tui else []),
                "broken/after",
                "alpha/stage_a",
            ],
        )

        assert tui_app.call_count == (1 if tui else 0), "only the tui case goes through the TUI"
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
        helpers.write_pipeline(cwd, "m", _MAPPED_SHARED_INPUT_CODE)
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
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


def test_repro_all_dry_run_keep_going_lists_the_stages_the_run_runs(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5 on A1: the dry run's stages are those the run on the same tree runs or skips; the
    reader of the missing tracked file and the stage downstream of it are absent, and one
    `Skipping:` line on stderr names the reader."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)

        dry_run = runner.invoke(cli.cli, helpers.DRY_RUN_KEEP_GOING)
        ran = runner.invoke(cli.cli, ["repro", "--all", "--keep-going", "--jsonl"])

        _helper_assert_only_alpha_runs(dry_run, ran, [_READER_SKIPPED])


def test_repro_all_dry_run_keep_going_checkout_missing_counts_a_cached_file_as_present(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5, D7: with --checkout-missing, the reader of a missing tracked file whose object the
    cache holds is listed, with the reason the run gives after it restores the file (its .pvt
    names other bytes than the reader last read); without the flag it is left out. The reader
    of a tracked file with no object is left out either way, and the dry run restores nothing."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_reader_project(cwd)
        helpers.write_pipeline(
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

        without = runner.invoke(cli.cli, helpers.DRY_RUN_KEEP_GOING)
        dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": False, "reason": "unchanged"},
                "kept/read_0": {"would_run": True, "reason": "Input dependencies changed"},
            }
        }, dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [_READER_SKIPPED], dry_run.output
        assert without.exit_code == 0, without.output
        assert json.loads(without.stdout)["stages"].keys() == {"alpha/stage_a"}, without.output
        assert ran.exit_code == 0, ran.output
        assert helpers.stage_statuses(ran.stdout) == {
            "alpha/stage_a": "cached",
            "kept/read_0": "ran",
        }, ran.output
        assert helpers.completed_stages(ran.stdout)["kept/read_0"]["reason"] == (
            "Input dependencies changed"
        ), ran.output
        assert cached.read_text() == "cached bytes"


def test_repro_all_dry_run_keep_going_checkout_missing_counts_a_file_reached_through_a_symlink(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """A5, D7: kept/read_0 reads data/raw/alias/x.txt, where alias links to inputs/sub and the
    cache can restore the missing tracked file inputs/sub/x.txt. With --checkout-missing the
    dry run lists kept/read_0, as the run restores the file and runs it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        helpers.write_pipeline(
            cwd, "kept", _helper_broken_pipeline_code(1, name="kept", input_name="alias/x.txt")
        )
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        tracked = cwd / "data" / "raw" / "inputs" / "sub" / "x.txt"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("x bytes")
        tracking = runner.invoke(cli.cli, ["track", "data/raw/inputs/sub/x.txt"])
        assert tracking.exit_code == 0, tracking.output
        tracked.unlink()
        (cwd / "data" / "raw" / "alias").symlink_to("inputs/sub")

        dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": True, "reason": "No previous run"},
                "kept/read_0": {"would_run": True, "reason": "No previous run"},
            }
        }, dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output
        assert ran.exit_code == 0, ran.output
        assert helpers.stage_statuses(ran.stdout) == {
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
        helpers.write_pipeline(cwd, "dirs", _helper_directory_reader_code("images/sub"))
        helpers.write_pipeline(
            cwd, "cat", _helper_broken_pipeline_code(1, name="cat", input_name="images/cat.png")
        )
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        images = cwd / "data" / "raw" / "images"
        (images / "sub").mkdir(parents=True)
        (images / "a.png").write_text("a")
        (images / "sub" / "b.png").write_text("b")
        tracking = runner.invoke(cli.cli, ["track", "data/raw/images"])
        assert tracking.exit_code == 0, tracking.output
        shutil.rmtree(images)

        dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        skipped = [
            "Skipping: Stage 'cat/read_0' depends on input 'images/cat.png' "
            + "(path data/raw/images/cat.png), which does not exist"
        ]
        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": True, "reason": "No previous run"},
                "dirs/scan": {"would_run": True, "reason": "No previous run"},
            }
        }, dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == skipped, dry_run.output
        assert ran.exit_code == 0, ran.output
        assert helpers.stage_statuses(ran.stdout) == {
            "alpha/stage_a": "ran",
            "dirs/scan": "ran",
        }, ran.output
        assert helpers.skipping_lines(ran.stderr) == skipped, ran.output
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
        helpers.write_pipeline(cwd, "dirs", _helper_directory_reader_code("images"))
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        images = cwd / "data" / "raw" / "images"
        images.mkdir(parents=True)
        (images / "a.png").write_text("a")
        tracking = runner.invoke(cli.cli, ["track", "data/raw/images"])
        assert tracking.exit_code == 0, tracking.output
        shutil.rmtree(images)
        (cwd / "data" / "raw" / "extra.pvt").write_text(
            "path: images/extra.bin\nhash: 0123456789abcdef\nsize: 10\n"
        )

        dry_run, ran = _helper_checkout_missing_dry_run_and_run(runner)

        _helper_assert_only_alpha_runs(
            dry_run,
            ran,
            [
                "Skipping: Stage 'dirs/scan' depends on directory 'images' (path "
                + "data/raw/images), which holds tracked file 'data/raw/images/extra.bin' that "
                + "does not exist"
            ],
        )
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
        helpers.write_pipeline(
            cwd,
            "sub",
            'from pivot.compose import Pipeline\n\nwith Pipeline("sub") as pipeline:\n    pass\n',
        )

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "sub/pipeline.py"])
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--jsonl", "sub/pipeline.py"]
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": {"would_run": True, "reason": "No previous run"}}
        }, dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [_READER_SKIPPED], dry_run.output
        assert ran.exit_code == 0, ran.output
        assert helpers.stage_statuses(ran.stdout) == {"alpha/stage_a": "ran"}, ran.output


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
        assert helpers.skipping_lines(result.output) == [_READER_SKIPPED], result.output


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
        helpers.write_pipeline(cwd, "m", _MAPPED_SHARED_INPUT_CODE)
        helpers.track_without_data(cwd / "data" / "m" / "plan" / "c.txt")

        result = runner.invoke(cli.cli, helpers.DRY_RUN_KEEP_GOING)
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
