"""The notice of `pivot repro --all --keep-going --dry-run --checkout-missing` (and `--explain`)
for a mapped stage it shows as not yet expanded although the run may expand it
(``cli/repro.py`` ``_log_restore_limits``; ``status.restore_limits`` owns the rule). Each test
name holds ``dry_run_keep_going``, check c4's ``-k`` word.
"""

from __future__ import annotations

import functools
import json
import shutil
from typing import TYPE_CHECKING, NamedTuple

import pytest

from pivot import cli
from tests import helpers
from tests.conftest import isolated_pivot_dir

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from click.testing import CliRunner

# p/cut is mapped over p/plan's files, and p/plan reads the directory data/raw/dir/, writing one
# file for each .txt file it holds
_MAPPED_OVER_A_READ_DIRECTORY_CODE = """\
import pathlib
from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def plan(folder: pathlib.Path) -> Annotated[dict[str, str], directory]:
    return {entry.name: entry.read_text() for entry in sorted(folder.glob("*.txt"))}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan(pipeline.input("dir", t=pathlib.Path))))
"""

# The same, with p/prep reading data/raw/dir/ upstream of p/plan
_MAPPED_BELOW_A_READ_DIRECTORY_CODE = """\
import pathlib
from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def prep(folder: pathlib.Path) -> str:
    return ",".join(sorted(entry.name for entry in folder.glob("*.txt")))


@stage
def plan(names: str) -> Annotated[dict[str, str], directory]:
    return {name: name for name in names.split(",")}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan(prep(pipeline.input("dir", t=pathlib.Path)))))
"""

# The same, with p/plan reading data/raw/dir/ too, beside p/prep's output
_MAPPED_OVER_AND_BELOW_A_READ_DIRECTORY_CODE = """\
import pathlib
from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def prep(folder: pathlib.Path) -> str:
    return ",".join(sorted(entry.name for entry in folder.glob("*.txt")))


@stage
def plan(folder: pathlib.Path, names: str) -> Annotated[dict[str, str], directory]:
    return {name: (folder / name).read_text() for name in names.split(",")}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    folder = pipeline.input("dir", t=pathlib.Path)
    cut(pipeline.map(plan(folder, prep(folder))))
"""

# _MAPPED_OVER_A_READ_DIRECTORY_CODE with params for p/plan, which params.yaml can set
_MAPPED_OVER_A_READ_DIRECTORY_WITH_PARAMS_CODE = """\
import pathlib
from typing import Annotated

from pivot import StageParams
from pivot.compose import Pipeline, directory, stage


class PlanParams(StageParams):
    tag: str = "one"


@stage
def plan(folder: pathlib.Path, params: PlanParams) -> Annotated[dict[str, str], directory]:
    return {entry.name: entry.read_text() + params.tag for entry in sorted(folder.glob("*.txt"))}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan(pipeline.input("dir", t=pathlib.Path), params=PlanParams())))
"""

# p/cut is mapped over p/plan's files, and p/plan reads the file data/raw/plan.txt, writing one
# file for each name it holds
_MAPPED_OVER_A_READ_FILE_CODE = """\
from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def plan(names: str) -> Annotated[dict[str, str], directory]:
    return {f"{name}.txt": name for name in names.split()}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan(pipeline.input("plan.txt", t=str))))
"""

# p/cut is mapped over p/plan's files, and p/plan reads data/raw/corpus/a.txt, a file of the
# tracked directory data/raw/corpus/, and data/raw/other.txt
_MAPPED_OVER_A_MEMBER_CODE = """\
from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def plan(member: str, other: str) -> Annotated[dict[str, str], directory]:
    return {"member.txt": member, "other.txt": other}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    member = pipeline.input("corpus/a.txt", t=str)
    cut(pipeline.map(plan(member, pipeline.input("other.txt", t=str))))
"""

# _MAPPED_OVER_A_READ_DIRECTORY_CODE with p/plan reading data/raw/extra.txt too
_MAPPED_OVER_A_READ_DIRECTORY_AND_A_FILE_CODE = _MAPPED_OVER_A_READ_DIRECTORY_CODE.replace(
    "def plan(folder: pathlib.Path)", "def plan(folder: pathlib.Path, extra: str)"
).replace(
    'plan(pipeline.input("dir", t=pathlib.Path))',
    'plan(pipeline.input("dir", t=pathlib.Path), pipeline.input("extra.txt", t=str))',
)

# _MAPPED_BELOW_A_READ_DIRECTORY_CODE with p/plan reading data/raw/extra.txt too
_MAPPED_BELOW_A_READ_DIRECTORY_AND_A_FILE_CODE = _MAPPED_BELOW_A_READ_DIRECTORY_CODE.replace(
    "def plan(names: str)", "def plan(names: str, extra: str)"
).replace(
    'plan(prep(pipeline.input("dir", t=pathlib.Path)))',
    'plan(prep(pipeline.input("dir", t=pathlib.Path)), pipeline.input("extra.txt", t=str))',
)

# _MAPPED_OVER_A_READ_DIRECTORY_CODE with p/cut reading data/raw/missing.txt too
_MAPPED_OVER_A_READ_DIRECTORY_CUT_READING_A_FILE_CODE = _MAPPED_OVER_A_READ_DIRECTORY_CODE.replace(
    "def cut(entry: str) -> str:", "def cut(entry: str, shared: str) -> str:"
).replace(
    'pipeline.input("dir", t=pathlib.Path))))',
    'pipeline.input("dir", t=pathlib.Path))), pipeline.input("missing.txt", t=str))',
)

# _MAPPED_OVER_A_READ_DIRECTORY_CODE with p/plan reading data/raw/images/sub/, which the tests
# make a subdirectory of the tracked directory data/raw/images/
_MAPPED_OVER_A_SUBDIRECTORY_CODE = _MAPPED_OVER_A_READ_DIRECTORY_CODE.replace(
    'pipeline.input("dir", t=pathlib.Path)', 'pipeline.input("images/sub", t=pathlib.Path)'
)

# p/prep reads the directory data/raw/a/; p/plan reads p/prep's output and the directory
# data/raw/b/, writing one file for each name p/prep gives and each .txt file of b
_MAPPED_BELOW_TWO_READ_DIRECTORIES_CODE = """\
import pathlib
from typing import Annotated

from pivot.compose import Pipeline, directory, stage


@stage
def prep(folder: pathlib.Path) -> str:
    return ",".join(sorted(entry.name for entry in folder.glob("*.txt")))


@stage
def plan(names: str, folder: pathlib.Path) -> Annotated[dict[str, str], directory]:
    found = sorted(entry.name for entry in folder.glob("*.txt"))
    return {name: name for name in [*names.split(","), *found]}


@stage
def cut(entry: str) -> str:
    return entry


with Pipeline("p") as pipeline:
    folder = pipeline.input("b", t=pathlib.Path)
    cut(pipeline.map(plan(prep(pipeline.input("a", t=pathlib.Path)), folder)))
"""

_UNCHANGED = {"would_run": False, "reason": "unchanged"}
_INPUTS_CHANGED = {"would_run": True, "reason": "Input dependencies changed"}


def _helper_waiting(reason: str) -> dict[str, dict[str, object]]:
    """The rows of p/plan, which runs for ``reason``, and of p/cut, which waits for it."""
    return {
        "p/plan": {"would_run": True, "reason": reason},
        "p/cut": {"would_run": True, "reason": f"Not yet expanded: waits for 'p/plan': {reason}"},
    }


def _helper_track(runner: CliRunner, cwd: pathlib.Path, name: str, text: str) -> pathlib.Path:
    """Write ``text`` to ``cwd / name`` and track it with the real ``pivot track``, which puts its
    object in the cache; return its path."""
    path = cwd / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    tracking = runner.invoke(cli.cli, ["track", name])
    assert tracking.exit_code == 0, tracking.output
    return path


def _helper_first_run(runner: CliRunner) -> None:
    first = runner.invoke(cli.cli, ["repro", "--all", "--keep-going"])
    assert first.exit_code == 0, first.output


def _helper_notices(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if "Notice:" in line]


def _helper_restore_limit_project(
    runner: CliRunner, cwd: pathlib.Path, code: str, *, ran: bool = True
) -> pathlib.Path:
    """Pipeline p of ``code`` and the healthy alpha/stage_a. data/raw/dir/ holds present.txt and
    the tracked file f.txt. With ``ran``, a first run expands p/cut over both files. Then f.txt
    is removed from the workspace; the cache keeps its object. Returns the path of f.txt."""
    helpers.write_pipeline(cwd, "p", code)
    helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    tracked = _helper_track(runner, cwd, "data/raw/dir/f.txt", "f")
    (tracked.parent / "present.txt").write_text("present")
    if ran:
        _helper_first_run(runner)
    tracked.unlink()
    return tracked


def _helper_restore_limit_notice(
    reader: str, path: str = "data/raw/dir", tracked: str = "data/raw/dir/f.txt"
) -> str:
    """The notice for p/cut when ``reader``, p/plan or p/prep upstream of it, reads ``path``,
    which changes when --checkout-missing restores the tracked path ``tracked``."""
    who = (
        "Its producer 'p/plan'"
        if reader == "p/plan"
        else f"Stage '{reader}', upstream of its producer 'p/plan',"
    )
    return (
        f"Notice: mapped stage 'p/cut' is shown as not yet expanded. {who} reads '{path}', "
        + f"which changes when --checkout-missing restores tracked path '{tracked}'. "
        + f"The dry run explains '{reader}' from the files on disk, so the run may restore "
        + f"'{tracked}', find 'p/plan' current and add the instances of 'p/cut'."
    )


def test_repro_all_dry_run_keep_going_explain_checkout_missing_notes_a_mapped_stage(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """--explain makes the same decisions as the dry run: on the tree where p/plan reads
    data/raw/dir/, which holds the missing tracked file f.txt whose object the cache holds, it
    shows p/cut as not yet expanded and prints the same one notice on stderr."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_restore_limit_project(runner, cwd, _MAPPED_OVER_A_READ_DIRECTORY_CODE)

        result = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--explain", "--checkout-missing"]
        )

        assert result.exit_code == 0, result.output
        assert (
            "Stage: p/cut\n  Status: NOT YET EXPANDED\n"
            + "  Reason: Not yet expanded: waits for 'p/plan': Input dependencies changed\n"
        ) in result.output, result.output
        assert _helper_notices(result.stderr) == [_helper_restore_limit_notice("p/plan")]


def test_repro_all_dry_run_keep_going_checkout_missing_notes_no_refused_mapped_stage(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """No notice for a mapped stage that is not expanded because the run refuses it, while its
    producer is current. p/plan reads data/raw/dir/, which holds the missing tracked file f.txt
    whose object the cache holds; f.txt was tracked after the first run, and .pivotignore leaves
    its .pvt out of the directory's hash, so p/plan stays current. data/p/cut@present.txt, the
    output p/cut@present would write, is a missing tracked file that no cache holds. The dry run
    with --checkout-missing shows p/cut as one row, which would not run, with the run's reason for
    refusing it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / ".pivotignore").write_text("*.pvt\n")
        helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_READ_DIRECTORY_CODE)
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        held = cwd / "data" / "raw" / "dir"
        held.mkdir(parents=True)
        (held / "present.txt").write_text("present")
        _helper_first_run(runner)
        _helper_track(runner, cwd, "data/raw/dir/f.txt", "f").unlink()
        output = cwd / "data" / "p" / "cut@present.txt"
        output.unlink()
        helpers.track_without_data(output)

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])

        assert dry_run.exit_code == 0, dry_run.output
        stages = json.loads(dry_run.stdout)["stages"]
        assert stages.keys() == {"alpha/stage_a", "p/plan", "p/cut"}, dry_run.output
        assert stages["p/plan"] == _UNCHANGED, dry_run.output
        assert stages["p/cut"]["reason"].startswith(
            "cannot expand over the files of 'p/plan': The following tracked files are missing:\n"
        ), dry_run.output
        assert "data/p/cut@present.txt" in stages["p/cut"]["reason"], dry_run.output
        assert stages["p/cut"]["would_run"] is False, dry_run.output
        assert _helper_notices(dry_run.stderr) == [], dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output


def _helper_member_project(runner: CliRunner, cwd: pathlib.Path) -> pathlib.Path:
    """Pipeline p of ``_MAPPED_OVER_A_MEMBER_CODE`` and the healthy alpha/stage_a, with
    data/raw/other.txt and the tracked directory data/raw/corpus/, whose a.txt holds "a" and
    whose objects and listing the cache holds. Returns the path of corpus."""
    helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_MEMBER_CODE)
    helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    corpus = cwd / "data" / "raw" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("a")
    (corpus.parent / "other.txt").write_text("other")
    tracking = runner.invoke(cli.cli, ["track", "data/raw/corpus"])
    assert tracking.exit_code == 0, tracking.output
    return corpus


def _helper_extra_changed(runner: CliRunner, cwd: pathlib.Path, code: str) -> None:
    """``_helper_restore_limit_project`` of ``code``, whose p/plan reads data/raw/extra.txt too;
    extra.txt changes after the first run."""
    extra = cwd / "data" / "raw" / "extra.txt"
    extra.parent.mkdir(parents=True)
    extra.write_text("one")
    _helper_restore_limit_project(runner, cwd, code)
    extra.write_text("two")


def _helper_params_changed(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/; params.yaml changes its params after the first run."""
    _helper_restore_limit_project(runner, cwd, _MAPPED_OVER_A_READ_DIRECTORY_WITH_PARAMS_CODE)
    (cwd / "params.yaml").write_text("p/plan:\n  tag: two\n")


def _helper_code_changed(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/; its code changes after the first run."""
    code = _MAPPED_OVER_A_READ_DIRECTORY_CODE
    _helper_restore_limit_project(runner, cwd, code)
    (cwd / "p" / "pipeline.py").write_text(
        code.replace('sorted(folder.glob("*.txt"))', 'sorted(folder.glob("*.txt"), key=str)')
    )


def _helper_tracked_file_code_changed(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads the tracked file data/raw/plan.txt itself, whose hash the dry run takes from
    its .pvt. After the first run its code changes and plan.txt is removed; the cache keeps it."""
    helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_READ_FILE_CODE)
    helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    tracked = _helper_track(runner, cwd, "data/raw/plan.txt", "a b")
    _helper_first_run(runner)
    (cwd / "p" / "pipeline.py").write_text(
        _MAPPED_OVER_A_READ_FILE_CODE.replace("names.split()", "names.split(' ')")
    )
    tracked.unlink()


def _helper_directory_beside(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/, which holds no tracked file: the tracked file
    data/raw/dir-backup/f.txt is beside it. After the first run a file is added to dir and f.txt
    is removed; the cache keeps it."""
    helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_READ_DIRECTORY_CODE)
    helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    held = cwd / "data" / "raw" / "dir"
    held.mkdir(parents=True)
    (held / "present.txt").write_text("present")
    tracked = _helper_track(runner, cwd, "data/raw/dir-backup/f.txt", "f")
    _helper_first_run(runner)
    (held / "new.txt").write_text("new")
    tracked.unlink()


def _helper_cut_left_out(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/, and p/cut also reads data/raw/missing.txt, a missing tracked
    file that no cache holds, so the run leaves p/cut out."""
    helpers.track_without_data(cwd / "data" / "raw" / "missing.txt")
    _helper_restore_limit_project(
        runner, cwd, _MAPPED_OVER_A_READ_DIRECTORY_CUT_READING_A_FILE_CODE
    )


def _helper_retrack_corpus(runner: CliRunner, corpus: pathlib.Path) -> None:
    """Give corpus/a.txt new contents and track corpus again with the real ``pivot track``."""
    (corpus / "a.txt").write_text("second")
    retracking = runner.invoke(cli.cli, ["track", "--force", "data/raw/corpus"])
    assert retracking.exit_code == 0, retracking.output


def _helper_member_retracked(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads corpus/a.txt and other.txt. After the first run a.txt gets new contents and
    corpus is tracked again, then corpus and its committed listing are removed: the explanation
    takes a.txt's hash from the cache's listing, and it is p/plan's only changed input."""
    corpus = _helper_member_project(runner, cwd)
    _helper_first_run(runner)
    _helper_retrack_corpus(runner, corpus)
    (corpus.parent / "corpus.pvt.listing").unlink()
    shutil.rmtree(corpus)


def _helper_stale_committed_listing(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads corpus/a.txt and other.txt and has not run. a.txt gets new contents and
    corpus is tracked again, then its committed listing is put back to the earlier one, whose
    bytes no longer hash to the .pvt's record, and corpus is removed. The cache holds the right
    listing and every object."""
    corpus = _helper_member_project(runner, cwd)
    committed = corpus.parent / "corpus.pvt.listing"
    stale = committed.read_bytes()
    _helper_retrack_corpus(runner, corpus)
    assert committed.read_bytes() != stale
    committed.write_bytes(stale)
    shutil.rmtree(corpus)


def _helper_subdirectory_removed(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/images/sub/, a subdirectory of the tracked directory
    data/raw/images/. After the first run images is removed; the cache keeps its listing and
    objects. So the path the notice takes at the lock's hash is missing from the disk."""
    helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_SUBDIRECTORY_CODE)
    helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    images = cwd / "data" / "raw" / "images"
    (images / "sub").mkdir(parents=True)
    (images / "sub" / "a.txt").write_text("a")
    (images / "other.txt").write_text("other")
    tracking = runner.invoke(cli.cli, ["track", "data/raw/images"])
    assert tracking.exit_code == 0, tracking.output
    _helper_first_run(runner)
    shutil.rmtree(images)


def _helper_current_upstream_reads_a_restored_path(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/prep reads data/raw/a/, and p/plan reads p/prep's output and data/raw/b/; .pivotignore
    leaves each .pvt out of a directory's hash. The first run reads a/present.txt, and
    b/keep.txt and the tracked b/g.txt. Then a/f.txt is tracked, and a/f.txt and b/g.txt are
    removed; the cache keeps both. p/prep is current for the disk without f.txt, and the
    restore adds f.txt to what it reads."""
    (cwd / ".pivotignore").write_text("*.pvt\n")
    helpers.write_pipeline(cwd, "p", _MAPPED_BELOW_TWO_READ_DIRECTORIES_CODE)
    helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    raw = cwd / "data" / "raw"
    (raw / "a").mkdir(parents=True)
    (raw / "a" / "present.txt").write_text("present")
    held = _helper_track(runner, cwd, "data/raw/b/g.txt", "g")
    (raw / "b" / "keep.txt").write_text("keep")
    _helper_first_run(runner)
    _helper_track(runner, cwd, "data/raw/a/f.txt", "f").unlink()
    held.unlink()


class _Row(NamedTuple):
    """One tree of ``test_..._notice_agrees_with_the_run``."""

    setup: Callable[[CliRunner, pathlib.Path], object]
    stages: dict[str, object]
    """The stages of the dry run's document."""
    reader: str | None
    """The stage the dry run's notice names (``_helper_restore_limit_notice``); None: none."""
    plan: str
    """The status of p/plan in the run."""
    expanded: bool
    """Whether the run expands p/cut."""
    args: tuple[str, ...] = ()
    """Given to the dry run and to the run."""
    read: tuple[str, str] = ("data/raw/dir", "data/raw/dir/f.txt")
    """The path the notice says ``reader`` reads, and the tracked path whose restore changes it."""


_NEVER_RAN = {"would_run": True, "reason": "No previous run"}


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            _Row(
                functools.partial(
                    _helper_restore_limit_project, code=_MAPPED_OVER_A_READ_DIRECTORY_CODE
                ),
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Input dependencies changed")},
                reader="p/plan",
                plan="cached",
                expanded=True,
            ),
            id="only-the-restored-input-changed",
        ),
        pytest.param(
            _Row(
                functools.partial(
                    _helper_restore_limit_project, code=_MAPPED_BELOW_A_READ_DIRECTORY_CODE
                ),
                {
                    "alpha/stage_a": _UNCHANGED,
                    "p/prep": _INPUTS_CHANGED,
                    **_helper_waiting("Upstream stale (p/prep)"),
                },
                reader="p/prep",
                plan="cached",
                expanded=True,
            ),
            id="upstream-stage-reads-the-restored-input",
        ),
        pytest.param(
            _Row(
                functools.partial(
                    _helper_restore_limit_project,
                    code=_MAPPED_OVER_AND_BELOW_A_READ_DIRECTORY_CODE,
                ),
                {
                    "alpha/stage_a": _UNCHANGED,
                    "p/prep": _INPUTS_CHANGED,
                    **_helper_waiting("Input dependencies changed"),
                },
                reader="p/plan",
                plan="cached",
                expanded=True,
            ),
            id="producer-and-upstream-stage-read-the-restored-input",
        ),
        pytest.param(
            _Row(
                functools.partial(
                    _helper_extra_changed, code=_MAPPED_OVER_A_READ_DIRECTORY_AND_A_FILE_CODE
                ),
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Input dependencies changed")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="a-second-changed-input",
        ),
        pytest.param(
            _Row(
                functools.partial(
                    _helper_extra_changed, code=_MAPPED_BELOW_A_READ_DIRECTORY_AND_A_FILE_CODE
                ),
                {
                    "alpha/stage_a": _UNCHANGED,
                    "p/prep": _INPUTS_CHANGED,
                    **_helper_waiting("Input dependencies changed"),
                },
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="upstream-stage-reads-the-restored-input-producer-has-a-changed-input",
        ),
        pytest.param(
            _Row(
                _helper_params_changed,
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Params changed")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="a-changed-param",
        ),
        pytest.param(
            _Row(
                _helper_code_changed,
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Code changed")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="a-code-change",
        ),
        pytest.param(
            _Row(
                functools.partial(
                    _helper_restore_limit_project,
                    code=_MAPPED_OVER_A_READ_DIRECTORY_CODE,
                    ran=False,
                ),
                {"alpha/stage_a": _NEVER_RAN, **_helper_waiting("No previous run")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="never-run",
        ),
        pytest.param(
            _Row(
                functools.partial(
                    _helper_restore_limit_project, code=_MAPPED_OVER_A_READ_DIRECTORY_CODE
                ),
                {
                    "alpha/stage_a": {"would_run": True, "reason": "forced"},
                    **_helper_waiting("Input dependencies changed"),
                },
                reader=None,
                plan="ran",
                expanded=True,
                args=("--force",),
            ),
            id="forced",
        ),
        pytest.param(
            _Row(
                _helper_tracked_file_code_changed,
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Code changed")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="no-restore-changed-input-reads-the-tracked-file-itself",
        ),
        pytest.param(
            _Row(
                _helper_directory_beside,
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Input dependencies changed")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="no-restore-changed-input-reads-a-directory-beside-it",
        ),
        pytest.param(
            _Row(
                _helper_cut_left_out,
                {"alpha/stage_a": _UNCHANGED, "p/plan": _INPUTS_CHANGED},
                reader=None,
                plan="cached",
                expanded=False,
            ),
            id="mapped-stage-left-out-for-an-unrestorable-input",
        ),
        pytest.param(
            _Row(
                _helper_member_retracked,
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Input dependencies changed")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="member-read-from-the-cached-listing-changed",
        ),
        pytest.param(
            _Row(
                _helper_stale_committed_listing,
                {"alpha/stage_a": _NEVER_RAN, **_helper_waiting("No previous run")},
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="stale-committed-listing",
        ),
        pytest.param(
            _Row(
                _helper_subdirectory_removed,
                {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Missing deps: images/sub")},
                reader="p/plan",
                plan="cached",
                expanded=True,
                read=("data/raw/images/sub", "data/raw/images"),
            ),
            id="restored-input-missing-from-the-disk",
        ),
        pytest.param(
            _Row(
                _helper_current_upstream_reads_a_restored_path,
                {
                    "alpha/stage_a": _UNCHANGED,
                    "p/prep": _UNCHANGED,
                    **_helper_waiting("Input dependencies changed"),
                },
                reader=None,
                plan="ran",
                expanded=True,
            ),
            id="current-upstream-stage-reads-a-restored-path",
        ),
    ],
)
def test_repro_all_dry_run_keep_going_checkout_missing_notice_agrees_with_the_run(
    runner: CliRunner, tmp_path: pathlib.Path, row: _Row
) -> None:
    """On each row's tree --checkout-missing restores a missing tracked file whose object the
    cache holds. The dry run with it exits 0 with the row's document and prints the row's
    notice, or none. The run with it on the same tree then finds p/plan cached and expands p/cut
    exactly where the dry run printed the notice."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        row.setup(runner, cwd)

        dry_run = runner.invoke(
            cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing", *row.args]
        )
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl", *row.args]
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {"stages": row.stages}, dry_run.output
        notices = _helper_notices(dry_run.stderr)
        assert notices == (
            [] if row.reader is None else [_helper_restore_limit_notice(row.reader, *row.read)]
        ), dry_run.output
        assert ran.exit_code == 0, ran.output
        events = [json.loads(line) for line in ran.stdout.splitlines()]
        expanded = any(
            event["type"] == "stages_expanded" and event["template"] == "p/cut" for event in events
        )
        plan = helpers.completed_stages(ran.stdout)["p/plan"]["status"]
        assert (plan, expanded) == (row.plan, row.expanded), ran.output
        assert bool(notices) == (plan == "cached" and expanded), "the dry run and the run agree"


# =============================================================================
# The notice's limit: trees where it is false, kept out of the agreement table above
# =============================================================================


def _helper_other_file_changed(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/, which holds the tracked file f.txt. After the first run
    present.txt, the other file of dir, changes."""
    tracked = _helper_restore_limit_project(runner, cwd, _MAPPED_OVER_A_READ_DIRECTORY_CODE)
    (tracked.parent / "present.txt").write_text("changed")


def _helper_file_tracked_again(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/. After the first run f.txt gets new contents and is tracked
    again with the real ``pivot track --force``, then removed: its .pvt and the cache hold the new
    contents, not the ones p/plan read."""
    tracked = _helper_restore_limit_project(runner, cwd, _MAPPED_OVER_A_READ_DIRECTORY_CODE)
    tracked.write_text("g")
    retracking = runner.invoke(cli.cli, ["track", "--force", "data/raw/dir/f.txt"])
    assert retracking.exit_code == 0, retracking.output
    tracked.unlink()


def _helper_executable_file_read(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/, and the first run reads f.txt executable: after ``pivot track``
    f.txt is replaced by a file of the same contents with the executable bit. A directory's hash
    holds each file's executable bit, and a single file's .pvt records none: the restore gives
    f.txt back from the cache object, which ``pivot track`` stored read-only and not executable.
    """
    # Unlinked by the helper, not written through: it may be a link to the cache object
    tracked = _helper_restore_limit_project(
        runner, cwd, _MAPPED_OVER_A_READ_DIRECTORY_CODE, ran=False
    )
    tracked.write_text("f")
    tracked.chmod(0o755)
    _helper_first_run(runner)
    tracked.unlink()


def _helper_file_added(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/, which holds the tracked file f.txt. After the first run
    new.txt is added to dir."""
    tracked = _helper_restore_limit_project(runner, cwd, _MAPPED_OVER_A_READ_DIRECTORY_CODE)
    (tracked.parent / "new.txt").write_text("new")


def _helper_file_tracked_after_the_run(runner: CliRunner, cwd: pathlib.Path) -> None:
    """p/plan reads data/raw/dir/, and its first run reads present.txt alone. After it f.txt is
    created in dir and tracked with the real ``pivot track``, then removed: p/plan never read
    it, and its .pvt is a new file of dir."""
    helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_READ_DIRECTORY_CODE)
    helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
    held = cwd / "data" / "raw" / "dir"
    held.mkdir(parents=True)
    (held / "present.txt").write_text("present")
    _helper_first_run(runner)
    _helper_track(runner, cwd, "data/raw/dir/f.txt", "f").unlink()


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(_helper_other_file_changed, id="another-file-of-the-read-directory-changed"),
        pytest.param(_helper_file_added, id="a-file-was-added-to-the-read-directory"),
        pytest.param(_helper_file_tracked_again, id="the-file-was-tracked-again-with-new-contents"),
        pytest.param(_helper_executable_file_read, id="the-restored-file-is-not-executable"),
        pytest.param(_helper_file_tracked_after_the_run, id="the-producer-never-read-the-file"),
    ],
)
def test_repro_all_dry_run_keep_going_checkout_missing_notice_limit_restore_differs(
    runner: CliRunner, tmp_path: pathlib.Path, setup: Callable[[CliRunner, pathlib.Path], object]
) -> None:
    """A known false notice, pinned as it is today. The notice explains p/plan again with
    data/raw/dir/ at the hash p/plan's lock recorded, as if the restore of f.txt gave back
    exactly what p/plan last read. On each tree it does not: the dry run with --checkout-missing
    prints the notice, and the run with it restores f.txt, still finds dir changed and runs
    p/plan again instead of finding it cached. Hashing the read path as the restore leaves it
    would fix the notice. These trees are not rows of the agreement table, where the notice and
    the run agree."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        setup(runner, cwd)

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": _UNCHANGED, **_helper_waiting("Input dependencies changed")}
        }, dry_run.output
        assert _helper_notices(dry_run.stderr) == [_helper_restore_limit_notice("p/plan")], (
            dry_run.output
        )
        assert ran.exit_code == 0, ran.output
        assert (cwd / "data" / "raw" / "dir" / "f.txt").is_file(), "the run restores f.txt"
        assert helpers.completed_stages(ran.stdout)["p/plan"]["status"] == "ran", (
            "the run does not find p/plan cached, so the notice was false"
        )
