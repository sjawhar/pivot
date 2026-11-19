"""The notice of `pivot repro --all --keep-going --dry-run --checkout-missing` (and `--explain`)
for a mapped stage it shows as not yet expanded although the run may expand it: the dry run
explains the producer as not current only because it, or a stage upstream of it, reads from the
disk a path that the restore changes (``cli/repro.py`` ``_log_restore_limits``,
``status.restore_limits``). Each test name holds ``dry_run_keep_going``, check c4's ``-k`` word.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from pivot import cli
from tests import helpers
from tests.conftest import isolated_pivot_dir

if TYPE_CHECKING:
    import pathlib

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


def _helper_restore_limit_notice(reader: str) -> str:
    """The notice for p/cut when ``reader``, p/plan or p/prep upstream of it, reads data/raw/dir/,
    which holds the tracked file f.txt that --checkout-missing restores."""
    who = (
        "Its producer 'p/plan'"
        if reader == "p/plan"
        else f"Stage '{reader}', upstream of its producer 'p/plan',"
    )
    return (
        f"Notice: mapped stage 'p/cut' is shown as not yet expanded. {who} reads 'data/raw/dir', "
        + "which changes when --checkout-missing restores tracked path 'data/raw/dir/f.txt'. "
        + f"The dry run explains '{reader}' from the files on disk, so the run may restore "
        + "'data/raw/dir/f.txt', find 'p/plan' current and add the instances of 'p/cut'."
    )


@pytest.mark.parametrize(
    ("code", "reader"),
    [
        pytest.param(_MAPPED_OVER_A_READ_DIRECTORY_CODE, "p/plan", id="producer"),
        pytest.param(_MAPPED_BELOW_A_READ_DIRECTORY_CODE, "p/prep", id="upstream"),
    ],
)
def test_repro_all_dry_run_keep_going_checkout_missing_notes_a_mapped_stage_it_cannot_expand(
    runner: CliRunner, tmp_path: pathlib.Path, code: str, reader: str
) -> None:
    """D7's limit, named: ``reader`` (p/plan, or p/prep upstream of it) reads data/raw/dir/,
    which holds the missing tracked file f.txt whose object the cache holds. With
    --checkout-missing the dry run explains the directory from the disk, so it shows p/cut as
    one row not yet expanded, stdout stays one JSON document, and one notice on stderr names
    p/cut, its producer, ``reader``, the directory and the file. The run on the same tree
    restores f.txt, finds p/plan current and adds p/cut's instances. Without --checkout-missing
    the dry run leaves ``reader`` out and prints no notice; once the run restored f.txt, the dry
    run with it prints none either and lists p/cut's instances."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        tracked = _helper_restore_limit_project(runner, cwd, code)

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])
        without = runner.invoke(cli.cli, helpers.DRY_RUN_KEEP_GOING)
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
        )
        restored = tracked.read_text()
        after = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])

        upstream = reader != "p/plan"
        why = "Upstream stale (p/prep)" if upstream else "Input dependencies changed"
        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": _UNCHANGED,
                **({"p/prep": _INPUTS_CHANGED} if upstream else {}),
                **_helper_waiting(why),
            }
        }, dry_run.output
        assert _helper_notices(dry_run.stderr) == [_helper_restore_limit_notice(reader)]
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output
        assert without.exit_code == 0, without.output
        assert json.loads(without.stdout) == {"stages": {"alpha/stage_a": _UNCHANGED}}
        assert helpers.skipping_lines(without.stderr) == [
            f"Skipping: Stage '{reader}' depends on directory 'dir', which holds tracked file "
            + "'data/raw/dir/f.txt' that does not exist"
        ], without.output
        assert _helper_notices(without.stderr) == [], without.output
        assert ran.exit_code == 0, ran.output
        assert restored == "f"
        events = [json.loads(line) for line in ran.stdout.splitlines()]
        instances = ["p/cut@f", "p/cut@present"]
        assert [event for event in events if event["type"] == "stages_expanded"] == [
            {"type": "stages_expanded", "template": "p/cut", "stages": instances}
        ], ran.output
        assert {
            stage: event["status"] for stage, event in helpers.completed_stages(ran.stdout).items()
        } == {
            "alpha/stage_a": "cached",
            **({"p/prep": "cached"} if upstream else {}),
            "p/plan": "cached",
            **dict.fromkeys(instances, "cached"),
        }, ran.output
        assert after.exit_code == 0, after.output
        assert json.loads(after.stdout) == {
            "stages": {
                "alpha/stage_a": _UNCHANGED,
                **({"p/prep": _UNCHANGED} if upstream else {}),
                "p/plan": _UNCHANGED,
                **dict.fromkeys(instances, _UNCHANGED),
            }
        }, after.output
        assert _helper_notices(after.stderr) == [], after.output


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


def test_repro_all_dry_run_keep_going_checkout_missing_notice_names_the_producers_own_read(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """p/plan and p/prep, upstream of it, both read data/raw/dir/, which holds the missing
    tracked file f.txt whose object the cache holds, and both run by their own explanations:
    the notice names p/plan's own read, not that of the stage upstream of it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _helper_restore_limit_project(runner, cwd, _MAPPED_OVER_AND_BELOW_A_READ_DIRECTORY_CODE)

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": _UNCHANGED,
                "p/prep": _INPUTS_CHANGED,
                **_helper_waiting("Input dependencies changed"),
            }
        }, dry_run.output
        assert _helper_notices(dry_run.stderr) == [_helper_restore_limit_notice("p/plan")]


@pytest.mark.parametrize("reads", ["the-tracked-file", "a-directory-beside-it"])
def test_repro_all_dry_run_keep_going_checkout_missing_notes_no_mapped_stage_stale_otherwise(
    runner: CliRunner, tmp_path: pathlib.Path, reads: str
) -> None:
    """No notice for a mapped stage not yet expanded because its producer is stale for a reason
    the restore does not hide, while a missing tracked file the cache holds exists. With
    `the-tracked-file`, p/plan reads the tracked file data/raw/plan.txt itself, whose hash the
    dry run takes from its .pvt, and its code changed. With `a-directory-beside-it`, p/plan reads
    data/raw/dir/, which holds no tracked file (data/raw/dir-backup/f.txt is beside it, not
    inside it), and a file was added to it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        helpers.write_pipeline(cwd, "alpha", helpers.one_stage_pipeline_code("alpha", "stage_a"))
        raw = cwd / "data" / "raw"
        if reads == "the-tracked-file":
            helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_READ_FILE_CODE)
            tracked = _helper_track(runner, cwd, "data/raw/plan.txt", "a b")
            _helper_first_run(runner)
            (cwd / "p" / "pipeline.py").write_text(
                _MAPPED_OVER_A_READ_FILE_CODE.replace("names.split()", "names.split(' ')")
            )
            reason = "Code changed"
        else:
            helpers.write_pipeline(cwd, "p", _MAPPED_OVER_A_READ_DIRECTORY_CODE)
            (raw / "dir").mkdir(parents=True)
            (raw / "dir" / "present.txt").write_text("present")
            tracked = _helper_track(runner, cwd, "data/raw/dir-backup/f.txt", "f")
            _helper_first_run(runner)
            (raw / "dir" / "new.txt").write_text("new")
            reason = "Input dependencies changed"
        tracked.unlink()

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": _UNCHANGED, **_helper_waiting(reason)}
        }, dry_run.output
        assert _helper_notices(dry_run.stderr) == [], dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output


@pytest.mark.parametrize(
    ("case", "alpha", "reason"),
    [
        pytest.param("code-change", _UNCHANGED, "Code changed", id="code-change"),
        pytest.param("params-change", _UNCHANGED, "Params changed", id="params-change"),
        pytest.param(
            "never-run",
            {"would_run": True, "reason": "No previous run"},
            "No previous run",
            id="never-run",
        ),
        pytest.param(
            "forced",
            {"would_run": True, "reason": "forced"},
            "Input dependencies changed",
            id="forced",
        ),
        pytest.param(
            "own-input",
            _UNCHANGED,
            "Input dependencies changed",
            id="producer-stale-for-its-own-input",
        ),
    ],
)
def test_repro_all_dry_run_keep_going_checkout_missing_notes_no_held_directory_stale_otherwise(
    runner: CliRunner,
    tmp_path: pathlib.Path,
    case: str,
    alpha: dict[str, object],
    reason: str,
) -> None:
    """No notice where p/plan, or p/prep upstream of it, reads data/raw/dir/, which holds the
    missing tracked file f.txt whose object the cache holds, when the run runs p/plan whatever
    the restore brings back. With `code-change`, p/plan's code changed, and with `params-change`
    params.yaml changed its params. With `never-run`, nothing ran, so p/plan has no lock. With
    `forced`, --force forces every stage. With `producer-stale-for-its-own-input`, p/prep reads
    the directory, and p/plan, which reads p/prep's output, runs by its own explanation: its
    input data/raw/extra.txt changed."""
    code = _MAPPED_OVER_A_READ_DIRECTORY_CODE
    if case == "params-change":
        code = _MAPPED_OVER_A_READ_DIRECTORY_WITH_PARAMS_CODE
    if case == "own-input":
        code = _MAPPED_BELOW_A_READ_DIRECTORY_CODE.replace(
            "def plan(names: str)", "def plan(names: str, extra: str)"
        ).replace(
            'plan(prep(pipeline.input("dir", t=pathlib.Path)))',
            'plan(prep(pipeline.input("dir", t=pathlib.Path)), pipeline.input("extra.txt", t=str))',
        )
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        extra = cwd / "data" / "raw" / "extra.txt"
        if case == "own-input":
            extra.parent.mkdir(parents=True)
            extra.write_text("one")
        _helper_restore_limit_project(runner, cwd, code, ran=case != "never-run")
        if case == "code-change":
            (cwd / "p" / "pipeline.py").write_text(
                code.replace(
                    'sorted(folder.glob("*.txt"))', 'sorted(folder.glob("*.txt"), key=str)'
                )
            )
        if case == "params-change":
            (cwd / "params.yaml").write_text("p/plan:\n  tag: two\n")
        if case == "own-input":
            extra.write_text("two")

        dry_run = runner.invoke(
            cli.cli,
            [
                *helpers.DRY_RUN_KEEP_GOING,
                "--checkout-missing",
                *(["--force"] if case == "forced" else []),
            ],
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": alpha,
                **({"p/prep": _INPUTS_CHANGED} if case == "own-input" else {}),
                **_helper_waiting(reason),
            }
        }, dry_run.output
        assert _helper_notices(dry_run.stderr) == [], dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output


def test_repro_all_dry_run_keep_going_checkout_missing_notes_no_refused_mapped_stage(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """No notice for a mapped stage that is not expanded because the run refuses it, while its
    producer is current. p/plan reads data/raw/dir/, which holds the missing tracked file f.txt
    whose object the cache holds; f.txt was tracked after the first run, and .pivotignore leaves
    its .pvt out of the directory's hash, so p/plan stays current. data/p/cut@present.txt, the
    output p/cut@present would write, is a missing tracked file that no cache holds. The dry run
    with --checkout-missing shows p/cut as one row with the run's reason for refusing it."""
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
            "Not yet expanded: cannot expand over the files of 'p/plan': "
            + "The following tracked files are missing:\n"
        ), dry_run.output
        assert "data/p/cut@present.txt" in stages["p/cut"]["reason"], dry_run.output
        assert _helper_notices(dry_run.stderr) == [], dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output


def test_repro_all_dry_run_keep_going_checkout_missing_notes_no_left_out_mapped_stage(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """No notice for a mapped stage the run leaves out: p/plan reads data/raw/dir/, which holds
    the missing tracked file f.txt whose object the cache holds, but p/cut also reads the missing
    tracked file missing.txt, which no cache holds. The dry run with --checkout-missing leaves
    p/cut out with its `Skipping:` line, shows no row for it and prints no notice."""
    code = _MAPPED_OVER_A_READ_DIRECTORY_CODE.replace(
        "def cut(entry: str) -> str:", "def cut(entry: str, shared: str) -> str:"
    ).replace(
        'pipeline.input("dir", t=pathlib.Path))))',
        'pipeline.input("dir", t=pathlib.Path))), pipeline.input("missing.txt", t=str))',
    )
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        helpers.track_without_data(cwd / "data" / "raw" / "missing.txt")
        _helper_restore_limit_project(runner, cwd, code)

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {"alpha/stage_a": _UNCHANGED, "p/plan": _INPUTS_CHANGED}
        }, dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [
            "Skipping: Stage 'p/cut' depends on input 'missing.txt' "
            + "(path data/raw/missing.txt), which does not exist"
        ], dry_run.output
        assert _helper_notices(dry_run.stderr) == [], dry_run.output


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


def test_repro_all_dry_run_keep_going_checkout_missing_reads_a_stale_committed_listing_from_the_cache(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """The notice's rule reads a tracked directory's listing with the cache, as the dry run's
    explanation does. data/raw/corpus/ is missing, and its committed corpus.pvt.listing is stale
    (an earlier version's, whose bytes no longer hash to the .pvt's record), while the cache
    holds the right listing and every object. p/plan reads corpus/a.txt and has not run, so
    p/cut is shown as not yet expanded. The dry run with --checkout-missing exits 0 with its
    document and no notice, and the run restores corpus, then runs p/plan and p/cut's
    instances."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        corpus = _helper_member_project(runner, cwd)
        committed = corpus.parent / "corpus.pvt.listing"
        stale = committed.read_bytes()
        (corpus / "a.txt").write_text("second")
        retracking = runner.invoke(cli.cli, ["track", "--force", "data/raw/corpus"])
        assert retracking.exit_code == 0, retracking.output
        assert committed.read_bytes() != stale
        committed.write_bytes(stale)
        shutil.rmtree(corpus)

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])
        ran = runner.invoke(
            cli.cli, ["repro", "--all", "--keep-going", "--checkout-missing", "--jsonl"]
        )

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": {"would_run": True, "reason": "No previous run"},
                **_helper_waiting("No previous run"),
            }
        }, dry_run.output
        assert _helper_notices(dry_run.stderr) == [], dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output
        assert ran.exit_code == 0, ran.output
        assert (corpus / "a.txt").read_text() == "second"
        assert {
            stage: event["status"] for stage, event in helpers.completed_stages(ran.stdout).items()
        } == {
            "alpha/stage_a": "ran",
            "p/plan": "ran",
            "p/cut@member": "ran",
            "p/cut@other": "ran",
        }, ran.output


def test_repro_all_dry_run_keep_going_checkout_missing_notes_no_member_read_from_the_cached_listing(
    runner: CliRunner, tmp_path: pathlib.Path
) -> None:
    """The notice's rule takes a member's hash where the dry run's explanation takes it. After a
    first run, data/raw/corpus/ is missing and its committed listing is gone, while the cache
    holds the listing and every object. p/plan reads corpus/a.txt, whose hash the explanation
    takes from that cached listing, and runs only because its other input data/raw/other.txt
    changed: the run runs it whatever the restore brings back, so no notice is printed."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        corpus = _helper_member_project(runner, cwd)
        _helper_first_run(runner)
        (corpus.parent / "corpus.pvt.listing").unlink()
        shutil.rmtree(corpus)
        (corpus.parent / "other.txt").write_text("changed")

        dry_run = runner.invoke(cli.cli, [*helpers.DRY_RUN_KEEP_GOING, "--checkout-missing"])

        assert dry_run.exit_code == 0, dry_run.output
        assert json.loads(dry_run.stdout) == {
            "stages": {
                "alpha/stage_a": _UNCHANGED,
                **_helper_waiting("Input dependencies changed"),
            }
        }, dry_run.output
        assert _helper_notices(dry_run.stderr) == [], dry_run.output
        assert helpers.skipping_lines(dry_run.stderr) == [], dry_run.output
