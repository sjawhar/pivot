"""`.pivotignore` with the installed CLI: what it keeps from making a stage stale (files in a
directory input) and what it never touches (outputs, tracked data, the cache)."""

# pyright: reportMissingImports=false, reportImplicitRelativeImport=false

from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import TYPE_CHECKING, Protocol, cast

import boto3
import pytest

from pivot import ignore
from tests import helpers

if TYPE_CHECKING:
    import pathlib
    import subprocess


class _S3Client(Protocol):
    """The one boto3 S3 call this module makes; boto3 ships no stubs."""

    def create_bucket(self, **kwargs: object) -> object: ...


_PYC_IGNORE = "*.pyc\n__pycache__/\n"

_INPUT_PIPELINE = """\
import pathlib

from pivot import Pipeline, stage


@stage
def lane(server: pathlib.Path) -> str:
    return (server / "grade.py").read_text()


with Pipeline("p") as pipeline:
    lane(pipeline.input("server", path="tasks/server", t=pathlib.Path))
"""

_OUTPUT_PIPELINE = """\
import pathlib
import tempfile

from pivot import Pipeline, StageParams, stage


class BuildParams(StageParams):
    bytecode: str = "one"


@stage
def build(raw: str, params: BuildParams) -> pathlib.Path:
    out = pathlib.Path(tempfile.mkdtemp()) / "pkg"
    (out / "__pycache__").mkdir(parents=True)
    (out / "mod.py").write_text(raw)
    (out / "__pycache__" / "mod.cpython-312.pyc").write_text(params.bytecode)
    return out


@stage
def use(pkg: pathlib.Path) -> str:
    return (pkg / "mod.py").read_text()


with Pipeline("p") as pipeline:
    use(build(pipeline.input("raw.txt"), params=BuildParams()))
"""


def _ok(result: subprocess.CompletedProcess[str]) -> str:
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def _project(tmp_path: pathlib.Path, pipeline: str) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    _ok(helpers.run_cli(root, home, "git", "init", "-q"))
    _ok(helpers.run_pivot(root, home, "init"))
    (root / ".pivotignore").write_text(_PYC_IGNORE)
    (root / "pipeline.py").write_text(pipeline)
    return home, root


def _statuses(root: pathlib.Path, home: pathlib.Path) -> dict[str, str]:
    out = _ok(helpers.run_pivot(root, home, "-q", "status", "--json"))
    return {stage["name"]: stage["status"] for stage in json.loads(out)["stages"]}


def _ran(root: pathlib.Path, home: pathlib.Path) -> dict[str, str]:
    """Run ``pivot repro`` and return each stage's completion status."""
    result = helpers.run_pivot(root, home, "repro", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    return {e["stage"]: e["status"] for e in events if e.get("type") == "stage_complete"}


def _source_dir(tmp_path: pathlib.Path, root: pathlib.Path, *, linked: bool) -> pathlib.Path:
    """``tasks/server`` holding ``grade.py``: in the project, or a symlink to a directory
    outside it. Returns the real directory."""
    (root / "tasks").mkdir()
    source = tmp_path / "elsewhere" / "server" if linked else root / "tasks" / "server"
    source.mkdir(parents=True)
    (source / "grade.py").write_text("print('grade')\n")
    if linked:
        (root / "tasks" / "server").symlink_to(source, target_is_directory=True)
    return source


@pytest.mark.parametrize("linked", [False, True], ids=["local", "symlinked-from-outside"])
def test_bytecode_written_into_an_input_directory_leaves_its_stage_cached(
    tmp_path: pathlib.Path, linked: bool
) -> None:
    """A1: Python writes __pycache__/*.pyc into a source directory a stage reads; .pivotignore
    names them, so the stage stays cached, for a local directory and a linked one alike."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=linked)
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    (source / "stray.pyc").write_bytes(b"\x00stray")

    assert _statuses(root, home) == {"p/lane": "cached"}
    check = _ok(helpers.run_pivot(root, home, "check-ignore", "tasks/server/__pycache__/x.pyc"))
    assert "tasks/server/__pycache__/x.pyc" in check, "check-ignore agrees with the hash"
    assert _ran(root, home) == {"p/lane": "cached"}


@pytest.mark.parametrize("linked", [False, True], ids=["local", "symlinked-from-outside"])
def test_edit_to_a_file_pivotignore_does_not_name_makes_the_stage_stale(
    tmp_path: pathlib.Path, linked: bool
) -> None:
    """A2: every file .pivotignore does not name still counts."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=linked)
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / "grade.py").write_text("print('graded')\n")
    (source / "helper.txt").write_text("new file\n")

    assert _statuses(root, home) == {"p/lane": "stale"}
    assert _ran(root, home) == {"p/lane": "ran"}


def test_a_directory_output_holding_bytecode_restores_it_after_checkout(
    tmp_path: pathlib.Path,
) -> None:
    """A3: outputs are cached and restored whole; .pivotignore never applies to them."""
    home, root = _project(tmp_path, _OUTPUT_PIPELINE)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "raw.txt").write_text("x = 1\n")
    assert _ran(root, home) == {"p/build": "ran", "p/use": "ran"}
    output = root / "data" / "p" / "build.dat"
    pyc = output / "__pycache__" / "mod.cpython-312.pyc"
    assert pyc.read_text() == "one"

    shutil.rmtree(output)
    _ok(helpers.run_pivot(root, home, "checkout", "p/build"))

    assert pyc.read_text() == "one", "the .pyc came back from the cache"
    assert (output / "mod.py").read_text() == "x = 1\n"
    assert _statuses(root, home) == {"p/build": "cached", "p/use": "cached"}


def test_a_tracked_directory_holding_bytecode_restores_it_after_checkout(
    tmp_path: pathlib.Path,
) -> None:
    """A3: `pivot track` records and restores every file, .pivotignore notwithstanding."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    tracked = root / "data" / "raw" / "pkg"
    (tracked / "__pycache__").mkdir(parents=True)
    (tracked / "mod.py").write_text("x = 1\n")
    (tracked / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00bytecode")
    _ok(helpers.run_pivot(root, home, "track", "data/raw/pkg"))
    members = (root / "data" / "raw" / "pkg.pvt.listing").read_text()
    assert "__pycache__/mod.cpython-312.pyc" in members, members

    shutil.rmtree(tracked)
    _ok(helpers.run_pivot(root, home, "checkout", "data/raw/pkg"))

    assert (tracked / "__pycache__" / "mod.cpython-312.pyc").read_bytes() == b"\x00bytecode"
    assert (tracked / "mod.py").read_text() == "x = 1\n"


def test_bytecode_in_an_input_directory_changes_neither_its_listing_nor_its_lock(
    tmp_path: pathlib.Path,
) -> None:
    """A directory input's committed listing holds the members its dependency hash counts, so
    it leaves out what .pivotignore names, as the hash in the lock beside it does. Writing a
    .pyc into the input changes neither file."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=False)
    assert _ran(root, home) == {"p/lane": "ran"}
    stages = root / ".pivot" / "stages"
    input_listing = stages / "=inputs" / "tasks%2Fserver.listing"
    stage_lock = stages / "p" / "lane.lock"
    before = (input_listing.read_text(), stage_lock.read_text())
    assert "grade.py" in before[0], before[0]

    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    assert _ran(root, home) == {"p/lane": "cached"}

    assert (input_listing.read_text(), stage_lock.read_text()) == before
    assert "__pycache__" not in input_listing.read_text()


def _helper_tracked_input_project(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """A project whose stage reads the tracked directory ``data/raw/server``, which holds a .pyc
    that .pivotignore names, run once. Returns its home, root and tracked directory."""
    home, root = _project(tmp_path, _INPUT_PIPELINE.replace("tasks/server", "data/raw/server"))
    tracked = root / "data" / "raw" / "server"
    (tracked / "__pycache__").mkdir(parents=True)
    (tracked / "grade.py").write_text("print('grade')\n")
    (tracked / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    _ok(helpers.run_pivot(root, home, "track", "data/raw/server"))
    assert _ran(root, home) == {"p/lane": "ran"}
    return home, root, tracked


def _helper_s3_remote(
    root: pathlib.Path, home: pathlib.Path, moto_services: dict[str, str]
) -> dict[str, str]:
    """Make an empty bucket on the moto server, configure it as the project's remote, and
    return the environment that points pivot at the server."""
    bucket = f"pivotignore-{uuid.uuid4().hex[:8]}"
    s3 = cast(
        "_S3Client",
        boto3.client(
            "s3",
            endpoint_url=moto_services["s3"],
            region_name="us-east-1",
            aws_access_key_id="x",
            aws_secret_access_key="x",
        ),
    )
    s3.create_bucket(Bucket=bucket)
    _ok(helpers.run_pivot(root, home, "config", "set", "remotes.origin", f"s3://{bucket}/cache"))
    return {
        "AWS_ENDPOINT_URL": moto_services["s3"],
        "AWS_ACCESS_KEY_ID": "x",
        "AWS_SECRET_ACCESS_KEY": "x",
        "AWS_DEFAULT_REGION": "us-east-1",
    }


def test_pull_fetches_every_file_of_a_tracked_directory_input(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """A stage's lock leaves out what .pivotignore names, but a tracked directory restores
    whole, so `pivot pull <tracked path>` fetches every file of its .pvt, the ignored ones too."""
    home, root, tracked = _helper_tracked_input_project(tmp_path)
    env = _helper_s3_remote(root, home, moto_services)
    _ok(helpers.run_pivot(root, home, "push", extra_env=env))

    shutil.rmtree(root / ".pivot" / "cache")
    shutil.rmtree(tracked)
    _ok(helpers.run_pivot(root, home, "pull", "data/raw/server", extra_env=env))

    assert (tracked / "__pycache__" / "grade.cpython-312.pyc").read_bytes() == b"\x00bytecode"
    assert (tracked / "grade.py").read_text() == "print('grade')\n"


def test_verify_allow_missing_passes_after_targeted_pushes_of_a_tracked_input(
    tmp_path: pathlib.Path, moto_services: dict[str, str]
) -> None:
    """`verify --allow-missing` asks the remote for what restores a tracked input, its .pvt's
    listing and files, which `push <path>` sends. It never asks for the listing of the lock's
    filtered hash: only the machine that ran the stage holds that, and it is derived from the
    whole listing wherever it is needed."""
    home, root, tracked = _helper_tracked_input_project(tmp_path)
    env = _helper_s3_remote(root, home, moto_services)
    _ok(helpers.run_pivot(root, home, "push", "data/raw/server", extra_env=env))
    _ok(helpers.run_pivot(root, home, "push", "p/lane", extra_env=env))

    shutil.rmtree(root / ".pivot" / "cache")
    shutil.rmtree(tracked)
    result = helpers.run_pivot(root, home, "verify", "--allow-missing", extra_env=env)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_a_changed_tracked_input_whose_listing_cannot_be_read_would_run(
    tmp_path: pathlib.Path, damage: str
) -> None:
    """`--allow-missing` takes a missing tracked input's hash from its .pvt. When the
    `.pvt.listing` is missing or corrupt the filtered hash cannot be derived, so the .pvt's
    whole hash stands, as 0.6.0 compares it: a changed input would run, never read unchanged."""
    home, root, tracked = _helper_tracked_input_project(tmp_path)
    (tracked / "grade.py").write_text("print('graded')\n")
    _ok(helpers.run_pivot(root, home, "track", "--force", "data/raw/server"))
    listing_file = root / "data" / "raw" / "server.pvt.listing"
    if damage == "missing":
        listing_file.unlink()
    else:
        listing_file.write_text("not a listing\n")
    shutil.rmtree(root / ".pivot" / "cache")
    shutil.rmtree(tracked)

    result = helpers.run_pivot(root, home, "repro", "--dry-run", "--allow-missing", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout[result.stdout.index("{") :])
    assert plan["stages"]["p/lane"]["would_run"] is True, plan
    assert "Input dependencies changed" in plan["stages"]["p/lane"]["reason"], plan


def test_a_negation_with_starstar_after_a_literal_prefix_follows_git_2_52(
    tmp_path: pathlib.Path,
) -> None:
    """In `!data/raw/src/docs**` the `**` follows the literal byte `s`, so it starts no whole
    path component and is a `*`, as git 2.52 matches it (git commit 1940a02dc1,
    "match_pathname(): give fnmatch one char of prefix context"): git keeps docsx.txt and
    leaves out docs/notes.txt like any other `*.txt`. An edit to docsx.txt makes the stage
    stale, and one to docs/notes.txt does not. Before 2.52 git took that `**` to span
    directories and kept docs/notes.txt too."""
    home, root = _project(tmp_path, _INPUT_PIPELINE.replace("tasks/server", "data/raw/src"))
    (root / ".pivotignore").write_text(
        "\n".join(ignore.get_default_patterns()) + "\n*.txt\n!data/raw/src/docs**\n"
    )
    source = root / "data" / "raw" / "src"
    (source / "docs").mkdir(parents=True)
    (source / "grade.py").write_text("print('grade')\n")
    (source / "docs" / "notes.txt").write_text("notes\n")
    (source / "docsx.txt").write_text("kept\n")
    (source / "scratch.txt").write_text("ignored\n")
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / "scratch.txt").write_text("still ignored\n")
    assert _statuses(root, home) == {"p/lane": "cached"}
    (source / "docs" / "notes.txt").write_text("edited\n")
    assert _statuses(root, home) == {"p/lane": "cached"}
    (source / "docsx.txt").write_text("edited\n")
    assert _statuses(root, home) == {"p/lane": "stale"}
    assert _ran(root, home) == {"p/lane": "ran"}


def test_a_file_a_negation_of_257_starstar_components_keeps_reruns_its_stage(
    tmp_path: pathlib.Path,
) -> None:
    """`!` and 257 "**/" before `keep.txt` match what `!**/keep.txt` matches, so git keeps
    sub/keep.txt, and an edit to it makes the stage stale. At 9c8acdce the line was over the
    limit of 256 runs of "*" and not used, so the edit left the stage cached."""
    home, root = _project(tmp_path, _INPUT_PIPELINE.replace("tasks/server", "data/raw/src"))
    (root / ".pivotignore").write_text(
        "\n".join(ignore.get_default_patterns()) + "\n*.txt\n!" + "**/" * 257 + "keep.txt\n"
    )
    source = root / "data" / "raw" / "src"
    (source / "sub").mkdir(parents=True)
    (source / "grade.py").write_text("print('grade')\n")
    (source / "sub" / "keep.txt").write_text("keep\n")
    (source / "scratch.txt").write_text("ignored\n")
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / "scratch.txt").write_text("still ignored\n")
    assert _statuses(root, home) == {"p/lane": "cached"}
    (source / "sub" / "keep.txt").write_text("edited\n")
    assert _statuses(root, home) == {"p/lane": "stale"}
    assert _ran(root, home) == {"p/lane": "ran"}


@pytest.mark.parametrize(
    ("input_path", "lines", "edited"),
    [
        pytest.param("build", "!" + "*a" * 257 + "/", "counted.txt", id="negation-input-build"),
        pytest.param(
            "build/sub", "!" + "*a" * 257 + "/", "counted.txt", id="negation-input-below-build"
        ),
        pytest.param(
            "a" * 128 + "/" + "a" * 129,
            "*a" * 128 + "/" + "*a" * 129,
            "counted.txt",
            id="pattern-git-ignores-the-input",
        ),
        pytest.param(
            "tasks/server",
            "!tasks/server/" + "*a" * 128 + "/" + "*a" * 129 + "*.txt",
            "a" * 128 + "/" + "a" * 129 + ".txt",
            id="negation-re-includes-a-member",
        ),
    ],
)
def test_an_over_limit_line_never_leaves_out_a_file_the_input_counts(
    tmp_path: pathlib.Path, input_path: str, lines: str, edited: str
) -> None:
    """A line over Pivot's limit of 256 runs of `*` errs toward counting a file, so an edit to
    a file the input counts under git's rule makes its stage stale. After `*.txt`:

    - At build, or at build/sub below it, the starter rule `build/` ignores the input. A
      directory-only negation over the limit matches every *member*, but decides neither that
      the input directory nor that a directory above it is not ignored, so the input is hashed
      whole. At e5821017 member_filter used that fallback for the input too and left
      counted.txt out; with it skipped only for the input itself, build/sub still did.
    - git ignores the 258-byte input directory a×128/a×129 by a line of 257 runs. Pivot takes
      it to match every input directory, so the input is hashed whole. Before, the line was
      dropped and `*.txt` left counted.txt out.
    - A negation of 258 runs re-includes a×128/a×129.txt inside the filtered input
      tasks/server; it matches every member, so the file counts (mutant G1, matching members
      as input directories, left it out)."""
    home, root = _project(tmp_path, _INPUT_PIPELINE.replace("tasks/server", input_path))
    (root / ".pivotignore").write_text(
        "\n".join(ignore.get_default_patterns()) + "\n*.txt\n" + lines + "\n"
    )
    source = root / input_path
    (source / edited).parent.mkdir(parents=True)
    (source / "grade.py").write_text("print('grade')\n")
    (source / edited).write_text("one\n")
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / edited).write_text("two\n")

    # The over-limit warning follows the JSON in this command's combined test output.
    status_output = _ok(helpers.run_pivot(root, home, "-q", "status", "--json"))
    status, _ = json.JSONDecoder().raw_decode(status_output[status_output.index("{") :])
    assert {stage["name"]: stage["status"] for stage in status["stages"]} == {"p/lane": "stale"}
    assert _ran(root, home) == {"p/lane": "ran"}


def test_a_pivotignore_that_is_not_utf8_fails_no_command(tmp_path: pathlib.Path) -> None:
    """git reads .gitignore as bytes, and so Pivot reads .pivotignore: an undecodable line
    matches the same bytes in a name, and status and repro run."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=False)
    with (root / ".pivotignore").open("ab") as pivotignore:
        pivotignore.write(b"\xff\xfe*.tmp\n")
    assert _ran(root, home) == {"p/lane": "ran"}

    (source / os.fsdecode(b"\xff\xfeX.tmp")).write_bytes(b"odd")

    assert _statuses(root, home) == {"p/lane": "cached"}
    assert _ran(root, home) == {"p/lane": "cached"}
    (source / "X.tmp").write_bytes(b"counted")
    assert _statuses(root, home) == {"p/lane": "stale"}


def test_a_downstream_stage_depends_on_a_directory_output_whole(tmp_path: pathlib.Path) -> None:
    """A4: a dependency on an upstream directory output is that output's whole hash. A stray
    .pyc in the output is removed by the upstream stage's restore, so nothing reruns; a .pyc
    the upstream stage itself writes differently reruns the consumer, as any output change does."""
    home, root = _project(tmp_path, _OUTPUT_PIPELINE)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "raw.txt").write_text("x = 1\n")
    assert _ran(root, home) == {"p/build": "ran", "p/use": "ran"}
    output = root / "data" / "p" / "build.dat"

    (output / "__pycache__" / "stray.cpython-312.pyc").write_bytes(b"\x00stray")
    assert _statuses(root, home) == {"p/build": "cached", "p/use": "cached"}
    assert _ran(root, home) == {"p/build": "cached", "p/use": "cached"}
    assert not (output / "__pycache__" / "stray.cpython-312.pyc").exists(), (
        "the upstream restore put its output back as recorded"
    )

    (root / "params.yaml").write_text("p/build:\n  bytecode: two\n")
    assert _ran(root, home) == {"p/build": "ran", "p/use": "ran"}


def test_an_input_hashed_before_pivotignore_applied_reruns_once(tmp_path: pathlib.Path) -> None:
    """A lock written when every file counted (pivot 0.5.20, or no .pivotignore) reruns a stage
    whose input holds an ignored file exactly once; then the stage stays cached."""
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    source = _source_dir(tmp_path, root, linked=False)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    (root / ".pivotignore").unlink()
    assert _ran(root, home) == {"p/lane": "ran"}

    (root / ".pivotignore").write_text(_PYC_IGNORE)

    assert _ran(root, home) == {"p/lane": "ran"}
    (source / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00recompiled")
    assert _ran(root, home) == {"p/lane": "cached"}


def test_an_input_with_no_ignored_file_keeps_its_hash_when_pivotignore_applies(
    tmp_path: pathlib.Path,
) -> None:
    home, root = _project(tmp_path, _INPUT_PIPELINE)
    _source_dir(tmp_path, root, linked=False)
    (root / ".pivotignore").unlink()
    assert _ran(root, home) == {"p/lane": "ran"}

    (root / ".pivotignore").write_text(_PYC_IGNORE)

    assert _ran(root, home) == {"p/lane": "cached"}


def test_a_missing_tracked_input_is_judged_by_its_pvt_without_ignored_files(
    tmp_path: pathlib.Path,
) -> None:
    """`--allow-missing` compares a missing tracked input by its .pvt, which keeps every file;
    the files .pivotignore names are taken out first, so the stage is not called stale."""
    pipeline = _INPUT_PIPELINE.replace('path="tasks/server"', 'path="data/raw/server"')
    home, root = _project(tmp_path, pipeline)
    tracked = root / "data" / "raw" / "server"
    (tracked / "__pycache__").mkdir(parents=True)
    (tracked / "grade.py").write_text("print('grade')\n")
    (tracked / "__pycache__" / "grade.cpython-312.pyc").write_bytes(b"\x00bytecode")
    _ok(helpers.run_pivot(root, home, "track", "data/raw/server"))
    assert _ran(root, home) == {"p/lane": "ran"}

    shutil.rmtree(tracked)
    result = helpers.run_pivot(root, home, "repro", "--dry-run", "--allow-missing", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout[result.stdout.index("{") :])
    assert plan["stages"] == {"p/lane": {"would_run": False, "reason": "unchanged"}}, plan
