"""Round two of #530 (BET-331): the gate's red tests R1-R5 (PR530Gate, at 3577dbc1), as written,
and one test per behaviour a surviving mutant broke.

A stage's console line is one line (`display_reason`) only when pivot's own I/O on state it owns
failed; `--verbose` prints the traceback the record holds; the JSONL `reason` keeps main's bytes. A
cause inside a PivotError keeps main's text. Any other stage failure, out of the user's code or a
loader or writer (pivot's own included), keeps main's traceback (round four).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import pathlib
import subprocess
import sys

import click
import pytest

from pivot import exceptions, project
from pivot.cli import _run_common, doctor
from pivot.cli import errors as cli_errors
from pivot.cli import verify as verify_mod
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.types import PipelineStatus, PipelineStatusInfo

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_EACCES = os.strerror(errno.EACCES)

_BIG = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def big() -> Annotated[dict[str, str], directory]:
    return {f"f{index}.txt": f"{index}\\n" for index in range(3)}


with Pipeline("p") as pipeline:
    big()
"""


def _pivot(cwd: pathlib.Path, *args: str, env: dict[str, str] | None = None):
    base = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT, *args],
        cwd=cwd,
        env={**base, "COLUMNS": "400", **(env or {})},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _project(tmp_path: pathlib.Path, pipeline: str) -> pathlib.Path:
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    assert _pivot(root, "init").returncode == 0
    (root / "pipeline.py").write_text(pipeline)
    first = _pivot(root, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    return root


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode 000")
def test_r1_repro_names_an_unreadable_cache_in_one_line_without_a_traceback(
    tmp_path: pathlib.Path,
) -> None:
    """B1: `pivot repro --force` whose output cannot be cached (cache directory at mode 000)
    prints the OS's message and the path, not a Python traceback; --verbose adds the traceback."""
    root = _project(tmp_path, _BIG)
    files = root / ".pivot" / "cache" / "files"
    files.chmod(0)
    try:
        default = _pivot(root, "repro", "--force")
        verbose = _pivot(root, "--verbose", "repro", "--force")
    finally:
        files.chmod(0o755)

    out = default.stdout + default.stderr
    assert default.returncode == 1, out
    assert "Traceback" not in out, out
    assert "[Errno" not in out, out
    assert f"{_EACCES}: {files}" in out, out
    assert "Traceback" in verbose.stdout + verbose.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode 000")
def test_r2_repro_names_an_unopenable_artifact_lock_once_and_keeps_the_jsonl_reason(
    tmp_path: pathlib.Path,
) -> None:
    """B2: a stage whose artifact lock cannot be opened fails through the engine's worker-failure
    path. The human output names the error once, as `Permission denied: <lock>`; the JSONL
    `stage_complete.reason` stays `str(e)`."""
    root = _project(tmp_path, _BIG)
    locks = [p for p in (root / ".pivot" / "locks").iterdir() if len(p.name) == 64]
    assert locks, "the first run left an artifact lock"
    for lock in locks:
        lock.chmod(0)
    try:
        plain = _pivot(root, "repro", "--force")
        as_json = _pivot(root, "repro", "--force", "--json")
    finally:
        for lock in locks:
            lock.chmod(0o644)

    out = plain.stdout + plain.stderr
    assert plain.returncode == 1, out
    assert "[Errno" not in out, out
    assert sum(out.count(f"{_EACCES}: {lock}") for lock in locks) == 1, out
    reasons = [
        json.loads(line)["reason"]
        for line in as_json.stdout.splitlines()
        if line.startswith("{") and json.loads(line)["type"] == "stage_complete"
    ]
    assert len(reasons) == 1, as_json.stdout
    assert reasons[0].startswith("[Errno 13] Permission denied: '"), reasons
    assert any(f"'{lock}'" in reasons[0] for lock in locks), reasons


def test_r3_verify_allow_missing_keeps_the_remote_error_text_of_main(
    tmp_path: pathlib.Path,
) -> None:
    """B3: the remote-setup failure is pivot's own error; its text prints as at main, with the
    library cause kept and no `Unexpected <type>:`. Since #522 (0.5.12) main words it
    `Cannot set up S3 access for <url>: <cause>` with a tip; before #530 the head added
    `Unexpected ProfileNotFound:` to the cause, which this pins out."""
    root = _project(tmp_path, _BIG)
    assert _pivot(root, "config", "set", "remotes.origin", "s3://bucket/prefix").returncode == 0
    assert _pivot(root, "config", "set", "default_remote", "origin").returncode == 0
    files = root / ".pivot" / "cache" / "files"
    for prefix in files.iterdir():
        for obj in prefix.iterdir():
            obj.chmod(0o644)
            obj.unlink()

    proc = _pivot(
        root,
        "verify",
        "--allow-missing",
        env={
            "AWS_PROFILE": "pr530-gate-no-such-profile",
            "AWS_CONFIG_FILE": str(tmp_path / "none"),
        },
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Unexpected" not in proc.stderr, proc.stderr
    assert (
        "Error: Cannot set up S3 access for s3://bucket/prefix: "
        + "The config profile (pr530-gate-no-such-profile) could not be found"
    ) in proc.stderr.splitlines(), proc.stderr


def _one_remote_configured() -> dict[str, str]:
    """One configured remote, so `--allow-missing` goes on to create it."""
    return {"origin": "s3://b/p"}


def test_a_remote_error_keeps_the_text_of_a_wrapped_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B3, a second cause: the remote connection wrap keeps a non-pivot cause's own text, as main
    printed it (`connection refused`, not `Unexpected RuntimeError: connection refused`)."""

    def refuse(_name: object) -> object:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(remote_config, "list_remotes", _one_remote_configured)
    monkeypatch.setattr(transfer, "create_remote_from_name", refuse)

    with pytest.raises(exceptions.RemoteError) as raised:
        verify_mod._create_remote_if_needed(True)

    assert str(raised.value) == "Failed to create remote connection: connection refused"


def test_the_remote_existence_check_keeps_the_text_of_its_cause(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3, the second wrap: `verify --allow-missing` asks the remote which objects it holds. A
    failure there is pivot's own error too, so its cause keeps the library's text."""

    class _Remote:
        async def bulk_exists(self, hashes: list[str]) -> dict[str, bool]:
            raise RuntimeError("connection reset by peer")

    def one_remote(_allow_missing: bool) -> _Remote:
        return _Remote()

    def no_local_hashes(_cache_dir: pathlib.Path) -> set[str]:
        return set[str]()

    monkeypatch.setattr(verify_mod, "_create_remote_if_needed", one_remote)
    monkeypatch.setattr(transfer, "get_local_cache_hashes", no_local_hashes)
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)

    def one_missing_hash(
        stage_name: str, local_hashes: set[str], allow_missing: bool, project_root: pathlib.Path
    ) -> dict[str, list[str]]:
        return {"a000000000000001": ["data/p/big"]}

    monkeypatch.setattr(verify_mod, "_get_stage_missing_hashes", one_missing_hash)
    cached = PipelineStatusInfo(
        name="p/big", status=PipelineStatus.CACHED, reason="", upstream_stale=[]
    )

    with pytest.raises(exceptions.RemoteError) as raised:
        verify_mod._verify_stages([cached], tmp_path, True)

    assert str(raised.value) == "Failed to check remote existence: connection reset by peer"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(
            exceptions.RemoteError("Cannot resolve ref 'v2'"),
            "Cannot resolve ref 'v2'",
            id="remote",
        ),
        pytest.param(
            exceptions.RemoteNotFoundError("Unknown remote: origin"),
            "Unknown remote: origin",
            id="subclass-with-tip",
        ),
        pytest.param(exceptions.PivotError(), "PivotError", id="empty"),
    ],
)
def test_r4_describe_error_keeps_a_pivot_error_message(error: BaseException, expected: str) -> None:
    """B4 (M08): update, status --check-imports, push/pull and doctor word a PivotError with
    describe_error; its text stays the error's own message."""
    assert exceptions.describe_error(error) == expected


_TWO_STAGES = """\
from typing import Annotated

from pivot import Pipeline, directory, stage


@stage
def big() -> Annotated[dict[str, str], directory]:
    return {f"f{index}.txt": f"{index}\\n" for index in range(3)}


@stage
def bad() -> str:
    raise ValueError("bad")


with Pipeline("p") as pipeline:
    big()
    bad()
"""


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode 000")
def test_a_user_stage_keeps_its_traceback_and_pivots_own_error_is_one_line(
    tmp_path: pathlib.Path,
) -> None:
    """One run, two failures: the stage whose own code raises `ValueError("bad")` prints its
    traceback as main does, with no --verbose; the stage whose output pivot cannot cache (cache
    directory at mode 000) prints one line. So the console follows where the error was raised, not
    the command."""
    root = _project(tmp_path, _BIG)
    (root / "pipeline.py").write_text(_TWO_STAGES)
    files = root / ".pivot" / "cache" / "files"
    files.chmod(0)
    try:
        proc = _pivot(root, "repro", "--force", "--keep-going")
    finally:
        files.chmod(0o755)

    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "Traceback (most recent call last)" in out, out
    assert 'raise ValueError("bad")' in out, out
    assert "ValueError: bad" in out, out
    assert "Unexpected ValueError" not in out, out
    assert f"{_EACCES}: {files}" in out, out
    assert "PermissionError: [Errno" not in out, out


_ESCAPE = "\x1b"

_RAISES_AN_ESCAPE = """\
from pivot import Pipeline, stage


@stage
def bad() -> str:
    raise ValueError("bad \\x1b[31mred\\x1b[0m")


with Pipeline("p") as pipeline:
    bad()
"""


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode 000")
def test_a_pivot_side_one_line_escapes_a_control_character_in_the_path(
    tmp_path: pathlib.Path,
) -> None:
    """The one line a pivot-owned failure prints is terminal-safe: a control character in the
    project's own path is written as visible text, never sent to the terminal. `describe_error`
    quotes a path that would not print on one line, and #534's escaping covers the rest."""
    root = tmp_path / f"pro{_ESCAPE}ject"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    assert _pivot(root, "init").returncode == 0
    (root / "pipeline.py").write_text(_BIG)
    assert _pivot(root, "repro").returncode == 0
    (root / "pipeline.py").write_text(_BIG.replace('f"{index}\\n"', 'f"{index} again\\n"'))
    files = root / ".pivot" / "cache" / "files"
    files.chmod(0)
    try:
        proc = _pivot(root, "repro", "--force")
    finally:
        files.chmod(0o755)

    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "Traceback" not in out, out
    assert f"{_EACCES}: '{tmp_path}/pro\\x1bject/.pivot/cache/files/" in out, out
    assert _ESCAPE not in out, "a control character reached the terminal"


def test_a_user_stages_traceback_escapes_the_control_characters_it_carries(
    tmp_path: pathlib.Path,
) -> None:
    """A failure out of the user's code keeps its traceback, and that traceback is terminal-safe
    too: the escape bytes in the message print as visible text."""
    root = _project(tmp_path, _BIG)
    (root / "pipeline.py").write_text(_RAISES_AN_ESCAPE)

    proc = _pivot(root, "repro")

    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "Traceback (most recent call last)" in out, out
    assert "ValueError: bad \\x1b[31mred\\x1b[0m" in out, out
    assert _ESCAPE not in out, "a control character reached the terminal"


def test_a_remote_urls_credentials_are_hidden_in_the_one_line(tmp_path: pathlib.Path) -> None:
    """A pivot error naming a remote URL prints that URL with its user-info hidden (#534), and
    keeps the cause's own text (#530's rule for a PivotError)."""
    root = _project(tmp_path, _BIG)
    url = "s3://reader:sesame@bucket/prefix"
    assert _pivot(root, "config", "set", "remotes.origin", url).returncode == 0
    assert _pivot(root, "config", "set", "default_remote", "origin").returncode == 0
    files = root / ".pivot" / "cache" / "files"
    for prefix in files.iterdir():
        for obj in prefix.iterdir():
            obj.chmod(0o644)
            obj.unlink()

    proc = _pivot(
        root,
        "verify",
        "--allow-missing",
        env={
            "AWS_PROFILE": "pr530-no-such-profile",
            "AWS_CONFIG_FILE": str(tmp_path / "none"),
        },
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "sesame" not in proc.stdout + proc.stderr, "the password reached the terminal"
    assert (
        "Error: Cannot set up S3 access for s3://***@bucket/prefix: "
        + "The config profile (pr530-no-such-profile) could not be found"
    ) in proc.stderr.splitlines(), proc.stderr


_USER_CODE = """\
import dataclasses
import functools
import pathlib
from typing import Annotated, override

from pivot import Pipeline, directory, incremental, loaders, plot, stage


@dataclasses.dataclass(frozen=True)
class BadSave(loaders.Writer[str]):
    @override
    def save(self, data: str, path: pathlib.Path) -> None:
        raise ValueError("custom writer bug in save")


@dataclasses.dataclass(frozen=True)
class BadLoad(loaders.Loader[str]):
    @override
    def load(self, path: pathlib.Path) -> str:
        raise KeyError("custom loader bug in load")

    @override
    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data)

    @override
    def empty(self) -> str:
        return ""


@dataclasses.dataclass(frozen=True)
class BadEmpty(loaders.Loader[str]):
    @override
    def load(self, path: pathlib.Path) -> str:
        return path.read_text()

    @override
    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data)

    @override
    def empty(self) -> str:
        raise LookupError("custom loader bug in empty")


def bad_factory() -> dict[str, int]:
    raise RuntimeError("custom empty_factory bug")


def first_state_with(start: int) -> dict[str, int]:
    raise ValueError(f"bug in the user's partial-wrapped empty_factory ({start})")


def label(x: float, pos: int) -> str:
    return ["only one label"][int(x)]


class BadState:
    def __getstate__(self) -> object:
        raise ValueError("bug in the user's __getstate__")


class BadStr:
    def __str__(self) -> str:
        raise ValueError("bug in the user's __str__")


@dataclasses.dataclass(frozen=True)
class SubclassedText(loaders.Text):
    @override
    def save(self, data: str, path: pathlib.Path) -> None:
        raise ValueError("bug in a subclass of a pivot loader")


"""

_USER_CODE_CASES = [
    # The user's own loader and writer.
    pytest.param(
        """\
@stage
def custom_save() -> Annotated[str, BadSave()]:
    return "x"


with Pipeline("p") as pipeline:
    custom_save()
""",
        "save",
        'raise ValueError("custom writer bug in save")',
        "ValueError: custom writer bug in save",
        id="writer-save",
    ),
    pytest.param(
        """\
@stage
def produce() -> Annotated[str, BadLoad()]:
    return "x"


@stage
def consume(text: str) -> str:
    return text


with Pipeline("p") as pipeline:
    consume(produce())
""",
        "load",
        'raise KeyError("custom loader bug in load")',
        "KeyError: 'custom loader bug in load'",
        id="loader-load",
    ),
    pytest.param(
        """\
@stage
def produce() -> Annotated[dict[str, str], directory, BadLoad()]:
    return {"a.txt": "a"}


@stage
def consume(parts: dict[str, str]) -> str:
    return ",".join(parts)


with Pipeline("p") as pipeline:
    consume(produce())
""",
        "load",
        'raise KeyError("custom loader bug in load")',
        "KeyError: 'custom loader bug in load'",
        id="loader-load-of-a-directory-member",
    ),
    pytest.param(
        """\
@stage
def grow(previous: str) -> Annotated[str, incremental, BadEmpty()]:
    return previous + "x"


with Pipeline("p") as pipeline:
    grow(previous=pipeline.previous())
""",
        "empty",
        'raise LookupError("custom loader bug in empty")',
        "LookupError: custom loader bug in empty",
        id="loader-empty",
    ),
    pytest.param(
        """\
@stage
def grow(
    previous: dict[str, int],
) -> Annotated[dict[str, int], incremental, loaders.JSON(empty_factory=bad_factory)]:
    return {"n": len(previous)}


with Pipeline("p") as pipeline:
    grow(previous=pipeline.previous())
""",
        "bad_factory",
        'raise RuntimeError("custom empty_factory bug")',
        "RuntimeError: custom empty_factory bug",
        id="pivot-loader-calling-a-user-function",
    ),
    pytest.param(
        """\
@stage
def custom_text() -> Annotated[str, SubclassedText()]:
    return "x"


with Pipeline("p") as pipeline:
    custom_text()
""",
        "save",
        'raise ValueError("bug in a subclass of a pivot loader")',
        "ValueError: bug in a subclass of a pivot loader",
        id="user-subclass-of-a-pivot-loader",
    ),
    # The user's code that pivot's own loaders run: round three's review (R3-1), and the case
    # no rule by call site or loader can reach, a value's __str__ that pandas calls.
    pytest.param(
        """\
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@stage
def chart() -> Annotated[Figure, plot]:
    fig, ax = plt.subplots()
    ax.plot(range(5))
    ax.xaxis.set_major_formatter(label)
    return fig


with Pipeline("p") as pipeline:
    chart()
""",
        "label",
        'return ["only one label"][int(x)]',
        "IndexError: list index out of range",
        id="plot-tick-formatter-in-matplotlib-save",
    ),
    pytest.param(
        """\
@stage
def pickled() -> Annotated[BadState, loaders.Pickle()]:
    return BadState()


with Pipeline("p") as pipeline:
    pickled()
""",
        "__getstate__",
        'raise ValueError("bug in the user\'s __getstate__")',
        "ValueError: bug in the user's __getstate__",
        id="getstate-in-pickle-save",
    ),
    pytest.param(
        """\
@stage
def grow(
    previous: dict[str, int],
) -> Annotated[
    dict[str, int], incremental, loaders.JSON(empty_factory=functools.partial(first_state_with, 1))
]:
    return {"n": len(previous)}


with Pipeline("p") as pipeline:
    grow(previous=pipeline.previous())
""",
        "first_state_with",
        'raise ValueError(f"bug in the user\'s partial-wrapped empty_factory ({start})")',
        "ValueError: bug in the user's partial-wrapped empty_factory (1)",
        id="partial-wrapped-empty-factory",
    ),
    pytest.param(
        """\
import pandas as pd


@stage
def table() -> Annotated[pd.DataFrame, loaders.CSV()]:
    return pd.DataFrame({"a": [BadStr()]})


with Pipeline("p") as pipeline:
    table()
""",
        "__str__",
        'raise ValueError("bug in the user\'s __str__")',
        "ValueError: bug in the user's __str__",
        id="str-called-by-pandas-in-csv-save",
    ),
]


@pytest.mark.parametrize(("stages", "function", "raised", "last_line"), _USER_CODE_CASES)
def test_a_failure_outside_pivots_own_io_keeps_its_traceback(
    tmp_path: pathlib.Path, stages: str, function: str, raised: str, last_line: str
) -> None:
    """Only pivot's own I/O on state it owns prints one line. A failure out of the user's code
    anywhere else prints main's traceback, ending at the user's file and line, with no
    --verbose: a loader or writer the user wrote (`save`, `load` of a dependency or of each file
    of a directory dependency, `empty` on an incremental output's first run), a subclass of a
    pivot loader, and the user's code that pivot's own loaders run (a function given as an
    option, plain or wrapped in `functools.partial`, a plot's tick formatter, a value's
    `__getstate__`, a value's `__str__` that pandas calls)."""
    pipeline = _USER_CODE + stages
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    assert _pivot(root, "init").returncode == 0
    (root / "pipeline.py").write_text(pipeline)

    proc = _pivot(root, "repro")

    out = proc.stdout + proc.stderr
    line = next(number for number, text in enumerate(pipeline.splitlines(), 1) if raised in text)
    assert proc.returncode == 1, out
    assert "Traceback (most recent call last)" in out, out
    assert f'pipeline.py", line {line}, in {function}' in out, out
    assert last_line in out, out
    assert f"Unexpected {last_line.partition(':')[0]}" not in out, out


# Makes a write into the directory PIVOT_TEST_FULL_DIR fail with ENOSPC, in pivot and in every
# worker it starts, as a full disk does.
_FULL_DISK_HOOK = """\
import builtins
import errno
import os

_FULL = os.environ["PIVOT_TEST_FULL_DIR"]
_open, _os_open, _link = builtins.open, os.open, os.link


def _full(path):
    try:
        return os.path.abspath(os.fsdecode(path)).startswith(_FULL)
    except TypeError:
        return False


def _no_space(path):
    return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), os.fsdecode(path))


def _open_full(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in "wax+") and _full(file):
        raise _no_space(file)
    return _open(file, mode, *args, **kwargs)


def _os_open_full(path, flags, *args, **kwargs):
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT) and _full(path):
        raise _no_space(path)
    return _os_open(path, flags, *args, **kwargs)


def _link_full(src, dst, *args, **kwargs):
    if _full(dst):
        raise _no_space(dst)
    return _link(src, dst, *args, **kwargs)


builtins.open, os.open, os.link = _open_full, _os_open_full, _link_full
"""


def _helper_full_disk(tmp_path: pathlib.Path, full: pathlib.Path) -> dict[str, str]:
    hook = tmp_path / "hook"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(_FULL_DISK_HOOK)
    path = os.pathsep.join(filter(None, [str(hook), os.environ.get("PYTHONPATH")]))
    return {"PIVOT_TEST_FULL_DIR": str(full), "PYTHONPATH": path}


_CAUSES = [
    pytest.param("cache-000", errno.EACCES, "PermissionError", id="cache-directory-mode-000"),
    pytest.param("cache-is-a-file", errno.ENOTDIR, "NotADirectoryError", id="cache-is-a-file"),
    pytest.param("output-dir-555", errno.EACCES, "PermissionError", id="read-only-output-dir"),
    pytest.param("full-disk", errno.ENOSPC, "OSError", id="full-disk"),
]

_NEW_FILE_STAGE = """\
from pivot import Pipeline, stage


@stage
def fresh() -> str:
    return "one"


with Pipeline("p") as pipeline:
    fresh()
"""


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode 000 and 555")
@pytest.mark.parametrize(("cause", "number", "kind"), _CAUSES)
def test_a_pivot_side_os_error_is_one_console_line_and_the_jsonl_reason_is_unchanged(
    tmp_path: pathlib.Path, cause: str, number: int, kind: str
) -> None:
    """For each cause pivot's own code meets while it stores a stage's output: the console prints
    `<strerror>: <path>` under the stage and no traceback, --verbose prints the traceback, and the
    JSONL `reason` is main's: the traceback, ending in Python's own line for the error."""
    root = _project(tmp_path, _BIG)
    # New content, so each run writes a cache object and the output again rather than skip both
    (root / "pipeline.py").write_text(_BIG.replace('f"{index}\\n"', 'f"{index} again\\n"'))
    files = root / ".pivot" / "cache" / "files"
    output_parent = root / "data" / "p"
    env: dict[str, str] = {}
    if cause == "cache-000":
        named, undo = files, lambda: files.chmod(0o755)
        files.chmod(0)
    elif cause == "cache-is-a-file":
        aside = root / ".pivot" / "cache" / "aside"
        files.rename(aside)
        files.write_text("")
        named = files

        def undo() -> None:
            files.unlink()
            aside.rename(files)

    elif cause == "output-dir-555":
        # A stage whose output file does not exist yet: pivot writes it into the read-only
        # parent, where the directory output above is only rewritten member by member.
        (root / "pipeline.py").write_text(_NEW_FILE_STAGE)
        named, undo = output_parent, lambda: output_parent.chmod(0o755)
        output_parent.chmod(0o555)
    else:
        named, undo = files, lambda: None
        env = _helper_full_disk(tmp_path, files)
    try:
        default = _pivot(root, "repro", "--force", env=env)
        verbose = _pivot(root, "--verbose", "repro", "--force", env=env)
        as_json = _pivot(root, "repro", "--force", "--json", env=env)
    finally:
        undo()

    out = default.stdout + default.stderr
    assert default.returncode == 1, out
    assert "Traceback" not in out, out
    assert "[Errno" not in out, out
    assert f"{os.strerror(number)}: {named}" in out, out
    assert "Traceback (most recent call last)" in verbose.stdout + verbose.stderr
    [reason] = [
        json.loads(line)["reason"]
        for line in as_json.stdout.splitlines()
        if line.startswith("{") and json.loads(line)["type"] == "stage_complete"
    ]
    assert reason.startswith("Traceback (most recent call last):\n"), reason
    last = reason.rstrip("\n").splitlines()[-1]
    assert last.startswith(f"{kind}: [Errno {number}] {os.strerror(number)}: '{named}"), reason


def _helper_log_a_stage_failure(level: int) -> None:
    """Log a failed stage as the engine does, under the TUI's console hold, with the root logger
    at ``level``; what printed is on stderr once the TUI lets go of the terminal."""
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(level)
    logger = logging.getLogger("pivot.engine.engine")
    try:
        with _run_common.defer_console_logging():
            try:
                raise RuntimeError("worker died")
            except RuntimeError as error:
                logger.error("Stage %s failed: %s", "demo/train", exceptions.describe_error(error))
                logger.debug("Traceback of stage %s's failure", "demo/train", exc_info=True)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


def test_r5_verbose_tui_keeps_the_stage_failure_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """B5: `pivot --verbose repro --tui` shows a failed stage's traceback after the TUI exits."""
    _helper_log_a_stage_failure(logging.DEBUG)  # what --verbose sets
    after = capsys.readouterr()

    assert "Stage demo/train failed: Unexpected RuntimeError: worker died" in after.err, after.err
    assert "Traceback (most recent call last)" in after.err, after.err


def test_tui_without_verbose_prints_the_stage_failure_line_and_no_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """B5's control: without --verbose the TUI run prints the one line after it exits."""
    _helper_log_a_stage_failure(logging.INFO)
    after = capsys.readouterr()

    assert "Stage demo/train failed: Unexpected RuntimeError: worker died" in after.err, after.err
    assert "Traceback" not in after.err, after.err


def test_doctor_words_a_project_root_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """B4 (M21): `pivot doctor`'s project-root check words an OSError as its message and path."""

    def unsearchable() -> pathlib.Path:
        raise PermissionError(errno.EACCES, _EACCES, "/srv/proj")

    monkeypatch.setattr(project, "get_project_root", unsearchable)

    event, root = doctor._check_project_root()

    assert root is None
    assert event["details"] == {"error": f"{_EACCES}: /srv/proj"}


def test_doctor_words_a_remote_url_os_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4 (M22): `pivot doctor`'s check of one remote words an OSError as its message and path."""
    from pivot.remote import config as remote_config

    def unreadable(_url: str) -> None:
        raise PermissionError(errno.EACCES, _EACCES, "/etc/pivot/ca.pem")

    monkeypatch.setattr(remote_config, "list_remotes", lambda: {"origin": "s3://bucket/prefix"})
    monkeypatch.setattr(remote_config, "validate_remote_url", unreadable)

    [event] = doctor._check_remote_connectivity(tmp_path)

    assert event["details"] == {"valid": False, "error": f"{_EACCES}: /etc/pivot/ca.pem"}


def test_a_click_exception_in_a_group_keeps_its_message() -> None:
    """B4 (M27): a click exception inside an exception group prints its own message."""
    group = ExceptionGroup("errors", [click.ClickException("'x' is not a tracked file")])

    assert cli_errors.format_unhandled_error(group) == "'x' is not a tracked file"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores mode 000")
def test_status_json_reason_words_an_unreadable_cache_object(tmp_path: pathlib.Path) -> None:
    """B4 (M24): `pivot status --json` holds the explanation failure as `Error: <strerror>:
    <path>`, the same words as its warning."""
    root = _project(tmp_path, _BIG)
    output = root / "data" / "p" / "big"
    for member in output.iterdir():
        member.unlink()
    output.rmdir()
    assert _pivot(root, "checkout", "--checkout-mode", "symlink", "data/p/big").returncode == 0
    link = output / "f0.txt"
    cache_object = link.parent / os.readlink(link)
    cache_object.parent.chmod(0)
    try:
        proc = _pivot(root, "status", "--json")
    finally:
        cache_object.parent.chmod(0o755)

    reasons = {stage["name"]: stage["reason"] for stage in json.loads(proc.stdout)["stages"]}
    assert reasons["p/big"] == f"Error: {_EACCES}: {cache_object}", proc.stdout + proc.stderr
