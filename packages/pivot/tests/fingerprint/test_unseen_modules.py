# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""User modules no recorder saw compile (`loaded_source.record_unseen_modules`): one imported
before Pivot's recorder, judged against a change time that keeps whole seconds, and one a
stage's module loads with a loader it calls itself, which a worker judges as discovery does.
"""

import importlib.machinery
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid
from typing import cast

import pytest

from pivot import loaded_source

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"


# --- Finding 1: a change time stamped in whole seconds -------------------------------------


@pytest.fixture
def isolated_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty record, as a new process has (as `test_loaded_source._isolated_record`)."""
    monkeypatch.setattr(loaded_source, "_by_file", {})
    monkeypatch.setattr(loaded_source, "_by_module", {})
    monkeypatch.setattr(loaded_source, "_recorded_objects", {})
    monkeypatch.setattr(loaded_source, "_line_entries", {})
    monkeypatch.setattr(loaded_source, "_token", None)
    monkeypatch.setattr(loaded_source, "_coordinator", None)
    monkeypatch.setattr(loaded_source, "_sent_token", None)
    monkeypatch.setattr(loaded_source, "_unseen", set[str]())
    monkeypatch.setattr(loaded_source, "_warned", set[str]())


def _whole_second_ctime(result: os.stat_result) -> os.stat_result:
    """`result` as a file system that keeps change times in whole seconds reports it (ext4 with
    128-byte inodes, FAT, HFS+, an SFTP mount): the sub-second part is gone."""
    ctime_ns = result.st_ctime_ns // 1_000_000_000 * 1_000_000_000
    fields = list(result[:10])
    fields[9] = ctime_ns // 1_000_000_000
    extra = {
        "st_atime": result.st_atime,
        "st_mtime": result.st_mtime,
        "st_ctime": float(ctime_ns // 1_000_000_000),
        "st_atime_ns": result.st_atime_ns,
        "st_mtime_ns": result.st_mtime_ns,
        "st_ctime_ns": ctime_ns,
    }
    return os.stat_result(fields, extra)


def test_a_change_time_in_whole_seconds_cannot_show_an_edit_came_before_the_start(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, isolated_record: None
) -> None:
    """The module was imported before Pivot's recorder (as `sitecustomize` does) and its file was
    then edited, 0.2 s after the process started. A file system with whole-second change times
    stamps that edit with the start's second, 0.2 s before the start, so the change time cannot
    show the file is unchanged since the start: within the 2 s margin it counts as changed after
    it. The module holds v1, the file holds v2, and the record must not be the file's bytes.

    Through the CLI (ext4 made with `mkfs.ext4 -I 128`), before the 2 s margin: `sitecustomize`
    imported the in-root helper and edited it at startup; run 1 ran plain-v1 under a v2
    fingerprint with no mark, and runs 2 and 3 reported `use_plain` cached with plain-v1 while
    helpers.py held v2.
    """
    path = tmp_path / "coarse_preloaded.py"
    path.write_text('def value():\n    return "v1"\n')
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    name = f"coarse_preloaded_{uuid.uuid4().hex}"
    # A plain sys.path import: CPython's own source file loader, no Pivot recorder.
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    assert type(spec.loader) is importlib.machinery.SourceFileLoader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    path.write_text('def value():\n    return "v2"\n')
    edit_second = path.stat().st_ctime_ns // 1_000_000_000 * 1_000_000_000
    monkeypatch.setattr(loaded_source, "_process_start_ns", lambda: edit_second + 200_000_000)
    real_fstat, real_lstat = os.fstat, os.lstat
    inode = path.stat().st_ino
    monkeypatch.setattr(
        os,
        "fstat",
        lambda fd: (r if (r := real_fstat(fd)).st_ino != inode else _whole_second_ctime(r)),
    )
    monkeypatch.setattr(
        os,
        "lstat",
        lambda p, **kw: (
            r if (r := real_lstat(p, **kw)).st_ino != inode else _whole_second_ctime(r)
        ),
    )
    assert module.value() == "v1"

    loaded_source.record_unseen_modules()
    record = loaded_source.digests()[name]

    assert loaded_source._record_is_unreadable(record), record


# --- Finding 2: a module a stage's module loads with its own loader, in a worker -----------

_HELPER_V1 = 'def plain():\n    return "plain-v1"\n'

_LOADS = {
    "load_module": 'helpers2 = importlib.machinery.SourceFileLoader("helpers2", {helper!r}).load_module()\n',
    "spec_from_loader": (
        '_loader = importlib.machinery.SourceFileLoader("helpers2", {helper!r})\n'
        'helpers2 = importlib.util.module_from_spec(importlib.util.spec_from_loader("helpers2", _loader))\n'
        'sys.modules["helpers2"] = helpers2\n'
        "_loader.exec_module(helpers2)\n"
    ),
}

_EXT_STAGES = """\
import importlib.machinery
import importlib.util
import sys

from pivot import stage

{load}

def _impl() -> str:
    return helpers2.plain()


@stage
def use_plain(ready: str) -> str:
    return _impl()
"""

_PIPELINE = """\
import pathlib

import ext_stages
from pivot import Pipeline, stage

HELPER = {helper!r}


@stage
def edit_helper(seed: str) -> str:
    path = pathlib.Path(HELPER)
    path.write_text(path.read_text().replace("-v1", "-v2"))
    return "ready"


with Pipeline("p") as pipeline:
    ext_stages.use_plain(edit_helper(pipeline.input("seed.txt")))
"""


def _pivot(project: pathlib.Path, ext: pathlib.Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env["PYTHONPATH"] = str(ext)
    env["COLUMNS"] = "1000"
    done = subprocess.run(
        [str(_PIVOT), *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@pytest.mark.parametrize("load", sorted(_LOADS))
def test_helper_a_stage_module_loads_with_its_own_loader_edited_mid_run_is_not_cached_stale(
    tmp_path: pathlib.Path, load: str
) -> None:
    """`ext_stages` (on PYTHONPATH) loads `helpers2` from a file outside `sys.path` with a
    `SourceFileLoader` it builds itself, the pattern for a plugin or a file without a `.py` name.
    The coordinator records helpers2 v1 (`record_unseen_modules` at the end of discovery). The
    one worker imports `ext_stages` only for `use_plain`, after `edit_helper` edited helpers2, so
    it runs plain-v2; no recorder sees that load in the worker, and `check_stage` treats a used
    module the worker holds but never recorded as matching. The lock records v1 with no mark and
    files a run-cache entry for plain-v2. After the user restores v1, the run cache hands back
    plain-v2 as cached."""
    private = tmp_path / "private"
    private.mkdir()
    helper = private / "helpers2.py"
    helper.write_text(_HELPER_V1)
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "ext_stages.py").write_text(
        _EXT_STAGES.format(load=_LOADS[load].format(helper=str(helper)))
    )
    project = tmp_path / "project"
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "seed.txt").write_text("seed\n")
    (project / "pipeline.py").write_text(_PIPELINE.format(helper=str(helper)))
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    _pivot(project, ext, "init")
    _pivot(project, ext, "config", "set", "core.max_workers", "1")

    _pivot(project, ext, "repro")
    _pivot(project, ext, "repro")
    helper.write_text(_HELPER_V1)
    # Older than the next process start by more than the 2 s margin, as a restore made by hand.
    time.sleep(4 * loaded_source._CHANGE_TIME_SLACK_NS / 1e9)
    third = {
        event["stage"]: event["status"]
        for line in _pivot(project, ext, "repro", "--json").splitlines()
        if line.startswith("{")
        and (event := cast("dict[str, str]", json.loads(line))).get("type") == "stage_complete"
    }

    output = (project / "data" / "p" / "use_plain.txt").read_text()
    assert output == "plain-v1", (third, output)
