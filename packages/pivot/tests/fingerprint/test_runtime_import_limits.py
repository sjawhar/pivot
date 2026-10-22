# pyright: reportImplicitStringConcatenation=false

"""Stated limits of the worker's stage-import check, driven through the installed ``pivot`` CLI.

A stage whose static manifest is unchanged may reuse a result an earlier Pivot cached. A stage
whose static walk gains entries (a user class method or a user module) reruns once after an
upgrade; after that its current manifest reuses normally. `runpy` loads that never register a
module are not seen. The docs list each case with its remedy.
"""

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_REPO = pathlib.Path(__file__).resolve().parents[4]
_UNCOVERED = "imported user modules its fingerprint does not cover"
_LIB = {"helpers/__init__.py": "", "helpers/lib.py": "VALUE = 1\n", "toplib.py": "VALUE = 1\n"}


def _run(project: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING")
    }
    return subprocess.run(
        [str(_PIVOT), *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
    )


def _flat(result: subprocess.CompletedProcess[str]) -> str:
    return " ".join((result.stdout + result.stderr).split())


def _project(
    tmp_path: pathlib.Path, files: dict[str, str], lines: str = "a\nb\nc\n"
) -> pathlib.Path:
    project = tmp_path / "project"
    for rel, text in {**files, "data/raw/lines.txt": lines}.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = _run(project, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    return project


def _value(project: pathlib.Path) -> object:
    return yaml.safe_load((project / "data" / "t" / "count.yaml").read_text())["v"]


def _section(doc: str, start: str) -> str:
    text = " ".join((_REPO / "docs" / doc / "fingerprinting.md").read_text().split())
    assert start in text, f"{doc}: section {start!r} not found"
    return text[text.index(start) :]


# A result cached by an earlier Pivot ---------------------------------------------------------

_RUNTIME_IMPORT = """\
from pivot import Pipeline, stage


@stage
def count(text: str) -> dict:
    scope = {}
    exec("import helpers.lib as loaded", scope)
    return {"v": scope["loaded"].VALUE + len(text.splitlines())}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""

# The lock `pivot repro` at main 19fd78cb (release 0.5.2) wrote for `_RUNTIME_IMPORT` with
# `_LINES_0_5_2` as its input. This version's own lock for the same project is these bytes plus
# one `uncovered-import:` line.
_LINES_0_5_2 = "Local Set Default Proof Using\nSet Default Proof Using\nLemma a\n"
_LOCK_0_5_2 = """\
schema_version: 2
code_manifest:
  dep:text:loader:pivot.loaders.Text:empty: f346543fc28fb29c
  dep:text:loader:pivot.loaders.Text:load: 69b78870ffe7932d
  dep:text:loader:pivot.loaders.Text:save: f187f23051043f60
  out:t/count:loader:pivot.loaders.YAML:config: cfc61af456ae42e3
  out:t/count:loader:pivot.loaders.YAML:empty: a66d45b00a1a3a57
  out:t/count:loader:pivot.loaders.YAML:load: 4a31bd3a307ad2f0
  out:t/count:loader:pivot.loaders.YAML:save: 5240a272b83ad042
  self:count: 44514e69896aabf0
  wire:text: lines.txt
  wire:text:path: data/raw/lines.txt
params: {}
deps:
- producer: lines.txt
  key: null
  hash: 133a2387bf822f09
outs:
- key: null
  hash: 2ad953167adbd9db
  path: data/t/count.yaml
"""


def test_result_cached_by_0_5_2_is_reused_until_a_forced_run_marks_it(
    tmp_path: pathlib.Path,
) -> None:
    """A stage whose 0.5.2 lock matches its unchanged static fingerprint is served from that lock
    without running, so the stage-import check has nothing to see. A static walk that gains a
    user class method or a user module reruns once as Code changed; after that it reuses normally.
    `pivot repro --force <stage>` runs a dynamically importing stage once under this version: the
    check marks it, and the next edit to the module it imports reruns it. Both docs list the
    boundary."""
    project = _project(tmp_path, {**_LIB, "pipeline.py": _RUNTIME_IMPORT}, lines=_LINES_0_5_2)
    flagged = _run(project, "repro")
    assert flagged.returncode == 0, flagged.stdout + flagged.stderr
    assert _value(project) == 4
    lock = project / ".pivot" / "stages" / "t" / "count.lock"
    head_lock = yaml.safe_load(lock.read_text())
    fixture = yaml.safe_load(_LOCK_0_5_2)
    assert head_lock["deps"] == fixture["deps"], "input bytes differ from the 0.5.2 fixture"
    assert head_lock["outs"] == fixture["outs"], "output bytes differ from the 0.5.2 fixture"
    lock.write_text(_LOCK_0_5_2)

    upgraded = _run(project, "repro", "--json")

    assert upgraded.returncode == 0, upgraded.stdout + upgraded.stderr
    assert '"stage": "t/count", "status": "cached"' in upgraded.stdout, upgraded.stdout
    assert _UNCOVERED not in _flat(upgraded)
    forced = _run(project, "repro", "--force", "t/count")
    assert forced.returncode == 0, forced.stdout + forced.stderr
    assert _flat(forced).count(_UNCOVERED) == 1, _flat(forced)
    manifest = yaml.safe_load(lock.read_text())["code_manifest"]
    assert any(key.startswith("uncovered-import:") for key in manifest), manifest
    (project / "helpers" / "lib.py").write_text("VALUE = 100\n")
    edited = _run(project, "repro")
    assert edited.returncode == 0, edited.stdout + edited.stderr
    assert _value(project) == 103
    concepts = _section("concepts", "That check does not see every way")
    architecture = _section("architecture", "The check does not see these paths")
    for doc, text in (("concepts", concepts), ("architecture", architecture)):
        assert "earlier Pivot version" in text, f"{doc}: the upgrade case is not a stated limit"
        assert "static fingerprint is unchanged" in text, f"{doc}: unchanged-lock boundary missing"
        assert "static walk reaches a user class method or a user module" in text, (
            f"{doc}: one-time expanded-manifest rerun missing"
        )
        assert "pivot repro --force <stage>" in text, f"{doc}: the upgrade remedy is missing"


# runpy -----------------------------------------------------------------------------------------

_RUNPY = """\
import runpy

from pivot import Pipeline, stage


@stage
def count(text: str) -> dict:
    ns = {call}
    return {{"v": ns["VALUE"] + len(text.splitlines())}}


with Pipeline("t") as pipeline:
    count(pipeline.input("lines.txt", t=str))
"""


@pytest.mark.parametrize(
    ("call", "edited_file"),
    [
        pytest.param('runpy.run_path("helpers/lib.py")', "helpers/lib.py", id="run_path"),
        pytest.param('runpy.run_module("toplib")', "toplib.py", id="run_module-top-level"),
    ],
)
def test_d2_runpy_load_of_a_user_file_is_caught_or_listed_as_a_limit(
    tmp_path: pathlib.Path, call: str, edited_file: str
) -> None:
    """docs/architecture/fingerprinting.md lists `runpy` among the spellings whose user-module
    loads are "caught by the worker's check below". `runpy.run_path` and `runpy.run_module`
    (default `alter_sys=False`, no parent package) execute the file without `__import__`,
    `import_module` or a lasting `sys.modules` entry, so the head warns nothing, caches, and
    serves the stale result after an edit, as main does. Either the check catches it, or the
    architecture doc's limits table names `runpy` (and the sentence stops calling it caught)."""
    project = _project(tmp_path, {**_LIB, "pipeline.py": _RUNPY.format(call=call)})
    first = _run(project, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    (project / edited_file).write_text("VALUE = 100\n")
    edited = _run(project, "repro", "--json")
    assert edited.returncode == 0, edited.stdout + edited.stderr

    if _UNCOVERED in _flat(first):
        assert _value(project) == 103
        return
    limits = _section("architecture", "The check does not see these paths")
    assert "runpy" in limits, "architecture: `runpy` is called caught but is not caught"
