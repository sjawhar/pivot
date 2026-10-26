"""An older pivot's run-cache entry restores a stage after an upgrade, whatever its
dependencies: one that reads a directory output whole, one that reads a pipeline input
directory whole, the directory's producer, a consumer of one file in it, and a file stage.

Pivot 0.5.10 (and #516) hashed a whole directory as the JSON of its inline file list; this
pivot hashes it as its listing. The entry an older pivot filed is looked up under the key that
pivot computed: unmarked, with each directory dependency hashed the older way from its members
(``worker.older_pivot_input_hash``, ``listing.legacy_digest``). From the PR #523 round-2
recheck (slice ``cip``); needs uv (``PIVOT_TEST_UV``) and PyPI, like
tests/cli/test_format_compatibility.py.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")

_PIPELINE = """import pathlib
from typing import Annotated

from pivot import Pipeline, directory, loaders, stage


@stage
def split(raw: str) -> Annotated[dict[str, str], directory, loaders.Text()]:
    words = raw.split()
    return {"x.txt": words[0], "y.txt": words[1], "sub/z.txt": raw}


@stage
def whole(parts: dict[str, str]) -> dict:
    return {"n": len(parts), "items": [parts[k] for k in sorted(parts)]}


@stage
def one(x: str) -> dict:
    return {"x": x}


@stage
def upper(seed: str) -> str:
    return seed.upper()


@stage
def scan(docs: pathlib.Path) -> dict:
    return {
        "files": sorted(
            [str(p.relative_to(docs)), p.read_text()] for p in docs.rglob("*") if p.is_file()
        )
    }


with Pipeline("r") as pipeline:
    parts = split(pipeline.input("t.txt"))
    whole(parts)
    one(parts["x.txt"])
    upper(pipeline.input("seed.txt"))
    scan(pipeline.input("docs", t=pathlib.Path))
"""

_INPUTS = {
    "v1": {"data/raw/t.txt": "a b\n", "data/raw/seed.txt": "s1\n", "data/raw/docs/a.txt": "d1\n"},
    "v2": {"data/raw/t.txt": "c d\n", "data/raw/seed.txt": "s2\n", "data/raw/docs/a.txt": "d2\n"},
}


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PIVOT_CACHE_DIR")}


def _run(cwd: pathlib.Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, env=_env(), capture_output=True, text=True, timeout=600, check=False
    )


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _released(version: str) -> list[str]:
    uv = os.environ.get("PIVOT_TEST_UV") or shutil.which("uv")
    assert uv is not None, "this test runs released pivots with `uv tool run`"
    return [uv, "tool", "run", "--quiet", "--from", f"pivot=={version}", "pivot"]


def _set_inputs(root: pathlib.Path, version: str) -> None:
    for rel, text in _INPUTS[version].items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        path.write_text(text)


@pytest.fixture(scope="module")
def old_project(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Pivot 0.5.10 runs the pipeline at inputs v1, then v2 (a run-cache entry for each), and the
    inputs are put back to v1."""
    old = _released("0.5.10")
    root = tmp_path_factory.mktemp("rc") / "src"
    root.mkdir()
    _ok(_run(root, "git", "init", "-q", "-b", "main"))
    _ok(_run(root, *old, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / ".gitignore").write_text("__pycache__/\n/data/r/\n")
    _set_inputs(root, "v1")
    _ok(_run(root, *old, "repro"))
    _set_inputs(root, "v2")
    _ok(_run(root, *old, "repro"))
    _set_inputs(root, "v1")
    return root


def _repro_reasons(root: pathlib.Path) -> dict[str, tuple[str, str]]:
    repro = _run(root, _PIVOT, "repro", "--jsonl")
    assert repro.returncode == 0, repro.stdout + repro.stderr
    events = [json.loads(line) for line in repro.stdout.splitlines() if line.startswith("{")]
    return {e["stage"]: (e["status"], e["reason"]) for e in events if e["type"] == "stage_complete"}


@pytest.mark.parametrize(
    "stage",
    [
        "r/scan",  # a pipeline input directory, read whole
        "r/whole",  # another stage's directory output, read whole
        "r/one",  # control: one file of that directory
        "r/split",  # control: the directory's producer (file input)
        "r/upper",  # control: a file-only stage
    ],
)
def test_an_old_run_cache_entry_restores_a_stage_after_an_upgrade(
    old_project: pathlib.Path, tmp_path: pathlib.Path, stage: str
) -> None:
    root = tmp_path / "w"
    shutil.copytree(old_project, root, symlinks=True)

    reasons = _repro_reasons(root)

    assert reasons[stage] == ("cached", "unchanged (run cache)"), reasons
