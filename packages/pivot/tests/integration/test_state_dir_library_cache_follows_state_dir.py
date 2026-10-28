"""A library caller's pipeline keeps its cache objects beside its locks, in the state directory
the pipeline resolved when it loaded, so a later `pivot checkout` restores what the run made.

The default cache is `<state dir>/cache`. It is derived from the state-directory record where it
is used, never stored in the process-wide merged config, so a pipeline loaded after an edit of
`core.state_dir` (or after the link the value names was pointed elsewhere) writes its cache
objects in the new directory together with its locks. Here a library program runs `a -> b`,
changes the state directory and the input, discovers the pipeline again and runs it again. A
fresh `pivot` process then deletes nothing but the workspace outputs and restores them.
"""

from __future__ import annotations

import os
import subprocess
import sys
import typing

import pytest

from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def a(value: str) -> str:
    return value.upper()


@stage
def b(value: str) -> str:
    return value + "b"


with Pipeline("test") as pipeline:
    b(a(pipeline.input("input.txt", t=str)))
"""

_LIBRARY_CALLER = """\
import os
import pathlib
import sys


def main() -> None:
    os.chdir(sys.argv[1])
    change = sys.argv[2]

    from pivot import discovery, executor

    first = executor.run(pipeline=discovery.discover_pipeline())
    assert all(r["status"] == "ran" for r in first.values()), first
    if change == "edit":
        pathlib.Path(".pivot/config.yaml").write_text("core:\\n  state_dir: otherstate\\n")
    else:
        link = pathlib.Path("link")
        link.unlink()
        pathlib.Path("s2").mkdir()
        link.symlink_to("s2")
    pathlib.Path("data/raw/input.txt").write_text("second input\\n")
    second = executor.run(pipeline=discovery.discover_pipeline())
    assert all(r["status"] == "ran" for r in second.values()), second


if __name__ == "__main__":
    main()
"""


def _cache_objects(cache_dir: pathlib.Path) -> list[str]:
    """The content objects under ``cache_dir/files``: ``<2 hex>/<rest>``, not refs or dotfiles."""
    return sorted(
        path.relative_to(cache_dir).as_posix()
        for path in (cache_dir / "files").glob("*/*")
        if path.is_file() and path.parent.name != "refs" and not path.name.startswith(".")
    )


@pytest.mark.parametrize(
    ("change", "first", "second"),
    [("edit", "libstate", "otherstate"), ("retarget", "s1", "s2")],
    ids=["edit", "retarget"],
)
def test_checkout_restores_a_library_run_made_after_the_state_dir_changed(
    tmp_path: pathlib.Path, change: str, first: str, second: str
) -> None:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    if change == "edit":
        value = first
    else:
        (root / "s1").mkdir()
        (root / "link").symlink_to("s1")
        value = "link"
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("first input\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    caller = tmp_path / "caller.py"
    caller.write_text(_LIBRARY_CALLER)

    ran = subprocess.run(
        [sys.executable, str(caller), str(root), change],
        cwd=root,
        env=helpers.cli_env(home),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr

    second_dir = root / second
    assert (second_dir / "stages" / "test" / "a.lock").is_file(), ran.stderr
    outputs = {
        path.name: path.read_text() for path in sorted((root / "data" / "test").glob("*.txt"))
    }
    assert outputs == {"a.txt": "SECOND INPUT\n", "b.txt": "SECOND INPUT\nb"}, outputs

    for name in outputs:
        (root / "data" / "test" / name).unlink()
    checkout = helpers.run_pivot(root, home, "checkout")

    assert checkout.returncode == 0, checkout.stdout + checkout.stderr
    restored = {
        name: (root / "data" / "test" / name).read_text()
        for name in outputs
        if os.path.exists(root / "data" / "test" / name)
    }
    assert restored == outputs, checkout.stdout + checkout.stderr
    # The second run's objects are beside its locks: one object for each of its two outputs.
    assert len(_cache_objects(second_dir / "cache")) == 2, (
        _cache_objects(root / first / "cache"),
        _cache_objects(second_dir / "cache"),
    )
