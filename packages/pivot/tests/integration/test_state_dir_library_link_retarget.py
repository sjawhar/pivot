"""A library caller's pipeline resolves `core.state_dir` as the kernel opens it when the
pipeline loads, also when the value names a symbolic link that was pointed elsewhere since an
earlier load.

The contract: a configured value is resolved with its symbolic links and `..` as the kernel
opens it, every state reader and writer takes its path from that result, and nothing outside a
state-directory record remembers it. Each library entry point (discovery, the pipeline's load,
an engine run) holds one record for its call, and a pipeline's run starts from the directory
the pipeline resolved, so one run keeps all of its state in one directory. Here
`core.state_dir: link`, `link -> s1`. The program runs the pipeline, points `link` at `s2`
without changing `.pivot/config.yaml`, discovers the pipeline again and runs it again. The
second run's directory is `s2`: its stages' locks, its run record and the lookup the program
makes afterwards all name `s2`, while the first run's record stays in `s1`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import typing

from pivot.storage import state
from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def first(value: str) -> str:
    return value.upper()


@stage
def second(value: str) -> str:
    return value + "-second"


with Pipeline("lib") as pipeline:
    second(first(pipeline.input("input.txt", t=str)))
"""

_LIBRARY_CALLER = """\
import json
import os
import pathlib
import sys


def main() -> None:
    os.chdir(sys.argv[1])

    from pivot import discovery, executor
    from pivot.config import io as config_io

    first = executor.run(pipeline=discovery.discover_pipeline())
    link = pathlib.Path("link")
    link.unlink()
    pathlib.Path("s2").mkdir()
    link.symlink_to("s2")
    second = executor.run(pipeline=discovery.discover_pipeline())
    print(
        "RESULT "
        + json.dumps(
            {
                "first": {name: r["status"] for name, r in first.items()},
                "second": {name: r["status"] for name, r in second.items()},
                "lookup": str(config_io.locate_state_dir()),
            }
        )
    )


if __name__ == "__main__":
    main()
"""


def _run_keys(state_dir: pathlib.Path) -> int:
    with state.StateDB(state_dir, readonly=True) as db:
        return len(list(db.iter_prefix(b"run:")))


def test_a_library_run_after_the_state_dir_link_moves_uses_the_link_s_new_target(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / "s1").mkdir()
    (root / "link").symlink_to("s1")
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: link\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("hello\n")
    (root / "pipeline.py").write_text(_PIPELINE)
    caller = tmp_path / "caller.py"
    caller.write_text(_LIBRARY_CALLER)

    ran = subprocess.run(
        [sys.executable, str(caller), str(root)],
        cwd=root,
        env=helpers.cli_env(home),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )

    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, output
    line = next(ln for ln in ran.stdout.splitlines() if ln.startswith("RESULT "))
    result = json.loads(line[len("RESULT ") :])
    s2 = os.path.realpath(root / "s2")
    assert result["lookup"] == s2, (result, output)
    # The second run starts in s2, which holds no lock yet, so both stages run there.
    assert result["second"] == {"lib/first": "ran", "lib/second": "ran"}, (result, output)
    assert (root / "s2" / "stages" / "lib" / "first.lock").is_file(), sorted(
        p.relative_to(root).as_posix() for p in (root / "s2").rglob("*")
    )
    assert (root / "s2" / "stages" / "lib" / "second.lock").is_file(), output
    assert _run_keys(root / "s1") == 1, output
    assert _run_keys(root / "s2") == 1, output
