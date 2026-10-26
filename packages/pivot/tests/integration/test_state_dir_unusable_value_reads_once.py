"""One `pivot status`, or one library fingerprint of every stage, reads the configuration a fixed
number of times, whatever the number of stages, also when `core.state_dir` cannot be used.

A state-directory record keeps the outcome of each project root's resolution, a failure
included: `''` falls back to `.pivot`, and `5` names no path. A pipeline keeps the outcome it
loaded with as well, so each of its entry points answers every cache entry from the record. The
program counts the parses of `.pivot/config.yaml` and of the global config in one cold run, for
a pipeline of 2 and of 6 independent stages.
"""

from __future__ import annotations

import json
import subprocess
import sys
import typing

import pytest

from tests import helpers

if typing.TYPE_CHECKING:
    import pathlib

_COUNTING_CALLER = """\
import json
import os
import pathlib
import sys


def main() -> None:
    os.chdir(sys.argv[1])
    mode = sys.argv[2]

    from pivot import config_files

    parse = config_files.parse_yaml
    parses = [0]

    def counted(stream, source, *, as_dict):
        if pathlib.Path(str(source)).name == "config.yaml":
            parses[0] += 1
        return parse(stream, source, as_dict=as_dict)

    config_files.parse_yaml = counted
    code = 0
    if mode == "status":
        import pivot.cli

        sys.argv = ["pivot", "status"]
        try:
            pivot.cli.main()
        except SystemExit as e:
            code = e.code or 0
    else:
        from pivot import discovery

        pipeline = discovery.discover_pipeline()
        for name in pipeline.list_stages():
            pipeline.ensure_fingerprint(name)
    print("COUNT " + json.dumps({"parses": parses[0], "code": code}))


if __name__ == "__main__":
    main()
"""


def _pipeline(stages: int) -> str:
    """``stages`` independent stages, each with a helper of its own to fingerprint."""
    lines = ["from pivot import Pipeline, stage", ""]
    for index in range(stages):
        lines += [
            "",
            f"def helper_{index}(value: str) -> str:",
            f"    return value + '{index}'",
            "",
            "",
            "@stage",
            f"def stage_{index}(value: str) -> str:",
            f"    return helper_{index}(value.upper())",
            "",
        ]
    lines += ["", 'with Pipeline("wide") as pipeline:']
    lines += [f'    stage_{index}(pipeline.input("input.txt", t=str))' for index in range(stages)]
    return "\n".join(lines) + "\n"


def _count(tmp_path: pathlib.Path, value: str, mode: str, stages: int) -> dict[str, int]:
    home = tmp_path / "home"
    root = tmp_path / f"project-{stages}"
    root.mkdir()
    assert helpers.run_cli(root, home, "git", "init", "-q").returncode == 0
    assert helpers.run_pivot(root, home, "init").returncode == 0
    (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {value}\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "pipeline.py").write_text(_pipeline(stages))
    caller = tmp_path / "caller.py"
    caller.write_text(_COUNTING_CALLER)
    ran = subprocess.run(
        [sys.executable, str(caller), str(root), mode],
        cwd=root,
        env=helpers.cli_env(home),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=300,
        check=False,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    line = next(ln for ln in ran.stdout.splitlines() if ln.startswith("COUNT "))
    return json.loads(line[len("COUNT ") :])


@pytest.mark.parametrize("mode", ["status", "fingerprint"])
@pytest.mark.parametrize("value", ["''", "5"], ids=["empty", "integer"])
def test_config_parses_do_not_grow_with_the_stage_count(
    tmp_path: pathlib.Path, value: str, mode: str
) -> None:
    few = _count(tmp_path, value, mode, stages=2)
    many = _count(tmp_path, value, mode, stages=6)

    assert few["code"] == many["code"], (few, many)
    assert many["parses"] == few["parses"], (few, many)
