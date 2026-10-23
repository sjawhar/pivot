"""The warning and the docs state the rule for an unusable local `core.state_dir`.

The rule: every command 0.5.2 ran keeps 0.5.2's exit code and uses the default `.pivot`; every
command 0.5.2 failed still fails, in one line. Under `core.state_dir: afile` (a regular file),
`pivot track` writes state, exits 0 as on 0.5.2, and writes its cache under `.pivot`, so neither
the warning nor `docs/concepts/pipelines.md` may say that commands which use state fail.
Run from `packages/pivot/tests/cli/` of a source tree; the docs check reads that tree's docs.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_DOCS = pathlib.Path(__file__).resolve().parents[4] / "docs" / "concepts" / "pipelines.md"


def _run(cwd: pathlib.Path, home: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def test_warning_does_not_say_only_commands_without_state_use_the_default(
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "p"
    root.mkdir()
    assert _run(root, home, "git", "init", "-q").returncode == 0
    assert _run(root, home, _PIVOT, "init").returncode == 0
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "input.txt").write_text("input\n")
    (root / "afile").write_text("not a directory\n")
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: afile\n")

    result = _run(root, home, _PIVOT, "track", "data/raw/input.txt")

    assert result.returncode == 0, result.stdout + result.stderr
    assert any((root / ".pivot" / "cache").rglob("*")), "track wrote its cache elsewhere"
    assert "commands that do not need it use the default .pivot" not in result.stderr, result.stderr


def test_docs_do_not_say_state_commands_fail() -> None:
    text = " ".join(_DOCS.read_text().split())
    assert "fail with a one-line error until the file is fixed" not in text
