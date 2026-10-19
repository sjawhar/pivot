"""The suite runs the pivot installed beside the interpreter running pytest, whatever PATH holds."""

from __future__ import annotations

import pathlib
import re

_PACKAGES = pathlib.Path(__file__).resolve().parents[2]
# An argv list literal whose program is a bare uv, pivot or python, which the OS finds on PATH.
_PATH_PROGRAM_RE = re.compile(r"""\[\s*["'](?:uv|uvx|pivot|python3?)["']\s*,""")


def test_tests_start_programs_without_searching_path() -> None:
    """Tests start pivot as ``pathlib.Path(sys.executable).parent / "pivot"``.

    A shell whose PATH has neither uv nor the venv, such as a non-login ssh shell, then runs
    the same suite as CI.
    """
    sources = sorted(_PACKAGES.glob("*/tests/**/*.py"))
    assert sources, f"no test sources under {_PACKAGES}"
    found = list[str]()
    for path in sources:
        text = path.read_text("utf-8")
        for match in _PATH_PROGRAM_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found.append(f"{path.relative_to(_PACKAGES)}:{line}")
    assert not found, (
        "these tests find their program on PATH; use pathlib.Path(sys.executable).parent / "
        + f'"pivot" (or sys.executable for python): {found}'
    )
