"""A tracked file under a directory whose name is not valid UTF-8 works like any other.

Linux file names are bytes; os.scandir hands a name that is not UTF-8 back with surrogate
escapes. Pivot keys its hash cache and its restore locks by path, so each key is built with
``os.fsencode``, which keeps such a name, rather than ``str.encode``, which refuses it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli

if TYPE_CHECKING:
    import click.testing

_PIPELINE = """\
from pivot import Pipeline, stage


@stage
def shout(raw: str) -> str:
    return raw.upper()


with Pipeline("t") as pipeline:
    shout(pipeline.input("input.txt", t=str))
"""


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("status", "--tracked-only", "--json"), id="status"),
        pytest.param(("repro",), id="repro"),
        pytest.param(("checkout",), id="checkout"),
    ],
)
def test_a_tracked_file_under_a_non_utf8_directory_is_handled(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, args: tuple[str, ...]
) -> None:
    with isolated_pivot_dir(runner, tmp_path):
        pathlib.Path("pipeline.py").write_text(_PIPELINE)
        pathlib.Path("data", "raw").mkdir(parents=True)
        pathlib.Path("data", "raw", "input.txt").write_text("x\n")
        pathlib.Path("data.txt").write_text("hello\n")
        result = runner.invoke(cli.cli, ["track", "data.txt"])
        assert result.exit_code == 0, result.output
        bad = pathlib.Path(os.fsdecode(b"bad\xffdir"))
        bad.mkdir()
        shutil.copy2("data.txt", bad / "extra.txt")
        (bad / "extra.txt.pvt").write_text(
            pathlib.Path("data.txt.pvt").read_text().replace("path: data.txt", "path: extra.txt")
        )

        result = runner.invoke(cli.cli, list(args))

        assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
        assert (bad / "extra.txt").read_text() == "hello\n"
