"""Helpers shared by benchmark runner signal tests."""

from __future__ import annotations

import contextlib
import os
import signal
from typing import TYPE_CHECKING

import run

if TYPE_CHECKING:
    import pathlib

# A pivot stand-in whose `repro` records SIGINT and keeps waiting, as a pivot whose shutdown
# outlasts the grace period would. It writes its session id first.
DEAF_STAND_IN = """#!/bin/sh
case "$1" in
  init) mkdir -p .pivot ;;
  repro) trap 'echo int > {sigint_file}' INT
         sleep 600 & ps -o sid= -p $$ > {sid_file}; while :; do wait; done ;;
esac
"""


def make_stand_in(tmp_path: pathlib.Path, script: str) -> pathlib.Path:
    pivot = tmp_path / "stand-in-pivot"
    sid_file, sigint_file = tmp_path / "sid", tmp_path / "got-sigint"
    pivot.write_text(script.format(sid_file=sid_file, sigint_file=sigint_file))
    pivot.chmod(0o755)
    return pivot


def run_argv(tmp_path: pathlib.Path, pivot: pathlib.Path) -> list[str]:
    argv = ["--project", "chain-10", "--variant", f"a={pivot}", "--reps", "1"]
    return [*argv, "--work", str(tmp_path / "work"), "--out-dir", str(tmp_path / "out")]


def kill_session(sid: int) -> None:
    for pid in run.session_members(sid):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
