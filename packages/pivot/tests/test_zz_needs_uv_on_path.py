"""Throwaway negative control: a test that needs uv on PATH must fail in CI's test step."""

import subprocess


def test_runs_uv_from_path() -> None:
    subprocess.run(["uv", "--version"], check=True)
