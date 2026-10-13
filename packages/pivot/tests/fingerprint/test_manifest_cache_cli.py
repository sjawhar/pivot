"""CLI tests: the stage-manifest cache must never hide a module-level value change."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from tests.conftest import init_git_repo

_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")


def _helper_pivot(cwd: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    return subprocess.run(
        [_PIVOT_BIN, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _helper_project(root: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    init_git_repo(root)
    init = _helper_pivot(root, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    for rel_path, text in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _helper_ok(proc: subprocess.CompletedProcess[str]) -> str:
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    return output


_TUPLE_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage

COL = ("a",)


@stage
def pick(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[[COL[0]]]


with Pipeline("k") as pipeline:
    pick(pipeline.input("t.csv"))
"""


def test_module_level_tuple_read_by_stage_is_an_error(tmp_path: pathlib.Path) -> None:
    root = _helper_project(
        tmp_path, {"pipeline.py": _TUPLE_PIPELINE, "data/raw/t.csv": "a,b,c\n1,2,3\n"}
    )

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"tuple capture was accepted:\n{output}"
    assert "COL" in output, output


_MODULE_ATTR_PIPELINE = """\
import pandas as pd

import settings
from pivot import Pipeline, stage


@stage
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * settings.FACTOR


with Pipeline("ms") as pipeline:
    scale(pipeline.input("t.csv"))
"""


def test_editing_sibling_module_scalar_reruns_stage(tmp_path: pathlib.Path) -> None:
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _MODULE_ATTR_PIPELINE,
            "settings.py": "FACTOR = 10\n",
            "data/raw/t.csv": "a\n1\n2\n",
        },
    )
    _helper_ok(_helper_pivot(root, "repro"))
    assert (root / "data/ms/scale.jsonl").read_text() == '{"a":10}\n{"a":20}\n'

    # Different size, so Python's mtime+size bytecode check cannot reuse stale bytecode.
    (root / "settings.py").write_text("FACTOR = 200\n")
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert (root / "data/ms/scale.jsonl").read_text() == '{"a":200}\n{"a":400}\n', output


_FROM_IMPORT_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage
from settings import FACTOR


@stage
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * FACTOR


with Pipeline("ms") as pipeline:
    scale(pipeline.input("t.csv"))
"""


def test_editing_from_imported_scalar_reruns_stage(tmp_path: pathlib.Path) -> None:
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _FROM_IMPORT_PIPELINE,
            "settings.py": "FACTOR = 10\n",
            "data/raw/t.csv": "a\n1\n2\n",
        },
    )
    _helper_ok(_helper_pivot(root, "repro"))

    (root / "settings.py").write_text("FACTOR = 200\n")
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert (root / "data/ms/scale.jsonl").read_text() == '{"a":200}\n{"a":400}\n', output


_REBOUND_HELPER_PIPELINE = """\
import pandas as pd

import settings
from pivot import Pipeline, stage


@stage
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return settings.transform(raw)


with Pipeline("ms") as pipeline:
    scale(pipeline.input("t.csv"))
"""


def test_rebinding_module_attribute_callable_reruns_stage(tmp_path: pathlib.Path) -> None:
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _REBOUND_HELPER_PIPELINE,
            "transforms.py": "def double(df):\n    return df * 2\n\n\ndef triple(df):\n    return df * 3\n",
            "settings.py": "from transforms import double as transform\n",
            "data/raw/t.csv": "a\n1\n2\n",
        },
    )
    _helper_ok(_helper_pivot(root, "repro"))

    (root / "settings.py").write_text("from transforms import triple as transform\n")
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert (root / "data/ms/scale.jsonl").read_text() == '{"a":3}\n{"a":6}\n', output
