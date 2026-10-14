"""CLI tests: the stage-manifest cache must never hide a module-level value change."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
from tests.conftest import init_git_repo

_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")


def _helper_pivot(
    cwd: pathlib.Path, *args: str, hash_seed: int | None = None
) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = str(hash_seed)
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


_LIST_DEFAULT_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage

COLUMNS = ["a"]


@stage
def pick(raw: pd.DataFrame, cols: list[str] = COLUMNS) -> pd.DataFrame:
    return raw[cols]


with Pipeline("d") as pipeline:
    pick(pipeline.input("t.csv"))
"""


def test_non_scalar_module_value_as_stage_default_is_an_error(tmp_path: pathlib.Path) -> None:
    root = _helper_project(
        tmp_path, {"pipeline.py": _LIST_DEFAULT_PIPELINE, "data/raw/t.csv": "a,b\n1,2\n"}
    )

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"list default was accepted:\n{output}"
    assert "parameter 'cols'" in output, output
    assert "COLUMNS" in output, output


_SCALAR_DEFAULT_PIPELINES = {
    "global": (
        {
            "pipeline.py": """\
import pandas as pd

from pivot import Pipeline, stage

FACTOR = 2


@stage
def scale(raw: pd.DataFrame, factor: int = FACTOR) -> pd.DataFrame:
    return raw * factor


with Pipeline("d") as pipeline:
    scale(pipeline.input("t.csv"))
"""
        },
        ("pipeline.py", "FACTOR = 2", "FACTOR = 50"),
    ),
    "module-attribute": (
        {
            "settings.py": "FACTOR = 2\n",
            "pipeline.py": """\
import pandas as pd

import settings
from pivot import Pipeline, stage


@stage
def scale(raw: pd.DataFrame, factor: int = settings.FACTOR) -> pd.DataFrame:
    return raw * factor


with Pipeline("d") as pipeline:
    scale(pipeline.input("t.csv"))
""",
        },
        ("settings.py", "FACTOR = 2", "FACTOR = 50"),
    ),
    "helper": (
        {
            "pipeline.py": """\
import pandas as pd

from pivot import Pipeline, stage

FACTOR = 2


def multiply(raw, factor=FACTOR):
    return raw * factor


@stage
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return multiply(raw)


with Pipeline("d") as pipeline:
    scale(pipeline.input("t.csv"))
"""
        },
        ("pipeline.py", "FACTOR = 2", "FACTOR = 50"),
    ),
}


@pytest.mark.parametrize(
    ("files", "edit"), _SCALAR_DEFAULT_PIPELINES.values(), ids=_SCALAR_DEFAULT_PIPELINES.keys()
)
def test_editing_scalar_module_value_used_as_default_reruns_stage(
    tmp_path: pathlib.Path, files: dict[str, str], edit: tuple[str, str, str]
) -> None:
    root = _helper_project(tmp_path, {**files, "data/raw/t.csv": "a\n1\n2\n"})
    _helper_ok(_helper_pivot(root, "repro"))
    assert (root / "data/d/scale.jsonl").read_text() == '{"a":2}\n{"a":4}\n'

    rel_path, old, new = edit
    path = root / rel_path
    path.write_text(path.read_text().replace(old, new))
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert (root / "data/d/scale.jsonl").read_text() == '{"a":50}\n{"a":100}\n', output


def _helper_same_named_pipeline(origin: str) -> str:
    return f"""\
import pandas as pd

from pivot.compose import Pipeline, stage


@stage
def make(x: pd.DataFrame) -> dict:
    return {{"from": "{origin}"}}


@stage
def use_{origin}(m: dict) -> dict:
    return {{"saw": m["from"]}}


with Pipeline("rep") as pipeline:
    use_{origin}(make(pipeline.input("x.csv")))
"""


def test_same_stage_name_from_another_pipeline_file_is_not_a_cache_hit(
    tmp_path: pathlib.Path,
) -> None:
    """Two pipeline.py files declaring `Pipeline("rep")` must not share a cached manifest."""
    root = _helper_project(
        tmp_path,
        {
            "a/pipeline.py": _helper_same_named_pipeline("a"),
            "b/pipeline.py": _helper_same_named_pipeline("b"),
            "data/raw/x.csv": "a\n1\n",
        },
    )
    _helper_ok(_helper_pivot(root / "a", "repro"))
    assert (root / "data/rep/make.yaml").read_text() == "from: a\n"

    explain = _helper_ok(_helper_pivot(root / "b", "status", "--explain", "rep/make"))
    assert "Code changed" in explain, explain

    output = _helper_ok(_helper_pivot(root / "b", "repro"))
    assert (root / "data/rep/make.yaml").read_text() == "from: b\n", output
    assert (root / "data/rep/use_b.yaml").read_text() == "saw: b\n", output


_CACHED_HELPER_PIPELINE = """\
import functools

import pandas as pd

from pivot import Pipeline, stage

FACTOR = 2


@functools.cache
def factor() -> int:
    return FACTOR


@stage
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return raw * factor()


with Pipeline("d") as pipeline:
    scale(pipeline.input("t.csv"))
"""


@pytest.mark.parametrize("decorator", ["@functools.cache", "@functools.lru_cache(maxsize=4)"])
def test_editing_scalar_read_by_cached_helper_reruns_stage(
    tmp_path: pathlib.Path, decorator: str
) -> None:
    pipeline = _CACHED_HELPER_PIPELINE.replace("@functools.cache", decorator)
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.csv": "a\n1\n2\n"})
    _helper_ok(_helper_pivot(root, "repro"))
    assert (root / "data/d/scale.jsonl").read_text() == '{"a":2}\n{"a":4}\n'

    (root / "pipeline.py").write_text(pipeline.replace("FACTOR = 2", "FACTOR = 7"))
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert (root / "data/d/scale.jsonl").read_text() == '{"a":7}\n{"a":14}\n', output


def test_non_scalar_read_by_cached_helper_is_an_error(tmp_path: pathlib.Path) -> None:
    pipeline = _CACHED_HELPER_PIPELINE.replace("FACTOR = 2", 'COLUMNS = ["a"]').replace(
        "def factor() -> int:\n    return FACTOR", "def factor() -> int:\n    return len(COLUMNS)"
    )
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.csv": "a\n1\n2\n"})

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"list read by a cached helper was accepted:\n{output}"
    assert "captures non-scalar variable 'COLUMNS'" in output, output


_SET_PARAMS_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, StageParams, stage


class Params(StageParams):
    tags: frozenset[str] = frozenset({"alpha", "beta", "gamma", "delta", "eps"})


@stage
def tagged(raw: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    return raw.assign(n=len(params.tags))


with Pipeline("sp") as pipeline:
    tagged(pipeline.input("t.csv"))
"""


def test_set_valued_params_stay_cached_across_processes(tmp_path: pathlib.Path) -> None:
    """A frozenset field is recorded and hashed in one order, so new processes stay cached."""
    root = _helper_project(
        tmp_path, {"pipeline.py": _SET_PARAMS_PIPELINE, "data/raw/t.csv": "a\n1\n"}
    )
    _helper_ok(_helper_pivot(root, "repro", hash_seed=0))

    for seed in range(1, 4):
        _helper_ok(_helper_pivot(root, "fingerprint", "reset", hash_seed=seed))
        status = _helper_ok(_helper_pivot(root, "status", hash_seed=seed))
        assert "1 cached, 0 stale" in status, f"PYTHONHASHSEED={seed}:\n{status}"


_HANDLE_DEFAULT_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage

with Pipeline("h") as pipeline:
    RAW = pipeline.input("t.csv")
    PARTS = [pipeline.input("t.csv")]


    @stage
    def scale(raw: pd.DataFrame = RAW) -> pd.DataFrame:
        return raw * 3


    @stage
    def total(parts: list[pd.DataFrame] = PARTS) -> pd.DataFrame:
        return sum(parts[1:], parts[0])


    scale()
    total()
"""


def test_artifact_handle_default_is_wired_as_an_input(tmp_path: pathlib.Path) -> None:
    """A handle default is a real pipeline input, not module state, so it is not refused."""
    root = _helper_project(
        tmp_path, {"pipeline.py": _HANDLE_DEFAULT_PIPELINE, "data/raw/t.csv": "a\n1\n2\n"}
    )

    output = _helper_ok(_helper_pivot(root, "repro"))

    assert (root / "data/h/scale.jsonl").read_text() == '{"a":3}\n{"a":6}\n', output
    assert (root / "data/h/total.jsonl").read_text() == '{"a":1}\n{"a":2}\n', output

    (root / "data/raw/t.csv").write_text("a\n5\n")
    output = _helper_ok(_helper_pivot(root, "repro"))
    assert (root / "data/h/scale.jsonl").read_text() == '{"a":15}\n', output
