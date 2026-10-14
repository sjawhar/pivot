"""CLI tests: the stage-manifest cache must never hide a module-level value change."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest
from tests.conftest import init_git_repo

_PIVOT_BIN = str(pathlib.Path(sys.executable).parent / "pivot")


def _helper_pivot(
    cwd: pathlib.Path,
    *args: str,
    hash_seed: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    full_env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    if hash_seed is not None:
        full_env["PYTHONHASHSEED"] = str(hash_seed)
    full_env.update(env or {})
    return subprocess.run(
        [_PIVOT_BIN, *args],
        cwd=cwd,
        env=full_env,
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


_LOADER_CONFIG_PIPELINE = """\
import dataclasses
import json
import pathlib
from typing import Annotated

import pydantic

from pivot import Pipeline, loaders, stage


class Item(pydantic.BaseModel):
    name: str = "none"


def make_empty() -> dict:
    return {}


@dataclasses.dataclass(frozen=True)
class ModelJSON(loaders.Loader[dict, dict]):
    model: type[pydantic.BaseModel] = Item

    def load(self, path: pathlib.Path) -> dict:
        return self.model.model_validate_json(path.read_text()).model_dump()

    def save(self, data: dict, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.model.model_validate(data).model_dump()))


@stage
def make(raw: dict) -> Annotated[dict, LOADER]:
    return {"name": "x"}


with Pipeline("m") as pipeline:
    make(pipeline.input("t.json", format=loaders.JSON()))
"""


@pytest.mark.parametrize(
    "loader",
    [
        pytest.param("loaders.JSON(empty_factory=Item)", id="json-model-class"),
        pytest.param("loaders.YAML(empty_factory=Item)", id="yaml-model-class"),
        pytest.param("ModelJSON()", id="custom-loader-model-field"),
    ],
)
def test_loader_config_with_model_class_stays_cached(tmp_path: pathlib.Path, loader: str) -> None:
    """A pydantic model class in a loader's config is hashed as a class, not dumped."""
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _LOADER_CONFIG_PIPELINE.replace("LOADER", loader),
            "data/raw/t.json": "{}\n",
        },
    )

    first = _helper_ok(_helper_pivot(root, "repro"))
    assert "1 ran" in first, first

    second = _helper_ok(_helper_pivot(root, "repro"))
    assert "0 ran, 1 cached" in second, second


@pytest.mark.parametrize(
    "loader",
    [
        pytest.param("loaders.JSON(empty_factory=make_empty)", id="user-function"),
        pytest.param(
            "loaders.JSON(empty_factory=functools.partial(make_empty))", id="partial-of-function"
        ),
    ],
)
def test_loader_config_with_user_function_stays_cached_and_tracks_edits(
    tmp_path: pathlib.Path, loader: str
) -> None:
    """A function in a loader's config hashes by its name and code, not its memory address."""
    pipeline = "import functools\n" + _LOADER_CONFIG_PIPELINE.replace("LOADER", loader)
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.json": "{}\n"})
    _helper_ok(_helper_pivot(root, "repro"))

    second = _helper_ok(_helper_pivot(root, "repro"))
    assert "0 ran, 1 cached" in second, second

    (root / "pipeline.py").write_text(pipeline.replace("return {}", 'return {"n": 0}'))
    edited = _helper_ok(_helper_pivot(root, "repro"))
    assert "1 ran" in edited, edited


_OBJECT_CONFIG_PIPELINE = """\
import dataclasses

from pivot import Pipeline, loaders, stage


class Opts:
    def __init__(self, depth: int) -> None:
        self.depth = depth


@dataclasses.dataclass(frozen=True)
class OptsYAML(loaders.YAML):
    opts: object = None


@stage
def keep(raw: dict) -> dict:
    return {"n": len(raw)}


with Pipeline("o") as pipeline:
    keep(pipeline.input("t.yaml", format=OptsYAML(opts=Opts(depth=2))))
"""


def test_loader_config_holding_a_plain_object_stays_cached_and_tracks_edits(
    tmp_path: pathlib.Path,
) -> None:
    """An object with the default repr hashes by its class and attributes, not its address."""
    root = _helper_project(
        tmp_path, {"pipeline.py": _OBJECT_CONFIG_PIPELINE, "data/raw/t.yaml": "a: 1\n"}
    )
    first = _helper_ok(_helper_pivot(root, "repro"))
    assert "1 ran" in first, first

    second = _helper_ok(_helper_pivot(root, "repro"))
    assert "0 ran, 1 cached" in second, second

    (root / "pipeline.py").write_text(_OBJECT_CONFIG_PIPELINE.replace("depth=2", "depth=3"))
    edited = _helper_ok(_helper_pivot(root, "repro"))
    assert "1 ran" in edited, edited


_TRANSFORM_LOADER_PIPELINE = """\
import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Annotated

from pivot import Pipeline, loaders, stage

FACTOR = 2
COLS = ("a",)


def scale_rows(data: dict) -> dict:
    return {key: value * FACTOR for key, value in data.items()}


def make_scaler(factor: int) -> Callable[[dict], dict]:
    def scale(data: dict) -> dict:
        return {key: value * factor for key, value in data.items()}

    return scale


def keep_cols(data: dict) -> dict:
    return {key: data[key] for key in COLS}


def identity(data: dict) -> dict:
    return data


class Scaler:
    def __call__(self, data: dict) -> dict:
        return {key: value * FACTOR for key, value in data.items()}


@dataclasses.dataclass(frozen=True)
class Transformed(loaders.Loader[dict, dict]):
    transform: Callable[[dict], dict] = identity

    def load(self, path: pathlib.Path) -> dict:
        return json.loads(path.read_text())

    def save(self, data: dict, path: pathlib.Path) -> None:
        path.write_text(json.dumps(self.transform(data)))


OUT = Transformed(transform=TRANSFORM)


@stage
def make(raw: dict) -> Annotated[dict, OUT]:
    return {"a": 3}


with Pipeline("l") as pipeline:
    make(pipeline.input("t.json", format=loaders.JSON()))
"""


@pytest.mark.parametrize(
    ("transform", "edit", "before", "after"),
    [
        pytest.param("scale_rows", ("FACTOR = 2", "FACTOR = 7"), 6, 21, id="global-read"),
        pytest.param(
            "make_scaler(2)", ("make_scaler(2)", "make_scaler(5)"), 6, 15, id="closure-value"
        ),
    ],
)
def test_value_read_by_loader_config_function_reruns_stage(
    tmp_path: pathlib.Path, transform: str, edit: tuple[str, str], before: int, after: int
) -> None:
    """A function in loader config is walked like a stage helper, so what it reads is tracked."""
    pipeline = _TRANSFORM_LOADER_PIPELINE.replace("TRANSFORM", transform)
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.json": "{}\n"})
    _helper_ok(_helper_pivot(root, "repro"))
    (output,) = (root / "data/l").iterdir()
    assert json.loads(output.read_text()) == {"a": before}

    (root / "pipeline.py").write_text(pipeline.replace(*edit))
    rerun = _helper_ok(_helper_pivot(root, "repro"))

    assert json.loads(output.read_text()) == {"a": after}, rerun


_PARTIAL_ARGUMENT_PIPELINE = """\
import functools

import pandas as pd

from pivot import Pipeline, stage

FACTOR = 2
COLS = ("a",)


def factor() -> int:
    return FACTOR


def cols() -> list[str]:
    return list(COLS)


def apply(raw: pd.DataFrame, get) -> pd.DataFrame:
    return raw * get()


def select(raw: pd.DataFrame, get) -> pd.DataFrame:
    return raw[get()]


class Scaler:
    def __call__(self) -> int:
        return FACTOR


BOUND = BINDING


@stage
def scale(raw: pd.DataFrame) -> pd.DataFrame:
    return BOUND(raw)


with Pipeline("d") as pipeline:
    scale(pipeline.input("t.csv"))
"""


@pytest.mark.parametrize(
    "binding",
    [
        pytest.param("functools.partial(apply, get=factor)", id="keyword"),
        pytest.param(
            "functools.partial(apply, get=functools.partial(factor))", id="nested-partial"
        ),
    ],
)
def test_value_read_by_partial_argument_function_reruns_stage(
    tmp_path: pathlib.Path, binding: str
) -> None:
    """A function bound as a partial argument is walked like a stage helper."""
    pipeline = _PARTIAL_ARGUMENT_PIPELINE.replace("BINDING", binding)
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.csv": "a\n1\n2\n"})
    _helper_ok(_helper_pivot(root, "repro"))
    assert (root / "data/d/scale.jsonl").read_text() == '{"a":2}\n{"a":4}\n'

    (root / "pipeline.py").write_text(pipeline.replace("FACTOR = 2", "FACTOR = 7"))
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert (root / "data/d/scale.jsonl").read_text() == '{"a":7}\n{"a":14}\n', output


@pytest.mark.parametrize(
    ("pipeline", "files"),
    [
        pytest.param(
            _PARTIAL_ARGUMENT_PIPELINE.replace("BINDING", "functools.partial(select, get=cols)"),
            {"data/raw/t.csv": "a\n1\n"},
            id="partial-argument",
        ),
        pytest.param(
            _TRANSFORM_LOADER_PIPELINE.replace("TRANSFORM", "keep_cols"),
            {"data/raw/t.json": "{}\n"},
            id="loader-config",
        ),
    ],
)
def test_non_scalar_read_by_bound_function_is_an_error(
    tmp_path: pathlib.Path, pipeline: str, files: dict[str, str]
) -> None:
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, **files})

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"tuple read by a bound function was accepted:\n{output}"
    assert "captures non-scalar variable 'COLS' (type: tuple)" in output, output


@pytest.mark.parametrize(
    ("pipeline", "files", "stage", "binding"),
    [
        pytest.param(
            _TRANSFORM_LOADER_PIPELINE.replace("TRANSFORM", "Scaler()"),
            {"data/raw/t.json": "{}\n"},
            "l/make",
            "pipeline.Transformed.transform",
            id="loader-config",
        ),
        pytest.param(
            _PARTIAL_ARGUMENT_PIPELINE.replace("BINDING", "functools.partial(apply, get=Scaler())"),
            {"data/raw/t.csv": "a\n1\n"},
            "d/scale",
            "pipeline.BOUND.kwargs[get]",
            id="partial-argument",
        ),
    ],
)
def test_user_callable_instance_bound_in_config_is_an_error(
    tmp_path: pathlib.Path, pipeline: str, files: dict[str, str], stage: str, binding: str
) -> None:
    """A callable object's `__call__` reads are not walked, so it is refused like a stage read."""
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, **files})

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"callable instance was accepted:\n{output}"
    assert f"Stage '{stage}'" in output, output
    assert f"captures non-scalar variable '{binding}' (type: Scaler)" in output, output


def test_stage_and_loader_config_refusals_are_reported_together(tmp_path: pathlib.Path) -> None:
    pipeline = (
        _TRANSFORM_LOADER_PIPELINE.replace("TRANSFORM", "keep_cols")
        .replace('COLS = ("a",)', 'COLS = ("a",)\nSTAGE_KEYS = ["k"]')
        .replace('return {"a": 3}', 'return {"a": len(STAGE_KEYS)}')
    )
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.json": "{}\n"})

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "'STAGE_KEYS' (type: list)" in output, output
    assert "'COLS' (type: tuple)" in output, output


def test_editing_model_class_used_by_loader_reruns_stage(tmp_path: pathlib.Path) -> None:
    """A user class in loader config is walked, so editing its definition reruns the stage."""
    pipeline = _LOADER_CONFIG_PIPELINE.replace("LOADER", "ModelJSON(model=Item)")
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.json": "{}\n"})
    _helper_ok(_helper_pivot(root, "repro"))
    (output,) = (root / "data/m").iterdir()
    assert json.loads(output.read_text()) == {"name": "x"}

    (root / "pipeline.py").write_text(
        pipeline.replace('    name: str = "none"\n', '    name: str = "none"\n    extra: int = 1\n')
    )
    rerun = _helper_ok(_helper_pivot(root, "repro"))

    assert json.loads(output.read_text()) == {"name": "x", "extra": 1}, rerun


_LOADER_METHOD_PIPELINE = """\
import dataclasses
import json
import pathlib
from typing import Annotated

from pivot import Pipeline, loaders, stage

FACTOR = 2
KEYS = ("a",)


@dataclasses.dataclass(frozen=True)
class Scaled(BASE):
    def load(self, path: pathlib.Path) -> dict:
        return json.loads(path.read_text())

    def save(self, data: dict, path: pathlib.Path) -> None:
        path.write_text(json.dumps(SAVED))


OUT = Scaled()


@stage
def make(raw: dict) -> Annotated[dict, OUT]:
    return {"a": 3}


with Pipeline("s") as pipeline:
    make(pipeline.input("t.json", format=loaders.JSON()))
"""


@pytest.mark.parametrize(
    "base",
    [
        pytest.param("loaders.Loader[dict, dict]", id="loader-subclass"),
        pytest.param("loaders.JSON[dict]", id="builtin-loader-subclass"),
    ],
)
def test_editing_scalar_read_by_custom_loader_method_reruns_stage(
    tmp_path: pathlib.Path, base: str
) -> None:
    """A user loader's own save() is walked like a stage helper, so what it reads is tracked."""
    pipeline = _LOADER_METHOD_PIPELINE.replace("BASE", base).replace(
        "SAVED", '{"a": data["a"] * FACTOR}'
    )
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.json": "{}\n"})
    _helper_ok(_helper_pivot(root, "repro"))
    (output,) = (root / "data/s").iterdir()
    assert json.loads(output.read_text()) == {"a": 6}

    (root / "pipeline.py").write_text(pipeline.replace("FACTOR = 2", "FACTOR = 7"))
    rerun = _helper_ok(_helper_pivot(root, "repro"))

    assert json.loads(output.read_text()) == {"a": 21}, rerun


def test_non_scalar_read_by_custom_loader_method_is_an_error(tmp_path: pathlib.Path) -> None:
    pipeline = _LOADER_METHOD_PIPELINE.replace("BASE", "loaders.Loader[dict, dict]").replace(
        "SAVED", "{key: data[key] for key in KEYS}"
    )
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/t.json": "{}\n"})

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"tuple read by a loader's save() was accepted:\n{output}"
    assert "Stage 's/make'" in output, output
    assert "captures non-scalar variable 'KEYS' (type: tuple)" in output, output


_SHARED_LOADER_PIPELINE = """\
import dataclasses
import json
import pathlib

from pivot import Pipeline, loaders, stage

KEYS = ("a",)


@dataclasses.dataclass(frozen=True)
class Picked(loaders.Loader[dict, dict]):
    def load(self, path: pathlib.Path) -> dict:
        return {key: json.loads(path.read_text())[key] for key in KEYS}

    def save(self, data: dict, path: pathlib.Path) -> None:
        path.write_text(json.dumps({key: data[key] for key in KEYS}))


@stage
def combine(first: dict, second: dict) -> dict:
    return {**first, **second}


with Pipeline("k") as pipeline:
    combine(
        pipeline.input("first.json", format=Picked()),
        pipeline.input("second.json", format=Picked()),
    )
"""


def test_variable_read_by_loader_on_two_inputs_is_named_once(tmp_path: pathlib.Path) -> None:
    """One refusal per variable: load() and save() on two inputs all read KEYS."""
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _SHARED_LOADER_PIPELINE,
            "data/raw/first.json": '{"a": 1}\n',
            "data/raw/second.json": '{"a": 2}\n',
        },
    )

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert output.count("captures non-scalar variable 'KEYS'") == 1, output


_ENV_SCALE_PIPELINE = """\
import os

import pandas as pd

from pivot import Pipeline, stage

SCALE = int(os.environ.get("SCALE", "2"))


@stage
def scale(x: pd.DataFrame) -> pd.DataFrame:
    return x * SCALE


with Pipeline("env") as pipeline:
    scale(pipeline.input("x.csv"))
"""


def test_scalar_read_from_environment_at_import_reruns_stage(tmp_path: pathlib.Path) -> None:
    """A cached manifest is reused only while every value the stage reads is unchanged."""
    root = _helper_project(
        tmp_path, {"pipeline.py": _ENV_SCALE_PIPELINE, "data/raw/x.csv": "a\n1\n"}
    )
    _helper_ok(_helper_pivot(root, "repro", env={"SCALE": "2"}))
    assert (root / "data/env/scale.jsonl").read_text() == '{"a":2}\n'

    status = _helper_ok(_helper_pivot(root, "status", env={"SCALE": "7"}))
    output = _helper_ok(_helper_pivot(root, "repro", env={"SCALE": "7"}))

    assert "1 stale" in status, status
    assert (root / "data/env/scale.jsonl").read_text() == '{"a":7}\n', output


_SETTINGS_FILE_PIPELINE = """\
import json
import pathlib

import pandas as pd

from pivot import Pipeline, stage

FACTOR = int(json.loads(pathlib.Path("settings.json").read_text())["factor"])


def make_stage(factor: int):
    @stage
    def made(x: pd.DataFrame) -> pd.DataFrame:
        return x * factor

    return made


made = make_stage(FACTOR)


@stage
def plain(x: pd.DataFrame) -> pd.DataFrame:
    return x * FACTOR


with Pipeline("cfg") as pipeline:
    made(pipeline.input("x.csv"))
    plain(pipeline.input("x.csv"))
"""


def test_scalar_read_from_settings_file_at_import_reruns_stages(tmp_path: pathlib.Path) -> None:
    """A settings file is not project source: the value read from it is checked on a hit."""
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _SETTINGS_FILE_PIPELINE,
            "settings.json": '{"factor": 2}\n',
            "data/raw/x.csv": "a\n1\n",
        },
    )
    _helper_ok(_helper_pivot(root, "repro"))
    assert (root / "data/cfg/made.jsonl").read_text() == '{"a":2}\n'
    assert (root / "data/cfg/plain.jsonl").read_text() == '{"a":2}\n'

    (root / "settings.json").write_text('{"factor": 7}\n')
    status = _helper_ok(_helper_pivot(root, "status"))
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert "2 stale" in status, status
    assert (root / "data/cfg/made.jsonl").read_text() == '{"a":7}\n', output
    assert (root / "data/cfg/plain.jsonl").read_text() == '{"a":7}\n', output


_FACTORY_STAGES = """\
import pandas as pd

from pivot import stage


def make_stage(factor: int):
    @stage
    def make(x: pd.DataFrame) -> pd.DataFrame:
        return x * factor

    return make
"""


def _factory_side(factor: int) -> str:
    return (
        "from pivot import Pipeline\nfrom shared.stages import make_stage\n\n"
        + f"make = make_stage({factor})\n\n"
        + 'with Pipeline("rep") as pipeline:\n    make(pipeline.input("x.csv"))\n'
    )


_INCLUDE_BOTH = (
    "from pivot import Pipeline\n"
    + "from left.pipeline import pipeline as left\n"
    + "from right.pipeline import pipeline as right\n\n"
    + 'with Pipeline("all") as pipeline:\n'
    + "    pipeline.include(left)\n"
    + "    pipeline.include(right)\n"
)


def test_include_refuses_two_factory_closures_under_one_stage_name(tmp_path: pathlib.Path) -> None:
    """make_stage(2) and make_stage(5) share a file and qualname; their closure values differ."""
    root = _helper_project(
        tmp_path,
        {
            "shared/stages.py": _FACTORY_STAGES,
            "left/pipeline.py": _factory_side(2),
            "right/pipeline.py": _factory_side(5),
            "pipeline.py": _INCLUDE_BOTH,
            "data/raw/x.csv": "a\n1\n",
        },
    )

    proc = _helper_pivot(root, "list")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert (
        "Stage 'rep/make' is defined twice with different closure values (factor=2 vs factor=5), "
        + "both calling make_stage.<locals>.make in shared/stages.py"
    ) in output, output


_FACTORY_LOADER_STAGES = """\
import dataclasses
from collections.abc import Callable
from typing import Any

import pandas as pd

from pivot import loaders, stage


def make_scaler(factor: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def scaler(df: pd.DataFrame) -> pd.DataFrame:
        return df * factor

    return scaler


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


@dataclasses.dataclass(frozen=True)
class ScaledCSV(loaders.CSV):
    transform: Callable[[pd.DataFrame], pd.DataFrame] = _identity

    def load(self, path: Any) -> pd.DataFrame:
        return self.transform(super().load(path))


@stage
def make(x: pd.DataFrame) -> pd.DataFrame:
    return x
"""


def _factory_loader_side(factor: int) -> str:
    return (
        "from pivot import Pipeline\nfrom shared.stages import ScaledCSV, make, make_scaler\n\n"
        + 'with Pipeline("rep") as pipeline:\n'
        + f'    make(pipeline.input("x.csv", format=ScaledCSV(transform=make_scaler({factor}))))\n'
    )


def test_include_refuses_one_stage_reading_through_two_factory_loaders(
    tmp_path: pathlib.Path,
) -> None:
    """A loader field holding make_scaler(2) or make_scaler(5) reads the input two ways."""
    root = _helper_project(
        tmp_path,
        {
            "shared/stages.py": _FACTORY_LOADER_STAGES,
            "left/pipeline.py": _factory_loader_side(2),
            "right/pipeline.py": _factory_loader_side(5),
            "pipeline.py": _INCLUDE_BOTH,
            "data/raw/x.csv": "a\n1\n",
        },
    )

    proc = _helper_pivot(root, "list")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "Stage 'rep/make' is defined twice with different input formats" in output, output


def test_same_named_pipelines_run_separately_keep_factory_closures_apart(
    tmp_path: pathlib.Path,
) -> None:
    """Right's factor=5 closure never gets left's cached factor=2 manifest."""
    root = _helper_project(
        tmp_path,
        {
            "shared/stages.py": _FACTORY_STAGES,
            "left/pipeline.py": _factory_side(2),
            "right/pipeline.py": _factory_side(5),
            "data/raw/x.csv": "a\n1\n",
        },
    )

    left = _helper_ok(_helper_pivot(root / "left", "repro"))
    assert (root / "data/rep/make.jsonl").read_text() == '{"a":2}\n', left
    right = _helper_ok(_helper_pivot(root / "right", "repro"))

    assert "1 ran" in right, right
    assert (root / "data/rep/make.jsonl").read_text() == '{"a":5}\n', right


_NAN_PREP_DEFAULT = """\
import pandas as pd

from pivot import Pipeline, stage


@stage
def drop_low(raw: pd.DataFrame, ceiling: float = float("nan")) -> pd.DataFrame:
    return raw[raw["score"] >= 0.5]


with Pipeline("prep") as pipeline:
    kept = drop_low(pipeline.input("scores.csv"))
"""

_NAN_PREP_FACTORY = """\
import pandas as pd

from pivot import Pipeline, stage

THRESHOLD = float("nan")


def make_stage(threshold: float):
    @stage
    def drop_low(raw: pd.DataFrame) -> pd.DataFrame:
        return raw[raw["score"] >= 0.5] if threshold != threshold else raw

    return drop_low


drop_low = make_stage(THRESHOLD)

with Pipeline("prep") as pipeline:
    kept = drop_low(pipeline.input("scores.csv"))
"""

_NAN_ROOT = """\
from pivot import Pipeline
from prep.pipeline import pipeline as prep

with Pipeline("all") as pipeline:
    pipeline.include(prep)
"""


@pytest.mark.parametrize(
    "prep",
    [
        pytest.param(_NAN_PREP_DEFAULT, id="nan-default"),
        pytest.param(_NAN_PREP_FACTORY, id="nan-closure-cell"),
    ],
)
def test_reimported_stage_with_a_nan_value_is_one_stage(tmp_path: pathlib.Path, prep: str) -> None:
    """`pivot repro prep/pipeline.py` imports prep again; its new NaN is the same value."""
    root = _helper_project(
        tmp_path,
        {
            "prep/pipeline.py": prep,
            "pipeline.py": _NAN_ROOT,
            "data/raw/scores.csv": "name,score\na,0.2\nb,0.7\n",
        },
    )

    output = _helper_ok(_helper_pivot(root, "repro", "prep/pipeline.py"))

    assert "1 ran" in output, output


_SELF_REFERENCE_PIPELINE = """\
import pandas as pd

from pivot import Pipeline, stage


class Node:
    def __init__(self) -> None:
        self.me = self
        self.factor = 2


NODE = Node()


{definition}


with Pipeline("cyc") as pipeline:
    scale(pipeline.input("x.csv"))
"""

_SELF_REFERENCE_CELL = """\
def make_stage(node: Node):
    @stage
    def scale(x: pd.DataFrame) -> pd.DataFrame:
        return x * node.factor

    return scale


scale = make_stage(NODE)"""

_SELF_REFERENCE_DEFAULT = """\
@stage
def scale(x: pd.DataFrame, node: Node = NODE) -> pd.DataFrame:
    return x * node.factor"""


@pytest.mark.parametrize(
    ("definition", "variable"),
    [
        pytest.param(_SELF_REFERENCE_CELL, "node", id="closure-cell"),
        pytest.param(_SELF_REFERENCE_DEFAULT, "NODE", id="default"),
    ],
)
def test_self_referencing_object_is_refused_not_a_crash(
    tmp_path: pathlib.Path, definition: str, variable: str
) -> None:
    """An object that holds itself is a D2 refusal, and unsafe mode warns and runs."""
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _SELF_REFERENCE_PIPELINE.format(definition=definition),
            "data/raw/x.csv": "a\n1\n",
        },
    )

    refused = _helper_pivot(root, "repro")
    unsafe = _helper_pivot(root, "repro", env={"PIVOT_UNSAFE_FINGERPRINTING": "1"})

    refused_output = refused.stdout + refused.stderr
    assert refused.returncode != 0, refused_output
    assert "recursion" not in refused_output, refused_output
    assert f"captures non-scalar variable '{variable}'" in refused_output, refused_output
    assert "(type: Node)" in refused_output, refused_output
    unsafe_output = _helper_ok(unsafe)
    assert "1 ran" in unsafe_output, unsafe_output
    assert (root / "data/cyc/scale.jsonl").read_text() == '{"a":2}\n'


_TWO_CLOSURES_MODULE = """\
def make(factor):
    def inner(x):
        return x * factor

    return inner


h2 = make(2)
h5 = make(5)


def stage(x):
    return h2(x) + h5(x)
"""

_TWO_CLOSURES_SCRIPT = """\
import json
import sys

sys.path.insert(0, sys.argv[1])
from pivot import fingerprint
import two_closures

print(json.dumps(fingerprint.get_stage_fingerprint(two_closures.stage), sort_keys=True))
"""


def test_two_closures_from_one_factory_keep_their_values_in_every_process(
    tmp_path: pathlib.Path,
) -> None:
    """`h2 = make(2)` and `h5 = make(5)` key their cells by binding; no hash seed picks one."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / "two_closures.py").write_text(_TWO_CLOSURES_MODULE)
    script = tmp_path / "script.py"
    script.write_text(_TWO_CLOSURES_SCRIPT)
    manifests = list[dict[str, str]]()
    for seed in ("0", "1", "2", "3"):
        env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, str(script), str(tmp_path)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        manifests.append(json.loads(proc.stdout))

    assert all(manifest == manifests[0] for manifest in manifests), manifests
    assert manifests[0]["const:two_closures.h2.factor"] == "2", manifests[0]
    assert manifests[0]["const:two_closures.h5.factor"] == "5", manifests[0]


_LIBRARY_BINDING_PIPELINE = """\
import functools
import os

import numpy as np
import pandas as pd

from pivot import Pipeline, stage

{binding}


@stage
def summarise(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({{"a": [float(agg(x["a"]))]}})


with Pipeline("lib") as pipeline:
    summarise(pipeline.input("x.csv"))
"""


def _helper_summary(root: pathlib.Path) -> float:
    return json.loads((root / "data/lib/summarise.jsonl").read_text())["a"]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param("agg = np.mean", "agg = np.median", id="module-attribute"),
        pytest.param(
            "from numpy import mean as agg", "from numpy import median as agg", id="from-import"
        ),
        pytest.param(
            "agg = functools.partial(np.mean, axis=0)",
            "agg = functools.partial(np.median, axis=0)",
            id="partial-function",
        ),
    ],
)
def test_rebinding_a_library_function_reruns_stage(
    tmp_path: pathlib.Path, before: str, after: str
) -> None:
    """A module-level name bound to a library function is recorded by what it names."""
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _LIBRARY_BINDING_PIPELINE.format(binding=before),
            "data/raw/x.csv": "a\n1\n2\n6\n",
        },
    )
    _helper_ok(_helper_pivot(root, "repro"))
    assert _helper_summary(root) == 3.0

    (root / "pipeline.py").write_text(_LIBRARY_BINDING_PIPELINE.format(binding=after))
    output = _helper_ok(_helper_pivot(root, "repro"))

    assert "1 ran" in output, output
    assert _helper_summary(root) == 2.0, output


def test_library_function_chosen_at_import_reruns_stage(tmp_path: pathlib.Path) -> None:
    """A cache hit re-reads which library function the name holds."""
    binding = 'agg = getattr(np, os.environ.get("AGG", "mean"))'
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _LIBRARY_BINDING_PIPELINE.format(binding=binding),
            "data/raw/x.csv": "a\n1\n2\n6\n",
        },
    )
    _helper_ok(_helper_pivot(root, "repro", env={"AGG": "mean"}))
    assert _helper_summary(root) == 3.0

    status = _helper_ok(_helper_pivot(root, "status", env={"AGG": "median"}))
    output = _helper_ok(_helper_pivot(root, "repro", env={"AGG": "median"}))

    assert "1 stale" in status, status
    assert _helper_summary(root) == 2.0, output


_LIBRARY_LOADER_PIPELINE = """\
import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class AggCSV(loaders.CSV):
    agg: Callable[..., Any] = np.median

    def load(self, path: Any) -> pd.DataFrame:
        return pd.DataFrame({"a": [float(self.agg(super().load(path)["a"]))]})


@stage
def summarise(x: pd.DataFrame) -> pd.DataFrame:
    return x


with Pipeline("lib") as pipeline:
    summarise(pipeline.input("x.csv", format=AggCSV(agg=np.mean)))
"""


def test_library_function_in_loader_config_stays_cached(tmp_path: pathlib.Path) -> None:
    """A library function in loader config is named by module and qualname, not its address."""
    root = _helper_project(
        tmp_path, {"pipeline.py": _LIBRARY_LOADER_PIPELINE, "data/raw/x.csv": "a\n1\n2\n6\n"}
    )
    first = _helper_ok(_helper_pivot(root, "repro"))
    assert "1 ran" in first, first

    second = _helper_ok(_helper_pivot(root, "repro"))

    assert "0 ran, 1 cached" in second, second
    assert _helper_summary(root) == 3.0


_BOUND_METHOD_PIPELINE = """\
import functools

import numpy as np
import pandas as pd

from pivot import Pipeline, stage


class Scaler:
    def __init__(self, factor: int) -> None:
        self.factor = factor

    def apply(self, x: float) -> float:
        return x * self.factor


{binding}


@stage
def summarise(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({{"a": [float(method(1.0))]}})


with Pipeline("lib") as pipeline:
    summarise(pipeline.input("x.csv"))
"""


@pytest.mark.parametrize(
    ("binding", "variable"),
    [
        pytest.param("method = Scaler(3).apply", "method", id="user-instance"),
        pytest.param("method = np.random.default_rng(0).normal", "method", id="seeded-generator"),
        pytest.param(
            "method = functools.partial(Scaler(3).apply)",
            "pipeline.method.func",
            id="partial-of-user-instance-method",
        ),
        pytest.param(
            "method = functools.partial(np.random.default_rng(0).normal, scale=1.0)",
            "pipeline.method.func",
            id="partial-of-seeded-generator-method",
        ),
    ],
)
def test_method_bound_to_an_instance_is_an_error(
    tmp_path: pathlib.Path, binding: str, variable: str
) -> None:
    """`Scaler(3).apply` carries state no code hash records, so the stage is refused."""
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _BOUND_METHOD_PIPELINE.format(binding=binding),
            "data/raw/x.csv": "a\n1\n",
        },
    )

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert f"captures non-scalar variable '{variable}' (type: method)" in output, output
    assert "StageParams" in output, output


@pytest.mark.parametrize(
    "binding",
    [
        pytest.param("method = ','.join", id="method-of-a-scalar"),
        pytest.param("method = dict.fromkeys", id="classmethod-of-a-builtin"),
        pytest.param("method = np.random.normal", id="library-module-export"),
    ],
)
def test_method_bound_to_a_scalar_class_or_library_is_allowed(
    tmp_path: pathlib.Path, binding: str
) -> None:
    """A method bound to a scalar, a class or a library module's own state is not refused."""
    pipeline = _BOUND_METHOD_PIPELINE.format(binding=binding).replace(
        "float(method(1.0))", "len(str(method))"
    )
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/x.csv": "a\n1\n"})

    output = _helper_ok(_helper_pivot(root, "repro"))

    assert "1 ran" in output, output


_SECRET_DEFAULT_PIPELINE = """\
import os

import pandas as pd
import pydantic

from pivot import Pipeline, StageParams, stage


class Creds(StageParams):
    token: pydantic.SecretStr = pydantic.SecretStr("unset")


def _length(creds: Creds = Creds(token=pydantic.SecretStr(os.environ["TOKEN"]))) -> int:
    return len(creds.token.get_secret_value())


@stage
def measure(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"a": [_length()]})


with Pipeline("sec") as pipeline:
    measure(pipeline.input("x.csv"))
"""

_SECRET_LOADER_PIPELINE = """\
import dataclasses
import os
from typing import Any

import pandas as pd
import pydantic

from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class TokenCSV(loaders.CSV):
    token: pydantic.SecretStr = pydantic.SecretStr("unset")

    def load(self, path: Any) -> pd.DataFrame:
        return pd.DataFrame({"a": [len(self.token.get_secret_value())]})


@stage
def measure(x: pd.DataFrame) -> pd.DataFrame:
    return x


with Pipeline("sec") as pipeline:
    token = pydantic.SecretStr(os.environ["TOKEN"])
    measure(pipeline.input("x.csv", format=TokenCSV(token=token)))
"""

_SECRET_IN_DATACLASS_LOADER_PIPELINE = """\
import dataclasses
import os
from typing import Any

import pandas as pd
import pydantic

from pivot import Pipeline, loaders, stage


@dataclasses.dataclass(frozen=True)
class Auth:
    token: pydantic.SecretStr


@dataclasses.dataclass(frozen=True)
class TokenCSV(loaders.CSV):
    auth: Auth = Auth(pydantic.SecretStr("unset"))

    def load(self, path: Any) -> pd.DataFrame:
        return pd.DataFrame({"a": [len(self.auth.token.get_secret_value())]})


@stage
def measure(x: pd.DataFrame) -> pd.DataFrame:
    return x


with Pipeline("sec") as pipeline:
    auth = Auth(pydantic.SecretStr(os.environ["TOKEN"]))
    measure(pipeline.input("x.csv", format=TokenCSV(auth=auth)))
"""

_SECRET_IN_DATACLASS_PARTIAL_PIPELINE = """\
import dataclasses
import functools
import os

import pandas as pd
import pydantic

from pivot import Pipeline, stage


@dataclasses.dataclass(frozen=True)
class Auth:
    token: pydantic.SecretStr


def _length(auth: Auth) -> int:
    return len(auth.token.get_secret_value())


LENGTH = functools.partial(_length, Auth(pydantic.SecretStr(os.environ["TOKEN"])))


@stage
def measure(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"a": [LENGTH()]})


with Pipeline("sec") as pipeline:
    measure(pipeline.input("x.csv"))
"""


@pytest.mark.parametrize(
    "pipeline",
    [
        pytest.param(_SECRET_DEFAULT_PIPELINE, id="stage-params-default"),
        pytest.param(_SECRET_LOADER_PIPELINE, id="loader-field"),
        pytest.param(_SECRET_IN_DATACLASS_LOADER_PIPELINE, id="dataclass-in-loader-field"),
        pytest.param(_SECRET_IN_DATACLASS_PARTIAL_PIPELINE, id="dataclass-in-partial-argument"),
    ],
)
def test_changing_a_secret_read_by_a_stage_reruns_it(tmp_path: pathlib.Path, pipeline: str) -> None:
    """A secret in a helper default or loader config is fingerprinted by its keyed digest."""
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/x.csv": "a\n1\n"})
    env = {"TOKEN": "first-secret"}
    _helper_ok(_helper_pivot(root, "repro", env=env))
    assert (root / "data/sec/measure.jsonl").read_text() == '{"a":12}\n'

    status = _helper_ok(_helper_pivot(root, "status", env={"TOKEN": "second-secret-x"}))
    output = _helper_ok(_helper_pivot(root, "repro", env={"TOKEN": "second-secret-x"}))

    assert "1 stale" in status, status
    assert (root / "data/sec/measure.jsonl").read_text() == '{"a":15}\n', output
    lock = (root / ".pivot/stages/sec/measure.lock").read_text()
    assert "first-secret" not in lock and "second-secret-x" not in lock, lock
    key_file = root / ".pivot/secret.key"
    assert key_file.stat().st_mode & 0o777 == 0o600
    ignored = subprocess.run(
        ["git", "check-ignore", ".pivot/secret.key"], cwd=root, capture_output=True, text=True
    )
    assert ignored.returncode == 0, "secret.key must be gitignored"


_CLASS_ATTRIBUTE_PIPELINE = """\
import enum
import os

import pandas as pd
import pydantic

from pivot import Pipeline, stage


class Config:
    SCALE = int(os.environ.get("SCALE", "2"))
    COLUMNS = ("a",)


class Mode(enum.Enum):
    DOUBLE = int(os.environ.get("MODE", "2"))


class Out(pydantic.BaseModel):
    scale: int = int(os.environ.get("OUT", "2"))


@stage
def scale(x: pd.DataFrame) -> pd.DataFrame:
    return x * {read}


with Pipeline("cls") as pipeline:
    scale(pipeline.input("x.csv"))
"""


@pytest.mark.parametrize(
    ("read", "variable"),
    [
        pytest.param("Config.SCALE", "SCALE", id="class-scalar"),
        pytest.param("Mode.DOUBLE.value", "MODE", id="enum-member"),
        pytest.param("Out().scale", "OUT", id="pydantic-default"),
    ],
)
def test_value_read_through_a_class_at_import_reruns_stage(
    tmp_path: pathlib.Path, read: str, variable: str
) -> None:
    """`Config.SCALE`, an Enum member and a pydantic field default are re-read on a hit."""
    root = _helper_project(
        tmp_path,
        {
            "pipeline.py": _CLASS_ATTRIBUTE_PIPELINE.format(read=read),
            "data/raw/x.csv": "a\n1\n",
        },
    )
    _helper_ok(_helper_pivot(root, "repro", env={variable: "2"}))
    assert (root / "data/cls/scale.jsonl").read_text() == '{"a":2}\n'
    cached = _helper_ok(_helper_pivot(root, "status", env={variable: "2"}))

    status = _helper_ok(_helper_pivot(root, "status", env={variable: "7"}))
    output = _helper_ok(_helper_pivot(root, "repro", env={variable: "7"}))

    assert "1 cached, 0 stale" in cached, cached
    assert "1 stale" in status, status
    assert (root / "data/cls/scale.jsonl").read_text() == '{"a":7}\n', output


def test_non_scalar_class_attribute_is_an_error(tmp_path: pathlib.Path) -> None:
    """A tuple read as `Config.COLUMNS` is refused like a module-level tuple."""
    pipeline = _CLASS_ATTRIBUTE_PIPELINE.format(read="len(Config.COLUMNS)")
    root = _helper_project(tmp_path, {"pipeline.py": pipeline, "data/raw/x.csv": "a\n1\n"})

    proc = _helper_pivot(root, "repro")

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "captures non-scalar variable 'pipeline.Config.COLUMNS' (type: tuple)" in output, output
    assert "StageParams" in output, output
