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
