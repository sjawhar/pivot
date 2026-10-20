# pyright: reportImplicitStringConcatenation=false

"""CLI regressions for fingerprint gaps found during PR #496 acceptance testing."""

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

_PIVOT = pathlib.Path(sys.executable).parent / "pivot"
_LINES = "Local Set Default Proof Using\nSet Default Proof Using\nLemma a\n"
_HEADER = """\
import functools
import importlib
import os

from pivot import Pipeline, stage
"""
_NORMALIZE = """

@stage
def normalize(text: str) -> str:
    return "\\n".join(line.strip() for line in text.splitlines() if line.strip())
"""
_FOOTER = """

with Pipeline("t") as pipeline:
    count(normalize(pipeline.input("lines.txt", t=str)))
"""
_LIB_VALUE = {"helpers/__init__.py": "", "helpers/lib.py": "VALUE = 1\n"}
_EDIT_LIB_VALUE = ("helpers/lib.py", "VALUE = 1\n", "VALUE = 100\n")


def _pipeline(imports: str, body: str) -> str:
    return _HEADER + imports + _NORMALIZE + "\n" + body + _FOOTER


def _run(project: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VIRTUAL_ENV", "PIVOT_UNSAFE_FINGERPRINTING")
    }
    return subprocess.run(
        [str(_PIVOT), *args], cwd=project, env=env, capture_output=True, text=True, timeout=300
    )


def _project(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    project = tmp_path / "project"
    for rel, text in files.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text)
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "raw" / "lines.txt").write_text(_LINES)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    init = _run(project, "init")
    assert init.returncode == 0, init.stdout + init.stderr
    return project


def _edit(project: pathlib.Path, rel: str, old: str, new: str) -> None:
    path = project / rel
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def _count(project: pathlib.Path) -> object:
    return yaml.safe_load((project / "data" / "t" / "count.yaml").read_text())["v"]


_DYNAMIC_IMPORT_SHAPES = {
    "local-alias-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    load = importlib.import_module\n"
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "global-alias-dunder-import": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\n_imp = __import__\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": _imp("helpers.lib", fromlist=["VALUE"]).VALUE'
                " + len(text.splitlines())}\n",
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "default-arg-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\n\ndef _load(name, loader=importlib.import_module):\n    return loader(name)\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": _load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "partial-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\nload = functools.partial(importlib.import_module)\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "literal-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                '    load = getattr(importlib, "import_module")\n'
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "local-alias-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    lookup = getattr\n"
                '    load = lookup(importlib, "import_module")\n'
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
}


@pytest.mark.parametrize("shape", sorted(_DYNAMIC_IMPORT_SHAPES))
def test_dynamic_import_spelling_is_refused(tmp_path: pathlib.Path, shape: str) -> None:
    files, _edit_spec = _DYNAMIC_IMPORT_SHAPES[shape]
    project = _project(tmp_path, files)

    result = _run(project, "repro")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "fingerprinting failed" in result.stdout + result.stderr
    assert "dynamic imports" in result.stdout + result.stderr


def test_unloaded_runtime_body_import_value_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BODY_IMPORT_BONUS", "1")
    project = _project(
        tmp_path,
        {
            "runtime_values.py": (
                "import os\n\nCONFIG = {'bonus': int(os.environ['BODY_IMPORT_BONUS'])}\n"
            ),
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    from runtime_values import CONFIG\n"
                '    return {"v": len(text.splitlines()) + CONFIG["bonus"]}\n',
            ),
        },
    )

    result = _run(project, "repro")

    assert result.returncode != 0, result.stdout + result.stderr
    message = result.stdout + result.stderr
    assert "runtime_values" in message
    assert "CONFIG" in message
    assert "StageParams" in message


def test_user_method_named_import_module_runs(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/formatter.py": (
                "class Formatter:\n"
                "    @staticmethod\n"
                "    def import_module(text: str) -> int:\n"
                "        return len(text.strip())\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.formatter import Formatter\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": Formatter.import_module(text)}\n',
            ),
        },
    )

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == len(_LINES.strip())


def test_decorated_method_wrapper_edit_reruns_stage(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import functools\n\n\ndef scaled(fn):\n    @functools.wraps(fn)\n"
                "    def wrapper(self, x):\n        return fn(self, x) * 2\n\n    return wrapper\n\n\n"
                "class Scaler:\n    @scaled\n    def apply(self, x):\n        return x + 1\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Scaler\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": Scaler().apply(len(text.splitlines()))}\n',
            ),
        },
    )
    assert _run(project, "repro").returncode == 0
    _edit(project, "helpers/model.py", "fn(self, x) * 2\n", "fn(self, x) * 200\n")

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 800


def test_decorator_wrapper_global_edit_reruns_stage(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import functools\n\nSCALE = 2\n\n\ndef scaled(fn):\n    @functools.wraps(fn)\n"
                "    def wrapper(self, x):\n        return fn(self, x) * SCALE\n\n    return wrapper\n\n\n"
                "class Scaler:\n    @scaled\n    def apply(self, x):\n        return x + 1\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Scaler\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": Scaler().apply(len(text.splitlines()))}\n',
            ),
        },
    )
    assert _run(project, "repro").returncode == 0
    _edit(project, "helpers/model.py", "SCALE = 2", "SCALE = 30")

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 120


def test_wrapped_helper_body_import_edit_reruns_stage(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "impl.py": "LIMIT = 1\n",
            "helpers/__init__.py": "",
            "helpers/lib.py": (
                "import functools\n\n\ndef limited(fn):\n    @functools.wraps(fn)\n"
                "    def wrapper(x):\n        from impl import LIMIT\n\n"
                "        return fn(x) + LIMIT\n\n    return wrapper\n\n\n"
                "@limited\ndef step(x):\n    return x\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.lib import step\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": step(len(text.splitlines()))}\n',
            ),
        },
    )
    assert _run(project, "repro").returncode == 0
    _edit(project, "impl.py", "LIMIT = 1", "LIMIT = 100")

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 103


_TRACKED_CLASS_MEMBER_SHAPES = {
    "dataclass-partial-member": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\nimport functools\n\n\ndef scale(x, k):\n    return x * k\n\n\n"
                "@dataclasses.dataclass\nclass Settings:\n    width: int = 1\n"
                "    multiply = functools.partial(scale, k=2)\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.multiply(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "return x * k\n", "return x * k * 3\n"),
        18,
    ),
    "nested-class-method": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\n\nSCALE = 2\n\n\n@dataclasses.dataclass\nclass Settings:\n"
                "    width: int = 1\n\n    class Rule:\n        def apply(self, x):\n"
                "            return x * SCALE\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.Rule().apply(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "SCALE = 2", "SCALE = 3"),
        9,
    ),
    "dataclass-field-default": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\n\nWIDTH = 2\n\n\n@dataclasses.dataclass\nclass Settings:\n"
                "    width: int = WIDTH\n\n"
                "    def slots(self, x):\n        return x * self.width\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.slots(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "WIDTH = 2", "WIDTH = 3"),
        9,
    ),
    "partialmethod-member": (
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import functools\n\n\ndef multiply(self, x, k):\n    return x * k\n\n\n"
                "class Scaler:\n    apply = functools.partialmethod(multiply, k=2)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Scaler\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": Scaler().apply(len(text.splitlines()))}\n',
            ),
        },
        ("helpers/model.py", "return x * k\n", "return x * k * 3\n"),
        18,
    ),
}


@pytest.mark.parametrize("shape", sorted(_TRACKED_CLASS_MEMBER_SHAPES))
def test_class_member_edit_reruns_stage(tmp_path: pathlib.Path, shape: str) -> None:
    files, edit, expected = _TRACKED_CLASS_MEMBER_SHAPES[shape]
    project = _project(tmp_path, files)
    assert _run(project, "repro").returncode == 0
    _edit(project, *edit)

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == expected


def test_dataclass_callable_member_is_refused(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/model.py": (
                "import dataclasses\n\n\nclass Scaler:\n    def __call__(self, x):\n"
                "        return x * 2\n\n\n@dataclasses.dataclass\nclass Settings:\n"
                "    scale = Scaler()\n\n\n"
                "def compute(settings: Settings, x):\n    return settings.scale(x)\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.model import Settings, compute\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": compute(Settings(), len(text.splitlines()))}\n',
            ),
        },
    )

    result = _run(project, "repro")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "captures non-scalar variable" in result.stdout + result.stderr
    assert "Settings.scale" in result.stdout + result.stderr


def test_guarded_in_body_import_is_not_imported_at_definition(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/gpu.py": 'raise RuntimeError("helpers.gpu needs a GPU")\n',
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    n = len(text.splitlines())\n"
                '    if os.environ.get("ACCEPT_USE_GPU"):\n        from helpers import gpu\n\n'
                '        n = gpu.boost(n)\n    return {"v": n}\n',
            ),
        },
    )
    first = _run(project, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert _count(project) == 3
    _edit(
        project, "pipeline.py", "n = len(text.splitlines())\n", "n = len(text.splitlines()) + 10\n"
    )

    second = _run(project, "repro")

    assert second.returncode == 0, second.stdout + second.stderr
    assert _count(project) == 13
