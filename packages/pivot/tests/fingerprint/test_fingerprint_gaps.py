# pyright: reportImplicitStringConcatenation=false

"""CLI regressions for fingerprint gaps found during PR #496 acceptance testing."""

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

from pivot import run_history
from pivot.storage import state

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


def _run_cache_entries(project: pathlib.Path) -> list[run_history.RunCacheEntry]:
    with state.StateDB(project / ".pivot", readonly=True) as db:
        entries = list(db.iter_prefix(b"runcache:t/count:"))
    return [run_history.deserialize_run_cache_entry(value) for _key, value in entries]


def _assert_not_reusable(project: pathlib.Path, result: subprocess.CompletedProcess[str]) -> None:
    """The stage ran, warned once, filed no run-cache entry, and its next repro reruns it."""
    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, message
    assert message.count("imported user modules its fingerprint does not cover") == 1, message
    assert _run_cache_entries(project) == []
    status = _run(project, "status", "--json")
    assert status.returncode == 0, status.stdout + status.stderr
    assert "Previous run imported modules its fingerprint does not cover" in status.stdout


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
    "chained-alias-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    first = getattr\n"
                "    lookup = first\n"
                '    load = lookup(importlib, "import_module")\n'
                '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
    "parameter-getattr-import-module": (
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "\n\ndef _load(lookup):\n"
                '    return lookup(importlib, "import_module")("helpers.lib").VALUE\n',
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": _load(getattr) + len(text.splitlines())}\n',
            ),
        },
        _EDIT_LIB_VALUE,
    ),
}


_RUNTIME_LOADERS = {
    "partial-getattr": (
        "lookup = functools.partial(getattr, importlib)\n"
        '    load = lookup("import_module")\n'
        '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}'
    ),
    "module-dict": (
        'load = importlib.__dict__["import_module"]\n'
        '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}'
    ),
    "vars-dict": (
        'load = vars(importlib)["import_module"]\n'
        '    return {"v": load("helpers.lib").VALUE + len(text.splitlines())}'
    ),
    "exec-import": (
        "scope = {}\n"
        '    exec("import helpers.lib as loaded", scope)\n'
        '    return {"v": scope["loaded"].VALUE + len(text.splitlines())}'
    ),
    "manual-loader": (
        "import importlib.util\n"
        "    import pathlib\n"
        "    import sys\n"
        "    path = pathlib.Path('helpers') / 'lib.py'\n"
        "    spec = importlib.util.spec_from_file_location('manual_loaded', path)\n"
        "    assert spec is not None and spec.loader is not None\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name] = module\n"
        "    spec.loader.exec_module(module)\n"
        '    return {"v": module.VALUE + len(text.splitlines())}'
    ),
}


@pytest.mark.parametrize("loader", sorted(_RUNTIME_LOADERS))
def test_runtime_dynamic_import_makes_result_not_reusable(
    tmp_path: pathlib.Path, loader: str
) -> None:
    """A user module loaded only at run time is outside the fingerprint: the stage still runs,
    warns, files no run-cache entry and reruns next time, so an edit to the module is seen."""
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    " + _RUNTIME_LOADERS[loader] + "\n",
            ),
        },
    )

    first = _run(project, "repro")

    _assert_not_reusable(project, first)
    assert _count(project) == 4
    _edit(project, *_EDIT_LIB_VALUE)
    second = _run(project, "repro")
    assert second.returncode == 0, second.stdout + second.stderr
    assert _count(project) == 103


def test_runtime_dynamic_import_under_no_commit_warns_that_commit_records_outputs(
    tmp_path: pathlib.Path,
) -> None:
    """`pivot commit` records outputs on disk against the static fingerprint, so under
    `--no-commit` the warning says so."""
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n    "
                + _RUNTIME_LOADERS["exec-import"]
                + "\n",
            ),
        },
    )

    result = _run(project, "repro", "--no-commit")

    message = " ".join((result.stdout + result.stderr).split())
    assert result.returncode == 0, message
    assert message.count("imported user modules its fingerprint does not cover") == 1, message
    assert "`pivot commit` will record these outputs as current" in message, message
    committed = _run(project, "repro")
    assert "`pivot commit`" not in " ".join((committed.stdout + committed.stderr).split())


def test_import_from_a_thread_outliving_its_stage_does_not_fail_the_next_stage(
    tmp_path: pathlib.Path,
) -> None:
    """A thread one stage leaves running can import while a later stage runs in the same
    worker. That later stage still succeeds; it only loses reuse and reruns next time."""
    project = _project(
        tmp_path,
        {
            "late.py": "VALUE = 1\n",
            "pipeline.py": (
                "import pathlib\nimport threading\nimport time\n\n"
                "from pivot import Pipeline, stage\n\n\n"
                "@stage\ndef start(text: str) -> dict:\n"
                '    gate = pathlib.Path("go")\n\n'
                "    def delayed_import() -> None:\n"
                "        while not gate.exists():\n"
                "            time.sleep(0.01)\n"
                '        exec("import late")\n\n'
                "    threading.Thread(target=delayed_import, daemon=True).start()\n"
                '    return {"v": len(text.splitlines())}\n\n\n'
                "@stage\ndef count(start: dict) -> dict:\n"
                '    pathlib.Path("go").write_text("open")\n'
                "    time.sleep(0.5)\n"
                '    return {"v": start["v"]}\n\n\n'
                'with Pipeline("t") as pipeline:\n'
                '    count(start(pipeline.input("lines.txt", t=str)))\n'
            ),
        },
    )
    configured = _run(project, "config", "set", "core.max_workers", "1")
    assert configured.returncode == 0, configured.stdout + configured.stderr

    first = _run(project, "repro")

    _assert_not_reusable(project, first)
    assert "late" in first.stdout + first.stderr
    assert _count(project) == 3
    second = _run(project, "repro", "t/count")
    assert second.returncode == 0, second.stdout + second.stderr
    assert "t/count" in second.stdout
    assert "cached" not in second.stdout.split("t/count", 1)[1].splitlines()[0]


@pytest.mark.parametrize("shape", sorted(_DYNAMIC_IMPORT_SHAPES))
def test_dynamic_import_spelling_is_refused(tmp_path: pathlib.Path, shape: str) -> None:
    files, _edit_spec = _DYNAMIC_IMPORT_SHAPES[shape]
    project = _project(tmp_path, files)

    result = _run(project, "repro")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "fingerprinting failed" in result.stdout + result.stderr
    assert "dynamic imports" in result.stdout + result.stderr


_UNLOADED_BODY_IMPORTS = {
    "from-import-function": (
        {"runtime_helper.py": "def value():\n    return 1\n"},
        "from runtime_helper import value",
        "runtime_helper",
    ),
    "import-module": (
        {"runtime_helper.py": "VALUE = 1\n"},
        "import runtime_helper",
        "runtime_helper",
    ),
    "package-submodule": (
        {"pkg/__init__.py": "", "pkg/sub.py": "VALUE = 1\n"},
        "import pkg.sub",
        "pkg.sub",
    ),
    "package-attribute-submodule": (
        _LIB_VALUE,
        "import helpers\n    _ = helpers.lib.VALUE",
        "helpers",
    ),
    "metaclass-hook": (
        {
            "runtime_helper.py": (
                "import pathlib\n\n\nclass Meta(type):\n"
                "    def __new__(mcs, name, bases, ns):\n"
                '        ns["BONUS"] = int(pathlib.Path("runtime.txt").read_text())\n'
                "        return super().__new__(mcs, name, bases, ns)\n\n\n"
                "class Config(metaclass=Meta):\n    pass\n"
            ),
            "runtime.txt": "1\n",
        },
        "from runtime_helper import Config",
        "runtime_helper",
    ),
}


@pytest.mark.parametrize("shape", sorted(_UNLOADED_BODY_IMPORTS))
def test_unloaded_user_body_import_is_refused_at_definition(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A body import of a user module discovery did not load is refused: Pivot would have to
    run the module to fingerprint it. The message names the module and the fix."""
    files, statement, module = _UNLOADED_BODY_IMPORTS[shape]
    project = _project(
        tmp_path,
        {
            **files,
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                f"    {statement}\n"
                '    return {"v": len(text.splitlines())}\n',
            ),
        },
    )

    result = _run(project, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert f"imports user module '{module}'" in message
    assert "import it at module level" in message
    assert "StageParams" in message
    assert not (project / "data" / "t" / "count.yaml").exists()


def test_body_import_of_a_library_module_is_not_refused(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "pipeline.py": _pipeline(
                "",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import json\n"
                "    import textwrap\n"
                '    return {"v": len(json.loads("[1, 2]")) + len(textwrap.dedent(text).split())}\n',
            ),
        },
    )

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr


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


def test_resolved_helper_accepts_import_module_argument_name(tmp_path: pathlib.Path) -> None:
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/names.py": (
                "def lookup(value: str, name: str) -> str:\n    return f'{value}:{name}'\n"
            ),
            "pipeline.py": _pipeline(
                "\nfrom helpers.names import lookup\n",
                "@stage\ndef count(text: str) -> dict:\n"
                '    return {"v": lookup("safe", "import_module")}\n',
            ),
        },
    )

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == "safe:import_module"


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
                "import functools\n\nimport impl  # noqa: F401 - discovery loads it\n\n\n"
                "def limited(fn):\n    @functools.wraps(fn)\n"
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


def test_guarded_unloaded_body_import_is_refused_without_running_it(
    tmp_path: pathlib.Path,
) -> None:
    """A guarded body import of a module discovery did not load is refused at definition, and
    the module is never run: one that raises on import does not break discovery."""
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

    result = _run(project, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "imports user module 'helpers'" in message
    assert "needs a GPU" not in message
    assert "StageParams" in message


def test_body_import_package_attribute_submodule_change_reruns_stage(
    tmp_path: pathlib.Path,
) -> None:
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "from . import lib\n",
            "helpers/lib.py": "VALUE = 1\n",
            "pipeline.py": _pipeline(
                "\nimport helpers.lib\n",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import helpers\n"
                '    return {"v": helpers.lib.VALUE + len(text.splitlines())}\n',
            ),
        },
    )
    assert _run(project, "repro").returncode == 0
    assert _count(project) == 4
    _edit(project, "helpers/lib.py", "VALUE = 1", "VALUE = 100")

    result = _run(project, "repro")

    assert result.returncode == 0, result.stdout + result.stderr
    assert _count(project) == 103


def test_body_import_of_unloaded_submodule_through_loaded_package_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """`import pkg.sub` where discovery loaded `pkg` but not `pkg.sub` names an unloaded user
    module, so it is refused like any other unloaded user body import."""
    project = _project(
        tmp_path,
        {
            "helpers/__init__.py": "",
            "helpers/lib.py": "VALUE = 1\n",
            "pipeline.py": _pipeline(
                "\nimport helpers\n",
                "@stage\ndef count(text: str) -> dict:\n"
                "    import helpers.lib\n"
                '    return {"v": helpers.lib.VALUE + len(text.splitlines())}\n',
            ),
        },
    )

    result = _run(project, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "imports user module 'helpers.lib'" in message


def test_preloaded_package_attribute_of_unloaded_package_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A stage that loads `helpers.lib` in its body so a later stage can read `helpers.lib`
    through `import helpers` is refused when discovery loaded neither module."""
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": (
                "from pivot import Pipeline, stage\n\n\n"
                "@stage\ndef preload(text: str) -> dict:\n"
                "    import helpers.lib\n"
                '    return {"v": len(text.splitlines())}\n\n\n'
                "@stage\ndef count(value: dict) -> dict:\n"
                "    import helpers\n"
                '    return {"v": helpers.lib.VALUE + value["v"]}\n\n\n'
                'with Pipeline("t") as pipeline:\n'
                '    count(preload(pipeline.input("lines.txt", t=str)))\n'
            ),
        },
    )

    result = _run(project, "repro")

    message = result.stdout + result.stderr
    assert result.returncode != 0, message
    assert "imports user module 'helpers" in message, message
    assert "import it at module level" in message
    assert not (project / "data" / "t" / "count.yaml").exists()


_LOADED_SUBMODULE_BODIES = {
    "import-submodule-without-attribute-read": (
        "@stage\ndef count(text: str) -> dict:\n"
        "    import helpers.lib\n"
        '    return {"v": len(text.splitlines())}\n',
        "t/count",
        3,
    ),
    "from-package-import-submodule": (
        "@stage\ndef count(text: str) -> dict:\n"
        "    from helpers import lib\n"
        '    return {"v": lib.VALUE + len(text.splitlines())}\n',
        "t/count",
        103,
    ),
    "preload-stage-then-package-attribute": (
        "@stage\ndef preload(text: str) -> dict:\n"
        "    import helpers.lib\n"
        '    return {"v": helpers.lib.VALUE + len(text.splitlines())}\n\n\n'
        "@stage\ndef count(value: dict) -> dict:\n"
        "    import helpers\n"
        '    return {"v": helpers.lib.VALUE + value["v"]}\n',
        "t/preload",
        203,
    ),
}


@pytest.mark.parametrize("shape", sorted(_LOADED_SUBMODULE_BODIES))
def test_static_body_import_of_loaded_submodule_is_reused(
    tmp_path: pathlib.Path, shape: str
) -> None:
    """A static body import of a user submodule discovery loaded is covered by the fingerprint,
    with its package: the stages run without a warning, stay cached on the next `repro`, and
    `count` shows an edit to the module it reads."""
    body, first_stage, after_edit = _LOADED_SUBMODULE_BODIES[shape]
    wiring = (
        'count(preload(pipeline.input("lines.txt", t=str)))'
        if first_stage == "t/preload"
        else 'count(pipeline.input("lines.txt", t=str))'
    )
    project = _project(
        tmp_path,
        {
            **_LIB_VALUE,
            "pipeline.py": (
                "import helpers.lib\n\nfrom pivot import Pipeline, stage\n\n\n"
                + body
                + f'\n\nwith Pipeline("t") as pipeline:\n    {wiring}\n'
            ),
        },
    )
    configured = _run(project, "config", "set", "core.max_workers", "1")
    assert configured.returncode == 0, configured.stdout + configured.stderr

    first = _run(project, "repro")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "does not cover" not in first.stdout + first.stderr
    assert _run_cache_entries(project) != []
    again = _run(project, "repro", "--json")
    assert again.returncode == 0, again.stdout + again.stderr
    for stage_name in {first_stage, "t/count"}:
        assert f'"stage": "{stage_name}", "status": "cached"' in again.stdout, again.stdout
    _edit(project, *_EDIT_LIB_VALUE)

    edited = _run(project, "repro")

    assert edited.returncode == 0, edited.stdout + edited.stderr
    assert "does not cover" not in edited.stdout + edited.stderr
    assert _count(project) == after_edit


def test_eval_lambda_is_hashed_by_its_code_object(tmp_path: pathlib.Path) -> None:
    """A lambda built by `eval` has no source; its code object is hashed, so its fingerprint is
    stable across runs and an edit to it reruns the stage."""
    pipeline = _pipeline(
        '\nhelper = eval("lambda x: x + 1")\n',
        '@stage\ndef count(text: str) -> dict:\n    return {"v": helper(len(text.splitlines()))}\n',
    )
    project = _project(tmp_path, {"pipeline.py": pipeline})
    assert _run(project, "repro").returncode == 0

    again = _run(project, "repro", "--json")

    assert '"stage": "t/count", "status": "cached"' in again.stdout, again.stdout
    _edit(project, "pipeline.py", "x + 1", "x + 100")
    edited = _run(project, "repro", "--json")
    assert '"stage": "t/count", "status": "ran"' in edited.stdout, edited.stdout
    assert _count(project) == 103
