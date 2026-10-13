import importlib.util
import pathlib
import sys
import types

from pivot import fingerprint

_SOURCE = """\
import functools


def decorator(func):
    @functools.wraps(func)
    def wrapper(value):
        return func(value) + 1

    return wrapper


@decorator
def stage(value):
    return value * 2
"""


def _load_module(path: pathlib.Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_wrapped_function_hash_ignores_source_path(tmp_path: pathlib.Path) -> None:
    first_path = tmp_path / "first" / "portable_hash.py"
    second_path = tmp_path / "second" / "portable_hash.py"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text(_SOURCE)
    second_path.write_text(_SOURCE)

    first_module = _load_module(first_path, "portable_hash_first")
    second_module = _load_module(second_path, "portable_hash_second")

    assert fingerprint.hash_function_ast(first_module.stage) == fingerprint.hash_function_ast(
        second_module.stage
    )
