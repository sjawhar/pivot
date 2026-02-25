# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import logging
import marshal
import pathlib
import sys
import types
import typing
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import BaseModel, Field

from pivot import ast_utils, exceptions, fingerprint, loaders, project
from pivot.config import io as config_io
from pivot.storage import state as state_module

if TYPE_CHECKING:
    from collections.abc import Callable

MODULE_CONST = 7


def _helper_identity(value: int) -> int:
    return value


def _helper_plus_one(value: int) -> int:
    return value + 1


def _helper_uses_global() -> int:
    return _helper_plus_one(MODULE_CONST)


def _helper_with_generator() -> int:
    return sum(_helper_plus_one(x) for x in [1, 2, 3])


def _helper_dynamic_getattr(name: str) -> Any:
    import math

    return getattr(math, name)


def _helper_literal_getattr() -> Any:
    import math

    return math.pi


def _helper_uses_globals() -> Any:
    return globals().get("MODULE_CONST")


def _helper_dynamic_import() -> Any:
    return importlib.import_module("math")


class _helper_dependency_class:
    pass


class _helper_class_with_annotations(_helper_dependency_class):
    value: _helper_dependency_class | None = None


@dataclasses.dataclass(frozen=True)
class _helper_frozen_dataclass:
    value: int


@dataclasses.dataclass
class _helper_mutable_dataclass:
    value: int


@dataclasses.dataclass
class _helper_dataclass_with_method:
    value: int

    def compute(self) -> int:
        return self.value + 1


class _helper_frozen_model(BaseModel):
    model_config = {"frozen": True}
    value: int


class _helper_nested_model(BaseModel):
    value: int = 1


class _helper_node_model(BaseModel):
    value: int = 0
    children: list[_helper_node_model] = Field(default_factory=list)


_helper_node_model.model_rebuild()


def _helper_wrapped_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


@_helper_wrapped_decorator
def _helper_wrapped_function(value: int) -> int:
    return value + 10


def _helper_unwrapped_function(value: int) -> int:
    return value + 10


def test_collecting_sources_restores_previous_map() -> None:
    fingerprint._active_source_map = {"existing.py": (1, 2, 3)}
    with fingerprint._collecting_sources() as source_map:
        source_map["new.py"] = (4, 5, 6)
        assert fingerprint._active_source_map is source_map
    assert fingerprint._active_source_map == {"existing.py": (1, 2, 3)}


def test_make_manifest_cache_key_includes_stage_and_versions() -> None:
    key = fingerprint._make_manifest_cache_key("stage_a")
    decoded = key.decode()
    assert decoded.startswith("sm:stage_a\x00")
    assert str(fingerprint._CACHE_SCHEMA_VERSION) in decoded


@pytest.mark.usefixtures("set_project_root")
def test_normalize_changed_paths_filters_non_project_paths(tmp_path: pathlib.Path) -> None:
    inside = tmp_path / "a.py"
    inside.write_text("x = 1\n")
    outside = pathlib.Path("/tmp/outside.py")

    normalized = fingerprint._normalize_changed_paths([inside, "a.py", outside])

    assert normalized == {"a.py"}


def test_manifest_references_paths_handles_invalid_payload() -> None:
    assert not fingerprint._manifest_references_paths(b"not-json", {"a.py"})
    assert not fingerprint._manifest_references_paths(json.dumps([1, 2]).encode(), {"a.py"})
    assert not fingerprint._manifest_references_paths(json.dumps({"s": []}).encode(), {"a.py"})


def test_manifest_references_paths_detects_changed_source() -> None:
    raw = json.dumps({"s": {"a.py": [1, 2, 3], "b.py": [4, 5, 6]}}).encode()
    assert fingerprint._manifest_references_paths(raw, {"b.py"})
    assert not fingerprint._manifest_references_paths(raw, {"c.py"})


class _FakeReadOnlyDB:
    def __init__(self, raw: bytes | None) -> None:
        self._raw = raw

    def get_raw(self, _key: bytes) -> bytes | None:
        return self._raw


@pytest.mark.usefixtures("set_project_root")
def test_try_manifest_cache_hit_returns_manifest_when_stats_match(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_file = tmp_path / "stages.py"
    source_file.write_text("def fn():\n    return 1\n")
    st = source_file.stat()
    payload = {
        "m": {"self:fn": "abc"},
        "s": {"stages.py": [st.st_mtime_ns, st.st_size, st.st_ino]},
    }
    db = _FakeReadOnlyDB(json.dumps(payload).encode())
    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: db)

    manifest = fingerprint._try_manifest_cache_hit("stage")

    assert manifest == {"self:fn": "abc"}


def test_try_manifest_cache_hit_rejects_corrupt_or_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"m": {"self:fn": "abc"}, "s": {"../evil.py": [1, 2, 3]}}
    db = _FakeReadOnlyDB(json.dumps(payload).encode())
    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: db)
    assert fingerprint._try_manifest_cache_hit("stage") is None


class _FakeStateDB:
    def __init__(
        self, *, readonly: bool, raw_rows: list[tuple[bytes, bytes]], deleted: list[bytes]
    ) -> None:
        self.readonly = readonly
        self.raw_rows = raw_rows
        self.deleted = deleted

    def __enter__(self) -> _FakeStateDB:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    def iter_prefix(self, prefix: bytes) -> list[tuple[bytes, bytes]]:
        assert prefix == b"sm:"
        return self.raw_rows

    def delete_raw_many(self, keys: list[bytes]) -> None:
        self.deleted.extend(keys)


@pytest.mark.usefixtures("set_project_root")
def test_invalidate_manifests_for_paths_removes_db_and_pending_entries(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "a.py"
    source_file.write_text("x = 1\n")

    raw_keep = json.dumps({"m": {"k": "v"}, "s": {"b.py": [1, 2, 3]}}).encode()
    raw_drop = json.dumps({"m": {"k": "v"}, "s": {"a.py": [1, 2, 3]}}).encode()
    deleted: list[bytes] = []
    rows = [(b"sm:keep", raw_keep), (b"sm:drop", raw_drop)]

    def _state_db_factory(_path: pathlib.Path, *, readonly: bool) -> _FakeStateDB:
        return _FakeStateDB(readonly=readonly, raw_rows=rows, deleted=deleted)

    monkeypatch.setattr(config_io, "get_state_db_path", lambda: tmp_path / "state.lmdb")
    monkeypatch.setattr(state_module, "StateDB", _state_db_factory)
    fingerprint._pending_manifest_writes = [(b"sm:pending", raw_drop), (b"sm:other", raw_keep)]

    fingerprint.invalidate_manifests_for_paths([source_file])

    assert b"sm:drop" in deleted
    assert b"sm:pending" not in [k for k, _ in fingerprint._pending_manifest_writes]


def test_collect_nested_code_globals_finds_generator_dependencies() -> None:
    names = fingerprint._collect_nested_code_globals(_helper_with_generator.__code__)
    assert "_helper_plus_one" in names


@pytest.mark.parametrize(
    ("func", "expected_message"),
    [
        pytest.param(_helper_dynamic_getattr, "getattr()", id="dynamic-getattr"),
        pytest.param(_helper_uses_globals, "globals()", id="globals"),
        pytest.param(_helper_dynamic_import, "dynamic imports", id="dynamic-import"),
    ],
)
def test_check_dynamic_name_access_rejects_unsafe_patterns(
    func: Callable[..., Any],
    expected_message: str,
) -> None:
    with pytest.raises(exceptions.StageDefinitionError, match=expected_message):
        fingerprint._check_dynamic_name_access(func)


def test_check_dynamic_name_access_allows_literal_getattr() -> None:
    fingerprint._check_dynamic_name_access(_helper_literal_getattr)


def test_get_stage_fingerprint_tracks_callable_and_constant_dependencies() -> None:
    manifest = fingerprint.get_stage_fingerprint(_helper_uses_global)
    assert "self:_helper_uses_global" in manifest
    assert "func:_helper_plus_one" in manifest
    assert manifest["const:MODULE_CONST"] == repr(MODULE_CONST)


def test_get_stage_fingerprint_handles_nested_global_references() -> None:
    manifest = fingerprint.get_stage_fingerprint(_helper_with_generator)
    assert "func:_helper_plus_one" in manifest


def test_get_stage_fingerprint_for_class_processes_class_body_dependencies() -> None:
    manifest = fingerprint.get_stage_fingerprint(_helper_class_with_annotations)
    assert "class:_helper_dependency_class" in manifest


@dataclasses.dataclass(frozen=True)
class _HelperReader(loaders.Reader[str]):
    suffix: str = "txt"

    def load(self, path: pathlib.Path) -> str:
        return path.read_text()


@dataclasses.dataclass(frozen=True)
class _HelperWriter(loaders.Writer[str]):
    suffix: str = "txt"

    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(data)


@dataclasses.dataclass(frozen=True)
class _HelperLoader(loaders.Loader[str]):
    prefix: str = "x"

    def load(self, path: pathlib.Path) -> str:
        return path.read_text()

    def save(self, data: str, path: pathlib.Path) -> None:
        path.write_text(f"{self.prefix}:{data}")

    def empty(self) -> str:
        return ""


def test_get_loader_fingerprint_for_reader_writer_and_loader() -> None:
    reader_manifest = fingerprint.get_loader_fingerprint(_HelperReader())
    writer_manifest = fingerprint.get_loader_fingerprint(_HelperWriter())
    loader_manifest = fingerprint.get_loader_fingerprint(_HelperLoader(prefix="p"))

    assert "loader:_HelperReader:load" in reader_manifest
    assert "loader:_HelperWriter:save" in writer_manifest
    assert "loader:_HelperLoader:load" in loader_manifest
    assert "loader:_HelperLoader:save" in loader_manifest
    assert "loader:_HelperLoader:empty" in loader_manifest
    assert "loader:_HelperLoader:config" in loader_manifest


def test_loader_config_change_changes_fingerprint() -> None:
    first = fingerprint.get_loader_fingerprint(_HelperLoader(prefix="a"))
    second = fingerprint.get_loader_fingerprint(_HelperLoader(prefix="b"))
    assert first["loader:_HelperLoader:config"] != second["loader:_HelperLoader:config"]


def test_is_unsafe_fingerprinting_enabled_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIVOT_UNSAFE_FINGERPRINTING", "1")
    assert fingerprint._is_unsafe_fingerprinting_enabled()


def test_check_mutable_capture_raises_when_safe_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIVOT_UNSAFE_FINGERPRINTING", raising=False)
    monkeypatch.setattr(fingerprint, "_is_unsafe_fingerprinting_enabled", lambda: False)
    with pytest.raises(exceptions.StageDefinitionError, match="closure captures mutable variable"):
        fingerprint._check_mutable_capture("items", [1, 2], "stage_a")


def test_check_mutable_capture_warns_when_unsafe_mode(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(fingerprint, "_is_unsafe_fingerprinting_enabled", lambda: True)
    with caplog.at_level("WARNING"):
        fingerprint._check_mutable_capture("items", [1, 2], "stage_a")
    assert "closure captures mutable variable" in caplog.text


def test_check_data_class_methods_rejects_user_methods() -> None:
    with pytest.raises(exceptions.StageDefinitionError, match="has methods"):
        fingerprint._check_data_class_methods(_helper_dataclass_with_method)


def test_is_frozen_dataclass_and_pydantic_detection() -> None:
    assert fingerprint._is_frozen_dataclass(_helper_frozen_dataclass(1))
    assert not fingerprint._is_frozen_dataclass(_helper_mutable_dataclass(1))
    assert fingerprint._is_frozen_pydantic(_helper_frozen_model(value=1))


class _HelperDeterministicRepr:
    def __repr__(self) -> str:
        return "Deterministic(value=1)"


class _HelperMemoryRepr:
    def __repr__(self) -> str:
        return "Thing at 0x1234"


def test_hash_unrecognized_closure_value_uses_repr_hash() -> None:
    manifest: dict[str, str] = {}
    fingerprint._hash_unrecognized_closure_value(
        "obj", _HelperDeterministicRepr(), manifest, "stage"
    )
    assert "const:obj" in manifest


def test_hash_unrecognized_closure_value_memory_repr_calls_mutable_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}

    def _mark_called(_name: str, _value: Any, _stage: str) -> None:
        called["value"] = True

    monkeypatch.setattr(fingerprint, "_check_mutable_capture", _mark_called)
    fingerprint._hash_unrecognized_closure_value("obj", _HelperMemoryRepr(), {}, "stage")
    assert called["value"]


def test_process_partial_dependency_tracks_args_kwargs_and_function() -> None:
    partial_obj = functools.partial(_helper_plus_one, value=10)
    manifest: dict[str, str] = {}
    fingerprint._process_partial_dependency("step", partial_obj, manifest, set())
    assert "partial:step.args" in manifest
    assert "partial:step.kwargs" in manifest
    assert "func:step.func" in manifest


def test_process_collection_dependency_tracks_only_user_callables() -> None:
    manifest: dict[str, str] = {}
    collection = {"a": _helper_plus_one, "b": len}
    fingerprint._process_collection_dependency("callbacks", collection, manifest, set())
    assert "func:callbacks['a']" in manifest
    assert "func:callbacks['b']" not in manifest


def test_is_primitive_collection_handles_nested_and_circular() -> None:
    assert fingerprint._is_primitive_collection({"a": [1, 2, {"b": "c"}]})
    circular: list[Any] = []
    circular.append(circular)
    assert not fingerprint._is_primitive_collection(circular)


def test_resolve_annotations_individually_keeps_resolvable_names() -> None:
    def _helper_annotation_func(x: _helper_dependency_class, y: typing.Any) -> None:
        return None

    resolved = fingerprint._resolve_annotations_individually(_helper_annotation_func)
    assert resolved["x"] is _helper_dependency_class
    assert "y" in resolved


def test_process_type_hint_rejects_dataclass_methods() -> None:
    with pytest.raises(exceptions.StageDefinitionError, match="has methods"):
        fingerprint._process_type_hint(_helper_dataclass_with_method, {}, set())


def test_strip_schema_metadata_recursively_removes_title_and_description() -> None:
    schema = {
        "title": "Root",
        "description": "desc",
        "$defs": {"Inner": {"title": "Inner", "description": "d", "type": "object"}},
        "properties": {"x": {"title": "X", "description": "dx", "type": "integer"}},
        "items": {"title": "Item", "description": "di", "type": "string"},
    }
    stripped = fingerprint._strip_schema_metadata(schema)
    assert "title" not in stripped
    assert "description" not in stripped
    assert "title" not in stripped["$defs"]["Inner"]
    assert "description" not in stripped["properties"]["x"]


def test_hash_pydantic_schema_includes_defaults_and_nested_types() -> None:
    manifest: dict[str, str] = {}
    fingerprint._hash_pydantic_schema(cast("Any", _helper_node_model), manifest, set())
    assert "schema:_helper_node_model" in manifest
    assert "class:_helper_node_model" in manifest


def test_serialize_value_for_hash_is_deterministic_for_sets() -> None:
    first = fingerprint._serialize_value_for_hash({"b", "a"})
    second = fingerprint._serialize_value_for_hash({"a", "b"})
    assert first == second


def test_process_module_dependency_tracks_module_attr_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("helper_mod")
    cast("Any", module).fn = _helper_plus_one
    cast("Any", module).NUM = 9

    monkeypatch.setattr(
        ast_utils,
        "extract_module_attr_usage",
        lambda _func: [("helper_mod", "fn"), ("helper_mod", "NUM"), ("helper_mod", "MISSING")],
    )

    manifest: dict[str, str] = {}
    fingerprint._process_module_dependency("helper_mod", module, _helper_identity, manifest, set())

    assert "mod:helper_mod.fn" in manifest
    assert manifest["mod:helper_mod.NUM"] == "9"
    assert manifest["mod:helper_mod.MISSING"] == "unknown"


def test_process_module_dependency_rejects_unsupported_module_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("helper_bad")
    cast("Any", module).BAD = object()
    monkeypatch.setattr(
        ast_utils, "extract_module_attr_usage", lambda _func: [("helper_bad", "BAD")]
    )
    with pytest.raises(TypeError, match="Cannot fingerprint module attribute"):
        fingerprint._process_module_dependency("helper_bad", module, _helper_identity, {}, set())


def test_get_qualname_for_cache_disambiguates_lambdas() -> None:
    def fn(x: int) -> int:
        return x + 1

    fn.__name__ = "<lambda>"
    fn.__qualname__ = "<lambda>"
    qualname = fingerprint._get_qualname_for_cache(fn)
    assert "<lambda>" in qualname
    assert qualname.count(":") >= 2


def test_should_skip_persistent_cache_for_locals_and_wrapped() -> None:
    def _helper_local() -> int:
        return 1

    assert fingerprint._should_skip_persistent_cache(_helper_local)
    assert fingerprint._should_skip_persistent_cache(_helper_wrapped_function)
    assert not fingerprint._should_skip_persistent_cache(_helper_unwrapped_function)


def test_hash_function_ast_persistent_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    fingerprint._hash_function_ast_cache = weak_cache = fingerprint.weakref.WeakKeyDictionary()  # type: ignore[attr-defined]

    class _DB:
        def get_ast_hash(self, *args: Any, **kwargs: Any) -> str:
            return "cached-hash"

    monkeypatch.setattr(fingerprint, "_should_skip_persistent_cache", lambda _func: False)
    monkeypatch.setattr(fingerprint, "_get_func_source_info", lambda _func: ("a.py", 1, 2, 3))
    monkeypatch.setattr(fingerprint, "_get_qualname_for_cache", lambda _func: "qual")
    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: _DB())

    result = fingerprint.hash_function_ast(_helper_plus_one)

    assert result == "cached-hash"
    assert weak_cache[_helper_plus_one] == "cached-hash"


def test_hash_function_ast_queues_persistent_write_on_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DB:
        def get_ast_hash(self, *args: Any, **kwargs: Any) -> None:
            return None

    fingerprint._hash_function_ast_cache = fingerprint.weakref.WeakKeyDictionary()  # type: ignore[attr-defined]
    fingerprint._pending_ast_writes = []
    monkeypatch.setattr(fingerprint, "_should_skip_persistent_cache", lambda _func: False)
    monkeypatch.setattr(fingerprint, "_get_func_source_info", lambda _func: ("a.py", 10, 20, 30))
    monkeypatch.setattr(fingerprint, "_get_qualname_for_cache", lambda _func: "qual")
    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: _DB())
    monkeypatch.setattr(fingerprint, "_compute_function_hash", lambda _func: "fresh")

    result = fingerprint.hash_function_ast(_helper_plus_one)

    assert result == "fresh"
    assert fingerprint._pending_ast_writes


def test_compute_function_hash_builtin_type_uses_stable_name() -> None:
    expected = fingerprint.xxhash.xxh64(b"builtin:list").hexdigest()  # type: ignore[attr-defined]
    assert fingerprint._compute_function_hash(list) == expected


def test_compute_function_hash_wrapped_uses_bytecode() -> None:
    expected = fingerprint.xxhash.xxh64(
        marshal.dumps(_helper_wrapped_function.__code__)
    ).hexdigest()  # type: ignore[attr-defined]
    assert fingerprint._compute_function_hash(_helper_wrapped_function) == expected


def test_compute_function_hash_exec_defined_function_uses_code_fallback() -> None:
    namespace: dict[str, Any] = {}
    exec("def generated(x):\n    return x + 1\n", namespace)
    generated = namespace["generated"]
    expected = fingerprint.xxhash.xxh64(marshal.dumps(generated.__code__)).hexdigest()  # type: ignore[attr-defined]
    assert fingerprint._compute_function_hash(generated) == expected


def test_compute_function_hash_syntax_error_falls_back_to_source_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fingerprint.inspect, "getsource", lambda _func: "def a():\n  return 1\n")
    monkeypatch.setattr(fingerprint.ast, "parse", lambda _src: (_ for _ in ()).throw(SyntaxError()))
    expected = fingerprint.xxhash.xxh64(b"def a():\n  return 1\n").hexdigest()  # type: ignore[attr-defined]
    assert fingerprint._compute_function_hash(_helper_plus_one) == expected


def test_normalize_ast_removes_docstrings_and_normalizes_function_name() -> None:
    tree = fingerprint.ast.parse(
        """
def named():
    \"\"\"doc\"\"\"
    return 1
"""
    )
    node = tree.body[0]
    normalized = fingerprint._normalize_ast(node)
    assert isinstance(normalized, fingerprint.ast.FunctionDef)
    assert normalized.name == "func"
    assert not fingerprint._has_docstring(normalized)


def test_is_user_code_classification() -> None:
    dynamic_module = types.ModuleType("dynamic_mod")
    assert not fingerprint.is_user_code(None)
    assert not fingerprint.is_user_code(sys)
    assert not fingerprint.is_user_code(fingerprint)
    assert fingerprint.is_user_code(dynamic_module)


def test_is_user_code_impl_namespace_package_in_site_packages() -> None:
    module = types.ModuleType("ns_pkg")
    module.__path__ = ["/tmp/site-packages/ns_pkg"]  # type: ignore[attr-defined]
    assert not fingerprint._is_user_code_impl(module)


def test_get_module_handles_builtins_and_modules() -> None:
    assert fingerprint._get_module(sys) is sys
    assert fingerprint._get_module(len) is None
    assert fingerprint._get_module(_helper_plus_one) is sys.modules[__name__]


def test_is_stdlib_path_respects_site_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    stdlib_root = tmp_path / "lib"
    module_file = stdlib_root / "pkg" / "mod.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("x = 1\n")
    monkeypatch.setattr(fingerprint, "_STDLIB_PATHS", (stdlib_root,))
    assert fingerprint._is_stdlib_path(module_file)

    sp_file = stdlib_root / "site-packages" / "pkg" / "mod.py"
    sp_file.parent.mkdir(parents=True)
    sp_file.write_text("x = 1\n")
    assert not fingerprint._is_stdlib_path(sp_file)


def test_add_callable_to_manifest_merges_child_without_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fingerprint, "hash_function_ast", lambda _func: "self-hash")
    monkeypatch.setattr(
        fingerprint,
        "get_stage_fingerprint",
        lambda _func, _visited=None: {"self:child": "a", "func:dep": "b"},
    )
    manifest: dict[str, str] = {}
    fingerprint._add_callable_to_manifest("func:root", _helper_plus_one, manifest, set())
    assert manifest == {"func:root": "self-hash", "func:dep": "b"}


def test_get_stage_fingerprint_cached_uses_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage: {"self:x": "cached"})
    result = fingerprint.get_stage_fingerprint_cached("stage", _helper_plus_one)
    assert result == {"self:x": "cached"}


def test_get_stage_fingerprint_cached_queues_manifest_on_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint._pending_manifest_writes = []
    monkeypatch.setattr(fingerprint, "_try_manifest_cache_hit", lambda _stage: None)
    monkeypatch.setattr(fingerprint, "get_stage_fingerprint", lambda _func: {"self:x": "fresh"})

    result = fingerprint.get_stage_fingerprint_cached("stage", _helper_plus_one)

    assert result == {"self:x": "fresh"}
    assert len(fingerprint._pending_manifest_writes) == 1


def test_flush_ast_hash_cache_success_and_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[fingerprint.AstHashEntry] = []

    class _StateDBOK:
        def __init__(self, _path: pathlib.Path, *, readonly: bool) -> None:
            assert not readonly

        def __enter__(self) -> _StateDBOK:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            return None

        def save_ast_hash_many(self, entries: list[fingerprint.AstHashEntry]) -> None:
            saved.extend(entries)

    monkeypatch.setattr(config_io, "get_state_db_path", lambda: tmp_path / "state.lmdb")
    monkeypatch.setattr(state_module, "StateDB", _StateDBOK)
    fingerprint._pending_ast_writes = [("a.py", 1, 2, 3, "q", "3.13", 2, "h")]
    fingerprint.flush_ast_hash_cache()
    assert saved
    assert not fingerprint._pending_ast_writes

    class _StateDBFail:
        def __init__(self, _path: pathlib.Path, *, readonly: bool) -> None:
            raise OSError("disk")

    monkeypatch.setattr(state_module, "StateDB", _StateDBFail)
    fingerprint._pending_ast_writes = [("b.py", 1, 2, 3, "q", "3.13", 2, "h")]
    fingerprint.flush_ast_hash_cache()
    assert fingerprint._pending_ast_writes


def test_flush_manifest_cache_success_and_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[tuple[bytes, bytes]] = []

    class _StateDBOK:
        def __init__(self, _path: pathlib.Path, *, readonly: bool) -> None:
            assert not readonly

        def __enter__(self) -> _StateDBOK:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            return None

        def put_raw_many(self, entries: list[tuple[bytes, bytes]]) -> None:
            saved.extend(entries)

    monkeypatch.setattr(config_io, "get_state_db_path", lambda: tmp_path / "state.lmdb")
    monkeypatch.setattr(state_module, "StateDB", _StateDBOK)
    fingerprint._pending_manifest_writes = [(b"k", b"v")]
    fingerprint.flush_manifest_cache()
    assert saved == [(b"k", b"v")]
    assert not fingerprint._pending_manifest_writes

    class _StateDBFail:
        def __init__(self, _path: pathlib.Path, *, readonly: bool) -> None:
            raise OSError("disk")

    monkeypatch.setattr(state_module, "StateDB", _StateDBFail)
    fingerprint._pending_manifest_writes = [(b"k2", b"v2")]
    fingerprint.flush_manifest_cache()
    assert fingerprint._pending_manifest_writes == [(b"k2", b"v2")]


def test_get_state_db_success_and_failed_init(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DB:
        def close(self) -> None:
            return None

    def _factory(_path: pathlib.Path, *, readonly: bool) -> _DB:
        assert readonly
        return _DB()

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False
    monkeypatch.setattr(config_io, "get_state_db_path", lambda: tmp_path / "state.lmdb")
    monkeypatch.setattr(state_module, "StateDB", _factory)
    db = fingerprint._get_state_db()
    assert db is not None
    assert fingerprint._get_state_db() is db

    def _raise_factory(_path: pathlib.Path, *, readonly: bool) -> _DB:
        raise OSError("boom")

    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = False
    monkeypatch.setattr(state_module, "StateDB", _raise_factory)
    assert fingerprint._get_state_db() is None
    assert fingerprint._state_db_init_attempted


def test_try_manifest_cache_hit_handles_decode_shape_and_stat_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DB:
        def __init__(self, raw: bytes | None, throw: bool = False) -> None:
            self.raw = raw
            self.throw = throw

        def get_raw(self, _key: bytes) -> bytes | None:
            if self.throw:
                raise OSError("db")
            return self.raw

    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: _DB(None, throw=True))
    assert fingerprint._try_manifest_cache_hit("stage") is None

    monkeypatch.setattr(
        fingerprint, "_get_state_db", lambda: _DB(json.dumps({"m": [], "s": {}}).encode())
    )
    assert fingerprint._try_manifest_cache_hit("stage") is None

    monkeypatch.setattr(
        fingerprint,
        "_get_state_db",
        lambda: _DB(json.dumps({"m": {"a": "b"}, "s": {"a.py": [1, 2]}}).encode()),
    )
    assert fingerprint._try_manifest_cache_hit("stage") is None


def test_collect_annotation_names_and_dotted_path_helpers() -> None:
    names: set[str] = set()
    dotted: list[tuple[str, ...]] = []
    expr = fingerprint.ast.parse("A | tuple[B, pkg.Inner] | 'pkg.Other'", mode="eval").body
    fingerprint._collect_annotation_names(expr, names, dotted)
    assert "A" in names
    assert "B" in names
    assert ("pkg", "Inner") in dotted
    assert ("pkg", "Other") in dotted

    attr_expr = fingerprint.ast.parse("pkg.sub.value", mode="eval").body
    assert isinstance(attr_expr, fingerprint.ast.Attribute)
    assert fingerprint._collect_dotted_path(attr_expr) == ("pkg", "sub", "value")

    ns = {"pkg": types.SimpleNamespace(sub=types.SimpleNamespace(value=123))}
    assert fingerprint._resolve_dotted_path(("pkg", "sub", "value"), ns) == 123
    assert fingerprint._resolve_dotted_path(("missing", "x"), ns) is None


def test_process_closure_values_covers_branching(monkeypatch: pytest.MonkeyPatch) -> None:
    hits: list[str] = []
    module = types.ModuleType("helper_mod")

    monkeypatch.setattr(
        fingerprint, "_process_partial_dependency", lambda *args, **kwargs: hits.append("partial")
    )
    monkeypatch.setattr(
        fingerprint, "_process_callable_dependency", lambda *args, **kwargs: hits.append("callable")
    )
    monkeypatch.setattr(
        fingerprint, "_process_module_dependency", lambda *args, **kwargs: hits.append("module")
    )
    monkeypatch.setattr(
        fingerprint,
        "_process_collection_dependency",
        lambda *args, **kwargs: hits.append("collection"),
    )
    monkeypatch.setattr(
        fingerprint, "_process_instance_dependency", lambda *args, **kwargs: hits.append("instance")
    )
    monkeypatch.setattr(
        fingerprint,
        "_hash_unrecognized_closure_value",
        lambda *args, **kwargs: hits.append("other"),
    )
    monkeypatch.setattr(
        fingerprint, "_check_mutable_capture", lambda *args, **kwargs: hits.append("mutable")
    )

    manifest: dict[str, str] = {}
    values = {
        "__dunder": 1,
        "part": functools.partial(_helper_plus_one, value=1),
        "user_func": _helper_plus_one,
        "stdlib_func": len,
        "mod": module,
        "logger": logging.getLogger("t"),
        "primitive": 10,
        "list_v": [1],
        "tuple_v": (1, 2),
        "instance_mut": _helper_mutable_dataclass(1),
        "instance_frozen": _helper_frozen_dataclass(2),
        "other": object(),
    }
    fingerprint._process_closure_values(
        values,
        _helper_plus_one,
        manifest,
        set(),
        skip_dunders=True,
        include_modules=True,
    )

    assert "partial" in hits
    assert "callable" in hits
    assert "module" in hits
    assert "collection" in hits
    assert hits.count("instance") >= 2
    assert "mutable" in hits
    assert "other" in hits
    assert manifest["const:primitive"] == "10"


def test_process_type_hint_origin_and_pydantic_paths() -> None:
    manifest: dict[str, str] = {}
    fingerprint._process_type_hint(type(None), manifest, set())
    fingerprint._process_type_hint(list[_helper_nested_model], manifest, set())
    assert "class:_helper_nested_model" in manifest
    assert "schema:_helper_nested_model" in manifest


def test_process_type_hint_dependencies_fallback_on_get_type_hints_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fn(x: _helper_dependency_class) -> None:
        return None

    monkeypatch.setattr(
        fingerprint.typing, "get_type_hints", lambda _func: (_ for _ in ()).throw(ValueError())
    )
    manifest: dict[str, str] = {}
    fingerprint._process_type_hint_dependencies(_fn, manifest, set())
    assert "class:_helper_dependency_class" in manifest


def test_strip_schema_metadata_handles_list_entries() -> None:
    schema = {
        "anyOf": [{"title": "A", "description": "d", "type": "string"}, "literal"],
        "allOf": [{"title": "B", "description": "d", "type": "integer"}],
    }
    stripped = fingerprint._strip_schema_metadata(schema)
    first_any = stripped["anyOf"][0]
    assert isinstance(first_any, dict)
    assert "title" not in first_any
    assert "description" not in first_any


class _HelperModelDump:
    def __init__(self, value: int) -> None:
        self.value = value

    def model_dump(self) -> dict[str, int]:
        return {"value": self.value}


def test_serialize_value_for_hash_model_dump_and_dict_paths() -> None:
    as_model = fingerprint._serialize_value_for_hash(_HelperModelDump(1))
    as_list = fingerprint._serialize_value_for_hash([_HelperModelDump(2)])
    as_dict = fingerprint._serialize_value_for_hash({"a": 1, "b": 2})
    assert "value" in as_model
    assert "value" in as_list
    assert '"a"' in as_dict


def test_process_collection_dependency_sequence_and_set_ordering() -> None:
    manifest: dict[str, str] = {}
    fingerprint._process_collection_dependency("seq", [_helper_plus_one, len], manifest, set())
    fingerprint._process_collection_dependency("setv", {_helper_plus_one, len}, manifest, set())
    assert any(key.startswith("func:seq[") for key in manifest)
    assert any(key.startswith("func:setv[") for key in manifest)


def test_hash_function_ast_active_source_map_and_exception_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint._hash_function_ast_cache = fingerprint.weakref.WeakKeyDictionary()  # type: ignore[attr-defined]
    fingerprint._active_source_map = {}
    monkeypatch.setattr(fingerprint, "_get_func_source_info", lambda _func: ("f.py", 1, 2, 3))
    monkeypatch.setattr(fingerprint, "_compute_function_hash", lambda _func: "h")
    monkeypatch.setattr(fingerprint, "_should_skip_persistent_cache", lambda _func: True)
    assert fingerprint.hash_function_ast(_helper_plus_one) == "h"
    assert "f.py" in fingerprint._active_source_map

    class _DB:
        def get_ast_hash(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("db")

    fingerprint._hash_function_ast_cache = fingerprint.weakref.WeakKeyDictionary()  # type: ignore[attr-defined]
    monkeypatch.setattr(fingerprint, "_should_skip_persistent_cache", lambda _func: False)
    monkeypatch.setattr(fingerprint, "_get_qualname_for_cache", lambda _func: "q")
    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: _DB())
    assert fingerprint.hash_function_ast(_helper_identity) == "h"
    assert fingerprint.hash_function_ast(len)


def test_compute_function_hash_warns_for_callable_without_source(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _HelperCallableNoCode:
        def __call__(self) -> int:
            return 1

    value = _HelperCallableNoCode()
    monkeypatch.setattr(
        fingerprint.inspect, "getsource", lambda _func: (_ for _ in ()).throw(TypeError())
    )
    with caplog.at_level("WARNING"):
        result = fingerprint._compute_function_hash(value)
    assert result
    assert "non-deterministic fallback" in caplog.text


def test_is_user_code_impl_namespace_and_stdlib_paths() -> None:
    ns_module = types.ModuleType("ns_user")
    ns_module.__path__ = ["/tmp/localpkg"]  # type: ignore[attr-defined]
    assert fingerprint._is_user_code_impl(ns_module)
    assert not fingerprint.is_user_code(json)
    assert fingerprint._get_module(object()) is None


def test_close_and_flush_pending_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DB:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    closed_db = _DB()
    fingerprint._state_db = closed_db
    fingerprint._close_state_db()
    assert closed_db.closed
    assert fingerprint._state_db is None

    calls: list[str] = []
    monkeypatch.setattr(fingerprint, "flush_ast_hash_cache", lambda: calls.append("ast"))
    monkeypatch.setattr(fingerprint, "flush_manifest_cache", lambda: calls.append("manifest"))
    fingerprint._flush_pending_caches()
    assert calls == ["ast", "manifest"]


@pytest.mark.usefixtures("set_project_root")
def test_get_func_source_info_success_and_builtin_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        project, "get_project_root", lambda: pathlib.Path(__file__).resolve().parents[4]
    )
    info = fingerprint._get_func_source_info(_helper_plus_one)
    assert info is not None
    assert info[0].endswith("test_fingerprint.py")
    assert fingerprint._get_func_source_info(len) is None


def test_get_state_db_skips_reinit_after_failed_attempt() -> None:
    fingerprint._state_db = None
    fingerprint._state_db_init_attempted = True
    assert fingerprint._get_state_db() is None


def test_try_manifest_cache_hit_db_none_and_missing_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: None)
    assert fingerprint._try_manifest_cache_hit("stage") is None

    class _DB:
        def get_raw(self, _key: bytes) -> None:
            return None

    monkeypatch.setattr(fingerprint, "_get_state_db", lambda: _DB())
    assert fingerprint._try_manifest_cache_hit("stage") is None


def test_process_class_body_dependencies_handles_dotted_refs() -> None:
    class _Pkg:
        class Inner(BaseModel):
            value: int

    pkg = _Pkg()

    class _Dotted:
        item: pkg.Inner = None  # pyright: ignore[reportInvalidTypeForm]

    cast("Any", sys.modules[__name__]).pkg = pkg
    manifest: dict[str, str] = {}
    fingerprint._process_class_body_dependencies(_Dotted, manifest, set())
    assert "class:Inner" in manifest
    assert "schema:Inner" in manifest


def test_is_unsafe_fingerprinting_enabled_config_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIVOT_UNSAFE_FINGERPRINTING", raising=False)

    class _Core:
        unsafe_fingerprinting = True

    class _Config:
        core = _Core()

    monkeypatch.setattr(config_io, "get_merged_config", lambda: _Config())
    assert fingerprint._is_unsafe_fingerprinting_enabled()

    monkeypatch.setattr(
        config_io, "get_merged_config", lambda: (_ for _ in ()).throw(AttributeError())
    )
    assert not fingerprint._is_unsafe_fingerprinting_enabled()

    monkeypatch.setattr(
        config_io, "get_merged_config", lambda: (_ for _ in ()).throw(RuntimeError())
    )
    assert not fingerprint._is_unsafe_fingerprinting_enabled()


def test_unrecognized_closure_value_repr_exception_and_large_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("nope")

    class _LargeRepr:
        def __repr__(self) -> str:
            return "x" * (fingerprint._REPR_SIZE_LIMIT + 1)

    hits: list[str] = []
    monkeypatch.setattr(
        fingerprint, "_check_mutable_capture", lambda *args, **kwargs: hits.append("hit")
    )
    fingerprint._hash_unrecognized_closure_value("a", _BadRepr(), {}, "stage")
    fingerprint._hash_unrecognized_closure_value("b", _LargeRepr(), {}, "stage")
    assert len(hits) == 2


def test_process_partial_dependency_skips_non_user_underlying() -> None:
    manifest: dict[str, str] = {}
    fingerprint._process_partial_dependency("m", functools.partial(len, [1, 2]), manifest, set())
    assert "func:m.func" not in manifest


def test_process_instance_dependency_and_user_class_instance_checks() -> None:
    manifest: dict[str, str] = {}
    inst = _helper_mutable_dataclass(1)
    assert fingerprint._is_user_class_instance(inst)
    assert not fingerprint._is_user_class_instance(1)
    fingerprint._process_instance_dependency("inst", inst, manifest, set())
    assert "class:inst.__class__" in manifest


def test_resolve_annotations_individually_covers_type_and_non_string_paths() -> None:
    def _fn(a: int) -> None:
        return None

    _fn.__annotations__["b"] = 123

    resolved = fingerprint._resolve_annotations_individually(_fn)
    assert resolved["a"] is int
    assert resolved["b"] == 123


def test_process_type_hint_dependencies_handles_non_weakref_function() -> None:
    manifest: dict[str, str] = {}
    fingerprint._process_type_hint_dependencies(len, manifest, set())
    assert manifest == {}


def test_process_type_hint_generic_origin_tracks_user_class() -> None:
    class _GenericBox[T]:
        pass

    manifest: dict[str, str] = {}
    fingerprint._process_type_hint(_GenericBox[int], manifest, set())
    assert "class:_GenericBox" in manifest


def test_serialize_value_for_hash_list_non_model_item() -> None:
    assert fingerprint._serialize_value_for_hash([1, 2]) == "[1, 2]"


def test_is_primitive_collection_circular_dict() -> None:
    d: dict[str, Any] = {}
    d["self"] = d
    assert not fingerprint._is_primitive_collection(d)


def test_process_module_dependency_additional_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    stdlib_module = json
    manifest: dict[str, str] = {}
    fingerprint._process_module_dependency("json", stdlib_module, _helper_identity, manifest, set())
    assert manifest == {}

    module = types.ModuleType("helper_mod_two")
    cast("Any", module).PRIMS = {"a": [1, 2]}
    monkeypatch.setattr(
        ast_utils,
        "extract_module_attr_usage",
        lambda _func: [("other", "x"), ("helper_mod_two", "PRIMS")],
    )
    fingerprint._process_module_dependency(
        "helper_mod_two", module, _helper_identity, manifest, set()
    )
    assert "mod:helper_mod_two.PRIMS" in manifest


def test_get_qualname_for_cache_non_lambda_and_no_code_path() -> None:
    assert fingerprint._get_qualname_for_cache(_helper_plus_one) == _helper_plus_one.__qualname__

    class _CallableNoCode:
        def __init__(self) -> None:
            self.__qualname__ = "<lambda>"

        def __call__(self) -> int:
            return 1

    value = _CallableNoCode()
    assert fingerprint._get_qualname_for_cache(value) == "<lambda>"


def test_hash_function_ast_handles_non_weakref_objects() -> None:
    result = fingerprint.hash_function_ast(max)
    assert isinstance(result, str)


def test_normalize_ast_inserts_pass_for_docstring_only_function() -> None:
    tree = fingerprint.ast.parse(
        """
def only_doc():
    \"\"\"doc\"\"\"
"""
    )
    node = tree.body[0]
    normalized = fingerprint._normalize_ast(node)
    assert isinstance(normalized, fingerprint.ast.FunctionDef)
    assert isinstance(normalized.body[0], fingerprint.ast.Pass)
