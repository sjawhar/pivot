import importlib.util
import pathlib
import sys
import types
from typing import Any

import pytest

from pivot import exceptions, fingerprint, project
from pivot.storage import state


def test_schema_41_cannot_hide_a_previously_dropped_descriptor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid old optimization manifest must not bypass the new descriptor admission rule."""
    (tmp_path / ".pivot").mkdir()
    source = tmp_path / "schema_stage.py"
    source.write_text(
        "import functools\nimport types\n"
        + "class Config:\n    value = types.DynamicClassAttribute(lambda self: 1)\n"
        + "HELD = functools.partial(str, Config())\n"
        + "def use():\n    return HELD()\n"
    )
    spec = importlib.util.spec_from_file_location("schema_stage", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "schema_stage", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: tmp_path / ".pivot")
    current = fingerprint._user_member_value

    def old_member(value: Any) -> Any:
        if type(value) is types.DynamicClassAttribute:
            return fingerprint._UNREAD
        return current(value)

    with state.StateDB(tmp_path / ".pivot") as db:
        monkeypatch.setattr(fingerprint, "_get_state_db", lambda: db)
        with monkeypatch.context() as old:
            old.setattr(fingerprint, "_CACHE_SCHEMA_VERSION", 41)
            old.setattr(fingerprint, "_user_member_value", old_member)
            fingerprint.get_stage_fingerprint_cached("p/use", module.use)
            fingerprint.flush_manifest_cache()
            assert fingerprint._try_manifest_cache_hit("p/use", module.use) is not None
        with pytest.raises(exceptions.StageDefinitionError, match=r"Config\.value"):
            fingerprint.get_stage_fingerprint_cached("p/use", module.use)


def test_schema_42_cannot_hide_a_previously_dropped_classmethod(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persisted old manifest must gain the held method and its bound class after upgrading."""
    (tmp_path / ".pivot").mkdir()
    source = tmp_path / "schema_bound.py"
    source.write_text(
        "class Other:\n    @classmethod\n    def cm(cls):\n        return 1\n"
        + "class Holder:\n    fn = Other.cm\n"
        + "def use():\n    return Holder().fn()\n"
    )
    spec = importlib.util.spec_from_file_location("schema_bound", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "schema_bound", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: tmp_path / ".pivot")
    current = fingerprint._user_member_value

    def old_member(value: Any) -> Any:
        if type(value) is types.MethodType:
            return fingerprint._UNREAD
        return current(value)

    with state.StateDB(tmp_path / ".pivot") as db:
        monkeypatch.setattr(fingerprint, "_get_state_db", lambda: db)
        with monkeypatch.context() as old:
            old.setattr(fingerprint, "_CACHE_SCHEMA_VERSION", 42)
            old.setattr(fingerprint, "_user_member_value", old_member)
            stale = fingerprint.get_stage_fingerprint_cached("p/use", module.use)
            fingerprint.flush_manifest_cache()
            assert fingerprint._try_manifest_cache_hit("p/use", module.use) == stale
            assert "func:schema_bound.Other.cm" not in stale

        manifest = fingerprint.get_stage_fingerprint_cached("p/use", module.use)

    assert "func:schema_bound.Other.cm" in manifest, manifest
    assert "class:schema_bound.Other" in manifest, manifest
