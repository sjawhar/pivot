import importlib.util
import pathlib
import sys
import types

import pytest

from pivot import fingerprint, project
from pivot.storage import state


def _load_module(
    root: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> types.ModuleType:
    (root / ".pivot").mkdir()
    path = root / "bound_receivers.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("bound_receivers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "bound_receivers", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(project, "_project_root_cache", root)
    return module


@pytest.mark.parametrize("wrapped", [False, True])
def test_a_method_bound_to_its_own_class_tracks_receiver_reads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, wrapped: bool
) -> None:
    source = (
        "import functools\nimport types\n\n"
        + "def cm(cls):\n    return cls.FACTOR\n\n"
        + "@functools.wraps(cm)\ndef wrapped(cls):\n    return cm(cls)\n\n"
        + "class Other:\n    FACTOR = 1\n\n"
        + f"Other.cm = types.MethodType({'wrapped' if wrapped else 'cm'}, Other)\n\n"
        + "def use():\n    return Other().cm()\n"
    )
    module = _load_module(tmp_path, monkeypatch, source)
    first = fingerprint.get_stage_fingerprint(module.use)
    assert module.use() == 1

    monkeypatch.setattr(module.Other, "FACTOR", 2000)

    assert module.use() == 2000
    assert fingerprint.get_stage_fingerprint(module.use) != first


def test_a_class_bound_method_chain_deeper_than_the_recursion_limit_is_fingerprinted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(
        tmp_path,
        monkeypatch,
        "import types\n\n"
        + "class Other:\n    pass\n\n"
        + "def cm(*args):\n    return len(args)\n\n"
        + "class Holder:\n    fn = cm\n\n"
        + "for _ in range(1000):\n    Holder.fn = types.MethodType(Holder.fn, Other)\n\n"
        + "def use():\n    return Holder().fn()\n",
    )

    first = fingerprint.get_stage_fingerprint(module.use)

    assert module.use() == 1000
    assert first == fingerprint.get_stage_fingerprint(module.use)


def test_a_cached_method_detects_a_rebound_receiver_without_a_source_edit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        "import types\n\n"
        + "def cm(cls):\n    return cls.VALUE\n\n"
        + "class First:\n    VALUE = 1\n\n"
        + "class Second:\n    VALUE = 2000\n\n"
        + "callback = types.MethodType(cm, First)\n\n"
        + "def use():\n    return callback()\n"
    )
    module = _load_module(tmp_path, monkeypatch, source)
    monkeypatch.setattr(fingerprint, "_producing_state_dir", lambda: tmp_path / ".pivot")
    with state.StateDB(tmp_path / ".pivot") as db:
        monkeypatch.setattr(fingerprint, "_get_state_db", lambda: db)
        first = fingerprint.get_stage_fingerprint_cached("p/use", module.use)
        fingerprint.flush_manifest_cache()
        assert module.use() == 1
        assert fingerprint._try_manifest_cache_hit("p/use", module.use) == first

        monkeypatch.setattr(module, "callback", types.MethodType(module.cm, module.Second))

        assert (tmp_path / "bound_receivers.py").read_text() == source
        assert module.use() == 2000
        assert fingerprint._try_manifest_cache_hit("p/use", module.use) is None
        assert fingerprint.get_stage_fingerprint_cached("p/use", module.use) != first
