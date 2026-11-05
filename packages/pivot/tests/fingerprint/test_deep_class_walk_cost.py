import sys
import types
from typing import Any

import pytest

from pivot import fingerprint


def test_a_standalone_stage_walk_reuses_each_base_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a surrounding engine pass, each ancestor must not restart its whole base walk."""
    module = types.ModuleType("deep_walk_cost")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(
        'DEEP = type("C0", (), {"marker": "A"})\n'
        + "for index in range(1, 60):\n"
        + '    DEEP = type(f"C{index}", (DEEP,), {})\n'
        + "def use(text):\n"
        + "    return text + DEEP().marker\n",
        vars(module),
    )
    computed = list[type[Any]]()
    original = fingerprint._computed_class_identity

    def counting(cls: type[Any]) -> str:
        computed.append(cls)
        return original(cls)

    monkeypatch.setattr(fingerprint, "_computed_class_identity", counting)
    first = fingerprint.get_stage_fingerprint(module.use)
    assert len(computed) < 3 * 60, len(computed)
    module.DEEP.__mro__[-2].marker = "B"
    second = fingerprint.get_stage_fingerprint(module.use)
    assert second != first, "A new standalone call must see the base's changed state"
