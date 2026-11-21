import ast
import importlib
import inspect
import pathlib
import sys
import textwrap
import types
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from pivot import fingerprint, loaded_source


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


def _shared_helper(text: str) -> str:
    return text.upper()


def _first_stage(text: str) -> str:
    return _shared_helper(text)


def _second_stage(text: str) -> str:
    return _shared_helper(text) + "!"


def _make_suffixer(suffix: str) -> Callable[[str], str]:
    def suffixer(text: str) -> str:
        return text + suffix

    return suffixer


_SUFFIX_A = _make_suffixer("a")
_SUFFIX_B = _make_suffixer("b")


def _stage_with_a(text: str) -> str:
    return _SUFFIX_A(text)


def _stage_with_b(text: str) -> str:
    return _SUFFIX_B(text)


@pytest.mark.parametrize(
    ("stages", "helpers"),
    [
        pytest.param((_first_stage, _second_stage), (_shared_helper,), id="one-helper"),
        pytest.param((_stage_with_a, _stage_with_b), (_SUFFIX_A, _SUFFIX_B), id="one-factory"),
    ],
)
def test_two_stages_of_one_pass_that_reach_one_helper_read_its_source_once(
    monkeypatch: pytest.MonkeyPatch,
    stages: tuple[Callable[[str], str], Callable[[str], str]],
    helpers: tuple[Callable[[str], str], ...],
) -> None:
    """The stages of a pipeline reach the same helpers, or closures one factory made, which share
    a code object. Within one pass that source is read and parsed once, not once for each stage
    whose walk reaches it; the tree each helper gets is its own source's, and each manifest is the
    one its stage gets alone."""
    alone = [fingerprint.get_stage_fingerprint(stage) for stage in stages]
    own = [ast.dump(ast.parse(textwrap.dedent(inspect.getsource(helper)))) for helper in helpers]
    reads = list[object]()
    original = loaded_source.getsource

    def counting(obj: Callable[..., Any] | types.CodeType) -> str:
        reads.append(obj)
        return original(obj)

    monkeypatch.setattr(loaded_source, "getsource", counting)
    with fingerprint.fingerprint_pass():
        together = [fingerprint.get_stage_fingerprint(stage) for stage in stages]
        trees = [fingerprint._source_tree(helper) for helper in helpers]

    code = helpers[0].__code__
    helper_reads = [obj for obj in reads if obj is code or any(obj is h for h in helpers)]
    assert len(helper_reads) == 1, reads
    assert [None if tree is None else ast.dump(tree) for tree in trees] == own
    assert together == alone


class _ExposesCode:
    SUFFIX: ClassVar[str] = "x"

    def apply(self, text: str) -> str:
        return text + self.SUFFIX

    __code__: ClassVar[types.CodeType] = apply.__code__


def _uses_exposed_code(text: str) -> str:
    return _ExposesCode().apply(text)


def test_a_class_that_sets_code_to_its_method_s_keeps_the_method_s_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A class whose `__code__` is its method's code is parsed as the class, apart from the
    method. Given the class's tree, the method would record none of the attributes it reads
    through `self`, and a change to one would not change the manifest."""
    with fingerprint.fingerprint_pass():
        before = fingerprint.get_stage_fingerprint(_uses_exposed_code)
    monkeypatch.setattr(_ExposesCode, "SUFFIX", "y")
    with fingerprint.fingerprint_pass():
        after = fingerprint.get_stage_fingerprint(_uses_exposed_code)

    assert after != before


def test_a_symlink_retargeted_within_a_pass_gives_the_new_file_s_path(
    set_project_root: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A function's source file is resolved on each use, so a module reloaded through a symlink
    retargeted within one pass is recorded at its new target, not at the one the pass saw first."""
    (set_project_root / "first.py").write_text("def helper() -> int:\n    return 1\n")
    (set_project_root / "second.py").write_text("def helper() -> int:\n    return 2\n")
    link = set_project_root / "retargeted_module.py"
    link.symlink_to(set_project_root / "first.py")
    monkeypatch.syspath_prepend(str(set_project_root))
    try:
        module = importlib.import_module("retargeted_module")
        with fingerprint.fingerprint_pass():
            first = fingerprint._get_func_source_info(module.helper)
            link.unlink()
            link.symlink_to(set_project_root / "second.py")
            module = importlib.reload(module)
            second = fingerprint._get_func_source_info(module.helper)
    finally:
        sys.modules.pop("retargeted_module", None)

    assert first is not None and first[0] == "first.py", first
    assert second is not None and second[0] == "second.py", second
