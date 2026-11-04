"""Property-based tests of code fingerprints and the run-cache key.

Stage bodies are generated expression trees over the stage's own argument and integer
literals. They read no module-level value, so these properties hold whichever module-level
types the fingerprinting policy allows.

- The same source gives the same manifest in this process and in fresh processes with
  different hash seeds.
- Formatting that leaves the syntax tree unchanged (spacing, redundant parentheses, comments,
  blank lines, docstrings) leaves the manifest unchanged.
- Changing one token of the body (a literal, the argument, an operator) changes the manifest.
- A ``params.yaml`` override changes a stage's run-cache key exactly when it changes the
  stage's effective params, and never changes another stage's key.
- No dependency, output field or stage name, each kind of manifest entry included, is read as
  a module a loader covers or as a by-name entry, and a lock keeps every entry but the stage's
  `byname:` ones.
- No dependency, output field or stage named like a kind of code entry marks a module as one
  whose code the stage used.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import uuid
from typing import TYPE_CHECKING

import hypothesis
import pydantic
import pytest
from hypothesis import strategies as st

from pivot import fingerprint, loaded_source, loaders, parameters, run_history, stage_def
from pivot.types import DepEntry

if TYPE_CHECKING:
    from collections.abc import Callable

_SETTINGS = hypothesis.settings(max_examples=60, deadline=None, derandomize=True, database=None)
_OPERATORS = ("+", "-", "*")


# =============================================================================
# Generated stage bodies
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _Lit:
    value: int


@dataclasses.dataclass(frozen=True)
class _Arg:
    pass


@dataclasses.dataclass(frozen=True)
class _Op:
    op: str
    left: _Expr
    right: _Expr


_Expr = _Lit | _Arg | _Op

_exprs: st.SearchStrategy[_Expr] = st.recursive(
    st.builds(_Lit, st.integers(0, 50)) | st.just(_Arg()),
    lambda children: st.builds(_Op, st.sampled_from(_OPERATORS), children, children),
    max_leaves=6,
)


@dataclasses.dataclass(frozen=True)
class _Style:
    """Formatting choices that never change the parsed syntax tree."""

    space: str
    extra_parens: bool
    comment_lines: int
    trailing_comment: bool
    blank_lines: int
    docstring: str | None


_styles = st.builds(
    _Style,
    space=st.sampled_from(["", " ", "  "]),
    extra_parens=st.booleans(),
    comment_lines=st.integers(0, 2),
    trailing_comment=st.booleans(),
    blank_lines=st.integers(0, 2),
    docstring=st.none() | st.text(alphabet="abc xyz", min_size=1, max_size=12),
)
_PLAIN = _Style(
    space=" ",
    extra_parens=False,
    comment_lines=0,
    trailing_comment=False,
    blank_lines=0,
    docstring=None,
)


def _render_expr(expr: _Expr, style: _Style) -> str:
    match expr:
        case _Lit(value):
            text = str(value)
        case _Arg():
            text = "x"
        case _Op(op, left, right):
            text = f"({_render_expr(left, style)}{style.space}{op}{style.space}"
            text += f"{_render_expr(right, style)})"
    return f"({text})" if style.extra_parens else text


def _render_module(expr: _Expr, style: _Style) -> str:
    lines = ["def stage(x):"]
    if style.docstring is not None:
        lines.append(f'    """{style.docstring}"""')
    lines.extend(f"    # comment {i}" for i in range(style.comment_lines))
    lines.extend("" for _ in range(style.blank_lines))
    trailing = "  # trailing" if style.trailing_comment else ""
    lines.append(f"    return {_render_expr(expr, style)}{trailing}")
    return "\n".join(lines) + "\n"


def _one_token_changes(expr: _Expr) -> list[_Expr]:
    """Every tree that differs from ``expr`` in exactly one leaf or operator."""
    match expr:
        case _Lit(value):
            return [_Lit(value + 1), _Arg()]
        case _Arg():
            return [_Lit(0)]
        case _Op(op, left, right):
            changes: list[_Expr] = [_Op(other, left, right) for other in _OPERATORS if other != op]
            changes.extend(_Op(op, new, right) for new in _one_token_changes(left))
            changes.extend(_Op(op, left, new) for new in _one_token_changes(right))
            return changes


def _load_stage(directory: pathlib.Path, source: str) -> tuple[str, Callable[..., object]]:
    """Import ``source`` as a uniquely named module and return (module name, its stage)."""
    module_name = f"_fp_prop_{uuid.uuid4().hex}"
    path = directory / f"{module_name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    stage = getattr(module, "stage")  # noqa: B009 - generated module, attribute known by contract
    return module_name, stage


def _manifest(directory: pathlib.Path, source: str) -> dict[str, str]:
    _, stage = _load_stage(directory, source)
    return fingerprint.get_stage_fingerprint(stage)


# =============================================================================
# Fingerprint manifests
# =============================================================================


@_SETTINGS
@hypothesis.given(_exprs, _styles)
def test_formatting_that_keeps_the_syntax_tree_keeps_the_manifest(
    expr: _Expr, style: _Style
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        plain = _manifest(directory, _render_module(expr, _PLAIN))
        restyled = _manifest(directory, _render_module(expr, style))
    assert restyled == plain, "whitespace, parentheses, comments and docstrings are not code"


@_SETTINGS
@hypothesis.given(_exprs, st.data())
def test_changing_one_token_of_the_body_changes_the_manifest(
    expr: _Expr, data: st.DataObject
) -> None:
    changed = data.draw(st.sampled_from(_one_token_changes(expr)))
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        before = _manifest(directory, _render_module(expr, _PLAIN))
        after = _manifest(directory, _render_module(changed, _PLAIN))
    assert after != before, (
        f"{_render_expr(expr, _PLAIN)} -> {_render_expr(changed, _PLAIN)} must change the manifest"
    )


_FRESH_PROCESS_SCRIPT = """\
import importlib.util, json, pathlib, sys
from pivot import fingerprint
manifests = {}
for path in sorted(pathlib.Path(sys.argv[1]).glob("_fp_prop_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    manifests[path.stem] = fingerprint.get_stage_fingerprint(module.stage)
print(json.dumps(manifests))
"""


def _manifests_in_fresh_process(
    modules: pathlib.Path, project: pathlib.Path, hash_seed: str
) -> dict[str, dict[str, str]]:
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_SCRIPT, str(modules)],
        cwd=project,
        env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@hypothesis.settings(_SETTINGS, max_examples=3)
@hypothesis.given(st.lists(st.tuples(_exprs, _styles), min_size=1, max_size=8))
def test_same_source_gives_same_manifest_in_fresh_processes(
    programs: list[tuple[_Expr, _Style]],
) -> None:
    """Workers fingerprint in their own process; the hash seed must not leak into manifests."""
    with tempfile.TemporaryDirectory() as tmp:
        modules = pathlib.Path(tmp) / "modules"
        project = pathlib.Path(tmp) / "project"
        modules.mkdir()
        (project / ".pivot").mkdir(parents=True)
        here = dict(_load_stage(modules, _render_module(expr, style)) for expr, style in programs)
        expected = {name: fingerprint.get_stage_fingerprint(stage) for name, stage in here.items()}

        first = _manifests_in_fresh_process(modules, project, "0")
        second = _manifests_in_fresh_process(modules, project, "4242")

    assert first == expected, "a fresh process must compute the manifest this process computed"
    assert second == first, "the manifest must not depend on PYTHONHASHSEED"


# =============================================================================
# Run-cache key from params.yaml overrides
# =============================================================================


class _TrainParams(stage_def.StageParams):
    factor: int = 2
    rate: float = 0.5
    label: str = "a"
    layers: list[int] = pydantic.Field(default_factory=lambda: [1, 2])


_STAGE = "demo/train"
_OTHER_STAGE = "demo/evaluate"
_MANIFEST = {"self:stage": "0123456789abcdef"}
_DEPS = [DepEntry(producer="demo/raw", key=None, hash="fedcba9876543210")]
_OUT_SPECS = [("demo/train", True)]

_overrides = st.fixed_dictionaries(
    {},
    optional={
        "factor": st.integers(-3, 3),
        "rate": st.integers(-2, 2) | st.floats(-2, 2, allow_nan=False, allow_infinity=False),
        "label": st.sampled_from(["a", "b", ""]),
        "layers": st.lists(st.integers(0, 3), max_size=3),
    },
)


def _effective(stage_name: str, overrides: parameters.ParamsOverrides) -> dict[str, object]:
    return parameters.get_effective_params(_TrainParams(), stage_name, overrides)


def _run_cache_key(stage_name: str, overrides: parameters.ParamsOverrides) -> str:
    return run_history.compute_input_hash(
        _MANIFEST, _effective(stage_name, overrides), _DEPS, _OUT_SPECS
    )


def _canonical(params: dict[str, object]) -> str:
    """Params as the run-cache key serializes them (run_history.compute_input_hash)."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


@_SETTINGS
@hypothesis.given(_overrides, _overrides)
@hypothesis.example({"rate": 0.0}, {"rate": -0.0})
def test_override_changes_run_cache_key_exactly_when_effective_params_change(
    first: dict[str, object], second: dict[str, object]
) -> None:
    """Params compare as the key serializes them: 0.0 == -0.0 in Python, but not in JSON."""
    first_overrides = {_STAGE: first}
    second_overrides = {_STAGE: second}

    same_params = _canonical(_effective(_STAGE, first_overrides)) == _canonical(
        _effective(_STAGE, second_overrides)
    )
    same_key = _run_cache_key(_STAGE, first_overrides) == _run_cache_key(_STAGE, second_overrides)

    assert same_key == same_params, (
        "the run-cache key must change for a changed param value and only then"
    )


@_SETTINGS
@hypothesis.given(_overrides)
def test_override_for_one_stage_leaves_other_stages_run_cache_keys(
    overrides: dict[str, object],
) -> None:
    baseline = _run_cache_key(_OTHER_STAGE, {})
    assert _run_cache_key(_OTHER_STAGE, {_STAGE: overrides}) == baseline


@_SETTINGS
@hypothesis.given(_overrides)
def test_base_stage_override_reaches_variants_unless_the_variant_overrides_it(
    overrides: dict[str, object],
) -> None:
    variant = f"{_STAGE}@v1"
    assert _effective(variant, {_STAGE: overrides}) == _effective(_STAGE, {_STAGE: overrides})
    assert _effective(variant, {_STAGE: overrides, variant: {"label": "variant"}})["label"] == (
        "variant"
    )


# =============================================================================
# Loader entries of a stage manifest
# =============================================================================

# Each kind of manifest entry, and the words of a loader's prefix and an output's tag: every one
# is a valid parameter, field and stage name, and must stay that name.
_ENTRY_KINDS = (
    *("self", "func", "class", "const", "default", "wire", "partial", "ext", "byname"),
    *("module", "loader", "schema", "builtin", "file", "dep", "out", "tag", "path", "paths"),
)


@pytest.mark.parametrize("keyed", [False, True], ids=["one-output", "fields"])
@pytest.mark.parametrize("kind", _ENTRY_KINDS)
@hypothesis.settings(_SETTINGS, max_examples=10)
@hypothesis.given(st.data())
def test_no_dependency_output_or_stage_name_is_read_as_a_module_or_by_name_entry(
    kind: str, keyed: bool, data: st.DataObject
) -> None:
    """Stage `<kind>/<kind>` has a dependency named each kind of entry, alone and as a list's
    item, and one output or a field named each kind. Its manifest is built as the registry
    builds it, from a framework loader whose own manifest names no module, plus a `module:` entry
    for each loader drawn to cover one: only those and the stage's own `byname:` entry name
    modules, and a lock keeps every entry but that one."""
    stage = f"{kind}/{kind}"
    outputs = {f"{stage}:{field}" for field in _ENTRY_KINDS} if keyed else {stage}
    prefixes = [f"dep:{name}:" for name in _ENTRY_KINDS]
    prefixes += [f"dep:{name}[0]:" for name in _ENTRY_KINDS]
    prefixes += [f"out:{key}:" for key in sorted(outputs)]
    manifest = {"self:use": "1", "byname:json": ""}
    for prefix in prefixes:
        fingerprint.merge_loader_fingerprint(manifest, prefix, loaders.YAML())
    manifest.update({f"out:{key}:tag": "data" for key in outputs})
    covering = data.draw(st.lists(st.sampled_from(prefixes), unique=True))
    covered = {f"covered{index}" for index in range(len(covering))}
    manifest.update({f"{prefix}module:covered{i}": "" for i, prefix in enumerate(covering)})

    assert fingerprint.covered_user_modules(manifest, outputs) == covered
    assert fingerprint.by_name_modules(manifest, outputs) == {"json", *covered}
    assert fingerprint.without_by_name(manifest) == {
        key: value for key, value in manifest.items() if key != "byname:json"
    }


# The kinds of entry that hash a module's code or values (`loaded_source._CODE_KINDS`).
_CODE_KINDS = ("func", "class", "const", "default", "schema", "partial", "loader")


_DIGEST = "0123456789abcdef"


@pytest.mark.parametrize("keyed", [False, True], ids=["one-output", "fields"])
@pytest.mark.parametrize("kind", _CODE_KINDS)
@hypothesis.settings(_SETTINGS, max_examples=10)
@hypothesis.given(st.data())
def test_no_dependency_output_field_or_stage_name_marks_a_module_used(
    kind: str, keyed: bool, data: st.DataObject
) -> None:
    """Stage `<kind>/<kind>` (function `<kind>`) has a dependency named each code kind, alone and
    as a list's item, wired as the registry wires it, and one output or a field named each; each
    of its loaders, and the stage itself, holds a loader (`held:<digest>:`). Every word of every
    key is a user module the worker knows. Only the entries drawn to hash a module's code, the
    stage's own or a loader's, mark it as one the stage used (`loaded_source._manifest_modules`)."""
    stage = f"{kind}/{kind}"
    outputs = {f"{stage}:{field}" for field in _CODE_KINDS} if keyed else {stage}
    prefixes = [f"dep:{name}:" for name in _CODE_KINDS]
    prefixes += [f"dep:{name}[0]:" for name in _CODE_KINDS]
    prefixes += [f"out:{key}:" for key in sorted(outputs)]
    prefixes += [f"{prefix}held:{_DIGEST}:" for prefix in ["", *prefixes]]
    manifest = {f"self:{kind}": "1", f"module:{kind}": "", f"ext:{kind}.value": f"{kind}.value"}
    for name in _CODE_KINDS:
        manifest.update({f"wire:{name}": "x", f"wire:{name}:path": "x", f"wire:{name}:paths": "[]"})
    for prefix in prefixes:
        fingerprint.merge_loader_fingerprint(manifest, prefix, loaders.YAML())
    manifest.update({f"out:{key}:tag": "data" for key in outputs})
    using = data.draw(st.lists(st.sampled_from(["", *prefixes]), unique=True))
    member = "Reads:load" if kind == "loader" else "value"
    manifest.update({f"{prefix}{kind}:used{i}.{member}": "h" for i, prefix in enumerate(using)})
    used = {f"used{index}" for index in range(len(using))}
    known = {*_ENTRY_KINDS, "load", "save", "empty", "config", "data", "x", "Reads", "held"}
    known.update({_DIGEST, *used})

    assert loaded_source._manifest_modules(manifest, outputs, known) == used
