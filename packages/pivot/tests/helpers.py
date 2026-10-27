"""Test helpers for building pipelines with the compositional API."""

# pyright: reportMissingImports=false, reportImplicitRelativeImport=false

from __future__ import annotations

import inspect
import os
import pathlib
import re
import stat
import textwrap
import typing
from typing import TYPE_CHECKING, Any

import anyio

from pivot import compose, exceptions, project, stage_def
from pivot import names as stage_names
from pivot.storage import state, store

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pytest

    from pivot.registry import PipelineLike

DepValue = compose.ArtifactHandle | str | list[compose.ArtifactHandle]


async def wait_for_socket(socket_path: pathlib.Path, timeout: float = 5.0) -> None:
    """Wait for Unix socket to be created and connectable.

    Args:
        socket_path: Path to the Unix socket file.
        timeout: Maximum time to wait in seconds.

    Raises:
        TimeoutError: If socket is not created within timeout.
    """
    deadline = anyio.current_time() + timeout
    while anyio.current_time() < deadline:
        if socket_path.exists():
            try:
                mode = socket_path.stat().st_mode
                if stat.S_ISSOCK(mode):
                    connection = await anyio.connect_unix(str(socket_path))
                    await connection.aclose()
                    return
            except (OSError, anyio.BrokenResourceError, anyio.ClosedResourceError):
                pass
        await anyio.sleep(0.05)
    msg = f"Socket {socket_path} not created and connectable within {timeout}s"
    raise TimeoutError(msg)


# Default pipeline for register_test_stage(). The conftest fixture
# bind_test_pipeline_helper sets it to the test_pipeline fixture's Pipeline for
# every test that uses test_pipeline (directly or via mock_discovery etc.).
_test_pipeline: PipelineLike | None = None


def get_test_pipeline() -> PipelineLike:
    """Get the current test pipeline.

    Raises RuntimeError if no test pipeline is set.
    """
    if _test_pipeline is None:
        raise RuntimeError(
            "No test pipeline available. Use the 'test_pipeline' fixture or call "
            + "set_test_pipeline() before registering stages."
        )
    return _test_pipeline


def set_test_pipeline(pipeline: PipelineLike | None) -> None:
    """Set the module-level test pipeline."""
    global _test_pipeline
    _test_pipeline = pipeline


def _original_func(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the undecorated function behind an @stage wrapper."""
    return inspect.unwrap(func)


def _param_types(func: Callable[..., Any]) -> dict[str, object]:
    """Resolve parameter type hints, unwrapping Annotated[T, ...] to T."""
    hints = typing.get_type_hints(_original_func(func), include_extras=True)
    result = dict[str, object]()
    for param_name, hint in hints.items():
        if param_name == "return":
            continue
        if typing.get_origin(hint) is typing.Annotated:
            hint = typing.get_args(hint)[0]
        result[param_name] = hint
    return result


def _params_class(hint: object) -> type[stage_def.StageParams] | None:
    if isinstance(hint, type) and issubclass(hint, stage_def.StageParams):
        return hint
    return None


def _annotation_params_class(
    func: Callable[..., Any], param: inspect.Parameter
) -> type[stage_def.StageParams] | None:
    """Resolve a parameter's StageParams class without evaluating other annotations.

    String annotations are looked up by name in the function's module globals; a
    name defined only in the generated file (``extra_code``) resolves to None.
    """
    annotation = param.annotation
    if isinstance(annotation, str):
        annotation = getattr(func, "__globals__", {}).get(annotation.strip())
    return _params_class(annotation)


def register_test_stage(
    func: Callable[..., Any],
    name: str | None = None,
    params: type[stage_def.StageParams] | stage_def.StageParams | None = None,
    resources: dict[str, int | str] | None = None,
    *,
    deps: Mapping[str, DepValue] | None = None,
    pipeline: PipelineLike | None = None,
) -> compose.ArtifactHandle:
    """Register a stage for testing and return its output handle.

    The stage is recorded with the compositional API (``with Pipeline(...)`` and
    an ``@stage`` call) and merged into the target pipeline, so it is registered
    as ``<pipeline name>/<name>``. Outputs come from the function's return type;
    pivot chooses their paths (see ``get_output_path``).

    Args:
        func: The stage function, plain or already decorated with ``@stage``.
        name: Stage name (defaults to the function name).
        params: StageParams class or instance. ``None`` instantiates the
            function's StageParams parameter type with its defaults.
        resources: Scheduler resources written into the registered stage info.
        deps: Value for each dependency parameter: an ArtifactHandle (another
            stage's output, e.g. a previous ``register_test_stage`` return value,
            or ``pipeline.input(...)``), a list of handles, or a ``str`` naming an
            input file relative to the project root.
        pipeline: Pipeline to register with. Defaults to the test pipeline set
            via ``set_test_pipeline`` (the ``test_pipeline`` fixture binds it).

    Raises:
        ValidationError: If a stage with the same name is already registered.
        TypeError: If ``params`` is given but the function has no StageParams parameter.
    """
    target = pipeline or get_test_pipeline()
    stage_func = func if hasattr(func, "__wrapped__") else compose.stage(func)
    param_types = _param_types(func)
    params_arg = next(
        (n for n, hint in param_types.items() if _params_class(hint) is not None), None
    )
    if params is not None and params_arg is None:
        raise TypeError(
            f"register_test_stage: {_original_func(func).__name__} has no StageParams "
            + "parameter, but params was given"
        )

    with compose.Pipeline(target.name, root=target.root) as builder:
        kwargs = dict[str, object]()
        for param_name, value in (deps or {}).items():
            if isinstance(value, str):
                hint = param_types.get(param_name)
                input_type = hint if isinstance(hint, type) else None
                if input_type is None:
                    kwargs[param_name] = builder.input(value, path=value)
                else:
                    kwargs[param_name] = builder.input(value, t=input_type, path=value)
            else:
                kwargs[param_name] = value
        if params_arg is not None:
            if isinstance(params, stage_def.StageParams):
                kwargs[params_arg] = params
            elif params is not None:
                kwargs[params_arg] = params()
            else:
                params_cls = _params_class(param_types[params_arg])
                assert params_cls is not None
                kwargs[params_arg] = params_cls()
        handle = typing.cast("compose.ArtifactHandle", stage_func(**kwargs))
        node = handle._source
        if name is not None and isinstance(node, compose._StageNode):
            node.name = name

    stage_name = name or _original_func(func).__name__
    qualified_name = f"{target.name}/{stage_name}"
    if qualified_name in target.list_stages():
        raise exceptions.ValidationError(f"Stage '{qualified_name}' already registered")
    target.include(builder)
    if resources:
        target.get_stage(qualified_name)["resources"] = dict(resources)
    return handle


def get_output_path(
    stage_name: str, key: str | None = None, *, pipeline: PipelineLike | None = None
) -> pathlib.Path:
    """Return the workspace path pivot writes a registered stage's output to.

    Args:
        stage_name: Stage name, bare (``"train"``) or qualified (``"test/train"``).
        key: Output key for TypedDict-returning stages; ``None`` for single outputs.
        pipeline: Pipeline holding the stage. Defaults to the test pipeline.
    """
    target = pipeline or get_test_pipeline()
    qualified_name = stage_names.resolve_stage_name(stage_name, dict.fromkeys(target.list_stages()))
    outs = target.get_stage(qualified_name)["outs"]
    ref = next((out for out in outs if out.identity.key == key), None)
    if ref is None:
        available = [out.identity.key for out in outs]
        raise KeyError(f"Stage '{qualified_name}' has no output {key!r}. Available: {available}")
    workspace = store.WorkspaceStore(
        project_root=project.get_project_root(),
        pipeline_name=target.name,
        input_bindings=target.input_bindings,
    )
    return workspace.resolve_display_path(ref)


def create_pipeline_py(
    stages: Sequence[Callable[..., Any]],
    *,
    path: pathlib.Path | None = None,
    extra_imports: str = "",
    extra_code: str = "",
    names: Mapping[str, str] | None = None,
    args: Mapping[str, Mapping[str, str]] | None = None,
) -> pathlib.Path:
    """Create a pipeline.py file that defines the given stages in a ``Pipeline("test")``.

    This is useful for CLI tests that use runner.isolated_filesystem()
    and need discover_pipeline() to find stages.

    The file holds each stage function's source, wraps it in ``@stage`` if needed,
    and calls the stages in order inside ``with Pipeline("test") as pipeline:``,
    binding each call's handle to ``<stage name>_out`` so later stages can use it.

    Args:
        stages: Stage functions to include, in call order. Their source code
            is copied into the generated file.
        path: Directory to create pipeline.py in. Defaults to current directory.
        extra_imports: Additional import statements to include at the top.
        extra_code: Additional code to include (e.g., StageParams classes).
        names: Mapping of function names to stage names; the copied function
            is renamed, e.g. {"_helper_process": "process"} defines ``process``.
        args: Mapping of stage name to {parameter: Python expression} for the
            stage call, e.g. {"process": {"data": "pipeline.input('in.csv', path='in.csv')"}}
            or {"train": {"data": "process_out"}}. A StageParams parameter
            without an expression gets ``ParamsType()``; a parameter with a default
            may be omitted; any other parameter must have an expression.

    Returns:
        Path to the created pipeline.py file.

    Raises:
        ValueError: If a dependency parameter has no expression in ``args``.
    """
    target_dir = path or pathlib.Path.cwd()
    pipeline_path = target_dir / "pipeline.py"
    names = names or {}
    args = args or {}

    lines = [
        "from __future__ import annotations",
        "",
        "import pathlib",
        "from typing import Annotated, TypedDict",
        "",
        "from pivot import compose, loaders, stage_def",
        "from pivot.compose import Pipeline, stage",
    ]
    if extra_imports:
        lines.append(extra_imports)
    lines.append("")
    if extra_code:
        lines.append(extra_code)
        lines.append("")

    calls = list[str]()
    for func in stages:
        original = _original_func(func)
        func_name = original.__name__
        stage_name = names.get(func_name, func_name)
        source = textwrap.dedent(inspect.getsource(original))
        if stage_name != func_name:
            source = re.sub(
                rf"^(\s*(?:async\s+)?def\s+){re.escape(func_name)}\(",
                rf"\g<1>{stage_name}(",
                source,
                count=1,
                flags=re.MULTILINE,
            )
        lines.append(source)
        if not hasattr(func, "__wrapped__"):
            lines.append(f"{stage_name} = stage({stage_name})")
        lines.append("")

        stage_args = args.get(stage_name, {})
        call_args = list[str]()
        for param_name, param in inspect.signature(original).parameters.items():
            if param_name in stage_args:
                call_args.append(f"{param_name}={stage_args[param_name]}")
                continue
            params_cls = _annotation_params_class(original, param)
            if params_cls is not None:
                call_args.append(f"{param_name}={params_cls.__name__}()")
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            raise ValueError(
                f"create_pipeline_py: stage {stage_name!r} parameter {param_name!r} "
                + "needs an expression in args"
            )
        calls.append(f"    {stage_name}_out = {stage_name}({', '.join(call_args)})")

    lines.append('with Pipeline("test") as pipeline:')
    lines.extend(calls or ["    pass"])
    lines.append("")

    pipeline_path.write_text("\n".join(lines))
    return pipeline_path


def mount_table(
    monkeypatch: pytest.MonkeyPatch, path: pathlib.Path, mounts: Sequence[tuple[int, str, str]]
) -> None:
    """Point the identity lookup at a ``/proc/self/mountinfo`` written to ``path`` that lists
    ``mounts``, each (device, mount point, filesystem type), with the kernel's escaping."""
    lines = list[str]()
    for mount_id, (device, point, fstype) in enumerate(mounts, 90):
        escaped = point.replace("\\", "\\134").replace(" ", "\\040")
        lines.append(
            f"{mount_id} 1 {os.major(device)}:{os.minor(device)} / {escaped} "
            + f"rw,relatime shared:{mount_id} - {fstype} /dev/fake{mount_id} rw\n"
        )
    path.write_text("".join(lines))
    monkeypatch.setattr(state, "_MOUNTINFO", path)
