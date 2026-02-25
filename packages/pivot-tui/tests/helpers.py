"""Test helpers for registering stages."""

# pyright: reportMissingImports=false

from __future__ import annotations

import inspect
import pathlib
import textwrap
import typing
from typing import TYPE_CHECKING, Any

import anyio

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pivot import outputs, stage_def
    from pivot.registry import PipelineLike


async def wait_for_socket(socket_path: pathlib.Path, timeout: float = 5.0) -> None:
    """Wait for Unix socket to be created and connectable.

    Args:
        socket_path: Path to the Unix socket file.
        timeout: Maximum time to wait in seconds.

    Raises:
        TimeoutError: If socket is not created within timeout.
    """
    import stat

    deadline = anyio.current_time() + timeout
    while anyio.current_time() < deadline:
        if socket_path.exists():
            try:
                mode = socket_path.stat().st_mode
                if stat.S_ISSOCK(mode):
                    return
            except OSError:
                pass  # Ignore if socket stat fails (race condition, permissions)
        await anyio.sleep(0.05)
    msg = f"Socket {socket_path} not created within {timeout}s"
    raise TimeoutError(msg)


# Module-level test pipeline for tests that don't have explicit Pipeline context.
# This is reset between tests by the clean_test_pipeline fixture in conftest.py.
_test_pipeline: PipelineLike | None = None


def get_test_pipeline() -> PipelineLike:
    """Get the current test pipeline.

    Raises RuntimeError if no test pipeline is set.
    """
    if _test_pipeline is None:
        raise RuntimeError(
            "No test pipeline available. Use the 'test_pipeline' fixture or call set_test_pipeline() before registering stages."
        )
    return _test_pipeline


def set_test_pipeline(pipeline: PipelineLike | None) -> None:
    """Set the module-level test pipeline."""
    global _test_pipeline
    _test_pipeline = pipeline


def register_test_stage(
    func: Callable[..., Any],
    name: str | None = None,
    params: type[stage_def.StageParams] | stage_def.StageParams | None = None,
    mutex: list[str] | None = None,
    variant: str | None = None,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    out_path_overrides: Mapping[str, outputs.PathType] | None = None,
    *,
    pipeline: PipelineLike | None = None,
) -> None:
    """Register a stage for testing.

    Args:
        func: The stage function.
        name: Stage name (defaults to function name).
        params: Stage parameters class or instance.
        mutex: Mutex groups.
        variant: Variant name.
        dep_path_overrides: Override paths for dependencies.
        out_path_overrides: Override paths for outputs.
        pipeline: Pipeline to register with. If not provided, uses the module-level
            test pipeline (set via set_test_pipeline or the test_pipeline fixture).

    Note: deps and outs are now defined via annotations on the function, not
    as parameters to this function. Use Annotated[T, Dep(...)] for deps and
    Annotated[T, Out(...)] return types for outputs.
    """
    from pivot import compose, stage_def

    target = pipeline or get_test_pipeline()

    def _param_type_from_annotation(annotation: object) -> type | None:
        if annotation is inspect._empty:
            return None
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            args = typing.get_args(annotation)
            if args:
                annotation = args[0]
        if isinstance(annotation, type):
            return annotation
        return None

    stage_func = func if getattr(func, "_is_stage", False) else compose.stage(func)

    with compose.Pipeline(target.name, root=target.root) as pipeline_builder:
        sig = inspect.signature(func)
        kwargs = dict[str, object]()
        for param_name, param in sig.parameters.items():
            param_type = _param_type_from_annotation(param.annotation)
            if param_type is not None and issubclass(param_type, stage_def.StageParams):
                if params is None:
                    kwargs[param_name] = param_type()
                elif isinstance(params, stage_def.StageParams):
                    kwargs[param_name] = params
                else:
                    kwargs[param_name] = params()
                continue

            override_path = None
            if dep_path_overrides is not None and param_name in dep_path_overrides:
                override = dep_path_overrides[param_name]
                if isinstance(override, (list, tuple)):
                    override_path = str(override[0])
                else:
                    override_path = str(override)
            if param_type is None:
                kwargs[param_name] = pipeline_builder.input(param_name, path=override_path)
            else:
                kwargs[param_name] = pipeline_builder.input(
                    param_name, t=param_type, path=override_path
                )

        _ = mutex
        _ = variant
        _ = out_path_overrides
        stage_func(**kwargs)

    target.include(pipeline_builder)


def create_pipeline_py(
    stages: Sequence[Callable[..., Any]],
    *,
    path: pathlib.Path | None = None,
    extra_imports: str = "",
    extra_code: str = "",
    names: Mapping[str, str] | None = None,
) -> pathlib.Path:
    """Create a pipeline.py file that registers the given stages.

    This is useful for CLI tests that use runner.isolated_filesystem()
    and need discover_pipeline() to find stages.

    Args:
        stages: Sequence of stage functions to include. Their source code
            will be extracted and included in the generated file.
        path: Directory to create pipeline.py in. Defaults to current directory.
        extra_imports: Additional import statements to include at the top.
        extra_code: Additional code to include (e.g., TypedDict definitions).
        names: Mapping of function names to registered stage names.
            E.g., {"_helper_process": "process"} registers with name="process".

    Returns:
        Path to the created pipeline.py file.

    Example:
        with runner.isolated_filesystem():
            create_pipeline_py([my_stage_func])
            result = runner.invoke(cli.cli, ["run"])
    """
    target_dir = path or pathlib.Path.cwd()
    pipeline_path = target_dir / "pipeline.py"
    names = names or {}

    # Build the pipeline.py content
    lines = [
        "from __future__ import annotations",
        "",
        "import pathlib",
        "from typing import Annotated, TypedDict",
        "",
        "from pivot import compose, loaders, outputs, stage_def",
    ]

    if extra_imports:
        lines.append(extra_imports)

    lines.extend(["", "pipeline_builder = compose.Pipeline('test')", ""])

    if extra_code:
        lines.append(extra_code)
        lines.append("")

    # Extract and include each stage function's source
    for func in stages:
        source = inspect.getsource(func)
        # Dedent in case the function was defined at non-zero indentation
        source = textwrap.dedent(source)
        lines.append(source)
        lines.append("")

    for func in stages:
        func_name = func.__name__
        lines.append(f"{func_name} = compose.stage({func_name})")

    lines.append("")
    lines.append("with pipeline_builder:")
    for func in stages:
        func_name = func.__name__
        list[str]()
        call_args = list[str]()
        for param in inspect.signature(func).parameters:
            if param == "params":
                lines.append(f"    {param} = stage_def.StageParams()")
                call_args.append(f"{param}={param}")
            else:
                lines.append(f"    {param} = pipeline_builder.input('{param}')")
                call_args.append(f"{param}={param}")
        call = ", ".join(call_args)
        if func_name in names:
            lines.append(f"    {func_name}({call})")
            lines.append(f"    {func_name}.__name__ = '{names[func_name]}'")
        else:
            lines.append(f"    {func_name}({call})")

    lines.append("")
    lines.append("pipeline = pipeline_builder")
    lines.append("")  # Final newline

    pipeline_path.write_text("\n".join(lines))
    return pipeline_path
