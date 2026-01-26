"""Test helpers for registering stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pivot import outputs, registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pivot import stage_def


def register_test_stage(
    func: Callable[..., Any],
    name: str | None = None,
    params: type[stage_def.StageParams] | stage_def.StageParams | None = None,
    mutex: list[str] | None = None,
    variant: str | None = None,
    dep_path_overrides: Mapping[str, outputs.PathType] | None = None,
    out_path_overrides: Mapping[str, registry.OutOverrideInput] | None = None,
) -> None:
    """Register a stage for testing.

    Note: deps and outs are now defined via annotations on the function, not
    as parameters to this function. Use Annotated[T, Dep(...)] for deps and
    Annotated[T, Out(...)] return types for outputs.
    """
    registry.REGISTRY.register(
        func=func,
        name=name,
        params=params,
        mutex=mutex,
        variant=variant,
        dep_path_overrides=dep_path_overrides,
        out_path_overrides=out_path_overrides,
    )
