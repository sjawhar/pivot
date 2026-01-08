from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING, Any

import pydantic

from pivot import outputs, parameters, registry

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Sequence


class PipelineError(Exception):
    """Error in pipeline configuration."""


class Pipeline:
    """Programmatic pipeline definition.

    Example:
        from mymodule import preprocess, train
        from pivot import Pipeline

        pipeline = Pipeline()
        pipeline.add_stage(preprocess, deps=['data/raw.csv'], outs=['data/clean.csv'])
        pipeline.add_stage(train, deps=['data/clean.csv'], outs=['models/model.pkl'],
                           params={'learning_rate': 0.05})
    """

    def __init__(self) -> None:
        self._stages: list[str] = []

    def add_stage(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        deps: Sequence[str] = (),
        outs: Sequence[outputs.OutSpec] = (),
        metrics: Sequence[outputs.OutSpec] = (),
        plots: Sequence[outputs.OutSpec] = (),
        params: dict[str, Any] | pydantic.BaseModel | None = None,
        mutex: Sequence[str] = (),
        cwd: str | pathlib.Path | None = None,
    ) -> None:
        """Add a stage to the pipeline and register it globally."""
        stage_name = name or func.__name__

        if stage_name in self._stages:
            raise PipelineError(f"Stage '{stage_name}' already added to pipeline")

        # Combine all outputs, converting metrics/plots strings to typed outputs
        all_outs: list[outputs.OutSpec] = list(outs)
        all_outs += [
            m if isinstance(m, outputs.BaseOut) else outputs.Metric(path=m) for m in metrics
        ]
        all_outs += [p if isinstance(p, outputs.BaseOut) else outputs.Plot(path=p) for p in plots]

        # Resolve params from dict if needed
        params_instance = _resolve_params_from_dict(func, params, stage_name)

        registry.REGISTRY.register(
            func=func,
            name=stage_name,
            deps=deps,
            outs=all_outs,
            params=params_instance,
            mutex=mutex,
            cwd=cwd,
        )
        self._stages.append(stage_name)

    @property
    def stages(self) -> list[str]:
        """Get list of stage names in registration order."""
        return list(self._stages)


def _resolve_params_from_dict(
    func: Callable[..., Any],
    params: dict[str, Any] | pydantic.BaseModel | None,
    stage_name: str,
) -> pydantic.BaseModel | None:
    """Resolve params dict to BaseModel instance by introspecting function signature."""
    if params is None:
        return None

    # If already a BaseModel instance, use directly
    if isinstance(params, pydantic.BaseModel):
        return params

    # At this point, params is a dict - introspect the params class from signature
    sig = inspect.signature(func)
    if "params" not in sig.parameters:
        raise PipelineError(
            f"Stage '{stage_name}': params provided but function has no 'params' parameter"
        )

    try:
        type_hints = typing.get_type_hints(func)
    except NameError as e:
        raise PipelineError(
            f"Stage '{stage_name}': failed to resolve type hints for '{func.__name__}': {e}"
        ) from e

    if "params" not in type_hints:
        raise PipelineError(
            f"Stage '{stage_name}': function has 'params' parameter but no type hint"
        )

    params_cls = type_hints["params"]

    if not parameters.validate_params_cls(params_cls):
        raise PipelineError(
            f"Stage '{stage_name}': params type hint must be a Pydantic BaseModel, "
            + f"got {params_cls}"
        )

    try:
        return params_cls(**params)
    except pydantic.ValidationError as e:
        raise PipelineError(f"Stage '{stage_name}': invalid params: {e}") from e
