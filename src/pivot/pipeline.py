from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pivot import outputs, registry

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Sequence

    from pydantic import BaseModel


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
                           params=TrainParams(learning_rate=0.05))
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
        params: type[BaseModel] | BaseModel | None = None,
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

        registry.REGISTRY.register(
            func=func,
            name=stage_name,
            deps=deps,
            outs=all_outs,
            params=params,
            mutex=mutex,
            cwd=cwd,
        )
        self._stages.append(stage_name)

    @property
    def stages(self) -> list[str]:
        """Get list of stage names in registration order."""
        return list(self._stages)
