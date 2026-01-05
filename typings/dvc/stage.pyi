"""Type stubs for dvc.stage."""

from typing import Any

from dvc.output import Output

class HashInfo:
    """Hash info for parameters."""

    value: dict[str, Any] | None

class ParamDependency:
    """Parameter dependency."""

    hash_info: HashInfo | None

class PipelineStage:
    """DVC pipeline stage representation."""

    name: str | None
    cmd: str | None
    path: str
    deps: list[Output]
    outs: list[Output]
    params: list[ParamDependency]
    frozen: bool
    desc: str | None
