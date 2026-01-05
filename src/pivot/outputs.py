"""Rich output types for DVC-compatible stage outputs."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class BaseOut:
    """Base class for all output types."""

    path: str
    cache: bool = True
    persist: bool = False


@dataclasses.dataclass(frozen=True)
class Out(BaseOut):
    """Regular output file (cached by default)."""

    cache: bool = True


@dataclasses.dataclass(frozen=True)
class Metric(BaseOut):
    """Metrics file (git-tracked by default, NOT cached)."""

    cache: bool = False


@dataclasses.dataclass(frozen=True)
class Plot(BaseOut):
    """Plot file with visualization options."""

    cache: bool = True
    x: str | None = None
    y: str | None = None
    template: str | None = None


type OutSpec = str | BaseOut


def normalize_out(out: OutSpec) -> BaseOut:
    """Convert string or BaseOut subclass to BaseOut object."""
    if isinstance(out, str):
        return Out(path=out)
    return out
