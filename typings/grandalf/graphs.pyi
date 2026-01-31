"""Type stubs for grandalf.graphs module."""

from typing import Any

class Vertex:
    """A graph vertex with optional data and view for layout."""

    data: Any
    view: Any

    def __init__(self, data: Any = None) -> None: ...

class Edge:
    """A directed edge connecting two vertices."""

    v: tuple[Vertex, Vertex]

    def __init__(
        self,
        x: Vertex,
        y: Vertex,
        w: int = 1,
        data: Any = None,
        connect: bool = False,
    ) -> None: ...

class graph_core:
    """Base class for connected components."""

    sV: list[Vertex]  # noqa: N815 - matches grandalf API
    sE: list[Edge]  # noqa: N815 - matches grandalf API

class Graph:
    """A directed or undirected graph."""

    C: list[graph_core]

    def __init__(
        self,
        V: list[Vertex] | None = None,  # noqa: N803 - matches grandalf API
        E: list[Edge] | None = None,  # noqa: N803 - matches grandalf API
        directed: bool = True,
    ) -> None: ...
