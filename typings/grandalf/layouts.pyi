"""Type stubs for grandalf.layouts module."""

from grandalf.graphs import Edge, Vertex, graph_core

class SugiyamaLayout:
    """Sugiyama layered graph layout algorithm."""

    def __init__(self, g: graph_core) -> None: ...
    def init_all(
        self,
        roots: list[Vertex] | None = None,
        inverted_edges: list[Edge] | None = None,
        optimize: bool = False,
    ) -> None: ...
    def draw(self, N: float | None = None) -> None: ...  # noqa: N803 - matches grandalf API
