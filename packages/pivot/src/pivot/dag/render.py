from __future__ import annotations

import math
from typing import TYPE_CHECKING

import wcwidth
from grandalf import graphs as grandalf_graphs
from grandalf import layouts as grandalf_layouts

if TYPE_CHECKING:
    from pivot.engine.graph import GraphView

__all__ = [
    "render_ascii",
    "render_mermaid",
    "render_dot",
]


def _display_width(s: str) -> int:
    """Get display width of string, accounting for wide characters (CJK, emoji)."""
    width = wcwidth.wcswidth(s)
    # wcswidth returns -1 if string contains non-printable characters
    return width if width >= 0 else len(s)


def _escape_mermaid_label(label: str) -> str:
    """Escape special characters for Mermaid node labels.

    Uses HTML entities for characters with special meaning in Mermaid syntax.
    """
    return (
        label.replace("\\", "&#92;")
        .replace('"', "&quot;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("(", "&#40;")
        .replace(")", "&#41;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("#", "&#35;")
        .replace("\n", " ")
    )


def _escape_dot_label(label: str) -> str:
    """Escape special characters for DOT node labels."""
    return label.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("|", "\\|")


class _VertexView:
    """View for grandalf vertex layout."""

    w: int
    h: int
    xy: tuple[float, float]

    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h
        self.xy = (0.0, 0.0)


def render_ascii(view: GraphView, stages: bool = False) -> str:
    """Render graph as ASCII art using grandalf Sugiyama layout.

    Args:
        view: GraphView from extract_graph_view().
        stages: If True, render stage nodes; if False (default), render artifact nodes.

    Returns:
        ASCII art representation of the graph.
    """
    nodes, edges = (
        (view["stages"], view["stage_edges"])
        if stages
        else (view["artifacts"], view["artifact_edges"])
    )

    if not nodes:
        return "(empty graph)"

    if len(nodes) == 1:
        label = nodes[0]
        return _draw_box(label)

    # Build grandalf graph
    vertex_map = dict[str, grandalf_graphs.Vertex]()
    for label in nodes:
        v = grandalf_graphs.Vertex(label)
        # Width based on display width (handles CJK, emoji), height fixed
        box_width = _display_width(label) + 4  # +4 for "| " and " |"
        v.view = _VertexView(w=box_width, h=3)
        vertex_map[label] = v

    grandalf_edges = list[grandalf_graphs.Edge]()
    for src, dst in edges:
        if src in vertex_map and dst in vertex_map:
            grandalf_edges.append(grandalf_graphs.Edge(vertex_map[src], vertex_map[dst]))

    grandalf_g = grandalf_graphs.Graph(list(vertex_map.values()), grandalf_edges)

    # Handle disconnected components - lay them out side by side
    all_vertices = list[grandalf_graphs.Vertex]()
    x_offset = 0.0
    component_gap = 10  # Gap between components

    for component in grandalf_g.C:
        sug = grandalf_layouts.SugiyamaLayout(component)
        sug.init_all()
        sug.draw()

        # Find component bounds
        if component.sV:
            comp_min_x = min(v.view.xy[0] - v.view.w / 2 for v in component.sV)
            comp_max_x = max(v.view.xy[0] + v.view.w / 2 for v in component.sV)

            # Offset all vertices in this component
            shift = x_offset - comp_min_x
            for v in component.sV:
                old_xy = v.view.xy
                v.view.xy = (old_xy[0] + shift, old_xy[1])

            # Update x_offset for next component
            x_offset = x_offset + (comp_max_x - comp_min_x) + component_gap

        all_vertices.extend(component.sV)

    # Deduplicate edges before rendering
    unique_edges = list(dict.fromkeys(edges))
    return _render_ascii_from_layout(all_vertices, unique_edges, vertex_map)


def _draw_box(label: str) -> str:
    """Draw a single box around a label."""
    width = _display_width(label) + 2
    top = "+" + "-" * width + "+"
    mid = "| " + label + " |"
    bot = "+" + "-" * width + "+"
    return f"{top}\n{mid}\n{bot}"


def _render_ascii_from_layout(
    vertices: list[grandalf_graphs.Vertex],
    edges: list[tuple[str, str]],
    vertex_map: dict[str, grandalf_graphs.Vertex],
) -> str:
    """Render ASCII art from grandalf layout."""
    if not vertices:
        return "(empty graph)"

    # Find bounds
    min_x = min(v.view.xy[0] - v.view.w / 2 for v in vertices)
    max_x = max(v.view.xy[0] + v.view.w / 2 for v in vertices)
    min_y = min(v.view.xy[1] - v.view.h / 2 for v in vertices)
    max_y = max(v.view.xy[1] + v.view.h / 2 for v in vertices)

    # Create canvas with margin
    margin = 2
    width = math.ceil(max_x - min_x) + margin * 2 + 2
    height = math.ceil(max_y - min_y) + margin * 2 + 2

    # Prevent excessive memory usage for very large graphs
    max_canvas_dim = 10000
    if width > max_canvas_dim or height > max_canvas_dim:
        return f"(graph too large for ASCII: {width}x{height}, use --mermaid or --dot)"

    # Initialize canvas
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # Helper to convert coordinates
    def to_canvas(x: float, y: float) -> tuple[int, int]:
        cx = int(x - min_x) + margin
        cy = int(y - min_y) + margin
        return cx, cy

    # Draw edges first (so boxes overlay them)
    for src, dst in edges:
        if src not in vertex_map or dst not in vertex_map:
            continue
        v_src = vertex_map[src]
        v_dst = vertex_map[dst]

        # Get center points
        src_x, src_y = v_src.view.xy
        dst_x, dst_y = v_dst.view.xy

        # Draw from bottom of source to top of destination
        src_bottom_y = src_y + v_src.view.h / 2
        dst_top_y = dst_y - v_dst.view.h / 2

        # Simple line drawing
        cx_src, cy_src = to_canvas(src_x, src_bottom_y)
        cx_dst, cy_dst = to_canvas(dst_x, dst_top_y)

        _draw_line(canvas, cx_src, cy_src, cx_dst, cy_dst, width, height)

    # Draw boxes
    for v in vertices:
        label = str(v.data)
        cx, cy = to_canvas(v.view.xy[0], v.view.xy[1])
        _draw_box_on_canvas(canvas, cx, cy, label, width, height)

    # Convert to string, removing trailing whitespace
    lines = ["".join(row).rstrip() for row in canvas]
    # Remove empty lines at top and bottom
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _draw_line(
    canvas: list[list[str]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> None:
    """Draw a line using '*' characters."""
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    x, y = x1, y1
    while True:
        # Only draw if in bounds and not overwriting a box character
        if 0 <= x < width and 0 <= y < height and canvas[y][x] == " ":
            canvas[y][x] = "*"
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _draw_box_on_canvas(
    canvas: list[list[str]],
    cx: int,
    cy: int,
    label: str,
    width: int,
    height: int,
) -> None:
    """Draw a box centered at (cx, cy) on the canvas."""
    box_width = _display_width(label) + 4  # "| " + label + " |"
    box_height = 3

    left = cx - box_width // 2
    top = cy - box_height // 2

    # Top border
    _write_str(canvas, left, top, "+" + "-" * (box_width - 2) + "+", width, height)
    # Middle with label
    _write_str(canvas, left, top + 1, "| " + label + " |", width, height)
    # Bottom border
    _write_str(canvas, left, top + 2, "+" + "-" * (box_width - 2) + "+", width, height)


def _write_str(
    canvas: list[list[str]],
    x: int,
    y: int,
    s: str,
    width: int,
    height: int,
) -> None:
    """Write a string to the canvas at position (x, y)."""
    if y < 0 or y >= height:
        return
    for i, ch in enumerate(s):
        px = x + i
        if 0 <= px < width:
            canvas[y][px] = ch


def render_mermaid(view: GraphView, stages: bool = False) -> str:
    """Render graph as Mermaid flowchart TD format.

    Args:
        view: GraphView from extract_graph_view().
        stages: If True, render stage nodes; if False (default), render artifact nodes.

    Returns:
        Mermaid flowchart string.
    """
    nodes, edges = (
        (view["stages"], view["stage_edges"])
        if stages
        else (view["artifacts"], view["artifact_edges"])
    )

    if not nodes:
        return "flowchart TD"

    lines = ["flowchart TD"]

    # Create node IDs (sanitized for Mermaid)
    node_ids = dict[str, str]()
    for i, label in enumerate(sorted(nodes), 1):
        node_ids[label] = f"node{i}"

    # Emit node definitions
    for label in sorted(nodes):
        node_id = node_ids[label]
        escaped_label = _escape_mermaid_label(label)
        lines.append(f'    {node_id}["{escaped_label}"]')

    # Emit edges
    seen_edges = set[tuple[str, str]]()
    for src, dst in edges:
        if (src, dst) in seen_edges:
            continue
        seen_edges.add((src, dst))
        if src in node_ids and dst in node_ids:
            lines.append(f"    {node_ids[src]}-->{node_ids[dst]}")

    return "\n".join(lines)


def render_dot(view: GraphView, stages: bool = False) -> str:
    """Render graph as Graphviz DOT format.

    Args:
        view: GraphView from extract_graph_view().
        stages: If True, render stage nodes; if False (default), render artifact nodes.

    Returns:
        DOT format string.
    """
    nodes, edges = (
        (view["stages"], view["stage_edges"])
        if stages
        else (view["artifacts"], view["artifact_edges"])
    )

    if not nodes:
        return "digraph {\n}"

    lines = ["digraph {"]

    # Emit edges (DOT auto-creates nodes from edges)
    seen_edges = set[tuple[str, str]]()
    emitted_nodes = set[str]()

    for src, dst in edges:
        if (src, dst) in seen_edges:
            continue
        seen_edges.add((src, dst))
        src_escaped = _escape_dot_label(src)
        dst_escaped = _escape_dot_label(dst)
        lines.append(f'    "{src_escaped}" -> "{dst_escaped}"')
        emitted_nodes.add(src)
        emitted_nodes.add(dst)

    # Emit isolated nodes (no edges)
    for label in sorted(nodes):
        if label not in emitted_nodes:
            escaped = _escape_dot_label(label)
            lines.append(f'    "{escaped}"')

    lines.append("}")
    return "\n".join(lines)
