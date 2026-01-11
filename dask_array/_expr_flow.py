"""Expression flow visualization - shows data transformation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from dask.utils import funcname


@dataclass
class FlowNode:
    """A node in the flow visualization representing a unique shape."""

    shape: tuple
    chunks: tuple
    operations: list[str] = field(default_factory=list)
    expressions: list = field(default_factory=list)  # Original expr objects
    nbytes: int = 0
    row: int = 0
    col: int = 0

    @property
    def ndim(self):
        return len(self.shape)


@dataclass
class FlowEdge:
    """Connection between flow nodes."""

    source: FlowNode
    target: FlowNode


def _get_operation_name(expr) -> str:
    """Get a user-friendly operation name from an expression."""
    class_name = funcname(type(expr))

    # Special cases for nicer names
    if class_name == "FromArray":
        return "Load"

    # Try to extract meaningful name from _name attribute
    if hasattr(expr, "_name"):
        name = expr._name
        if "-" in name:
            prefix = name.rsplit("-", 1)[0]
            # Clean up common patterns
            prefix = prefix.replace("_", " ")
            prefix = prefix.replace("-aggregate", "")
            prefix = prefix.replace("-partial", "")
            prefix = prefix.strip().replace("-", " ")
            if prefix:
                # Capitalize nicely
                return prefix.title()

    return class_name


def _is_reduction_intermediate(expr) -> bool:
    """Check if this expression is an intermediate reduction shape.

    Conservative filter - only catches the most obvious tree_reduce intermediates.
    """
    from dask_array.reductions._reduction import PartialReduce

    if not isinstance(expr, PartialReduce):
        return False

    shape = expr.shape
    if not shape:
        return False

    # Only filter if ALL dimensions are small (clearly chunk counts, not user data)
    CHUNK_COUNT_MAX = 16
    return all(0 < d <= CHUNK_COUNT_MAX for d in shape)


def _walk_expr_tree(expr, visited=None):
    """Walk expression tree depth-first, yielding expressions from leaves to root."""
    if visited is None:
        visited = set()

    expr_id = id(expr)
    if expr_id in visited:
        return
    visited.add(expr_id)

    # Get array dependencies
    deps = [op for op in expr.dependencies() if hasattr(op, "chunks")]

    # Visit children first (leaves to root)
    for dep in deps:
        yield from _walk_expr_tree(dep, visited)

    yield expr


def _get_expr_inputs(expr):
    """Get the direct array expression inputs to this expression."""
    # Use dependencies() instead of operands - handles fused nodes correctly
    return [dep for dep in expr.dependencies() if hasattr(dep, "chunks")]


def build_flow_graph(expr):
    """Build a flow graph from an expression tree.

    Returns a tuple of (nodes, edges) where:
    - nodes: list of FlowNode objects
    - edges: list of FlowEdge objects

    Nodes are grouped by shape, with consecutive same-shape operations
    collapsed into a single node (only when they form a linear chain).
    """
    # Collect all expressions
    all_exprs = list(_walk_expr_tree(expr))

    # Filter out intermediate reduction shapes (but never filter the root expression)
    root_id = id(expr)
    filtered_exprs = [e for e in all_exprs if id(e) == root_id or not _is_reduction_intermediate(e)]

    if not filtered_exprs:
        return [], []

    # Build expression -> node mapping
    # Key insight: we only merge operations into the same node if:
    # 1. Same shape
    # 2. Single input that's already in a node
    # 3. That input node is the one we'd extend (linear chain)
    expr_to_node = {}
    nodes = []

    for e in filtered_exprs:
        shape = e.shape
        inputs = _get_expr_inputs(e)

        # Find which node(s) our inputs belong to
        input_nodes = [expr_to_node.get(inp) for inp in inputs if inp in expr_to_node]
        input_nodes = [n for n in input_nodes if n is not None]

        # Can only extend if: single input, same shape, and that input's node
        # has the same shape (indicating a linear chain)
        can_extend = len(input_nodes) == 1 and len(inputs) == 1 and shape == input_nodes[0].shape

        if can_extend:
            # Extend the existing node
            node = input_nodes[0]
            node.operations.append(_get_operation_name(e))
            node.expressions.append(e)
            expr_to_node[e] = node
            # Update nbytes to reflect latest expression
            try:
                node.nbytes = prod(shape) * e.dtype.itemsize
            except Exception:
                pass
        else:
            # Create a new node
            try:
                nbytes = prod(shape) * e.dtype.itemsize
            except Exception:
                nbytes = 0

            node = FlowNode(
                shape=shape,
                chunks=e.chunks,
                operations=[_get_operation_name(e)],
                expressions=[e],
                nbytes=nbytes,
            )
            nodes.append(node)
            expr_to_node[e] = node

    # Build edges based on expression dependencies
    # We need to trace through filtered intermediates to find actual sources
    edges = []
    seen_edges = set()

    def find_source_nodes(expr, visited=None):
        """Trace back through filtered expressions to find source nodes."""
        if visited is None:
            visited = set()
        if id(expr) in visited:
            return []
        visited.add(id(expr))

        node = expr_to_node.get(expr)
        if node is not None:
            return [node]

        # This expression was filtered - look at its inputs
        results = []
        for inp in _get_expr_inputs(expr):
            results.extend(find_source_nodes(inp, visited))
        return results

    for e in filtered_exprs:
        target_node = expr_to_node.get(e)
        if target_node is None:
            continue

        for inp in _get_expr_inputs(e):
            # Trace back through filtered intermediates
            for source_node in find_source_nodes(inp):
                if source_node != target_node:
                    edge_key = (id(source_node), id(target_node))
                    if edge_key not in seen_edges:
                        edges.append(FlowEdge(source=source_node, target=target_node))
                        seen_edges.add(edge_key)

    # Assign row/column positions for layout
    _assign_layout(nodes, edges)

    return nodes, edges


def _assign_layout(nodes, edges):
    """Assign row and column positions to nodes for rendering.

    Uses a simple algorithm:
    - Nodes with no incoming edges start at column 0
    - Each node's column = max(input columns) + 1
    - Nodes at the same column are stacked in rows
    """
    if not nodes:
        return

    # Build adjacency info
    node_inputs = {id(n): [] for n in nodes}
    for edge in edges:
        node_inputs[id(edge.target)].append(edge.source)

    # Assign columns (topological order)
    node_col = {}
    for node in nodes:
        inputs = node_inputs[id(node)]
        if not inputs:
            node_col[id(node)] = 0
        else:
            max_input_col = max(node_col.get(id(inp), 0) for inp in inputs)
            node_col[id(node)] = max_input_col + 1
        node.col = node_col[id(node)]

    # Assign rows within each column
    col_counts = {}
    for node in nodes:
        col = node.col
        row = col_counts.get(col, 0)
        node.row = row
        col_counts[col] = row + 1


def count_operations(expr) -> int:
    """Count total operations in an expression tree."""
    return len(list(_walk_expr_tree(expr)))


def _format_bytes(nbytes: int) -> str:
    """Format bytes with 2 significant figures."""
    for unit, threshold in [
        ("PiB", 2**50),
        ("TiB", 2**40),
        ("GiB", 2**30),
        ("MiB", 2**20),
        ("kiB", 2**10),
    ]:
        if nbytes >= threshold:
            value = nbytes / threshold
            if value >= 10:
                return f"{value:.0f} {unit}"
            else:
                return f"{value:.1f} {unit}"
    return f"{nbytes} B"


def _format_shape(shape: tuple) -> str:
    """Format shape tuple for display."""
    if not shape:
        return "scalar"
    return f"({', '.join(str(s) for s in shape)})"


# Card dimensions (fixed for consistency)
CARD_WIDTH = 130
CARD_HEIGHT = 150
CARD_SVG_REGION = 55  # Available space for SVG in card
CARD_GAP = 20
ARROW_WIDTH = 50


def _compute_emphasis(nodes, threshold: float = 0.5) -> dict:
    """Compute which nodes should be emphasized based on array size.

    Returns a dict mapping node id to bool (True = emphasize).
    Nodes with nbytes > threshold * max_nbytes are emphasized.
    """
    valid_bytes = [n.nbytes for n in nodes if n.nbytes > 0]
    if not valid_bytes:
        return {id(n): True for n in nodes}

    max_bytes = max(valid_bytes)
    if max_bytes <= 0:
        return {id(n): True for n in nodes}

    return {id(n): n.nbytes > threshold * max_bytes for n in nodes}


def render_flow_svg(expr) -> str:
    """Render expression flow as an SVG diagram with card-based layout.

    Parameters
    ----------
    expr : ArrayExpr
        The expression to visualize

    Returns
    -------
    str
        HTML with embedded SVG showing the data flow
    """
    nodes, edges = build_flow_graph(expr)
    if not nodes:
        return "<div>Empty expression</div>"

    max_col = max(n.col for n in nodes) + 1
    max_row = max(n.row for n in nodes) + 1

    # Compute which nodes to emphasize
    emphasis = _compute_emphasis(nodes)

    # Compute global max dimension for consistent scaling across all SVGs
    all_shapes = [n.shape for n in nodes if n.shape]
    global_max_dim = max(max(s) for s in all_shapes) if all_shapes else 1

    # Group nodes by column
    cols = {}
    for node in nodes:
        cols.setdefault(node.col, []).append(node)

    # Calculate SVG dimensions
    padding = 24
    col_width = CARD_WIDTH + ARROW_WIDTH
    row_height = CARD_HEIGHT + CARD_GAP
    svg_width = max_col * col_width - ARROW_WIDTH + 2 * padding
    svg_height = max_row * row_height - CARD_GAP + 2 * padding

    # Build node position map (center of each card)
    node_positions = {}
    for node in nodes:
        x = padding + node.col * col_width + CARD_WIDTH / 2
        y = padding + node.row * row_height + CARD_HEIGHT / 2
        node_positions[id(node)] = (x, y)

    # Start SVG
    svg_parts = [
        f'<svg width="{svg_width}" height="{svg_height}" '
        f'style="font-family: system-ui;" xmlns="http://www.w3.org/2000/svg">'
    ]

    # Add definitions for shadow
    svg_parts.append("""<defs>
    <filter id="card-shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.1"/>
    </filter>
  </defs>""")

    # Draw arrows first (so they appear behind cards)
    for edge in edges:
        src_x, src_y = node_positions[id(edge.source)]
        tgt_x, tgt_y = node_positions[id(edge.target)]

        # Arrow from right edge of source to left edge of target
        x1 = src_x + CARD_WIDTH / 2 + 4
        y1 = src_y
        x2 = tgt_x - CARD_WIDTH / 2 - 4
        y2 = tgt_y

        if abs(y1 - y2) < 5:
            # Straight horizontal arrow - use arrowhead
            # Draw line and separate arrowhead
            svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2 - 8}" y2="{y2}" stroke="#a8a29e" stroke-width="2"/>')
            # Arrowhead pointing right
            svg_parts.append(f'<polygon points="{x2},{y2} {x2 - 8},{y2 - 4} {x2 - 8},{y2 + 4}" fill="#a8a29e"/>')
        else:
            # Curved arrow for cross-row connections - no arrowhead, use dot instead
            mid_x = (x1 + x2) / 2
            svg_parts.append(
                f'<path d="M {x1} {y1} C {mid_x} {y1}, {mid_x} {y2}, {x2} {y2}" '
                f'stroke="#a8a29e" stroke-width="2" fill="none"/>'
            )
            # End with a small circle instead of arrowhead
            svg_parts.append(f'<circle cx="{x2}" cy="{y2}" r="4" fill="#a8a29e"/>')

    # Draw cards - use consistent SVG size across all cards
    for node in nodes:
        cx, cy = node_positions[id(node)]
        card_x = cx - CARD_WIDTH / 2
        card_y = cy - CARD_HEIGHT / 2
        emphasized = emphasis.get(id(node), False)
        svg_parts.append(_render_card(node, card_x, card_y, emphasized, global_max_dim))

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _render_card(node: FlowNode, x: float, y: float, emphasized: bool, global_max_dim: int) -> str:
    """Render a single flow node as an SVG card."""
    from dask_array._svg import svg, ratio_response

    parts = []

    # Card styling based on emphasis
    if emphasized:
        # Emphasized: stronger border, subtle warm tint
        fill = "#fffbf7"  # Very subtle orange tint
        stroke = "#a8a29e"  # Darker border
        stroke_width = "1.5"
    else:
        # De-emphasized: lighter, more muted
        fill = "white"
        stroke = "#e7e5e4"  # Lighter border
        stroke_width = "1"

    parts.append(
        f'<rect x="{x}" y="{y}" width="{CARD_WIDTH}" height="{CARD_HEIGHT}" '
        f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" filter="url(#card-shadow)"/>'
    )

    # Text colors based on emphasis
    title_color = "#44403c" if emphasized else "#78716c"
    info_color = "#57534e" if emphasized else "#a8a29e"
    secondary_color = "#a8a29e" if emphasized else "#d6d3d1"

    # Operation name at top (title)
    ops = node.operations
    if len(ops) > 2:
        ops_str = f"{ops[0]} → {ops[-1]}"
    elif len(ops) == 2:
        ops_str = f"{ops[0]} → {ops[1]}"
    else:
        ops_str = ops[0] if ops else ""

    # Truncate if too long
    if len(ops_str) > 18:
        ops_str = ops_str[:16] + "…"

    parts.append(
        f'<text x="{x + CARD_WIDTH / 2}" y="{y + 20}" '
        f'text-anchor="middle" font-size="11" font-weight="600" fill="{title_color}">'
        f"{ops_str}</text>"
    )

    # Divider line
    parts.append(
        f'<line x1="{x + 10}" y1="{y + 30}" x2="{x + CARD_WIDTH - 10}" y2="{y + 30}" '
        f'stroke="#e7e5e4" stroke-width="1"/>'
    )

    # SVG visualization (centered in middle region)
    svg_y = y + 35
    svg_region_height = 70
    svg_region_width = CARD_WIDTH - 20
    try:
        if node.chunks and all(node.chunks):
            # Compute sizes using global reference so dimensions are comparable across cards
            shape = node.shape
            # Ratio of global max to each dimension, with logarithmic compression
            ratios = [global_max_dim / max(0.1, d) for d in shape]
            ratios = [ratio_response(r) for r in ratios]
            sizes = tuple(CARD_SVG_REGION / r for r in ratios)

            node_svg = svg(node.chunks, size=CARD_SVG_REGION, sizes=sizes, labels=False)
            parts.append(
                f'<foreignObject x="{x + 10}" y="{svg_y}" width="{svg_region_width}" height="{svg_region_height}">'
                f'<div xmlns="http://www.w3.org/1999/xhtml" style="display:flex;justify-content:center;align-items:center;height:100%;overflow:hidden;">'
                f"{node_svg}"
                f"</div></foreignObject>"
            )
        else:
            # Scalar - show small circle
            cx = x + CARD_WIDTH / 2
            cy = svg_y + svg_region_height / 2
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="#fb923c" fill-opacity="0.7"/>')
    except (NotImplementedError, ValueError):
        # Fallback - empty area
        pass

    # Divider line before info section
    parts.append(
        f'<line x1="{x + 10}" y1="{y + 110}" x2="{x + CARD_WIDTH - 10}" y2="{y + 110}" '
        f'stroke="#e7e5e4" stroke-width="1"/>'
    )

    # Bottom section: shape and bytes, left-aligned
    shape_str = _format_shape(node.shape)
    bytes_str = _format_bytes(node.nbytes) if node.nbytes > 0 else ""
    left_margin = x + 12

    parts.append(
        f'<text x="{left_margin}" y="{y + 128}" '
        f'text-anchor="start" font-size="10" fill="{info_color}">'
        f"{shape_str}</text>"
    )

    if bytes_str:
        # Same font size as shape, but bold for emphasized (large) arrays
        bytes_weight = 'font-weight="600"' if emphasized else ""
        parts.append(
            f'<text x="{left_margin}" y="{y + 142}" '
            f'text-anchor="start" font-size="10" {bytes_weight} fill="{secondary_color}">'
            f"{bytes_str}</text>"
        )

    return "\n".join(parts)


class FlowDiagram:
    """Wrapper for flow diagram with Jupyter and terminal display support."""

    def __init__(self, expr):
        self._expr = expr
        self._html_cache = None

    def _repr_html_(self) -> str:
        """Jupyter notebook display."""
        if self._html_cache is None:
            self._html_cache = render_flow_svg(self._expr)
        return self._html_cache

    def __repr__(self) -> str:
        """Terminal display - show summary."""
        nodes, edges = build_flow_graph(self._expr)
        n_ops = count_operations(self._expr)
        shapes = [n.shape for n in nodes]
        shape_str = " → ".join(str(s) for s in shapes)
        return f"Expression: {n_ops} operations, {len(nodes)} shape(s): {shape_str}"


def expr_flow(expr) -> FlowDiagram:
    """Create a flow diagram visualization of an expression.

    Parameters
    ----------
    expr : ArrayExpr or Array
        The expression or array to visualize

    Returns
    -------
    FlowDiagram
        A displayable flow diagram (works in Jupyter and terminal)
    """
    # Handle both Array and ArrayExpr
    if hasattr(expr, "_expr"):
        expr = expr._expr
    return FlowDiagram(expr)
