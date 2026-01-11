"""Rich-based visualization for array expressions."""

from __future__ import annotations

import io
import math
from math import isnan, nan, prod

from dask.utils import funcname

# Color coding using Tango palette for readability
# Orange (warm) = sources, new data entering the computation
# Blue (cool) = reducers, data being reduced/consumed
SOURCE_COLOR = "#ce5c00"  # Tango orange dark
REDUCER_COLOR = "#3465a4"  # Tango sky blue


def format_bytes(nbytes: float) -> str:
    """Format bytes with 2 significant figures."""
    if math.isnan(nbytes):
        return "?"

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
    return f"{int(nbytes)} B"


class ExprTable:
    """Wrapper for rich Table with Jupyter and terminal display support."""

    def __init__(self, table):
        self._table = table
        self._html_cache = None
        self._text_cache = None

    def _repr_html_(self):
        """Jupyter notebook display."""
        if self._html_cache is None:
            from rich.console import Console

            console = Console(file=io.StringIO(), record=True, width=120, force_jupyter=False)
            console.print(self._table)
            self._html_cache = console.export_html(inline_styles=True, code_format="<pre>{code}</pre>")
        return self._html_cache

    def __repr__(self):
        """Terminal display."""
        if self._text_cache is None:
            from rich.console import Console

            console = Console(file=io.StringIO(), force_terminal=True, force_jupyter=False, width=120)
            console.print(self._table)
            self._text_cache = console.file.getvalue().rstrip()
        return self._text_cache

    def __str__(self):
        return self.__repr__()

    def print(self):
        """Print to the current console."""
        from rich.console import Console

        Console().print(self._table)


def _walk_expr(expr, prefix: str = "", is_last: bool = True):
    """Walk expression tree depth-first, yielding (expr, display_prefix) pairs."""
    yield expr, prefix

    deps = [op for op in expr.dependencies() if hasattr(op, "chunks")]

    for i, child in enumerate(deps):
        is_last_child = i == len(deps) - 1
        if prefix == "":
            child_prefix = ""
        else:
            child_prefix = prefix[:-2] + ("  " if is_last else "│ ")
        branch = "└ " if is_last_child else "├ "
        yield from _walk_expr(child, child_prefix + branch, is_last_child)


def _compute_row_emphasis(values: list[float], threshold: float = 0.5) -> list[bool]:
    """Compute which rows should be emphasized based on relative values."""
    valid_values = [v for v in values if not math.isnan(v)]
    if not valid_values:
        return [True] * len(values)

    max_value = max(valid_values)
    if max_value <= 0:
        return [True] * len(values)

    return [not math.isnan(v) and v > threshold * max_value for v in values]


def _get_op_display_name(node, use_label_for: frozenset) -> str:
    """Get the display name for an operation."""
    class_name = funcname(type(node))

    if class_name not in use_label_for:
        return class_name

    # Extract prefix from _name (everything before the hash)
    expr_name = node._name
    if "-" in expr_name:
        parts = expr_name.rsplit("-", 1)
        if len(parts) == 2 and len(parts[1]) >= 8:
            label = parts[0]
            label = label.replace("_", " ")
            for suffix in ["-aggregate", "-partial"]:
                if suffix in label:
                    label = label.replace(suffix, "")
            label = label.replace("-", " ").strip()
            return label.title()

    return class_name


def _get_op_color(node) -> str | None:
    """Determine operation color based on class hierarchy and data flow."""
    from dask_array._expr import ArrayExpr
    from dask_array.reductions._reduction import PartialReduce
    from dask_array.slicing._basic import Slice

    # Sources: no ArrayExpr dependencies (data enters here)
    deps = [op for op in node.operands if isinstance(op, ArrayExpr)]
    if not deps:
        return SOURCE_COLOR

    # Reducers: PartialReduce or Slice subclasses (data shrinks here)
    if isinstance(node, (PartialReduce, Slice)):
        return REDUCER_COLOR

    return None


def _get_nbytes(node) -> float:
    """Get the number of bytes for an expression, or NaN if unknown."""
    try:
        shape = node.shape
        if any(isnan(s) for s in shape):
            return nan
        return prod(shape) * node.dtype.itemsize
    except Exception:
        return nan


# Operations where we prefer showing the _name prefix as the primary name
_USE_LABEL_AS_NAME = frozenset({"Blockwise", "PartialReduce", "Elemwise", "Random", "SliceSlicesIntegers"})


def expr_table(expr, color: bool = True) -> ExprTable:
    """
    Display expression tree as a table.

    Parameters
    ----------
    expr : ArrayExpr
        The expression to visualize
    color : bool
        Whether to color-code operations by type

    Returns
    -------
    ExprTable
        A displayable table object (works in Jupyter and terminal)
    """
    from rich.table import Table
    from rich.text import Text

    table = Table(
        show_header=True,
        header_style="dim",
        box=None,
        padding=(0, 2),
        collapse_padding=True,
    )

    table.add_column("Operation", no_wrap=True)
    table.add_column("Shape", justify="right", no_wrap=True)
    table.add_column("Bytes", justify="right", no_wrap=True)
    table.add_column("Chunks", justify="right", no_wrap=True)

    # Collect nodes and compute emphasis based on bytes
    nodes_and_prefixes = list(_walk_expr(expr))
    node_bytes = [_get_nbytes(node) for node, _ in nodes_and_prefixes]
    row_emphasis = _compute_row_emphasis(node_bytes)

    for (node, prefix), nbytes, emphasize in zip(nodes_and_prefixes, node_bytes, row_emphasis):
        display_name = _get_op_display_name(node, _USE_LABEL_AS_NAME)
        data_style = None if color and emphasize else "dim"

        if color:
            op_color = _get_op_color(node)
            op_style = f"bold {op_color}" if op_color else "bold"
            op_text = Text()
            op_text.append(prefix, style="dim")
            op_text.append(display_name, style=op_style)
        else:
            op_text = f"{prefix}{display_name}"

        # Format shape and chunks
        shape_str = "()" if not node.shape else f"({', '.join(str(s) for s in node.shape)})"
        chunks_str = "×".join(str(c[0] if c else 0) for c in node.chunks) if node.chunks else ""

        table.add_row(
            op_text,
            Text(shape_str, style=data_style),
            Text(format_bytes(nbytes), style=data_style),
            Text(chunks_str, style=data_style),
        )

    return ExprTable(table)
