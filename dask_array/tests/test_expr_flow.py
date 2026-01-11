"""Tests for expression flow visualization."""

import numpy as np
import pytest

import dask_array as da
from dask_array._expr_flow import (
    FlowDiagram,
    FlowEdge,
    FlowNode,
    build_flow_graph,
    count_operations,
    expr_flow,
    render_flow_svg,
)


def test_linear_chain_single_node():
    """Linear chain with same shape creates single node."""
    x = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    y = x + 1
    z = y * 2
    result = z - 0.5

    nodes, edges = build_flow_graph(result._expr)
    assert len(nodes) == 1
    assert len(edges) == 0
    assert nodes[0].shape == (100, 100)
    assert len(nodes[0].operations) == 4  # Load, Add, Mul, Sub


def test_reduction_creates_nodes():
    """Reduction creates nodes with edges."""
    x = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    y = x + 1
    result = y.sum()

    nodes, edges = build_flow_graph(result._expr)
    # Should have at least source and final result (may include intermediates)
    assert len(nodes) >= 2
    assert len(edges) >= 1
    shapes = {n.shape for n in nodes}
    assert (100, 100) in shapes  # Source
    assert () in shapes  # Final scalar result


def test_axis_reduction():
    """Axis reduction shows shape change."""
    x = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    result = x.sum(axis=0)

    nodes, edges = build_flow_graph(result._expr)
    # Should have at least source and final result
    assert len(nodes) >= 2
    shapes = {n.shape for n in nodes}
    assert (100, 100) in shapes  # Source
    assert (100,) in shapes  # Final reduced result


def test_multi_input_separate_nodes():
    """Multi-input creates separate source nodes."""
    a = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    b = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    result = a + b

    nodes, edges = build_flow_graph(result._expr)
    assert len(nodes) == 3  # Two inputs + one output
    assert len(edges) == 2  # Two edges to output


def test_layout_assignment():
    """Layout assigns correct columns."""
    x = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    result = x.sum()

    nodes, edges = build_flow_graph(result._expr)
    # Source should be at column 0, result at highest column
    cols = {n.shape: n.col for n in nodes}
    assert cols[(100, 100)] == 0  # Source at start
    # Final result at end (may be column 1 or higher if intermediates shown)
    assert cols[()] >= 1


def test_count_operations():
    """count_operations returns correct count."""
    x = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    y = x + 1
    result = y.sum()

    count = count_operations(result._expr)
    assert count >= 2  # At least load and sum


def test_expr_flow_accepts_array():
    """expr_flow works with Array objects."""
    x = da.ones((10, 10), chunks=5)
    flow = expr_flow(x)
    assert isinstance(flow, FlowDiagram)


def test_expr_flow_accepts_expr():
    """expr_flow works with ArrayExpr objects."""
    x = da.ones((10, 10), chunks=5)
    flow = expr_flow(x._expr)
    assert isinstance(flow, FlowDiagram)


def test_flow_diagram_repr():
    """FlowDiagram has text repr."""
    x = da.ones((10, 10), chunks=5)
    flow = expr_flow(x)
    text = repr(flow)
    assert "Expression:" in text
    assert "operations" in text


def test_flow_diagram_html():
    """FlowDiagram generates HTML."""
    x = da.ones((10, 10), chunks=5)
    flow = expr_flow(x)
    html = flow._repr_html_()
    assert "<div" in html
    assert "svg" in html or "text-align" in html


def test_render_flow_svg():
    """render_flow_svg produces HTML."""
    x = da.ones((10, 10), chunks=5)
    html = render_flow_svg(x._expr)
    assert "<div" in html


def test_key_shapes_present():
    """Key shapes (source and result) are present in flow graph."""
    x = da.from_array(np.random.random((100, 100)), chunks=(50, 50))
    result = x.sum()

    nodes, edges = build_flow_graph(result._expr)
    # Main shapes should always be present
    shapes = {n.shape for n in nodes}
    assert (100, 100) in shapes  # Source data
    assert () in shapes  # Final scalar result
    # May include intermediates - that's OK with conservative filtering
