"""Tests for expression visualization."""

import dask_array as da
from dask_array._visualize import expr_table


def test_expr_table_contains_shapes():
    """Test that expr_table output contains array shapes."""
    x = da.ones((10, 100), chunks=(5, 50))
    y = x.sum()

    table = expr_table(y.expr)
    text = str(table)

    # Should contain the input shape
    assert "(10, 100)" in text
    # Should contain scalar output shape
    assert "()" in text


def test_expr_table_contains_bytes():
    """Test that expr_table output contains byte sizes."""
    x = da.ones((10, 100), chunks=(5, 50))
    y = x.sum()

    table = expr_table(y.expr)
    text = str(table)

    # 1000 float64 elements = 8000 bytes = 7.8 kiB
    assert "kiB" in text
    # Scalar output = 8 bytes
    assert "8 B" in text


def test_expr_table_contains_operation_names():
    """Test that expr_table shows operation names."""
    x = da.ones((10, 10), chunks=5)
    y = x + 1
    z = y.sum()

    table = expr_table(z.expr)
    text = str(table)

    assert "Reduction" in text
    assert "Ones" in text


def test_expr_table_styling_emphasis():
    """Test that large arrays are bold and small arrays are dim."""
    x = da.ones((100, 100), chunks=50)
    y = x.sum()  # Reduces to scalar

    table = expr_table(y.expr)
    text = str(table)

    # The repr includes ANSI codes for styling
    # [1m = bold, [2m = dim
    # Large array (input) should be bold, scalar output should be dim
    assert "[1m" in text  # Bold styling present
    assert "[2m" in text  # Dim styling present


def test_expr_table_html_output():
    """Test that HTML output is generated for Jupyter."""
    x = da.ones((10, 10), chunks=5)
    y = x.sum()

    table = expr_table(y.expr)
    html = table._repr_html_()

    assert "<pre>" in html
    assert "Reduction" in html
    assert "Ones" in html
