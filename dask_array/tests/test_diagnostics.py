import numpy as np
import pytest

import dask_array as da
from dask_array._diagnostics import explain, trace_rewrites


@pytest.fixture
def sliced_pipeline():
    x = da.from_array(np.ones((100, 100)), chunks=(10, 10))
    y = da.from_array(np.ones((100, 100)), chunks=(10, 10))
    return ((x + y) * 2).sum(axis=0)[:50]


def test_trace_records_slice_pushdown(sliced_pipeline):
    with trace_rewrites() as t:
        sliced_pipeline.expr.simplify()

    assert t.records
    # The slice was rewritten away: some rule fired on a SliceSlicesIntegers node
    assert any(r.before_type == "SliceSlicesIntegers" for r in t.records)
    # Every record names the rule that fired and what it produced
    for r in t.records:
        assert r.rule and r.after_type
        assert r.phase in ("simplify", "lower")


def test_trace_records_lowering(sliced_pipeline):
    with trace_rewrites() as t:
        sliced_pipeline.expr.simplify().lower_completely()

    lower_rules = {r.rule for r in t.records if r.phase == "lower"}
    # The abstract Sum lowered into the blockwise + tree-reduce cascade
    assert any(rule.endswith("._lower") for rule in lower_rules)
    assert any(rule.startswith("Sum.") for rule in lower_rules)


def test_trace_unpatches_on_exit(sliced_pipeline):
    with trace_rewrites() as t:
        sliced_pipeline.expr.simplify()
    n = len(t.records)

    # A fresh, differently-parameterized pipeline so simplification really runs
    # again (results of the traced run are cached by name).
    x = da.from_array(np.ones((60, 60)), chunks=(10, 10))
    ((x + 3) * 7).sum(axis=1)[:20].expr.simplify()
    assert len(t.records) == n


def test_trace_repr_aggregates(sliced_pipeline):
    with trace_rewrites() as t:
        sliced_pipeline.expr.simplify()
    text = repr(t)
    assert "rewrites" in text
    assert "SliceSlicesIntegers" in text


def test_explain_phases(sliced_pipeline):
    report = explain(sliced_pipeline)

    phases = report.phases
    assert [p.name for p in phases] == ["raw", "simplified", "lowered", "fused"]
    raw, simplified, lowered, fused = phases

    # Fusion and pushdown shrink the graph
    assert fused.tasks < raw.tasks
    # Slice absorption into FromArray regions shrinks bytes read at the leaves
    assert simplified.read_bytes < raw.read_bytes
    # Un-lowered phases don't have task counts
    assert simplified.tasks is None
    assert lowered.tasks >= fused.tasks

    # Rule attribution present
    assert report.simplify_rules
    assert any(rule.endswith("._lower") for rule in report.lower_rules)
    # One fused group covering several blockwise ops
    assert report.fusion_groups and max(report.fusion_groups) >= 2


def test_explain_repr(sliced_pipeline):
    text = repr(explain(sliced_pipeline))
    for token in ("raw", "simplified", "lowered", "fused", "rules"):
        assert token in text


def test_explain_trivial_expr():
    x = da.from_array(np.ones((10, 10)), chunks=(5, 5))
    report = explain(x)
    assert report.phases[0].nodes == 1
    assert report.phases[-1].tasks >= 4
    repr(report)  # renders without error


def test_explain_accepts_expr_or_collection(sliced_pipeline):
    a = explain(sliced_pipeline)
    b = explain(sliced_pipeline.expr)
    assert [p.tasks for p in a.phases] == [p.tasks for p in b.phases]
