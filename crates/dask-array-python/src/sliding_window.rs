//! Banded sliding/moving-window reduction layers (native-chunk rolling
//! reductions; `dask_array/reductions/_sliding_window.py`).
//!
//! Both layers share one task shape per output block:
//!
//!   `reduce_func(x[i], [total[q] …], [x[band] …], scalar, scalar)`
//!
//! plus shared `total_func(x[q])` tasks for the blocks any window covers
//! whole. The banded plan — which blocks form the band, their offsets, and
//! (moving) the truncation count — is computed by the expression in Python
//! and passed as flat integer rows; this layer only walks the block grid and
//! emits. Everything is `Dep`/`List`/`Scalar` slots, so both layers are
//! binary-records expressible.
//!
//! Forward (`SlidingWindowReductionLayer`, from
//! `reduction(sliding_window_view(x, W))`): only blocks with `out_len > 0`
//! emit; middles are `i+1..band_lo`; with `keepdims` the output coord gains a
//! 0 at `window_axis`.
//!
//! Trailing (`MovingWindowReductionLayer`, bottleneck `move_*` semantics):
//! every block emits; middles are `band_hi+1..i`; `band_lo == -1` marks the
//! array-start block, which has no band and no middles.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{
    grid_nbytes, to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num,
};

/// The band invariants below are guaranteed by the Python gates
/// (`supports_native_*` in `dask_array/reductions/_sliding_window.py`); if
/// gate and layer ever drift, degrade to the adapter tier (the collect walk
/// catches `NotImplementedError`) instead of panicking mid-emission.
fn check_plan(
    plan: &[Vec<i64>],
    numblocks: &[usize],
    axis: usize,
    ok: impl Fn(usize, &[i64]) -> bool,
) -> PyResult<()> {
    if axis >= numblocks.len()
        || plan.len() != numblocks[axis]
        || plan
            .iter()
            .enumerate()
            .any(|(i, row)| row.len() != 4 || !ok(i, row))
    {
        return Err(PyNotImplementedError::new_err(
            "banded sliding-window plan violates the band invariants",
        ));
    }
    Ok(())
}

// Name indices (also `Dep` name_idx).
const N_OUT: usize = 0;
const N_TOTAL: usize = 1;
const N_X: usize = 2;
// Func indices.
const F_REDUCE: usize = 0;
const F_TOTAL: usize = 1;

/// Walk every position of the non-axis dimensions, handing the callback a
/// closure that builds the full block coord for a given axis block.
fn for_each_non_axis_position(
    numblocks: &[usize],
    axis: usize,
    mut body: impl FnMut(&dyn Fn(u32) -> Vec<u32>),
) {
    let ndim = numblocks.len();
    let non_axis: Vec<usize> = (0..ndim).filter(|&d| d != axis).collect();
    let na_total: usize = non_axis.iter().map(|&d| numblocks[d]).product();
    let mut na_coord = vec![0u32; non_axis.len()];
    for _ in 0..na_total {
        let full_coord = |i: u32| -> Vec<u32> {
            let mut c = vec![0u32; ndim];
            c[axis] = i;
            for (k, &d) in non_axis.iter().enumerate() {
                c[d] = na_coord[k];
            }
            c
        };
        body(&full_coord);
        for k in (0..non_axis.len()).rev() {
            na_coord[k] += 1;
            if (na_coord[k] as usize) < numblocks[non_axis[k]] {
                break;
            }
            na_coord[k] = 0;
        }
    }
}

/// Emit the shared banded task set. Per plan row `i` with `emit(i)` true:
/// `reduce_func(x[i], [totals over mids(i)], [x over band(i)], scalar_a(i),
/// scalar_b(i))`, plus one `total_func(x[q])` per block in any mids range.
/// `out_coord` maps the input coord to the output coord (keepdims insertion).
#[allow(clippy::too_many_arguments)]
fn expand_banded<'a>(
    layer_names: &'a [String],
    funcs: &'a [Py<PyAny>],
    kwargs: &'a Py<PyAny>,
    numblocks: &[usize],
    axis: usize,
    chunks: &[Vec<i64>],
    total_itemsize: i64,
    emit: impl Fn(usize) -> bool,
    mids: impl Fn(usize) -> (usize, usize),
    band: impl Fn(usize) -> (i64, i64),
    scalars: impl Fn(usize) -> (i64, i64),
    out_coord: impl Fn(&[u32]) -> Vec<u32>,
) -> Expanded<'a> {
    let n_axis = numblocks[axis];
    let mut needed = vec![false; n_axis];
    for i in 0..n_axis {
        if emit(i) {
            let (lo, hi) = mids(i);
            needed[lo..hi].fill(true);
        }
    }

    let mut tasks: Vec<NeutralTask> = Vec::new();
    for_each_non_axis_position(numblocks, axis, |full_coord| {
        for (q, _) in needed.iter().enumerate().filter(|(_, &n)| n) {
            let coord = full_coord(q as u32);
            // A block total is one keepdims hyperplane (1 along the axis);
            // `total_itemsize` bundles the value + count planes when the
            // reducer tracks counts, and is 0 (no stamp) for unknown chunks.
            let ndim = numblocks.len();
            let nbytes = grid_nbytes(
                total_itemsize,
                (0..ndim).map(|d| {
                    if d == axis {
                        1
                    } else {
                        chunks[d][coord[d] as usize]
                    }
                }),
            );
            tasks.push(NeutralTask {
                nbytes,
                name_idx: N_TOTAL,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: F_TOTAL },
                slots: vec![ArgSlot::Dep {
                    name_idx: N_X,
                    coord,
                }],
            });
        }
        for i in 0..n_axis {
            if !emit(i) {
                continue;
            }
            let in_coord = full_coord(i as u32);
            let (mid_lo, mid_hi) = mids(i);
            let mid_deps = (mid_lo..mid_hi)
                .map(|q| ArgSlot::Dep {
                    name_idx: N_TOTAL,
                    coord: full_coord(q as u32),
                })
                .collect();
            let (band_lo, band_hi) = band(i);
            let band_deps = if band_lo < 0 {
                Vec::new()
            } else {
                (band_lo..=band_hi)
                    .map(|q| ArgSlot::Dep {
                        name_idx: N_X,
                        coord: full_coord(q as u32),
                    })
                    .collect()
            };
            let (a, b) = scalars(i);
            tasks.push(NeutralTask {
                nbytes: 0,
                name_idx: N_OUT,
                coord: out_coord(&in_coord),
                compute: Compute::Call { func_idx: F_REDUCE },
                slots: vec![
                    ArgSlot::Dep {
                        name_idx: N_X,
                        coord: in_coord,
                    },
                    ArgSlot::List(mid_deps),
                    ArgSlot::List(band_deps),
                    ArgSlot::Scalar(Num::Int(a)),
                    ArgSlot::Scalar(Num::Int(b)),
                ],
            });
        }
    });

    Expanded {
        names: layer_names.iter().map(|s| s.as_str()).collect(),
        funcs: funcs.iter().collect(),
        kwargs,
        literals: &[],
        dep_names: layer_names,
        tasks,
    }
}

#[pyclass]
pub struct SlidingWindowReductionLayer {
    /// `[output, total, x]`.
    names: Vec<String>,
    /// `[reduce_func, total_func]`.
    funcs: Vec<Py<PyAny>>,
    kwargs: Py<PyAny>,
    axis: usize,
    numblocks: Vec<usize>,
    keepdims: bool,
    window_axis: usize,
    /// Per axis block: `[out_len, band_offset, band_lo, band_hi]`.
    plan: Vec<Vec<i64>>,
    /// Input chunk sizes + effective per-element size of a block total —
    /// only for the `-total` expected-nbytes stamps (0 disables).
    chunks: Vec<Vec<i64>>,
    total_itemsize: i64,
}

#[pymethods]
impl SlidingWindowReductionLayer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        x_name: String,
        reduce_func: Py<PyAny>,
        total_func: Py<PyAny>,
        kwargs: Py<PyAny>,
        axis: usize,
        numblocks: Vec<usize>,
        keepdims: bool,
        window_axis: usize,
        plan: Vec<Vec<i64>>,
        chunks: Vec<Vec<i64>>,
        total_itemsize: i64,
    ) -> PyResult<Self> {
        // Emitting rows: the band starts past the block itself and ends in-grid.
        check_plan(&plan, &numblocks, axis, |i, row| {
            row[0] <= 0
                || (row[2] > i as i64 && row[2] <= row[3] && (row[3] as usize) < numblocks[axis])
        })?;
        if keepdims && window_axis > numblocks.len() {
            return Err(PyNotImplementedError::new_err(
                "window_axis outside the output grid",
            ));
        }
        Ok(Self {
            names: vec![name.clone(), format!("{name}-total"), x_name],
            funcs: vec![reduce_func, total_func],
            kwargs,
            axis,
            numblocks,
            keepdims,
            window_axis,
            plan,
            chunks,
            total_itemsize,
        })
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }

    fn to_records_chunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        crate::common::to_records_chunk(py, &self.expand())
    }
}

impl SlidingWindowReductionLayer {
    fn expand(&self) -> Expanded<'_> {
        expand_banded(
            &self.names,
            &self.funcs,
            &self.kwargs,
            &self.numblocks,
            self.axis,
            &self.chunks,
            self.total_itemsize,
            |i| self.plan[i][0] > 0,
            |i| (i + 1, self.plan[i][2] as usize),
            |i| (self.plan[i][2], self.plan[i][3]),
            |i| (self.plan[i][0], self.plan[i][1]),
            |in_coord| {
                let mut c = in_coord.to_vec();
                if self.keepdims {
                    c.insert(self.window_axis, 0);
                }
                c
            },
        )
    }
}

#[pyclass]
pub struct MovingWindowReductionLayer {
    /// `[output, total, x]`.
    names: Vec<String>,
    /// `[reduce_func, total_func]`.
    funcs: Vec<Py<PyAny>>,
    kwargs: Py<PyAny>,
    axis: usize,
    numblocks: Vec<usize>,
    /// Per axis block: `[n_trunc, band_offset, band_lo, band_hi]`;
    /// `band_lo == -1` means no band (the block starting the array).
    plan: Vec<Vec<i64>>,
    /// Input chunk sizes + effective per-element size of a block total —
    /// only for the `-total` expected-nbytes stamps (0 disables).
    chunks: Vec<Vec<i64>>,
    total_itemsize: i64,
}

#[pymethods]
impl MovingWindowReductionLayer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        x_name: String,
        reduce_func: Py<PyAny>,
        total_func: Py<PyAny>,
        kwargs: Py<PyAny>,
        axis: usize,
        numblocks: Vec<usize>,
        plan: Vec<Vec<i64>>,
        chunks: Vec<Vec<i64>>,
        total_itemsize: i64,
    ) -> PyResult<Self> {
        // A band (band_lo >= 0) ends before the block itself; the array-start
        // block has neither band nor middles (both markers negative).
        check_plan(&plan, &numblocks, axis, |i, row| {
            if row[2] < 0 {
                row[3] < 0
            } else {
                row[2] <= row[3] && row[3] < i as i64
            }
        })?;
        Ok(Self {
            names: vec![name.clone(), format!("{name}-total"), x_name],
            funcs: vec![reduce_func, total_func],
            kwargs,
            axis,
            numblocks,
            plan,
            chunks,
            total_itemsize,
        })
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }

    fn to_records_chunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        crate::common::to_records_chunk(py, &self.expand())
    }
}

impl MovingWindowReductionLayer {
    fn expand(&self) -> Expanded<'_> {
        expand_banded(
            &self.names,
            &self.funcs,
            &self.kwargs,
            &self.numblocks,
            self.axis,
            &self.chunks,
            self.total_itemsize,
            |_| true,
            |i| {
                let hi = self.plan[i][3];
                if hi < 0 {
                    (i, i) // empty
                } else {
                    (hi as usize + 1, i)
                }
            },
            |i| (self.plan[i][2], self.plan[i][3]),
            |i| (self.plan[i][0], self.plan[i][1]),
            |in_coord| in_coord.to_vec(),
        )
    }
}
