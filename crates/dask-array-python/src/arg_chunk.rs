//! ArgChunk layer: the per-block chunk step of an arg reduction (`argmin`/
//! `argmax`), non-ravel case.
//!
//! The reference `_layer` emits, for every input block index `k` (C order over
//! `x.numblocks`), a task `(chunk_func, (x.name, *k), axis, off)` — i.e.
//! `chunk_func(x_block, axis, off)`. `axis` is shared across all blocks; `off`
//! is the per-block start offset along the reduction axis
//! (`pluck(axis[0], offsets)` in the legacy non-ravel branch). So it fits the
//! neutral form exactly: one shared func, output coord == input coord (identity,
//! like coarsen), with three args per task — a single `ArgSlot::Dep` (the input
//! block), one `ArgSlot::Literal` (the shared `axis`), and one
//! `ArgSlot::Scalar(Num::Int(off))` (the per-block offset).
//!
//! Only the non-ravel case is modeled here: the ravel offset is a nested
//! `(offsets, x.shape)` tuple the simple `Scalar` slot can't carry, so the
//! routing raises `NotImplementedError` and falls back to legacy dask.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct ArgChunkLayer {
    /// Output layer name.
    name: String,
    /// The shared `chunk_func` (the partialed arg-chunk function).
    func: PyObject,
    /// Shared kwargs applied to every call — empty.
    kwargs: PyObject,
    /// Shared literals `[axis]`, referenced by `ArgSlot::Literal(0)`.
    literals: Vec<PyObject>,
    /// Name of the single input dependency.
    dep_names: Vec<String>,
    /// Number of blocks per dimension (input grid; arg-chunk preserves the grid,
    /// so it is also the output grid).
    numblocks: Vec<usize>,
    /// Per-block start offset along the reduction axis, one per block in C order.
    offs: Vec<i64>,
}

#[pymethods]
impl ArgChunkLayer {
    /// `axis`: shared per-block arg (the reduction axes tuple). `dep_name`: the
    /// input array name. `numblocks`: blocks per dimension. `offs`: the per-block
    /// offsets, already computed in Python (one per block, C order), mirroring the
    /// legacy non-ravel `pluck(axis[0], offsets)`.
    #[new]
    fn new(
        name: String,
        func: PyObject,
        kwargs: PyObject,
        axis: PyObject,
        dep_name: String,
        numblocks: Vec<usize>,
        offs: Vec<i64>,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            literals: vec![axis],
            dep_names: vec![dep_name],
            numblocks,
            offs,
        }
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }
}

impl ArgChunkLayer {
    /// Build the neutral form. One task per block, output coord == input coord, a
    /// single dependency at the same coord plus the shared `axis` literal and the
    /// per-block offset scalar. Iterates blocks in C order to match
    /// `itertools.product` (last axis fastest), so `offs[p]` lines up with the
    /// `p`-th block.
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total: usize = if ndim == 0 { 1 } else { self.numblocks.iter().product() };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for p in 0..total {
            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                // chunk_func(x_block, axis, off)
                slots: vec![
                    ArgSlot::Dep { name_idx: 0, coord: coord.clone() },
                    ArgSlot::Literal(0), // axis
                    ArgSlot::Scalar(Num::Int(self.offs[p])),
                ],
            });

            for d in (0..ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < self.numblocks[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        Expanded {
            names: vec![&self.name],
            funcs: vec![&self.func],
            kwargs: &self.kwargs,
            literals: &self.literals,
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
