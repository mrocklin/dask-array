//! GUfunc leaf layer: the output-splitting step of `apply_gufunc`.
//!
//! `GUfuncLeafExpr._layer` maps each block of the gufunc's (loop-chunked) result
//! array to one leaf-output block. Mirroring it exactly:
//!   - single output (`nout` falsy): the leaf block *aliases* the source block —
//!     `{(leaf, *coord, *core0): (array, *coord)}`.
//!   - multiple outputs (`nout` truthy): the leaf block picks output `i` out of
//!     the source block's result tuple — `getitem(array_block, i)`.
//!
//! The leaf's output coord is the source (loop) coord followed by one 0 per core
//! output dimension (each core dim is a single chunk). `getitem` is dask-array's
//! copy-if-small `chunk.getitem`, passed from the Python wrapper.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask, Num};

#[pyclass]
pub struct GUfuncLeafLayer {
    /// Leaf output name (produced).
    name: String,
    /// `chunk.getitem`, used only when `nout` (multi-output).
    getitem: Py<PyAny>,
    /// Shared kwargs (empty).
    kwargs: Py<PyAny>,
    /// Source (gufunc result) array name — the single dependency.
    array_name: String,
    /// Source loop-block grid.
    numblocks: Vec<usize>,
    /// Number of trailing single-chunk core-output dims (`len(ocd)`).
    n_core: usize,
    /// Multiple outputs? Then pick element `i`; else alias the whole block.
    nout: bool,
    /// Which output to pick (the leaf index).
    i: i64,
}

#[pymethods]
impl GUfuncLeafLayer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        getitem: Py<PyAny>,
        kwargs: Py<PyAny>,
        array_name: String,
        numblocks: Vec<usize>,
        n_core: usize,
        nout: bool,
        i: i64,
    ) -> Self {
        Self {
            name,
            getitem,
            kwargs,
            array_name,
            numblocks,
            n_core,
            nout,
            i,
        }
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

impl GUfuncLeafLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total: usize = if ndim == 0 {
            1
        } else {
            self.numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for _ in 0..total {
            // Output coord: the source loop coord + one 0 per core-output dim.
            let mut out_coord = coord.clone();
            out_coord.extend(std::iter::repeat(0u32).take(self.n_core));

            let source = ArgSlot::Dep {
                name_idx: 0,
                coord: coord.clone(),
            };
            let task = if self.nout {
                // getitem(array_block, i)
                NeutralTask {
                    nbytes: 0,
                    name_idx: 0,
                    coord: out_coord,
                    compute: Compute::Call { func_idx: 0 },
                    slots: vec![source, ArgSlot::Scalar(Num::Int(self.i))],
                }
            } else {
                // leaf block aliases the source block
                NeutralTask {
                    nbytes: 0,
                    name_idx: 0,
                    coord: out_coord,
                    compute: Compute::Alias,
                    slots: vec![source],
                }
            };
            tasks.push(task);

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
            funcs: vec![&self.getitem],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: std::slice::from_ref(&self.array_name),
            tasks,
        }
    }
}
