//! Concatenate layer: alias each output block to exactly one input block.
//!
//! `da.concatenate([a0, a1, ...], axis=ax)` stacks arrays along `ax`. Each
//! output block maps 1:1 to a block in exactly one input array — there is no
//! per-block computation, only routing. The layer therefore uses
//! [`Compute::Alias`] for every task: `funcs` is empty and `kwargs` is
//! unused; both converters skip them for Alias tasks.
//!
//! Index mapping (mirrors `Concatenate._layer()` exactly):
//!   - `cum_dims = [0, n0, n0+n1, ...]` — cumulative block counts along `axis`.
//!   - For output block `coord`:
//!     - `ci = coord[axis]` — which output block along the concat axis.
//!     - `a  = bisect_right(cum_dims, ci) - 1` — which source array (0-based).
//!     - source coord = `coord` with `coord[axis]` replaced by `ci - cum_dims[a]`.
//!
//! The Python `_frisky_layer` passes `blocks_per_arr` so this constructor can
//! compute `cum_dims` in Rust (avoids shipping the array to Rust).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct ConcatenateLayer {
    /// Name of the output array.
    out_name: String,
    /// Names of the input arrays, in order. Also used as dep_names.
    dep_names: Vec<String>,
    /// The axis along which arrays are concatenated.
    axis: usize,
    /// Cumulative block counts along `axis`:
    /// `cum_dims[0] = 0`, `cum_dims[k] = sum(blocks_per_arr[0..k])`.
    /// Length = `dep_names.len() + 1`.
    cum_dims: Vec<usize>,
    /// Number of output blocks per output dimension (len = ndim).
    out_numblocks: Vec<usize>,
    /// Empty dict, stored so `Expanded.kwargs` has a valid reference.
    empty_kwargs: PyObject,
}

#[pymethods]
impl ConcatenateLayer {
    /// `dep_names`: input array names (in concat order).
    /// `blocks_per_arr`: number of blocks along `axis` for each input array.
    /// `out_numblocks`: blocks per output dimension (len = ndim).
    #[new]
    fn new(
        py: Python<'_>,
        out_name: String,
        dep_names: Vec<String>,
        axis: usize,
        blocks_per_arr: Vec<usize>,
        out_numblocks: Vec<usize>,
    ) -> Self {
        // Compute cumulative block counts: cum_dims[i] = sum(blocks_per_arr[0..i]).
        let mut cum_dims = Vec::with_capacity(blocks_per_arr.len() + 1);
        let mut acc = 0usize;
        cum_dims.push(0);
        for &n in &blocks_per_arr {
            acc += n;
            cum_dims.push(acc);
        }
        let empty_kwargs = PyDict::new(py).unbind().into_any();
        Self { out_name, dep_names, axis, cum_dims, out_numblocks, empty_kwargs }
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }
}

impl ConcatenateLayer {
    /// Pure-Rust expansion. Iterates output blocks in C order; for each block
    /// finds its source array + source coord via `cum_dims`, then emits an
    /// `Alias` task. No funcs or kwargs are needed.
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.out_numblocks.len();
        let total: usize =
            if ndim == 0 { 1 } else { self.out_numblocks.iter().product() };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for _ in 0..total {
            // The concat-axis index for this output block.
            let ci = coord[self.axis] as usize;

            // bisect_right(cum_dims, ci) - 1: find the source array.
            // Find the first position where cum_dims[pos] > ci, then subtract 1.
            // This mirrors Python's `bisect(cum_dims, key[axis+1]) - 1`.
            let a = self
                .cum_dims
                .partition_point(|&c| c <= ci)
                - 1;

            // Source coord: same as output coord everywhere except `axis`,
            // where we subtract the cumulative offset of source array `a`.
            let src_coord: Vec<u32> = coord
                .iter()
                .enumerate()
                .map(|(d, &c)| {
                    if d == self.axis {
                        (ci - self.cum_dims[a]) as u32
                    } else {
                        c
                    }
                })
                .collect();

            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Alias,
                slots: vec![ArgSlot::Dep { name_idx: a, coord: src_coord }],
            });

            // Advance coord in C order (last axis fastest).
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < self.out_numblocks[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        Expanded {
            names: vec![&self.out_name],
            // No funcs: Alias tasks don't call any function.
            funcs: vec![],
            kwargs: &self.empty_kwargs,
            literals: &[],
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
