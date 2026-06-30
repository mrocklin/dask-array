//! Blocks layer: block-index selection (`x.blocks[...]`), pure aliasing.
//!
//! `x.blocks[index]` selects a subset of `x`'s blocks by *block* index (not
//! element index). Like concatenate, every output block maps 1:1 to exactly one
//! input block — there is no per-block computation, only routing. The layer
//! therefore uses [`Compute::Alias`] for every task: `funcs` is empty and
//! `kwargs` is unused; both converters skip them for Alias tasks.
//!
//! Index mapping (mirrors `Blocks._layer()` exactly):
//!   - Python pre-computes, per dimension `d`, a remap list
//!     `index_maps[d] = np.arange(numblocks[d])[index[d]]` — output position →
//!     input block index along `d`. The output numblocks is `len(index_maps[d])`.
//!   - For output block `coord`: `in_coord[d] = index_maps[d][coord[d]]`.
//!
//! The Python `_frisky_layer` does the `np.arange(...)[idx]` remap (reusing
//! numpy) and passes the plain `int` lists here, so this constructor needs no
//! knowledge of slices/lists — just the per-dimension lookup tables.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct BlocksLayer {
    /// Name of the output array.
    out_name: String,
    /// Name of the single input array. Stored as a 1-element `Vec` so it can be
    /// borrowed as `dep_names` (an `ArgSlot::Dep` always uses `name_idx == 0`).
    dep_names: Vec<String>,
    /// Per-dimension remap lists: `index_maps[d][out_pos] = input_block_index`.
    /// `index_maps.len() == ndim`; `index_maps[d].len()` is the output numblocks
    /// along `d`.
    index_maps: Vec<Vec<u32>>,
    /// Empty dict, stored so `Expanded.kwargs` has a valid reference.
    empty_kwargs: Py<PyAny>,
}

#[pymethods]
impl BlocksLayer {
    /// `dep_name`: the single input array name.
    /// `index_maps`: per-dimension output-position → input-block-index lists
    /// (already remapped through `np.arange(numblocks)[idx]` in Python). The
    /// output numblocks along each dimension is the length of its remap list.
    #[new]
    fn new(py: Python<'_>, out_name: String, dep_name: String, index_maps: Vec<Vec<u32>>) -> Self {
        let empty_kwargs = PyDict::new(py).unbind().into_any();
        Self {
            out_name,
            dep_names: vec![dep_name],
            index_maps,
            empty_kwargs,
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

impl BlocksLayer {
    /// Pure-Rust expansion. Iterates output blocks in C order; for each block
    /// remaps every output coordinate through `index_maps[d]` to get the source
    /// block coord, then emits an `Alias` task. No funcs or kwargs are needed.
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.index_maps.len();
        // Output numblocks per dimension is the length of each remap list.
        let out_numblocks: Vec<usize> = self.index_maps.iter().map(|m| m.len()).collect();
        let total: usize = if ndim == 0 {
            1
        } else {
            out_numblocks.iter().product()
        };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for _ in 0..total {
            // Source coord: remap each output coordinate through its dimension's
            // lookup list (mirrors `index_maps[d][out_key[d]]` in Python).
            let src_coord: Vec<u32> = coord
                .iter()
                .enumerate()
                .map(|(d, &c)| self.index_maps[d][c as usize])
                .collect();

            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Alias,
                slots: vec![ArgSlot::Dep {
                    name_idx: 0,
                    coord: src_coord,
                }],
            });

            // Advance coord in C order (last axis fastest).
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < out_numblocks[d] {
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
