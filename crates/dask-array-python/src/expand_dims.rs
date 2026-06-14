//! Dimension expansion layer (expand_dims).
//!
//! Each output block is `getitem(input_block, indexer)` where `indexer` is the
//! same for every block (shared literal, `ArgSlot::Literal(0)`).
//!
//! The output block coord is the input coord with `0` inserted at each of the
//! sorted expansion `axes`; equivalently, the input coord is the output coord
//! with those positions removed. Output has one block on each inserted axis
//! (the Python layer already validates `axes` and chunk shapes).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct ExpandDimsLayer {
    name: String,
    input_name: String,
    func: PyObject,
    kwargs: PyObject,
    /// The shared indexer tuple (one per layer, same for every block).
    indexer: PyObject,
    /// Block counts of the INPUT array (output has 1 block on each new axis).
    input_numblocks: Vec<usize>,
    /// Sorted expansion axes (positions in the OUTPUT coordinate).
    axes: Vec<usize>,
}

#[pymethods]
impl ExpandDimsLayer {
    #[new]
    fn new(
        name: String,
        input_name: String,
        func: PyObject,
        kwargs: PyObject,
        indexer: PyObject,
        input_numblocks: Vec<usize>,
        axes: Vec<usize>,
    ) -> Self {
        Self { name, input_name, func, kwargs, indexer, input_numblocks, axes }
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }
}

impl ExpandDimsLayer {
    /// Expand into the neutral form.
    ///
    /// Iterates input blocks in C order. For each input coord:
    /// - output coord = input coord with `0` inserted at each axis in `self.axes`.
    /// - slots = [Dep{input, input_coord}, Literal(0) (the indexer)]
    fn expand(&self) -> Expanded<'_> {
        let in_ndim = self.input_numblocks.len();
        let total: usize = if in_ndim == 0 { 1 } else { self.input_numblocks.iter().product() };
        let out_ndim = in_ndim + self.axes.len();
        let mut tasks = Vec::with_capacity(total);
        let mut in_coord = vec![0u32; in_ndim];

        // For each output dimension, mark whether it's an expansion axis.
        let axes_set: Vec<bool> = (0..out_ndim).map(|i| self.axes.contains(&i)).collect();

        for _ in 0..total {
            // Build output coord: walk output dims, inserting 0 at expansion
            // axes and consuming in_coord values at non-expansion axes.
            let mut out_coord = Vec::with_capacity(out_ndim);
            let mut in_pos = 0;
            for is_new in &axes_set {
                if *is_new {
                    out_coord.push(0u32);
                } else {
                    out_coord.push(in_coord[in_pos]);
                    in_pos += 1;
                }
            }

            tasks.push(NeutralTask {
                name_idx: 0,
                coord: out_coord,
                compute: Compute::Call { func_idx: 0 },
                slots: vec![
                    // dep_names[0] is the input array name
                    ArgSlot::Dep { name_idx: 0, coord: in_coord.clone() },
                    // literals[0] is the shared indexer tuple
                    ArgSlot::Literal(0),
                ],
            });

            // Increment in_coord in C order.
            for d in (0..in_ndim).rev() {
                in_coord[d] += 1;
                if (in_coord[d] as usize) < self.input_numblocks[d] {
                    break;
                }
                in_coord[d] = 0;
            }
        }

        Expanded {
            names: vec![&self.name],
            funcs: vec![&self.func],
            kwargs: &self.kwargs,
            literals: std::slice::from_ref(&self.indexer),
            dep_names: std::slice::from_ref(&self.input_name),
            tasks,
        }
    }
}
