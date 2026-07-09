//! Tree-reduction aggregate layer (`PartialReduce`).
//!
//! A reduction lowers to a per-chunk `Blockwise` (already handled) plus a
//! cascade of `PartialReduce` layers. Each `PartialReduce` output block applies
//! the shared aggregate `func` to a nested list of input blocks — dask builds
//! that list with `lol_tuples`: the reduced axes (`split_every` keys) become
//! nested lists, the kept axes are fixed coordinates. This layer reproduces that
//! structure as a single [`ArgSlot::List`] argument per task.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

#[pyclass]
pub struct PartialReduceLayer {
    name: String,
    func: Py<PyAny>,
    kwargs: Py<PyAny>,
    dep_name: String,
    numblocks: Vec<usize>,
    /// Per dimension: the `split_every` step for a reduced axis, or `0` for a
    /// kept axis. A dim is reduced iff its step is non-zero.
    steps: Vec<usize>,
    keepdims: bool,
}

#[pymethods]
impl PartialReduceLayer {
    #[new]
    fn new(
        name: String,
        func: Py<PyAny>,
        kwargs: Py<PyAny>,
        dep_name: String,
        numblocks: Vec<usize>,
        steps: Vec<usize>,
        keepdims: bool,
    ) -> Self {
        Self {
            name,
            func,
            kwargs,
            dep_name,
            numblocks,
            steps,
            keepdims,
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

/// Contiguous runs of length `size` over `0..n` (the last may be shorter),
/// matching `tlz.partition_all(size, range(n))`.
fn partition_all(size: usize, n: usize) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        let end = (i + size).min(n);
        out.push((i as u32..end as u32).collect());
        i = end;
    }
    out
}

impl PartialReduceLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        // Per-dim partitions of the input block ids; kept axes are singletons.
        let parts: Vec<Vec<Vec<u32>>> = (0..ndim)
            .map(|d| {
                let psize = if self.steps[d] != 0 { self.steps[d] } else { 1 };
                partition_all(psize, self.numblocks[d])
            })
            .collect();
        let n_out: Vec<usize> = parts.iter().map(|p| p.len()).collect();
        let total: usize = if ndim == 0 { 1 } else { n_out.iter().product() };

        let mut tasks = Vec::with_capacity(total);
        let mut oi = vec![0usize; ndim]; // output index into parts per dim

        for _ in 0..total {
            // Output key coord: full index when keepdims, else drop reduced axes.
            let coord: Vec<u32> = (0..ndim)
                .filter(|&d| self.keepdims || self.steps[d] == 0)
                .map(|d| oi[d] as u32)
                .collect();

            // The single nested-list argument (dask's lol_tuples).
            let mut leaf = vec![0u32; ndim];
            let arg = self.build_lol(0, &parts, &oi, &mut leaf);
            tasks.push(NeutralTask {
                nbytes: 0,
                name_idx: 0,
                coord,
                compute: Compute::Call { func_idx: 0 },
                slots: vec![arg],
            });

            // Advance the output index in C order.
            for d in (0..ndim).rev() {
                oi[d] += 1;
                if oi[d] < n_out[d] {
                    break;
                }
                oi[d] = 0;
            }
        }

        Expanded {
            names: vec![&self.name],
            funcs: vec![&self.func],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: std::slice::from_ref(&self.dep_name),
            tasks,
        }
    }

    /// Recursively build the lol_tuples nested argument: reduced dims expand into
    /// a list over their selected partition, kept dims fix a coordinate.
    fn build_lol(
        &self,
        dim: usize,
        parts: &[Vec<Vec<u32>>],
        oi: &[usize],
        leaf: &mut Vec<u32>,
    ) -> ArgSlot {
        let ndim = self.numblocks.len();
        if dim == ndim {
            return ArgSlot::Dep {
                name_idx: 0,
                coord: leaf.clone(),
            };
        }
        let selected = &parts[dim][oi[dim]];
        if self.steps[dim] != 0 {
            let items = selected
                .iter()
                .map(|&v| {
                    leaf[dim] = v;
                    self.build_lol(dim + 1, parts, oi, leaf)
                })
                .collect();
            ArgSlot::List(items)
        } else {
            leaf[dim] = selected[0];
            self.build_lol(dim + 1, parts, oi, leaf)
        }
    }
}
