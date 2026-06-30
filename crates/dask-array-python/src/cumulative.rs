//! Cumulative-reduction layer (`CumReduction`: cumsum, cumprod, …), the
//! sequential algorithm.
//!
//! This is the most intricate layer: within one layer it emits four task kinds
//! across four names, with a sequential carry along the reduction axis.
//! Mirroring `CumReduction._layer`:
//!   - **chunk** (`name-chunk`): per block, `chunk_func(x_block)` (a Python
//!     `partial(func, axis=…[, dtype=…])`, built in the wrapper).
//!   - **extra** (`name-extra`): the running carry. The first block along the
//!     axis is `full_like(meta, ident, dtype, shape=…)` (the identity, with size
//!     1 along the axis) — a per-block `shape` keyword, so `Compute::CallKw`.
//!     Subsequent blocks are `binop(extra[prev], tail[prev])`.
//!   - **tail** (`name-tail`): `getitem(chunk[blk], last-along-axis)`. The legacy
//!     nests this `getitem` inline inside the `binop` task; the neutral form has
//!     no nested-task arg, so we flatten it into its own named task.
//!   - **output** (`name`): first block along the axis aliases `chunk`;
//!     subsequent blocks are `binop(extra[blk], chunk[blk])`.
//!
//! The legacy keys the carry as `(name, "extra", *coord)` (a string in the key);
//! we instead give it its own layer name with an ordinary integer coord — the
//! value is identical, only the internal intermediate key differs.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{
    to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, IndexElem, NeutralTask,
};

// Name indices into the interned `names` list (also used as `Dep` name_idx).
const N_OUT: usize = 0;
const N_CHUNK: usize = 1;
const N_EXTRA: usize = 2;
const N_TAIL: usize = 3;
const N_X: usize = 4;
// Func indices.
const F_CHUNK: usize = 0;
const F_FULL_LIKE: usize = 1;
const F_GETITEM: usize = 2;
const F_BINOP: usize = 3;

#[pyclass]
pub struct CumReductionLayer {
    /// `[output, chunk, extra, tail, x]` names — produced names use the first
    /// four; `Dep`s reference chunk/extra/tail/x.
    names: Vec<String>,
    /// `[chunk_func, np.full_like, operator.getitem, binop]`.
    funcs: Vec<Py<PyAny>>,
    /// Shared kwargs (empty — the per-block `shape` rides in `CallKw`).
    kwargs: Py<PyAny>,
    /// `[meta, ident, dtype]` for the `full_like` identity blocks.
    literals: Vec<Py<PyAny>>,
    axis: usize,
    numblocks: Vec<usize>,
    /// Per-dimension chunk sizes (for the identity block shapes).
    chunks: Vec<Vec<i64>>,
}

#[pymethods]
impl CumReductionLayer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        chunk_func: Py<PyAny>,
        full_like: Py<PyAny>,
        getitem: Py<PyAny>,
        binop: Py<PyAny>,
        kwargs: Py<PyAny>,
        meta: Py<PyAny>,
        ident: Py<PyAny>,
        dtype: Py<PyAny>,
        x_name: String,
        axis: usize,
        numblocks: Vec<usize>,
        chunks: Vec<Vec<i64>>,
    ) -> Self {
        let names = vec![
            name.clone(),
            format!("{name}-chunk"),
            format!("{name}-extra"),
            format!("{name}-tail"),
            x_name,
        ];
        Self {
            names,
            funcs: vec![chunk_func, full_like, getitem, binop],
            kwargs,
            literals: vec![meta, ident, dtype],
            axis,
            numblocks,
            chunks,
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

impl CumReductionLayer {
    /// The `getitem` index that selects the last position along `axis`:
    /// `(:, …, slice(-1, None), …, :)`.
    fn tail_index(&self) -> Vec<IndexElem> {
        (0..self.numblocks.len())
            .map(|d| {
                if d == self.axis {
                    IndexElem::Slice {
                        start: Some(-1),
                        stop: None,
                        step: None,
                    }
                } else {
                    IndexElem::Slice {
                        start: None,
                        stop: None,
                        step: None,
                    }
                }
            })
            .collect()
    }

    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let n = self.numblocks[self.axis];
        let mut tasks: Vec<NeutralTask> = Vec::new();

        // 1) Per-block chunk tasks over the whole grid (C order).
        let total: usize = self.numblocks.iter().product();
        let mut coord = vec![0u32; ndim];
        for _ in 0..total {
            tasks.push(NeutralTask {
                name_idx: N_CHUNK,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: F_CHUNK },
                slots: vec![ArgSlot::Dep {
                    name_idx: N_X,
                    coord: coord.clone(),
                }],
            });
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if (coord[d] as usize) < self.numblocks[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        // 2) Sequential carry along `axis`, once per non-axis position.
        let non_axis: Vec<usize> = (0..ndim).filter(|&d| d != self.axis).collect();
        let na_total: usize = non_axis.iter().map(|&d| self.numblocks[d]).product();
        let mut na_coord = vec![0u32; non_axis.len()];
        // Build a full block coord from the non-axis position + an axis value.
        let full_coord = |na: &[u32], i: u32| -> Vec<u32> {
            let mut c = vec![0u32; ndim];
            c[self.axis] = i;
            for (k, &d) in non_axis.iter().enumerate() {
                c[d] = na[k];
            }
            c
        };

        for _ in 0..na_total {
            let c0 = full_coord(&na_coord, 0);

            // extra[pos, 0] = full_like(meta, ident, dtype, shape=<1 along axis>)
            let shape: Vec<i64> = (0..ndim)
                .map(|d| {
                    if d == self.axis {
                        1
                    } else {
                        self.chunks[d][c0[d] as usize]
                    }
                })
                .collect();
            tasks.push(NeutralTask {
                name_idx: N_EXTRA,
                coord: c0.clone(),
                compute: Compute::CallKw {
                    func_idx: F_FULL_LIKE,
                    kwargs: vec![("shape".to_string(), ArgSlot::IntTuple(shape))],
                },
                slots: vec![
                    ArgSlot::Literal(0),
                    ArgSlot::Literal(1),
                    ArgSlot::Literal(2),
                ],
            });
            // output[pos, 0] = chunk[pos, 0]  (alias)
            tasks.push(NeutralTask {
                name_idx: N_OUT,
                coord: c0.clone(),
                compute: Compute::Alias,
                slots: vec![ArgSlot::Dep {
                    name_idx: N_CHUNK,
                    coord: c0.clone(),
                }],
            });

            for i in 1..n as u32 {
                let old = full_coord(&na_coord, i - 1);
                let cur = full_coord(&na_coord, i);

                // tail[old] = getitem(chunk[old], last-along-axis)
                tasks.push(NeutralTask {
                    name_idx: N_TAIL,
                    coord: old.clone(),
                    compute: Compute::Call {
                        func_idx: F_GETITEM,
                    },
                    slots: vec![
                        ArgSlot::Dep {
                            name_idx: N_CHUNK,
                            coord: old.clone(),
                        },
                        ArgSlot::Index(self.tail_index()),
                    ],
                });
                // extra[cur] = binop(extra[old], tail[old])
                tasks.push(NeutralTask {
                    name_idx: N_EXTRA,
                    coord: cur.clone(),
                    compute: Compute::Call { func_idx: F_BINOP },
                    slots: vec![
                        ArgSlot::Dep {
                            name_idx: N_EXTRA,
                            coord: old.clone(),
                        },
                        ArgSlot::Dep {
                            name_idx: N_TAIL,
                            coord: old,
                        },
                    ],
                });
                // output[cur] = binop(extra[cur], chunk[cur])
                tasks.push(NeutralTask {
                    name_idx: N_OUT,
                    coord: cur.clone(),
                    compute: Compute::Call { func_idx: F_BINOP },
                    slots: vec![
                        ArgSlot::Dep {
                            name_idx: N_EXTRA,
                            coord: cur.clone(),
                        },
                        ArgSlot::Dep {
                            name_idx: N_CHUNK,
                            coord: cur,
                        },
                    ],
                });
            }

            for k in (0..non_axis.len()).rev() {
                na_coord[k] += 1;
                if (na_coord[k] as usize) < self.numblocks[non_axis[k]] {
                    break;
                }
                na_coord[k] = 0;
            }
        }

        Expanded {
            names: self.names.iter().map(|s| s.as_str()).collect(),
            funcs: self.funcs.iter().collect(),
            kwargs: &self.kwargs,
            literals: &self.literals,
            dep_names: &self.names,
            tasks,
        }
    }
}
