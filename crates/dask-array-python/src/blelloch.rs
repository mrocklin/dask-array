//! Blelloch parallel cumulative-reduction layer (`CumReductionBlelloch`), the
//! work-efficient parallel scan (`method="blelloch"`).
//!
//! Mirrors `CumReductionBlelloch._layer` exactly — the graph structure is fixed
//! by the block *count* along the reduction axis (`numblocks[axis]`), never the
//! chunk sizes. Three task kinds across two names, plus the input `x`:
//!   - **batch** (`name-batch`): per block, `preop(x_block, axis=…, keepdims=True)`
//!     (a partial bound in the Python wrapper).
//!   - **scan** (`name`, coord `(*index, level, i)`): the upsweep/downsweep
//!     combine tree — `binop(left, right)`, where each operand is a batch key or an
//!     earlier scan key. The Python `_layer` keys these `base_key + index +
//!     (level, i)`; the neutral form carries the same key as a longer coord under
//!     the same layer name (output blocks have a length-`ndim` coord, scan nodes
//!     length `ndim+2` — no collision).
//!   - **output** (`name`, coord `index`): the first block along the axis is
//!     `_prefixscan_first(x_block)`; the rest are `_prefixscan_combine(prefix,
//!     x_block)` (both partials with `func`/`binop`/`axis`/`dtype` pre-bound, so no
//!     literal args remain).
//!
//! The upsweep/downsweep `prefix_vals` bookkeeping (which key each axis position
//! currently resolves to) is replicated here verbatim, so the emitted deps match
//! the reference graph's wiring.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{
    grid_nbytes, to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask,
};

// Name indices (also `Dep` name_idx). `name` produces both outputs and scan
// nodes; `name-batch` the preop batches; `x` is referenced only.
const N_OUT: usize = 0;
const N_BATCH: usize = 1;
const N_X: usize = 2;
// Func indices.
const F_PREOP: usize = 0;
const F_BINOP: usize = 1;
const F_FIRST: usize = 2;
const F_COMBINE: usize = 3;

#[pyclass]
pub struct CumReductionBlellochLayer {
    /// `[name, name-batch, x_name]`.
    names: Vec<String>,
    /// `[preop_partial, binop, first_partial, combine_partial]`.
    funcs: Vec<Py<PyAny>>,
    /// Shared kwargs (empty).
    kwargs: Py<PyAny>,
    axis: usize,
    numblocks: Vec<usize>,
    /// Per-dimension chunk sizes — used only for expected-nbytes stamps (the
    /// plan depends on block counts alone). Empty when sizes are unknown.
    chunks: Vec<Vec<i64>>,
    /// Per-task-family item sizes for expected-nbytes stamps; 0 disables stamping.
    batch_itemsize: i64,
    scan_itemsize: i64,
    first_itemsize: i64,
    combine_itemsize: i64,
}

#[pymethods]
impl CumReductionBlellochLayer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        preop: Py<PyAny>,
        binop: Py<PyAny>,
        first: Py<PyAny>,
        combine: Py<PyAny>,
        kwargs: Py<PyAny>,
        x_name: String,
        axis: usize,
        numblocks: Vec<usize>,
        chunks: Vec<Vec<i64>>,
        batch_itemsize: i64,
        scan_itemsize: i64,
        first_itemsize: i64,
        combine_itemsize: i64,
    ) -> Self {
        let names = vec![name.clone(), format!("{name}-batch"), x_name];
        let (batch_itemsize, scan_itemsize, first_itemsize, combine_itemsize) =
            if chunks.len() == numblocks.len() {
                (
                    batch_itemsize,
                    scan_itemsize,
                    first_itemsize,
                    combine_itemsize,
                )
            } else {
                (0, 0, 0, 0)
            };
        Self {
            names,
            funcs: vec![preop, binop, first, combine],
            kwargs,
            axis,
            numblocks,
            chunks,
            batch_itemsize,
            scan_itemsize,
            first_itemsize,
            combine_itemsize,
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

impl CumReductionBlellochLayer {
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let axis = self.axis;
        let n = self.numblocks[axis];
        let na_dims: Vec<usize> = (0..ndim).filter(|&d| d != axis).collect();
        let na_count: usize = na_dims.iter().map(|&d| self.numblocks[d]).product();

        // Decode a non-axis position (C order over `na_dims`) into per-dim coords.
        let na_coord = |na_idx: usize| -> Vec<u32> {
            let mut rem = na_idx;
            let mut coords = vec![0u32; na_dims.len()];
            for k in (0..na_dims.len()).rev() {
                let sz = self.numblocks[na_dims[k]];
                coords[k] = (rem % sz) as u32;
                rem /= sz;
            }
            coords
        };
        // Full block coord: `axis` -> `axis_pos`, non-axis dims from `na_idx`.
        let full_coord = |axis_pos: u32, na_idx: usize| -> Vec<u32> {
            let nac = na_coord(na_idx);
            let mut c = vec![0u32; ndim];
            c[axis] = axis_pos;
            for (k, &d) in na_dims.iter().enumerate() {
                c[d] = nac[k];
            }
            c
        };

        let mut tasks: Vec<NeutralTask> = Vec::new();

        // Expected output sizes: outputs are whole blocks; batches and the
        // combine-tree scan nodes are one keepdims hyperplane (1 along the axis).
        let block_nbytes = |coord: &[u32], itemsize: i64| {
            grid_nbytes(
                itemsize,
                (0..ndim).map(|d| self.chunks[d][coord[d] as usize]),
            )
        };
        let plane_nbytes = |coord: &[u32], itemsize: i64| {
            grid_nbytes(
                itemsize,
                (0..ndim).map(|d| {
                    if d == axis {
                        1
                    } else {
                        self.chunks[d][coord[d] as usize]
                    }
                }),
            )
        };

        // Phase 1: preop batch per block over the whole grid (C order).
        let total: usize = if ndim == 0 {
            1
        } else {
            self.numblocks.iter().product()
        };
        let mut coord = vec![0u32; ndim];
        for _ in 0..total {
            tasks.push(NeutralTask {
                nbytes: plane_nbytes(&coord, self.batch_itemsize),
                name_idx: N_BATCH,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: F_PREOP },
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

        // Empty reduction axis (or no blocks): just the batches, like `_layer`'s
        // `if not full_indices: return dsk`.
        if n == 0 || na_count == 0 {
            return self.assemble(tasks);
        }

        // prefix_vals[axis_pos][na] = the key (name_idx, coord) currently holding
        // the prefix for that position; seeded from the batches for axis 0..n-1.
        let n_vals = n - 1;
        let mut prefix_vals: Vec<Vec<(usize, Vec<u32>)>> = (0..n_vals)
            .map(|ap| {
                (0..na_count)
                    .map(|na| (N_BATCH, full_coord(ap as u32, na)))
                    .collect()
            })
            .collect();

        let mut level: u32 = 0;
        let emit_binop = |tasks: &mut Vec<NeutralTask>,
                          prefix_vals: &mut Vec<Vec<(usize, Vec<u32>)>>,
                          i: usize,
                          stride: usize,
                          level: u32| {
            for na in 0..na_count {
                let index = full_coord(i as u32, na);
                let (ln, lc) = prefix_vals[i - stride][na].clone();
                let (rn, rc) = prefix_vals[i][na].clone();
                let nbytes = plane_nbytes(&index, self.scan_itemsize);
                let mut key_coord = index;
                key_coord.push(level);
                key_coord.push(i as u32);
                tasks.push(NeutralTask {
                    nbytes,
                    name_idx: N_OUT,
                    coord: key_coord.clone(),
                    compute: Compute::Call { func_idx: F_BINOP },
                    slots: vec![
                        ArgSlot::Dep {
                            name_idx: ln,
                            coord: lc,
                        },
                        ArgSlot::Dep {
                            name_idx: rn,
                            coord: rc,
                        },
                    ],
                });
                prefix_vals[i][na] = (N_OUT, key_coord);
            }
        };

        if n_vals >= 2 {
            // Upsweep: doubling strides.
            let mut stride = 1usize;
            let mut stride2 = 2usize;
            while stride2 <= n_vals {
                let mut i = stride2 - 1;
                while i < n_vals {
                    emit_binop(&mut tasks, &mut prefix_vals, i, stride, level);
                    i += stride2;
                }
                stride = stride2;
                stride2 *= 2;
                level += 1;
            }
            // Downsweep: halving strides. `stride2` starts at the smallest power
            // of two >= n_vals/2 (== 2**ceil(log2(n_vals//2))), min 2.
            let mut stride2 = (n_vals / 2).next_power_of_two().max(2);
            let mut stride = stride2 / 2;
            while stride > 0 {
                let mut i = stride2 + stride - 1;
                while i < n_vals {
                    emit_binop(&mut tasks, &mut prefix_vals, i, stride, level);
                    i += stride2;
                }
                stride2 = stride;
                stride /= 2;
                level += 1;
            }
        }

        // Phase 2: outputs. First block along the axis is `_prefixscan_first`.
        for na in 0..na_count {
            let c = full_coord(0, na);
            tasks.push(NeutralTask {
                nbytes: block_nbytes(&c, self.first_itemsize),
                name_idx: N_OUT,
                coord: c.clone(),
                compute: Compute::Call { func_idx: F_FIRST },
                slots: vec![ArgSlot::Dep {
                    name_idx: N_X,
                    coord: c,
                }],
            });
        }
        // Output at axis position k+1 is `_prefixscan_combine(prefix_vals[k], x)`.
        for k in 0..n_vals {
            let axis_pos = (k + 1) as u32;
            for na in 0..na_count {
                let c = full_coord(axis_pos, na);
                let (vn, vc) = prefix_vals[k][na].clone();
                tasks.push(NeutralTask {
                    nbytes: block_nbytes(&c, self.combine_itemsize),
                    name_idx: N_OUT,
                    coord: c.clone(),
                    compute: Compute::Call {
                        func_idx: F_COMBINE,
                    },
                    slots: vec![
                        ArgSlot::Dep {
                            name_idx: vn,
                            coord: vc,
                        },
                        ArgSlot::Dep {
                            name_idx: N_X,
                            coord: c,
                        },
                    ],
                });
            }
        }

        self.assemble(tasks)
    }

    fn assemble(&self, tasks: Vec<NeutralTask>) -> Expanded<'_> {
        Expanded {
            names: self.names.iter().map(|s| s.as_str()).collect(),
            funcs: self.funcs.iter().collect(),
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &self.names,
            tasks,
        }
    }
}
