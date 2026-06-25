//! Task-based rechunk layer (`TasksRechunk`).
//!
//! This is the multi-stage exemplar: within one layer it emits, per step, two
//! kinds of tasks with their own keys — `split` (`getitem(old_block, slices)`)
//! and `merge` (`concatenate3(nested blocks)` or an `Alias` when a single source
//! covers a whole new block). It exercises the full neutral-form vocabulary:
//! per-task func ([`Compute::Call`] with two `funcs`), free-form intermediate
//! keys (split names), [`Compute::Alias`], nested deps ([`ArgSlot::List`]) and
//! per-task slices ([`ArgSlot::Index`]).
//!
//! The intricate planning heuristic (`plan_rechunk`) stays in tested Python and
//! runs once; it hands this layer a list of steps (old/new chunk sizes + the
//! merge/split key names). Rust does the O(n_blocks) work: the 1-D chunk
//! intersection (`old_to_new`/`_intersect_1d`) and the per-block split/merge
//! expansion (`_compute_rechunk`), ported from `dask_array/_rechunk.py`.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::common::{
    to_dask_graph, to_records_chunk, to_task_records, ArgSlot, Compute, Expanded, IndexElem,
    NeutralTask,
};

struct Step {
    old_idx: usize,
    merge_idx: usize,
    split_idx: usize,
    old_chunks: Vec<Vec<i64>>,
    new_chunks: Vec<Vec<i64>>,
}

#[pyclass]
pub struct RechunkLayer {
    getitem: PyObject,
    concatenate3: PyObject,
    kwargs: PyObject,
    /// All key/dep names, interned; both `name_idx` and `Dep::name_idx` index it.
    names: Vec<String>,
    steps: Vec<Step>,
}

fn intern(s: String, names: &mut Vec<String>, index: &mut HashMap<String, usize>) -> usize {
    if let Some(&i) = index.get(&s) {
        i
    } else {
        let i = names.len();
        index.insert(s.clone(), i);
        names.push(s);
        i
    }
}

#[pymethods]
impl RechunkLayer {
    /// `steps`: each `(old_name, old_chunks, new_chunks, merge_name, split_name)`.
    /// Steps chain — a step's `old_name` is the previous step's `merge_name`.
    #[new]
    fn new(
        getitem: PyObject,
        concatenate3: PyObject,
        kwargs: PyObject,
        steps: Vec<(String, Vec<Vec<i64>>, Vec<Vec<i64>>, String, String)>,
    ) -> PyResult<Self> {
        let mut names: Vec<String> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut step_structs = Vec::with_capacity(steps.len());
        for (old_name, old_chunks, new_chunks, merge_name, split_name) in steps {
            // Guard the per-block expansion against malformed descriptors (the
            // normal path is validated upstream by `_validate_rechunk`): every
            // dimension must have at least one block on both sides, so the
            // intersection and old-chunk indexing below can't go out of bounds.
            if old_chunks.len() != new_chunks.len()
                || old_chunks
                    .iter()
                    .chain(new_chunks.iter())
                    .any(|c| c.is_empty())
            {
                return Err(PyValueError::new_err(
                    "rechunk step has empty or mismatched chunk dims",
                ));
            }
            let old_idx = intern(old_name, &mut names, &mut index);
            let merge_idx = intern(merge_name, &mut names, &mut index);
            let split_idx = intern(split_name, &mut names, &mut index);
            step_structs.push(Step {
                old_idx,
                merge_idx,
                split_idx,
                old_chunks,
                new_chunks,
            });
        }
        Ok(Self {
            getitem,
            concatenate3,
            kwargs,
            names,
            steps: step_structs,
        })
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }

    fn to_records_chunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        to_records_chunk(py, &self.expand())
    }
}

/// Cumulative chunk boundaries: `[0, c0, c0+c1, ...]`.
fn cum(chunks: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(chunks.len() + 1);
    let mut s = 0i64;
    out.push(0);
    for &c in chunks {
        s += c;
        out.push(s);
    }
    out
}

/// Merge old ('o', true) and new ('n', false) boundaries, sorted by position;
/// a stable sort keeps an 'o' before an 'n' at the same position.
fn breakpoints(old_cum: &[i64], new_cum: &[i64]) -> Vec<(bool, i64)> {
    let mut v: Vec<(bool, i64)> = Vec::with_capacity(old_cum.len() + new_cum.len());
    for &x in old_cum {
        v.push((true, x));
    }
    for &x in new_cum {
        v.push((false, x));
    }
    v.sort_by_key(|p| p.1);
    v
}

/// Port of `_intersect_1d`: for each new chunk on this dim, the list of
/// `(old_block_idx, start, stop)` contributions (slices relative to the old
/// block).
fn intersect_1d(breaks: &[(bool, i64)]) -> Vec<Vec<(u32, i64, i64)>> {
    let o_count = breaks.iter().filter(|b| b.0).count();
    let last_old_chunk_idx = o_count as i64 - 2;
    let last_o_br = breaks.iter().rev().find(|b| b.0).map(|b| b.1).unwrap_or(0);

    let mut start: i64;
    let mut last_end: i64 = 0;
    let mut old_idx: i64 = 0;
    let mut last_o_end: i64 = 0;
    let mut ret: Vec<Vec<(u32, i64, i64)>> = Vec::new();
    let mut ret_next: Vec<(u32, i64, i64)> = Vec::new();

    for idx in 1..breaks.len() {
        let (label_old, br) = breaks[idx];
        let (last_label_old, last_br) = breaks[idx - 1];
        if !last_label_old {
            start = last_end;
            if !ret_next.is_empty() {
                ret.push(std::mem::take(&mut ret_next));
            }
        } else {
            start = 0;
        }
        let end = br - last_br + start;
        last_end = end;
        if br == last_br {
            if label_old {
                old_idx += 1;
                last_o_end = end;
            }
            if !label_old && !last_label_old {
                if br == last_o_br {
                    ret_next.push((last_old_chunk_idx as u32, last_o_end, last_o_end));
                    continue;
                }
                // else: fall through to the append below
            } else {
                continue;
            }
        }
        ret_next.push((old_idx as u32, start, end));
        if label_old {
            old_idx += 1;
            last_o_end = end;
        }
    }
    if !ret_next.is_empty() {
        ret.push(ret_next);
    }
    ret
}

/// Reshape a flat C-order list of refs into an `ArgSlot::List` nested to `dims`.
fn nest(iter: &mut std::vec::IntoIter<ArgSlot>, dims: &[usize]) -> ArgSlot {
    match dims.split_first() {
        None => iter.next().unwrap(),
        Some((&first, rest)) => ArgSlot::List((0..first).map(|_| nest(iter, rest)).collect()),
    }
}

impl RechunkLayer {
    fn expand(&self) -> Expanded<'_> {
        let mut tasks: Vec<NeutralTask> = Vec::new();

        for step in &self.steps {
            let ndim = step.old_chunks.len();
            // 1-D old->new intersection per dimension.
            let o2n: Vec<Vec<Vec<(u32, i64, i64)>>> = (0..ndim)
                .map(|d| {
                    intersect_1d(&breakpoints(
                        &cum(&step.old_chunks[d]),
                        &cum(&step.new_chunks[d]),
                    ))
                })
                .collect();
            let new_nb: Vec<usize> = step.new_chunks.iter().map(|c| c.len()).collect();
            let total_new: usize = if ndim == 0 {
                1
            } else {
                new_nb.iter().product()
            };

            let mut new_idx = vec![0u32; ndim];
            let mut suffix: u32 = 0; // split-key counter, reset per step (dask matches)

            for _ in 0..total_new {
                let entries: Vec<&Vec<(u32, i64, i64)>> =
                    (0..ndim).map(|d| &o2n[d][new_idx[d] as usize]).collect();
                let subdims: Vec<usize> = entries.iter().map(|e| e.len()).collect();
                let ncontrib: usize = subdims.iter().product();

                // Each old-block contribution to this new block, in C order.
                let mut refs: Vec<ArgSlot> = Vec::with_capacity(ncontrib);
                let mut sel = vec![0usize; ndim];
                for _ in 0..ncontrib {
                    let mut full = true;
                    let mut old_coord = vec![0u32; ndim];
                    let mut slices: Vec<IndexElem> = Vec::with_capacity(ndim);
                    for d in 0..ndim {
                        let (oi, lo, hi) = entries[d][sel[d]];
                        old_coord[d] = oi;
                        slices.push(IndexElem::Slice {
                            start: Some(lo),
                            stop: Some(hi),
                            step: Some(1),
                        });
                        if !(lo == 0 && hi == step.old_chunks[d][oi as usize]) {
                            full = false;
                        }
                    }
                    let cur = suffix;
                    suffix += 1;
                    let r = if full {
                        // Whole old block — reference it directly (no split task).
                        ArgSlot::Dep {
                            name_idx: step.old_idx,
                            coord: old_coord,
                        }
                    } else {
                        tasks.push(NeutralTask {
                            name_idx: step.split_idx,
                            coord: vec![cur],
                            compute: Compute::Call { func_idx: 0 }, // getitem
                            slots: vec![
                                ArgSlot::Dep {
                                    name_idx: step.old_idx,
                                    coord: old_coord,
                                },
                                ArgSlot::Index(slices),
                            ],
                        });
                        ArgSlot::Dep {
                            name_idx: step.split_idx,
                            coord: vec![cur],
                        }
                    };
                    refs.push(r);

                    for d in (0..ndim).rev() {
                        sel[d] += 1;
                        if sel[d] < subdims[d] {
                            break;
                        }
                        sel[d] = 0;
                    }
                }

                if subdims.iter().all(|&s| s == 1) {
                    // Single source block covers the whole new block — alias it.
                    let r = refs.into_iter().next().unwrap();
                    tasks.push(NeutralTask {
                        name_idx: step.merge_idx,
                        coord: new_idx.clone(),
                        compute: Compute::Alias,
                        slots: vec![r],
                    });
                } else {
                    let mut it = refs.into_iter();
                    let nested = nest(&mut it, &subdims);
                    tasks.push(NeutralTask {
                        name_idx: step.merge_idx,
                        coord: new_idx.clone(),
                        compute: Compute::Call { func_idx: 1 }, // concatenate3
                        slots: vec![nested],
                    });
                }

                for d in (0..ndim).rev() {
                    new_idx[d] += 1;
                    if (new_idx[d] as usize) < new_nb[d] {
                        break;
                    }
                    new_idx[d] = 0;
                }
            }
        }

        Expanded {
            names: self.names.iter().map(|s| s.as_str()).collect(),
            funcs: vec![&self.getitem, &self.concatenate3],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &self.names,
            tasks,
        }
    }
}
