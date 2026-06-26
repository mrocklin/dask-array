//! Overlap neighbor assembly (`OverlapInternal`).
//!
//! Dask's `ArrayOverlapLayer` builds private getitem keys with fractional block
//! coordinates (for example `0.9` and `1.1`) and then concatenates the neighbor
//! pieces for each output block. The Frisky binary-record format uses integer
//! coordinates, so this layer preserves the public output keys and uses
//! integer-coded private getitem keys: `(source_coord..., side_code...)`, where
//! side code `0` is the previous/right-edge slice, `1` is the full block, and
//! `2` is the next/left-edge slice.

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::common::{
    to_dask_graph, to_records_chunk, to_task_records, ArgSlot, Compute, Expanded, IndexElem,
    NeutralTask,
};

#[derive(Clone, Copy)]
struct Depth {
    left: i64,
    right: i64,
}

#[pyclass]
pub struct OverlapLayer {
    names: Vec<String>,
    dep_names: Vec<String>,
    getitem: PyObject,
    concatenate_shaped: PyObject,
    kwargs: PyObject,
    numblocks: Vec<u32>,
    depths: Vec<Depth>,
}

#[pymethods]
impl OverlapLayer {
    /// `axis_depths`: `(axis, left_depth, right_depth)` for axes with overlap.
    #[new]
    fn new(
        name: String,
        dep_name: String,
        getitem: PyObject,
        concatenate_shaped: PyObject,
        kwargs: PyObject,
        numblocks: Vec<u32>,
        axis_depths: Vec<(usize, i64, i64)>,
    ) -> PyResult<Self> {
        if numblocks.iter().any(|n| *n == 0) {
            return Err(PyValueError::new_err(
                "overlap layer has an empty block dimension",
            ));
        }

        let mut depths = vec![Depth { left: 0, right: 0 }; numblocks.len()];
        for (axis, left, right) in axis_depths {
            if axis >= numblocks.len() {
                return Err(PyValueError::new_err("overlap depth axis out of range"));
            }
            if left < 0 || right < 0 {
                return Err(PyValueError::new_err("overlap depths must be non-negative"));
            }
            depths[axis] = Depth { left, right };
        }

        let getitem_name = format!("{name}-getitem");
        Ok(Self {
            names: vec![name, getitem_name.clone()],
            dep_names: vec![dep_name, getitem_name],
            getitem,
            concatenate_shaped,
            kwargs,
            numblocks,
            depths,
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

fn advance(pos: &mut [u32], limits: &[u32]) {
    for d in (0..pos.len()).rev() {
        pos[d] += 1;
        if pos[d] < limits[d] {
            break;
        }
        pos[d] = 0;
    }
}

fn product_u32(vals: &[u32]) -> usize {
    vals.iter().fold(1usize, |acc, v| acc * (*v as usize))
}

fn getitem_coord_from_choice(choices: &[Vec<(u32, u32)>], pos: &[u32]) -> Vec<u32> {
    let ndim = choices.len();
    let mut coord = Vec::with_capacity(ndim * 2);
    for d in 0..ndim {
        coord.push(choices[d][pos[d] as usize].0);
    }
    for d in 0..ndim {
        coord.push(choices[d][pos[d] as usize].1);
    }
    coord
}

impl OverlapLayer {
    fn choices_for_center(&self, center: &[u32]) -> Vec<Vec<(u32, u32)>> {
        let mut choices = Vec::with_capacity(center.len());
        for (axis, &block) in center.iter().enumerate() {
            let depth = self.depths[axis];
            let mut dim = Vec::with_capacity(3);
            if block > 0 && depth.left != 0 {
                dim.push((block - 1, 0));
            }
            dim.push((block, 1));
            if block + 1 < self.numblocks[axis] && depth.right != 0 {
                dim.push((block + 1, 2));
            }
            choices.push(dim);
        }
        choices
    }

    fn index_for_getitem(&self, coord: &[u32]) -> (Vec<u32>, Vec<IndexElem>) {
        let ndim = self.numblocks.len();
        let source = coord[..ndim].to_vec();
        let mut index = Vec::with_capacity(ndim);
        for axis in 0..ndim {
            let side = coord[ndim + axis];
            let depth = self.depths[axis];
            let elem = match side {
                0 => IndexElem::Slice {
                    start: Some(-depth.left),
                    stop: None,
                    step: None,
                },
                1 => IndexElem::Slice {
                    start: None,
                    stop: None,
                    step: None,
                },
                2 => IndexElem::Slice {
                    start: Some(0),
                    stop: Some(depth.right),
                    step: None,
                },
                _ => unreachable!("invalid overlap side code"),
            };
            index.push(elem);
        }
        (source, index)
    }

    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total_outputs = product_u32(&self.numblocks);
        let mut center = vec![0u32; ndim];
        let mut needed_getitems: HashSet<Vec<u32>> = HashSet::new();
        let mut finals: Vec<NeutralTask> = Vec::with_capacity(total_outputs);

        for _ in 0..total_outputs {
            let choices = self.choices_for_center(&center);
            let shape: Vec<i64> = choices.iter().map(|dim| dim.len() as i64).collect();
            let limits: Vec<u32> = choices.iter().map(|dim| dim.len() as u32).collect();
            let total_neighbors = product_u32(&limits);
            let mut pos = vec![0u32; ndim];
            let mut neighbors = Vec::with_capacity(total_neighbors);

            for _ in 0..total_neighbors {
                let getitem_coord = getitem_coord_from_choice(&choices, &pos);
                needed_getitems.insert(getitem_coord.clone());
                neighbors.push(ArgSlot::Dep {
                    name_idx: 1,
                    coord: getitem_coord,
                });
                advance(&mut pos, &limits);
            }

            finals.push(NeutralTask {
                name_idx: 0,
                coord: center.clone(),
                compute: Compute::Call { func_idx: 1 },
                slots: vec![ArgSlot::List(neighbors), ArgSlot::IntTuple(shape)],
            });
            advance(&mut center, &self.numblocks);
        }

        let mut getitem_coords: Vec<Vec<u32>> = needed_getitems.into_iter().collect();
        getitem_coords.sort();
        let mut tasks = Vec::with_capacity(getitem_coords.len() + finals.len());
        for coord in getitem_coords {
            let (source, index) = self.index_for_getitem(&coord);
            let all_full = index.iter().all(|elem| {
                matches!(
                    elem,
                    IndexElem::Slice {
                        start: None,
                        stop: None,
                        step: None
                    }
                )
            });

            if all_full {
                tasks.push(NeutralTask {
                    name_idx: 1,
                    coord,
                    compute: Compute::Alias,
                    slots: vec![ArgSlot::Dep {
                        name_idx: 0,
                        coord: source,
                    }],
                });
            } else {
                tasks.push(NeutralTask {
                    name_idx: 1,
                    coord,
                    compute: Compute::Call { func_idx: 0 },
                    slots: vec![
                        ArgSlot::Dep {
                            name_idx: 0,
                            coord: source,
                        },
                        ArgSlot::Index(index),
                    ],
                });
            }
        }
        tasks.extend(finals);

        Expanded {
            names: self.names.iter().map(String::as_str).collect(),
            funcs: vec![&self.getitem, &self.concatenate_shaped],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
