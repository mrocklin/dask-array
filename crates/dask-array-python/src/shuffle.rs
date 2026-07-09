//! Native expansion for `Shuffle`.
//!
//! Python still computes `_new_chunks` from the user indexer and chunk-size
//! policy. This layer expands the blockwise shuffle task geometry: each output
//! block either directly takes from one source block, or takes sorted pieces
//! from several source blocks and concatenates them back into output order.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::common::{
    grid_nbytes, to_dask_graph, to_records_chunk, to_task_records, ArgSlot, Compute, Expanded,
    NeutralTask, Num,
};

#[pyclass]
pub struct ShuffleLayer {
    names: Vec<String>,
    dep_names: Vec<String>,
    take: Py<PyAny>,
    concat: Py<PyAny>,
    data: Py<PyAny>,
    kwargs: Py<PyAny>,
    chunks: Vec<Vec<i64>>,
    axis: usize,
    new_chunks: Vec<Vec<i64>>,
    /// Dtype itemsize for expected-nbytes stamps; 0 disables stamping.
    itemsize: i64,
}

#[pymethods]
impl ShuffleLayer {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        dep_name: String,
        take: Py<PyAny>,
        concat: Py<PyAny>,
        data: Py<PyAny>,
        kwargs: Py<PyAny>,
        chunks: Vec<Vec<i64>>,
        axis: isize,
        new_chunks: Vec<Vec<i64>>,
        itemsize: i64,
    ) -> PyResult<Self> {
        if chunks.is_empty() {
            return Err(PyValueError::new_err(
                "shuffle layer needs at least one axis",
            ));
        }
        if chunks.iter().any(|dim| dim.is_empty()) {
            return Err(PyValueError::new_err(
                "shuffle layer has an empty chunk dimension",
            ));
        }
        if new_chunks.iter().any(|chunk| chunk.is_empty()) {
            return Err(PyValueError::new_err(
                "shuffle layer has an empty output chunk",
            ));
        }

        let ndim = chunks.len() as isize;
        let axis = if axis < 0 { ndim + axis } else { axis };
        if axis < 0 || axis >= ndim {
            return Err(PyValueError::new_err("shuffle axis out of range"));
        }
        let axis = axis as usize;

        let axis_len: i64 = chunks[axis].iter().sum();
        for chunk in &new_chunks {
            for &idx in chunk {
                if idx < 0 || idx >= axis_len {
                    return Err(PyValueError::new_err(
                        "shuffle index is out of bounds for axis chunks",
                    ));
                }
            }
        }

        let split_name = format!("{name}-split");
        let data_name = format!("{name}-data");
        Ok(Self {
            names: vec![name, split_name.clone(), data_name.clone()],
            dep_names: vec![dep_name, split_name, data_name],
            take,
            concat,
            data,
            kwargs,
            chunks,
            axis,
            new_chunks,
            itemsize,
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

fn argsort_i64(values: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by_key(|&i| (values[i], i));
    idx
}

fn argsort_usize(values: &[usize]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by_key(|&i| (values[i], i));
    idx
}

fn searchsorted_right(boundaries: &[i64], value: i64) -> usize {
    boundaries.partition_point(|&boundary| boundary <= value)
}

fn full_coord(non_axis: &[u32], axis_value: u32, axis: usize, ndim: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(ndim);
    let mut j = 0;
    for d in 0..ndim {
        if d == axis {
            out.push(axis_value);
        } else {
            out.push(non_axis[j]);
            j += 1;
        }
    }
    out
}

fn int_list_slot(values: Vec<i64>) -> ArgSlot {
    ArgSlot::List(
        values
            .into_iter()
            .map(|value| ArgSlot::Scalar(Num::Int(value)))
            .collect(),
    )
}

impl ShuffleLayer {
    fn chunk_boundaries(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.chunks[self.axis].len());
        let mut total = 0i64;
        for &chunk in &self.chunks[self.axis] {
            total += chunk;
            out.push(total);
        }
        out
    }

    fn source_segments(
        &self,
        sorted_taker: &[i64],
        boundaries: &[i64],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut sources = Vec::new();
        let mut starts = Vec::new();
        let mut last = None;

        for (i, &idx) in sorted_taker.iter().enumerate() {
            let source = searchsorted_right(boundaries, idx);
            if last != Some(source) {
                sources.push(source);
                starts.push(i);
                last = Some(source);
            }
        }
        starts.push(sorted_taker.len());
        (sources, starts)
    }

    fn expand(&self) -> Expanded<'_> {
        let ndim = self.chunks.len();
        let boundaries = self.chunk_boundaries();
        let non_axis_limits: Vec<u32> = self
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if i == self.axis {
                    None
                } else {
                    Some(dim.len() as u32)
                }
            })
            .collect();
        let non_axis_total = if non_axis_limits.is_empty() {
            1
        } else {
            non_axis_limits.iter().product::<u32>() as usize
        };

        let mut tasks = Vec::new();
        let mut split_idx = 0u32;

        for (new_chunk_idx, new_chunk_taker) in self.new_chunks.iter().enumerate() {
            let sorter = argsort_i64(new_chunk_taker);
            let sorted_taker: Vec<i64> = sorter.iter().map(|&i| new_chunk_taker[i]).collect();
            let (sources, starts) = self.source_segments(&sorted_taker, &boundaries);
            let sorter_coord = if sources.len() > 1 {
                let coord = vec![new_chunk_idx as u32, 0];
                tasks.push(NeutralTask {
                    // An index list; ~8 bytes per element as an int64 array.
                    nbytes: grid_nbytes(8, [sorter.len() as i64]),
                    name_idx: 2,
                    coord: coord.clone(),
                    compute: Compute::Call { func_idx: 2 },
                    slots: vec![int_list_slot(sorter.iter().map(|&i| i as i64).collect())],
                });
                Some(coord)
            } else {
                None
            };

            let mut taker_coords = Vec::with_capacity(sources.len());
            for (segment_idx, (&source, bounds)) in
                sources.iter().zip(starts.windows(2)).enumerate()
            {
                let start = bounds[0];
                let end = bounds[1];
                let source_start = if source == 0 {
                    0
                } else {
                    boundaries[source - 1]
                };
                let mut taker: Vec<i64> = sorted_taker[start..end]
                    .iter()
                    .map(|idx| idx - source_start)
                    .collect();

                if sources.len() == 1 {
                    let restore = argsort_usize(&sorter);
                    taker = restore.into_iter().map(|i| taker[i]).collect();
                }

                let taker_coord = vec![new_chunk_idx as u32, segment_idx as u32 + 1];
                tasks.push(NeutralTask {
                    // An index list; ~8 bytes per element as an int64 array.
                    nbytes: grid_nbytes(8, [taker.len() as i64]),
                    name_idx: 2,
                    coord: taker_coord.clone(),
                    compute: Compute::Call { func_idx: 2 },
                    slots: vec![int_list_slot(taker)],
                });
                taker_coords.push(taker_coord);
            }

            let mut non_axis_coord = vec![0u32; non_axis_limits.len()];
            for _ in 0..non_axis_total {
                let mut split_deps = Vec::with_capacity(sources.len());

                for ((segment_idx, &source), taker_coord) in
                    sources.iter().enumerate().zip(&taker_coords)
                {
                    let segment_len = (starts[segment_idx + 1] - starts[segment_idx]) as i64;
                    let source_coord = full_coord(&non_axis_coord, source as u32, self.axis, ndim);
                    let take_nbytes = grid_nbytes(
                        self.itemsize,
                        (0..ndim).map(|d| {
                            if d == self.axis {
                                segment_len
                            } else {
                                self.chunks[d][source_coord[d] as usize]
                            }
                        }),
                    );
                    let take_slots = vec![
                        ArgSlot::Dep {
                            name_idx: 0,
                            coord: source_coord,
                        },
                        ArgSlot::Dep {
                            name_idx: 2,
                            coord: taker_coord.clone(),
                        },
                        ArgSlot::Scalar(Num::Int(self.axis as i64)),
                    ];

                    if sources.len() == 1 {
                        tasks.push(NeutralTask {
                            nbytes: take_nbytes,
                            name_idx: 0,
                            coord: full_coord(
                                &non_axis_coord,
                                new_chunk_idx as u32,
                                self.axis,
                                ndim,
                            ),
                            compute: Compute::Call { func_idx: 0 },
                            slots: take_slots,
                        });
                    } else {
                        let coord = vec![split_idx];
                        split_idx += 1;
                        tasks.push(NeutralTask {
                            nbytes: take_nbytes,
                            name_idx: 1,
                            coord: coord.clone(),
                            compute: Compute::Call { func_idx: 0 },
                            slots: take_slots,
                        });
                        split_deps.push(ArgSlot::Dep { name_idx: 1, coord });
                    }
                }

                if sources.len() > 1 {
                    let out_nbytes = grid_nbytes(
                        self.itemsize,
                        (0..ndim).map(|d| {
                            if d == self.axis {
                                new_chunk_taker.len() as i64
                            } else {
                                let full = full_coord(&non_axis_coord, 0, self.axis, ndim);
                                self.chunks[d][full[d] as usize]
                            }
                        }),
                    );
                    tasks.push(NeutralTask {
                        nbytes: out_nbytes,
                        name_idx: 0,
                        coord: full_coord(&non_axis_coord, new_chunk_idx as u32, self.axis, ndim),
                        compute: Compute::Call { func_idx: 1 },
                        slots: vec![
                            ArgSlot::List(split_deps),
                            ArgSlot::Dep {
                                name_idx: 2,
                                coord: sorter_coord.as_ref().unwrap().clone(),
                            },
                            ArgSlot::Scalar(Num::Int(self.axis as i64)),
                        ],
                    });
                }

                advance(&mut non_axis_coord, &non_axis_limits);
            }
        }

        Expanded {
            names: self.names.iter().map(String::as_str).collect(),
            funcs: vec![&self.take, &self.concat, &self.data],
            kwargs: &self.kwargs,
            literals: &[],
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
