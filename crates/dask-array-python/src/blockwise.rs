//! Same-grid / broadcast elementwise blockwise layer.
//!
//! An output block `(i, j, ...)` is the function applied to the matching block
//! of each array input plus any literals. Contractions, new axes,
//! concatenation, adjusted chunks, and unaligned (un-lowered) inputs are
//! rejected by the Python `Blockwise._frisky_layer` before construction, so the
//! expansion here is a simple coordinate map.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::common::{to_dask_graph, to_task_records, ArgSlot, Compute, Expanded, NeutralTask};

/// How an input dimension's block id is derived from the output block id.
enum Axis {
    /// Input has a single block on this dimension (broadcast): always block 0.
    Const0,
    /// Input block id equals the output block id at this output position.
    FromOut(usize),
}

/// An argument position: a shared literal (by index) or a dependency (by
/// dep-name index) whose per-block coord is read off the output coord.
enum ArgTemplate {
    Lit(usize),
    Arr { dep_idx: usize, axes: Vec<Axis> },
}

#[pyclass]
pub struct BlockwiseLayer {
    name: String,
    func: PyObject,
    kwargs: PyObject,
    literals: Vec<PyObject>,
    dep_names: Vec<String>,
    numblocks: Vec<usize>,
    template: Vec<ArgTemplate>,
}

#[pymethods]
impl BlockwiseLayer {
    /// `args` items are `("literal", value)` or
    /// `("array", dep_name, ind, numblocks)`, index labels (ints) aligned with
    /// `out_ind`. The Python `_frisky_layer` has already rejected unsupported
    /// shapes; here we only sort args into literals + dependency templates.
    #[new]
    fn new(
        name: String,
        func: PyObject,
        kwargs: PyObject,
        numblocks: Vec<usize>,
        out_ind: Vec<i64>,
        args: Bound<'_, PyList>,
    ) -> PyResult<Self> {
        let mut pos: HashMap<i64, usize> = HashMap::with_capacity(out_ind.len());
        for (i, &label) in out_ind.iter().enumerate() {
            pos.insert(label, i);
        }

        let mut literals: Vec<PyObject> = Vec::new();
        let mut dep_names: Vec<String> = Vec::new();
        let mut template: Vec<ArgTemplate> = Vec::with_capacity(args.len());

        for item in args.iter() {
            let kind: String = item.get_item(0)?.extract()?;
            match kind.as_str() {
                "literal" => {
                    template.push(ArgTemplate::Lit(literals.len()));
                    literals.push(item.get_item(1)?.unbind());
                }
                "array" => {
                    let dep_name: String = item.get_item(1)?.extract()?;
                    let ind: Vec<i64> = item.get_item(2)?.extract()?;
                    let nb: Vec<usize> = item.get_item(3)?.extract()?;
                    if ind.len() != nb.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "ind and numblocks length mismatch",
                        ));
                    }
                    let mut axes = Vec::with_capacity(ind.len());
                    for (d, label) in ind.iter().enumerate() {
                        if nb[d] == 1 {
                            axes.push(Axis::Const0);
                        } else if let Some(&p) = pos.get(label) {
                            axes.push(Axis::FromOut(p));
                        } else {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "contracted dimension",
                            ));
                        }
                    }
                    let dep_idx = dep_names.len();
                    dep_names.push(dep_name);
                    template.push(ArgTemplate::Arr { dep_idx, axes });
                }
                _ => return Err(pyo3::exceptions::PyValueError::new_err("bad arg kind")),
            }
        }

        Ok(Self { name, func, kwargs, literals, dep_names, numblocks, template })
    }

    fn to_dask_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        to_dask_graph(py, &self.expand())
    }

    fn to_task_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_task_records(py, &self.expand())
    }
}

impl BlockwiseLayer {
    /// Pure-Rust expansion into the neutral form. Iterates output blocks in C
    /// order (matching `itertools.product`) and resolves each arg template.
    fn expand(&self) -> Expanded<'_> {
        let ndim = self.numblocks.len();
        let total: usize = if ndim == 0 { 1 } else { self.numblocks.iter().product() };
        let mut tasks = Vec::with_capacity(total);
        let mut coord = vec![0u32; ndim];

        for _ in 0..total {
            let slots = self
                .template
                .iter()
                .map(|t| match t {
                    ArgTemplate::Lit(i) => ArgSlot::Literal(*i),
                    ArgTemplate::Arr { dep_idx, axes } => {
                        let dep_coord = axes
                            .iter()
                            .map(|ax| match ax {
                                Axis::Const0 => 0u32,
                                Axis::FromOut(p) => coord[*p],
                            })
                            .collect();
                        ArgSlot::Dep { name_idx: *dep_idx, coord: dep_coord }
                    }
                })
                .collect();
            tasks.push(NeutralTask {
                name_idx: 0,
                coord: coord.clone(),
                compute: Compute::Call { func_idx: 0 },
                slots,
            });

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
            funcs: vec![&self.func],
            kwargs: &self.kwargs,
            literals: &self.literals,
            dep_names: &self.dep_names,
            tasks,
        }
    }
}
