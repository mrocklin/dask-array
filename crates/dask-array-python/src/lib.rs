//! Native task-generation accelerator for dask-array.
//!
//! The graph code lives in the per-layer modules: a layer type
//! (`blockwise::BlockwiseLayer`, `creation::CreationLayer`, ...) knows how to
//! expand itself into the individual tasks of its subgraph. Expansion produces
//! *neutral* tasks — a coordinate plus an argument list in which references to
//! other tasks are `common::DepRef` markers. The Python `Layer` base class
//! translates those neutral tasks generically into either a Dask task graph
//! (the correctness/legacy path, which therefore validates this Rust code) or,
//! later, Frisky wire tuples.
//!
//! It is deliberately self-contained: only pyo3, no frisky crate, and no
//! knowledge of dask's or frisky's task representations — those live in the
//! generic Python/Rust translators. Frisky stays agnostic to array semantics.
//!
//! Each layer is its own module so new layer kinds can be added in their own
//! files; the only shared edit when adding one is a single `add_class` line
//! below.

use pyo3::prelude::*;

mod arange;
mod arg_chunk;
mod blocks;
mod blockwise;
mod broadcast;
mod coarsen;
mod common;
mod concatenate;
mod creation;
mod cumulative;
mod diag;
mod expand_dims;
mod eye;
mod linspace;
mod rechunk;
mod reduction;
mod reshape;
mod slicing;
mod squeeze;
mod stack;

/// Protocol revision for the native extension. Python checks this on import so
/// a stale `.so` fails loudly instead of silently producing wrong tasks.
const PROTOCOL_REVISION: usize = 19;

#[pyfunction]
fn protocol_revision() -> usize {
    PROTOCOL_REVISION
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PROTOCOL_REVISION", PROTOCOL_REVISION)?;
    m.add_function(wrap_pyfunction!(protocol_revision, m)?)?;
    m.add_class::<arange::ArangeLayer>()?;
    m.add_class::<arg_chunk::ArgChunkLayer>()?;
    m.add_class::<blocks::BlocksLayer>()?;
    m.add_class::<blockwise::BlockwiseLayer>()?;
    m.add_class::<broadcast::BroadcastLayer>()?;
    m.add_class::<coarsen::CoarsenLayer>()?;
    m.add_class::<concatenate::ConcatenateLayer>()?;
    m.add_class::<creation::CreationLayer>()?;
    m.add_class::<cumulative::CumReductionLayer>()?;
    m.add_class::<diag::Diag1DLayer>()?;
    m.add_class::<diag::Diag2DSimpleLayer>()?;
    m.add_class::<expand_dims::ExpandDimsLayer>()?;
    m.add_class::<eye::EyeLayer>()?;
    m.add_class::<linspace::LinspaceLayer>()?;
    m.add_class::<reduction::PartialReduceLayer>()?;
    m.add_class::<rechunk::RechunkLayer>()?;
    m.add_class::<reshape::ReshapeLayer>()?;
    m.add_class::<slicing::SliceLayer>()?;
    m.add_class::<squeeze::SqueezeLayer>()?;
    m.add_class::<stack::StackLayer>()?;
    Ok(())
}
