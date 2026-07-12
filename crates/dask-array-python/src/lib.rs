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
use pyo3::types::PyBytes;

mod arange;
mod arg_chunk;
mod blelloch;
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
mod from_array;
mod from_map;
mod fused_blockwise;
mod gufunc;
mod linspace;
mod overlap;
mod rechunk;
mod reduction;
mod reshape;
mod shuffle;
mod slicing;
mod sliding_window;
mod squeeze;
mod stack;

/// Generation counter for this native build. Python (`base.py`) checks it on
/// import so a stale `.so` (source changed but not rebuilt) fails loudly instead
/// of silently mishandling a changed call. This is LOCAL to dask-array — it is
/// not a wire protocol and Frisky never reads it. The version Frisky *does*
/// coordinate on is the binary records grammar (`common::RECORDS_PROTOCOL_VERSION`
/// ↔ Frisky's `records_proto::CHUNK_GRAMMAR_VERSION`), which only moves when the
/// chunk byte-grammar changes — not when a layer is added.
const NATIVE_BUILD_GENERATION: usize = 43;

#[pyfunction]
fn native_build_generation() -> usize {
    NATIVE_BUILD_GENERATION
}

#[pyfunction]
fn stamp_expected_nbytes<'py>(
    py: Python<'py>,
    chunk: Bound<'py, PyBytes>,
    output_name: String,
    chunks: Vec<Vec<i64>>,
    itemsize: i64,
) -> PyResult<Bound<'py, PyBytes>> {
    let data = chunk.as_bytes().to_vec();
    let stamped = py.detach(move || {
        common::stamp_expected_nbytes_chunk(data, &output_name, &chunks, itemsize)
    })?;
    Ok(PyBytes::new(py, &stamped))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // base.py's freshness check calls native_build_generation(); the module
    // attribute exists for CI's build check (.github/workflows/ci.yml), which
    // asserts both forms agree with base.py.
    m.add("NATIVE_BUILD_GENERATION", NATIVE_BUILD_GENERATION)?;
    m.add_function(wrap_pyfunction!(native_build_generation, m)?)?;
    // The binary records grammar version (a Frisky-coordinated wire constant,
    // unlike the local build generation above). Exported so tests assert
    // against it instead of hardcoding the byte.
    m.add("RECORDS_PROTOCOL_VERSION", common::RECORDS_PROTOCOL_VERSION)?;
    m.add_function(wrap_pyfunction!(stamp_expected_nbytes, m)?)?;
    m.add_class::<arange::ArangeLayer>()?;
    m.add_class::<arg_chunk::ArgChunkLayer>()?;
    m.add_class::<blelloch::CumReductionBlellochLayer>()?;
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
    m.add_class::<from_array::FromArrayGetterLayer>()?;
    m.add_class::<from_map::FromMapLayer>()?;
    m.add_class::<from_map::FromMapBinaryLayer>()?;
    m.add_class::<fused_blockwise::FusedBlockwiseLayer>()?;
    m.add_class::<gufunc::GUfuncLeafLayer>()?;
    m.add_class::<linspace::LinspaceLayer>()?;
    m.add_class::<overlap::OverlapLayer>()?;
    m.add_class::<reduction::PartialReduceLayer>()?;
    m.add_class::<rechunk::RechunkLayer>()?;
    m.add_class::<reshape::ReshapeLayer>()?;
    m.add_class::<shuffle::ShuffleLayer>()?;
    m.add_class::<slicing::SliceLayer>()?;
    m.add_class::<sliding_window::SlidingWindowReductionLayer>()?;
    m.add_class::<sliding_window::MovingWindowReductionLayer>()?;
    m.add_class::<squeeze::SqueezeLayer>()?;
    m.add_class::<stack::StackLayer>()?;
    Ok(())
}
