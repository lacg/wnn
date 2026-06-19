//! IDSCacheBuilder — Phase 5 F-prep chunked accumulator.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// IDSCacheBuilder — Phase 5 F-prep chunked accumulator
// =============================================================================

/// Chunked builder for IDSCache. Lets Python feed train/eval data in slabs
/// and finalize into the same IDSCache structure that `IDSCacheWrapper.new_from_numpy`
/// produces.
///
/// Usage (Python):
///     b = IDSCacheBuilder(num_classes, total_features, num_parts, num_negatives,
///                         seed, balance_classes, single_cluster,
///                         undersample_majority, class_weight_multiplier)
///     for chunk_packed, chunk_labels in encoder.iter_chunks(df_train, chunk_size=1_000_000):
///         b.add_train_chunk(chunk_packed.ravel(), chunk_labels)
///     for chunk_packed, chunk_labels in encoder.iter_chunks(df_test, chunk_size=1_000_000):
///         b.add_eval_chunk(chunk_packed.ravel(), chunk_labels)
///     cache = b.finalize()
///
/// For Phase 2-onward (single-chunk case) prefer `IDSCacheWrapper.new_from_numpy`.
/// The builder is the entry point for Option F's streaming/memmap data path.
#[pyclass]
pub(crate) struct IDSCacheBuilderWrapper {
    /// Option<...> so finalize() can take ownership and consume the state.
    inner: Option<ids_cache::PartialFitState>,
}

#[pymethods]
impl IDSCacheBuilderWrapper {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (num_classes, total_features, num_parts, num_negatives, seed, balance_classes=false, single_cluster=false, undersample_majority=false, class_weight_multiplier=1.0))]
    fn new(
        num_classes: usize,
        total_features: usize,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        balance_classes: bool,
        single_cluster: bool,
        undersample_majority: bool,
        class_weight_multiplier: f32,
    ) -> Self {
        Self {
            inner: Some(ids_cache::PartialFitState::new(
                num_classes,
                total_features,
                num_parts,
                num_negatives,
                seed,
                balance_classes,
                single_cluster,
                undersample_majority,
                class_weight_multiplier,
            )),
        }
    }

    /// Append a chunk of training data.
    ///
    /// Args:
    ///   chunk_packed: numpy uint8 (np.packbits output, LSB-first within byte),
    ///                 shape (chunk_rows * bytes_per_row,) flattened.
    ///   chunk_labels: numpy int64 of length chunk_rows.
    fn add_train_chunk<'py>(
        &mut self,
        chunk_packed: PyReadonlyArray1<'py, u8>,
        chunk_labels: Vec<i64>,
    ) -> PyResult<()> {
        let state = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSCacheBuilder already finalized — cannot add more chunks",
            )
        })?;
        let slice = chunk_packed.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("chunk_packed not contiguous: {}", e))
        })?;
        let pb = ram_core::packed_bits::PackedBits::from_packed_bytes(slice.to_vec(), state.total_features());
        state.add_train_chunk(&pb, &chunk_labels);
        Ok(())
    }

    /// Append a chunk of evaluation data.
    fn add_eval_chunk<'py>(
        &mut self,
        chunk_packed: PyReadonlyArray1<'py, u8>,
        chunk_labels: Vec<i64>,
    ) -> PyResult<()> {
        let state = self.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSCacheBuilder already finalized — cannot add more chunks",
            )
        })?;
        let slice = chunk_packed.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("chunk_packed not contiguous: {}", e))
        })?;
        let pb = ram_core::packed_bits::PackedBits::from_packed_bytes(slice.to_vec(), state.total_features());
        state.add_eval_chunk(&pb, &chunk_labels);
        Ok(())
    }

    /// Number of training rows accumulated so far.
    fn num_train(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|s| s.num_train())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("IDSCacheBuilder already finalized")
            })
    }

    /// Number of eval rows accumulated so far.
    fn num_eval(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|s| s.num_eval())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("IDSCacheBuilder already finalized")
            })
    }

    /// Consume the builder and produce an IDSCacheWrapper. After this call,
    /// no more chunks may be added (the builder is one-shot).
    fn finalize(&mut self) -> PyResult<IDSCacheWrapper> {
        let state = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "IDSCacheBuilder already finalized — cannot finalize twice",
            )
        })?;
        Ok(IDSCacheWrapper {
            inner: state.finalize(),
        })
    }
}
