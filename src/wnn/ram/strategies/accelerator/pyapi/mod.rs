//! PyO3 surface of the accelerator, split by domain (D3, 11/06/2026).
//!
//! lib.rs stays the thin crate root: mod declarations, shared singletons,
//! ABI_VERSION, and the #[pymodule] registration. Every #[pyfunction] /
//! #[pyclass] wrapper lives here.

mod general;
pub(crate) use general::*;
mod ramlm_api;
pub(crate) use ramlm_api::*;
mod ramlm_numpy;
pub(crate) use ramlm_numpy::*;
mod sparse_api;
pub(crate) use sparse_api::*;
mod tiered_sparse;
pub(crate) use tiered_sparse::*;
mod per_cluster_api;
pub(crate) use per_cluster_api::*;
mod token_cache_api;
pub(crate) use token_cache_api::*;
mod ids_cache_api;
pub(crate) use ids_cache_api::*;
mod ids_builder_api;
pub(crate) use ids_builder_api::*;
mod ids_streamer_api;
pub(crate) use ids_streamer_api::*;
mod gating_api;
pub(crate) use gating_api::*;
mod bitwise_api;
pub(crate) use bitwise_api::*;
mod bitwise_cache_api;
pub(crate) use bitwise_cache_api::*;
mod multistage_api;
pub(crate) use multistage_api::*;
mod connections_api;
pub(crate) use connections_api::*;
