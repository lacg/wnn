//! `weightless._core` — the public Python surface over `ram_core`.
//!
//! One class, `SparseModel`: a uniform layout (num_clusters × neurons × bits)
//! of sparse RAM neurons in one `CellMode`, trained order-independently
//! (`ram_core::train::SparseTrainer`) and scored through the `Forward` trait
//! (`CpuForward` everywhere, `MetalForward` on Apple silicon). Everything
//! research-specific stays in the two research wheels.
//!
//! Inputs are PACKED bit rows: `uint8[n, ceil(total_bits/8)]`, LSB-first per
//! byte — exactly `numpy.packbits(bits, axis=1, bitorder="little")`.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use ram_core::cell_mode::CellMode;
use ram_core::forward::{ClusterLayout, CpuForward, Forward, ForwardRequest, ReadParams};
use ram_core::packed_bits::PackedBits;
use ram_core::sparse_memory::{SparseGpuExport, SparseLayerMemory};
use ram_core::train::{new_memory, SparseTrainer};

/// Bumped when the Python-visible contract changes. `weightless/__init__.py`
/// asserts it so a stale wheel fails loudly, the same rule as the research
/// facades.
pub const ABI_VERSION: u32 = 1;

fn value_err<E: std::fmt::Display>(e: E) -> PyErr
{
	PyValueError::new_err(e.to_string())
}

fn parse_mode(code: u8) -> PyResult<CellMode>
{
	CellMode::from_u8(code).ok_or_else(|| PyValueError::new_err(format!("unknown cell_mode code {code}")))
}

fn packed_rows(arr: &PyReadonlyArray2<'_, u8>, total_bits: usize) -> PyResult<PackedBits>
{
	let bytes_per_row = (total_bits + 7) / 8;
	let shape = arr.as_array().shape().to_vec();
	if shape.len() != 2 || shape[1] != bytes_per_row
	{
		return Err(PyValueError::new_err(format!(
			"packed input has {} bytes per row but total_bits={} needs {}",
			shape[1], total_bits, bytes_per_row
		)));
	}
	let data = arr
		.as_slice()
		.map_err(|_| PyValueError::new_err("packed input must be C-contiguous uint8"))?
		.to_vec();
	Ok(PackedBits::from_packed_bytes(data, total_bits))
}

// ---- backends --------------------------------------------------------------

// Device init + shader compilation happen once per thread and are reused by
// every later predict() call. Thread-local sidesteps any Send question.
thread_local! {
	#[cfg(target_os = "macos")]
	static METAL: std::cell::RefCell<Option<std::rc::Rc<ram_core::forward::MetalForward>>> =
		const { std::cell::RefCell::new(None) };
	static WGPU: std::cell::RefCell<Option<std::rc::Rc<ram_core::forward::WgpuForward>>> =
		const { std::cell::RefCell::new(None) };
}

#[cfg(target_os = "macos")]
fn metal() -> Option<std::rc::Rc<ram_core::forward::MetalForward>>
{
	METAL.with(|slot| {
		let mut slot = slot.borrow_mut();
		if slot.is_none()
		{
			if let Ok(m) = ram_core::forward::MetalForward::new()
			{
				*slot = Some(std::rc::Rc::new(m));
			}
		}
		slot.clone()
	})
}

fn wgpu_backend() -> Option<std::rc::Rc<ram_core::forward::WgpuForward>>
{
	WGPU.with(|slot| {
		let mut slot = slot.borrow_mut();
		if slot.is_none()
		{
			if let Ok(w) = ram_core::forward::WgpuForward::new()
			{
				*slot = Some(std::rc::Rc::new(w));
			}
		}
		slot.clone()
	})
}

/// "auto": Metal (Apple silicon) → wgpu (any GPU) → CPU. "gpu" = the first GPU
/// backend that exists here; "metal" / "wgpu" / "cpu" name one exactly.
fn with_backend<R>(name: &str, f: impl FnOnce(&dyn Forward) -> R) -> PyResult<R>
{
	match name
	{
		"cpu" => Ok(f(&CpuForward)),
		"metal" =>
		{
			#[cfg(target_os = "macos")]
			{
				if let Some(m) = metal()
				{
					return Ok(f(m.as_ref()));
				}
			}
			Err(PyRuntimeError::new_err("Metal backend unavailable on this machine"))
		}
		"wgpu" => match wgpu_backend()
		{
			Some(w) => Ok(f(w.as_ref())),
			None => Err(PyRuntimeError::new_err("wgpu backend unavailable on this machine (no GPU adapter)")),
		},
		"gpu" | "auto" =>
		{
			#[cfg(target_os = "macos")]
			{
				if let Some(m) = metal()
				{
					return Ok(f(m.as_ref()));
				}
			}
			if let Some(w) = wgpu_backend()
			{
				return Ok(f(w.as_ref()));
			}
			if name == "auto"
			{
				Ok(f(&CpuForward))
			}
			else
			{
				Err(PyRuntimeError::new_err("GPU backend unavailable on this machine (no Metal device, no wgpu adapter)"))
			}
		}
		other => Err(PyValueError::new_err(format!(
			"unknown backend {other:?}; use 'auto', 'cpu', 'gpu', 'metal' or 'wgpu'"
		))),
	}
}

/// Backends that can run here, in preference order; the wgpu entry names the
/// API it sits on, e.g. "wgpu:Metal", "wgpu:Vulkan".
#[pyfunction]
fn backends() -> Vec<String>
{
	let mut v = Vec::new();
	#[cfg(target_os = "macos")]
	{
		if metal().is_some()
		{
			v.push("metal".to_string());
		}
	}
	if let Some(w) = wgpu_backend()
	{
		v.push(format!("wgpu:{}", w.backend_name()));
	}
	v.push("cpu".to_string());
	v
}

#[pyfunction]
fn cell_mode_names() -> Vec<(u8, &'static str)>
{
	CellMode::ALL.iter().map(|m| (m.as_u8(), m.name())).collect()
}

// ---- the model --------------------------------------------------------------

#[pyclass(module = "weightless._core")]
struct SparseModel
{
	mode: CellMode,
	layout: ClusterLayout,
	total_bits: usize,
	connections: Vec<i64>,
	memory: SparseLayerMemory,
	trainer: SparseTrainer,
	export: Option<SparseGpuExport>,
}

#[pymethods]
impl SparseModel
{
	/// `connections`: int64[num_clusters * neurons_per_cluster * bits_per_neuron],
	/// neuron-major, each entry an input bit index in `0..total_bits`.
	#[new]
	fn new(
		num_clusters: usize,
		neurons_per_cluster: usize,
		bits_per_neuron: usize,
		total_bits: usize,
		cell_mode: u8,
		connections: PyReadonlyArray1<'_, i64>,
	) -> PyResult<Self>
	{
		let mode = parse_mode(cell_mode)?;
		let layout = ClusterLayout {
			num_clusters,
			neurons_per_cluster,
			bits_per_neuron,
		};
		if num_clusters == 0 || neurons_per_cluster == 0 || bits_per_neuron == 0 || total_bits == 0
		{
			return Err(PyValueError::new_err("num_clusters, neurons_per_cluster, bits_per_neuron and total_bits must be > 0"));
		}
		let connections = connections.as_slice().map_err(|_| PyValueError::new_err("connections must be contiguous int64"))?.to_vec();
		if connections.len() != layout.num_neurons() * bits_per_neuron
		{
			return Err(PyValueError::new_err(format!(
				"connections has {} entries, layout needs {}",
				connections.len(),
				layout.num_neurons() * bits_per_neuron
			)));
		}
		if let Some(&bad) = connections.iter().find(|&&c| c < 0 || c as usize >= total_bits)
		{
			return Err(PyValueError::new_err(format!("connection index {bad} outside 0..{total_bits}")));
		}
		Ok(Self {
			mode,
			layout,
			total_bits,
			connections: connections.clone(),
			memory: new_memory(mode, layout),
			trainer: SparseTrainer::new(mode, layout),
			export: None,
		})
	}

	#[getter]
	fn cell_mode(&self) -> u8
	{
		self.mode.as_u8()
	}

	#[getter]
	fn num_clusters(&self) -> usize
	{
		self.layout.num_clusters
	}

	#[getter]
	fn total_bits(&self) -> usize
	{
		self.total_bits
	}

	/// Accumulate `packed` rows with integer `labels` (0..num_clusters) and
	/// optional uint32 repeat-count `weights`, then commit. Calling again FOLDS
	/// the new batch into the same accumulators (order-independent), so
	/// `train(A); train(B)` == `train(A+B)`.
	#[pyo3(signature = (packed, labels, weights=None))]
	fn train(
		&mut self,
		py: Python<'_>,
		packed: PyReadonlyArray2<'_, u8>,
		labels: PyReadonlyArray1<'_, i64>,
		weights: Option<PyReadonlyArray1<'_, u32>>,
	) -> PyResult<usize>
	{
		let input = packed_rows(&packed, self.total_bits)?;
		let labels = labels.as_slice().map_err(|_| PyValueError::new_err("labels must be contiguous int64"))?.to_vec();
		let weights: Option<Vec<u32>> = match weights
		{
			Some(w) => Some(w.as_slice().map_err(|_| PyValueError::new_err("weights must be contiguous uint32"))?.to_vec()),
			None => None,
		};
		let trainer = &self.trainer;
		let memory = &self.memory;
		let conns = &self.connections;
		py.allow_threads(|| -> PyResult<()> {
			trainer.accumulate(&input, conns, &labels, weights.as_deref()).map_err(value_err)?;
			trainer.commit(memory);
			Ok(())
		})?;
		self.export = None;
		Ok(self.memory.total_cells())
	}

	/// Per-cluster scores, float32[n * num_clusters] row-major, in [0, 1].
	/// `backend`: "auto" | "cpu" | "gpu".
	#[pyo3(signature = (packed, empty_value=0.5, coverage_aware=false, run_seed=0, backend="auto", read_mode=None))]
	fn forward<'py>(
		&mut self,
		py: Python<'py>,
		packed: PyReadonlyArray2<'_, u8>,
		empty_value: f32,
		coverage_aware: bool,
		run_seed: u64,
		backend: &str,
		read_mode: Option<u8>,
	) -> PyResult<Bound<'py, PyArray1<f32>>>
	{
		let input = packed_rows(&packed, self.total_bits)?;
		// `read_mode` lets Python ask for the EXPECTED read of a stochastic
		// mode (QSR → QUAD_WEIGHTED, PLN → TERNARY@0.5) for predict_proba.
		let cell_mode = match read_mode
		{
			Some(code) => parse_mode(code)?,
			None => self.mode,
		};
		if self.export.is_none()
		{
			self.export = Some(self.memory.export_for_gpu());
		}
		let export = self.export.as_ref().expect("export just built");
		let req = ForwardRequest {
			input: &input,
			connections: &self.connections,
			memory: export,
			layout: self.layout,
			read: ReadParams {
				cell_mode,
				empty_value,
				coverage_aware,
				run_seed,
			},
		};
		let scores = py.allow_threads(|| with_backend(backend, |b| b.forward(&req)))?.map_err(value_err)?;
		Ok(PyArray1::from_vec(py, scores))
	}

	/// The sorted-key export: (offsets uint32, counts uint32, keys uint64, values uint8).
	fn export_keys<'py>(
		&mut self,
		py: Python<'py>,
	) -> PyResult<(
		Bound<'py, PyArray1<u32>>,
		Bound<'py, PyArray1<u32>>,
		Bound<'py, PyArray1<u64>>,
		Bound<'py, PyArray1<u8>>,
	)>
	{
		if self.export.is_none()
		{
			self.export = Some(self.memory.export_for_gpu());
		}
		let e = self.export.as_ref().expect("export just built");
		Ok((
			PyArray1::from_vec(py, e.offsets.clone()),
			PyArray1::from_vec(py, e.counts.clone()),
			PyArray1::from_vec(py, e.keys.clone()),
			PyArray1::from_vec(py, e.values.clone()),
		))
	}

	/// Stored (non-default) cells.
	fn total_cells(&self) -> usize
	{
		self.memory.total_cells()
	}

	/// Forget everything (memory AND accumulators).
	fn reset(&mut self)
	{
		self.memory = new_memory(self.mode, self.layout);
		self.trainer = SparseTrainer::new(self.mode, self.layout);
		self.export = None;
	}

	// ---- pickle ------------------------------------------------------------
	// State = the constructor args + the trainer's accumulators (NOT the
	// committed cells: they are a pure function of the accumulators, and
	// carrying the accumulators is what keeps partial_fit exact after a
	// save/load). numpy arrays throughout so the pickle is compact.

	fn __reduce__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<(PyObject, PyObject, PyObject)>
	{
		let cls = py.get_type::<Self>().into_any().unbind();
		let args = (
			slf.layout.num_clusters,
			slf.layout.neurons_per_cluster,
			slf.layout.bits_per_neuron,
			slf.total_bits,
			slf.mode.as_u8(),
			PyArray1::from_vec(py, slf.connections.clone()),
		)
			.into_pyobject(py)?
			.into_any()
			.unbind();
		let acc = slf.trainer.export_accum();
		let neurons: Vec<u32> = acc.iter().map(|e| e.0).collect();
		let keys: Vec<u64> = acc.iter().map(|e| e.1).collect();
		let values: Vec<i64> = acc.iter().map(|e| e.2).collect();
		let state = (
			ABI_VERSION,
			PyArray1::from_vec(py, neurons),
			PyArray1::from_vec(py, keys),
			PyArray1::from_vec(py, values),
		)
			.into_pyobject(py)?
			.into_any()
			.unbind();
		Ok((cls, args, state))
	}

	fn __setstate__(
		&mut self,
		state: (u32, PyReadonlyArray1<'_, u32>, PyReadonlyArray1<'_, u64>, PyReadonlyArray1<'_, i64>),
	) -> PyResult<()>
	{
		let (abi, neurons, keys, values) = state;
		if abi != ABI_VERSION
		{
			return Err(PyValueError::new_err(format!("pickled with weightless._core ABI {abi}, this build is {ABI_VERSION}")));
		}
		let neurons = neurons.as_slice().map_err(|_| PyValueError::new_err("bad pickle: neurons"))?;
		let keys = keys.as_slice().map_err(|_| PyValueError::new_err("bad pickle: keys"))?;
		let values = values.as_slice().map_err(|_| PyValueError::new_err("bad pickle: values"))?;
		if neurons.len() != keys.len() || keys.len() != values.len()
		{
			return Err(PyValueError::new_err("bad pickle: accumulator arrays differ in length"));
		}
		let entries: Vec<(u32, u64, i64)> = neurons
			.iter()
			.zip(keys)
			.zip(values)
			.map(|((n, k), v)| (*n, *k, *v))
			.collect();
		self.reset();
		self.trainer.import_accum(&entries).map_err(value_err)?;
		self.trainer.commit(&self.memory);
		Ok(())
	}

	/// Accumulator count (what a pickle carries).
	fn accum_len(&self) -> usize
	{
		self.trainer.touched()
	}
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()>
{
	m.add("ABI_VERSION", ABI_VERSION)?;
	m.add_function(wrap_pyfunction!(backends, m)?)?;
	m.add_function(wrap_pyfunction!(cell_mode_names, m)?)?;
	m.add_class::<SparseModel>()?;
	Ok(())
}
