//! `Forward` — the one public boundary for scoring a sparse RAM memory.
//!
//! Every backend scores the same way: for each example and each cluster, walk
//! the cluster's neurons, address each one through its connectivity, binary-
//! search the neuron's sorted keys (miss → the mode's default cell), turn the
//! cell into a weight (`cell_to_weight_rng`, seeded coin for QSR/PLN) and
//! average over the cluster. `CpuForward` IS that description, written once in
//! Rust; `MetalForward` wraps the existing `sparse_forward.metal` kernel. The
//! parity test at the bottom is the contract: every backend must match
//! `CpuForward` on every mode at a fixed `run_seed`.
//!
//! Training is NOT on this trait — RAM writes are DashMap on the CPU
//! (`sparse_memory::train_batch_sparse`); the GPU only ever reads the sorted
//! export. That is the existing architecture, stated on the public surface.

use crate::cell_mode::CellMode;
use crate::neuron_memory::{
	cell_to_weight_rng, compute_address_packed, pack_packed_to_u64, qsr_key,
};
use crate::packed_bits::PackedBits;
use crate::sparse_memory::SparseGpuExport;
use rayon::prelude::*;

/// How cells are read. `run_seed` only matters for the stochastic modes;
/// `empty_value` only for TERNARY (`CellMode::uses_empty_value`).
#[derive(Clone, Copy, Debug)]
pub struct ReadParams
{
	pub cell_mode: CellMode,
	pub empty_value: f32,
	pub coverage_aware: bool,
	pub run_seed: u64,
}

impl ReadParams
{
	pub fn deterministic(cell_mode: CellMode) -> Self
	{
		Self {
			cell_mode,
			empty_value: 0.5,
			coverage_aware: false,
			run_seed: 0,
		}
	}
}

/// A uniform layout: `num_clusters` discriminators of `neurons_per_cluster`
/// neurons, each observing `bits_per_neuron` input positions. Neuron `i` of
/// cluster `c` is global neuron `c * neurons_per_cluster + i`; its
/// connectivity is `connections[neuron * bits_per_neuron ..][..bits_per_neuron]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterLayout
{
	pub num_clusters: usize,
	pub neurons_per_cluster: usize,
	pub bits_per_neuron: usize,
}

impl ClusterLayout
{
	#[inline]
	pub fn num_neurons(&self) -> usize
	{
		self.num_clusters * self.neurons_per_cluster
	}
}

/// One scoring call. Borrows everything; a backend copies only what its device
/// needs. Output is row-major `(num_examples × num_clusters)` in `[0, 1]`.
pub struct ForwardRequest<'a>
{
	pub input: &'a PackedBits,
	pub connections: &'a [i64],
	pub memory: &'a SparseGpuExport,
	pub layout: ClusterLayout,
	pub read: ReadParams,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardError
{
	/// The request's shapes disagree with each other (caller bug).
	Shape(String),
	/// The backend cannot run here (no Metal device, feature off, …).
	Unavailable(String),
	/// The backend ran and failed.
	Backend(String),
}

impl std::fmt::Display for ForwardError
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		match self
		{
			ForwardError::Shape(s) => write!(f, "forward: shape error: {s}"),
			ForwardError::Unavailable(s) => write!(f, "forward: backend unavailable: {s}"),
			ForwardError::Backend(s) => write!(f, "forward: backend error: {s}"),
		}
	}
}

impl std::error::Error for ForwardError {}

impl<'a> ForwardRequest<'a>
{
	/// Every shape relation a kernel would otherwise trust blindly. Called by
	/// each backend before it touches a buffer.
	pub fn validate(&self) -> Result<(), ForwardError>
	{
		let l = self.layout;
		let shape = |msg: String| Err(ForwardError::Shape(msg));
		if l.num_clusters == 0 || l.neurons_per_cluster == 0 || l.bits_per_neuron == 0
		{
			return shape(format!("layout has a zero dimension: {l:?}"));
		}
		let want = l.num_neurons() * l.bits_per_neuron;
		if self.connections.len() != want
		{
			return shape(format!(
				"connections.len()={} but layout needs {} ({} neurons × {} bits)",
				self.connections.len(),
				want,
				l.num_neurons(),
				l.bits_per_neuron
			));
		}
		let total_bits = self.input.total_bits() as i64;
		if let Some(&bad) = self.connections.iter().find(|&&c| c < 0 || c >= total_bits)
		{
			return shape(format!(
				"connection index {bad} outside the input width {total_bits}"
			));
		}
		let m = self.memory;
		if m.offsets.len() != l.num_neurons() || m.counts.len() != l.num_neurons()
		{
			return shape(format!(
				"memory exports {} neurons (offsets) / {} (counts) but layout has {}",
				m.offsets.len(),
				m.counts.len(),
				l.num_neurons()
			));
		}
		if m.keys.len() != m.values.len()
		{
			return shape(format!(
				"memory keys.len()={} != values.len()={}",
				m.keys.len(),
				m.values.len()
			));
		}
		for n in 0..l.num_neurons()
		{
			let end = m.offsets[n] as usize + m.counts[n] as usize;
			if end > m.keys.len()
			{
				return shape(format!(
					"neuron {n}: offset+count={end} runs past keys.len()={}",
					m.keys.len()
				));
			}
		}
		Ok(())
	}
}

/// A scoring backend. `name()` is for logs and the parity report.
pub trait Forward
{
	fn name(&self) -> &'static str;
	fn forward(&self, req: &ForwardRequest) -> Result<Vec<f32>, ForwardError>;
}

// ============================================================================
// CPU — the reference implementation
// ============================================================================

/// Binary search of one neuron's sorted keys; a miss reads as `miss_cell`.
/// Mirrors `binary_search_lookup` in sparse_forward.metal.
#[inline]
fn lookup_or(memory: &SparseGpuExport, neuron: usize, address: u64, miss_cell: u8) -> u8
{
	let start = memory.offsets[neuron] as usize;
	let count = memory.counts[neuron] as usize;
	if count == 0
	{
		return miss_cell;
	}
	match memory.keys[start..start + count].binary_search(&address)
	{
		Ok(i) => memory.values[start + i],
		Err(_) => miss_cell,
	}
}

/// Rayon over examples; the per-cluster loop is the shader's `accumulate_sparse`
/// written once for every mode (`cell_to_weight_rng` collapses the shader's
/// per-mode branches: threshold for QUAD_BINARY, graded for QUAD_WEIGHTED,
/// coin for QSR/PLN, TRUE-only for BINARY, `empty_value` for TERNARY).
pub struct CpuForward;

impl Forward for CpuForward
{
	fn name(&self) -> &'static str
	{
		"cpu"
	}

	fn forward(&self, req: &ForwardRequest) -> Result<Vec<f32>, ForwardError>
	{
		req.validate()?;
		let l = req.layout;
		let r = req.read;
		let mode = r.cell_mode.as_u8();
		let miss = r.cell_mode.miss_cell(r.coverage_aware);
		let (words, wpe) = pack_packed_to_u64(req.input);
		let num_examples = req.input.num_rows();
		let mut out = vec![0.0f32; num_examples * l.num_clusters];
		out
			.par_chunks_mut(l.num_clusters)
			.enumerate()
			.for_each(|(ex, row)| {
				let w = &words[ex * wpe..(ex + 1) * wpe];
				for (c, slot) in row.iter_mut().enumerate()
				{
					let mut sum = 0.0f32;
					for n in 0..l.neurons_per_cluster
					{
						let neuron = c * l.neurons_per_cluster + n;
						let conns =
							&req.connections[neuron * l.bits_per_neuron..(neuron + 1) * l.bits_per_neuron];
						let address = compute_address_packed(w, conns, l.bits_per_neuron) as u64;
						let cell = lookup_or(req.memory, neuron, address, miss);
						let rng = qsr_key(r.run_seed, neuron as u64, address, ex as u64);
						sum += cell_to_weight_rng(cell as i64, mode, r.empty_value, rng);
					}
					*slot = sum / l.neurons_per_cluster as f32;
				}
			});
		Ok(out)
	}
}

// ============================================================================
// Metal — wraps the existing sparse_forward.metal kernel (macOS only)
// ============================================================================

#[cfg(target_os = "macos")]
pub struct MetalForward
{
	evaluator: crate::metal_sparse::MetalSparseEvaluator,
}

#[cfg(target_os = "macos")]
impl MetalForward
{
	pub fn new() -> Result<Self, ForwardError>
	{
		let evaluator =
			crate::metal_sparse::MetalSparseEvaluator::new().map_err(ForwardError::Unavailable)?;
		Ok(Self { evaluator })
	}
}

#[cfg(target_os = "macos")]
impl Forward for MetalForward
{
	fn name(&self) -> &'static str
	{
		"metal"
	}

	fn forward(&self, req: &ForwardRequest) -> Result<Vec<f32>, ForwardError>
	{
		req.validate()?;
		let l = req.layout;
		let r = req.read;
		let (words, wpe) = pack_packed_to_u64(req.input);
		self
			.evaluator
			.forward_batch_sparse(
				&words,
				req.connections,
				&req.memory.keys,
				&req.memory.values,
				&req.memory.offsets,
				&req.memory.counts,
				req.input.num_rows(),
				wpe,
				l.num_neurons(),
				l.bits_per_neuron,
				l.neurons_per_cluster,
				l.num_clusters,
				r.coverage_aware,
				r.cell_mode.as_u8(),
				r.empty_value,
				r.run_seed,
			)
			.map_err(ForwardError::Backend)
	}
}

#[cfg(feature = "gpu-wgpu")]
pub use crate::wgpu_forward::WgpuForward;

/// The best backend available here: Metal when a device exists, else wgpu
/// (feature `gpu-wgpu`), else CPU.
pub fn default_backend() -> Box<dyn Forward + Send + Sync>
{
	#[cfg(target_os = "macos")]
	{
		if let Ok(m) = MetalForward::new()
		{
			return Box::new(m);
		}
	}
	#[cfg(feature = "gpu-wgpu")]
	{
		if let Ok(w) = WgpuForward::new()
		{
			return Box::new(w);
		}
	}
	Box::new(CpuForward)
}

// ============================================================================
// Tests — the parity contract
// ============================================================================

#[cfg(test)]
mod tests
{
	use super::*;

	/// Tiny LCG so the test needs no rand feature plumbing.
	struct Lcg(u64);
	impl Lcg
	{
		fn next(&mut self) -> u64
		{
			self.0 = self
				.0
				.wrapping_mul(6364136223846793005)
				.wrapping_add(1442695040888963407);
			self.0 >> 11
		}
		fn below(&mut self, n: usize) -> usize
		{
			(self.next() % n as u64) as usize
		}
	}

	struct Fixture
	{
		input: PackedBits,
		connections: Vec<i64>,
		memory: SparseGpuExport,
		layout: ClusterLayout,
	}

	/// Random connectivity + random input rows; the memory holds the addresses
	/// the rows actually produce (for a random half of the neurons) with random
	/// in-range cells, so lookups HIT — a fresh random memory at 96 bits would
	/// never be hit and the parity would be vacuous.
	fn fixture(bits: usize, mode: CellMode, seed: u64) -> Fixture
	{
		let layout = ClusterLayout {
			num_clusters: 3,
			neurons_per_cluster: 4,
			bits_per_neuron: bits,
		};
		let total_bits = 128usize;
		let num_examples = 40usize;
		let mut rng = Lcg(seed);
		let connections: Vec<i64> = (0..layout.num_neurons() * bits)
			.map(|_| rng.below(total_bits) as i64)
			.collect();
		let bools: Vec<bool> = (0..num_examples * total_bits)
			.map(|_| rng.next() & 1 == 1)
			.collect();
		let input = PackedBits::from_bool_slice(&bools, total_bits);
		let (words, wpe) = pack_packed_to_u64(&input);
		let states = mode.num_states() as usize;
		let mut keys = Vec::new();
		let mut values = Vec::new();
		let mut offsets = Vec::new();
		let mut counts = Vec::new();
		for neuron in 0..layout.num_neurons()
		{
			offsets.push(keys.len() as u32);
			let mut entries: Vec<u64> = Vec::new();
			if neuron % 2 == 0
			{
				for ex in 0..num_examples
				{
					if rng.below(2) == 0
					{
						let w = &words[ex * wpe..(ex + 1) * wpe];
						let conns = &connections[neuron * bits..(neuron + 1) * bits];
						entries.push(compute_address_packed(w, conns, bits) as u64);
					}
				}
			}
			entries.sort_unstable();
			entries.dedup();
			counts.push(entries.len() as u32);
			for k in entries
			{
				keys.push(k);
				values.push(rng.below(states) as u8);
			}
		}
		let memory = SparseGpuExport {
			keys,
			values,
			offsets,
			counts,
			num_neurons: layout.num_neurons(),
		};
		Fixture {
			input,
			connections,
			memory,
			layout,
		}
	}

	fn score(backend: &dyn Forward, f: &Fixture, read: ReadParams) -> Vec<f32>
	{
		backend
			.forward(&ForwardRequest {
				input: &f.input,
				connections: &f.connections,
				memory: &f.memory,
				layout: f.layout,
				read,
			})
			.unwrap_or_else(|e| panic!("{} failed: {e}", backend.name()))
	}

	#[test]
	fn validate_rejects_bad_shapes()
	{
		let f = fixture(16, CellMode::QuadWeighted, 1);
		let read = ReadParams::deterministic(CellMode::QuadWeighted);
		let short = &f.connections[..f.connections.len() - 1];
		let req = ForwardRequest {
			input: &f.input,
			connections: short,
			memory: &f.memory,
			layout: f.layout,
			read,
		};
		assert!(matches!(req.validate(), Err(ForwardError::Shape(_))));
		let mut bad_conn = f.connections.clone();
		bad_conn[0] = 128;
		let req = ForwardRequest {
			input: &f.input,
			connections: &bad_conn,
			memory: &f.memory,
			layout: f.layout,
			read,
		};
		assert!(matches!(req.validate(), Err(ForwardError::Shape(_))));
		let mut narrow = f.layout;
		narrow.num_clusters = 2;
		let req = ForwardRequest {
			input: &f.input,
			connections: &f.connections,
			memory: &f.memory,
			layout: narrow,
			read,
		};
		assert!(matches!(req.validate(), Err(ForwardError::Shape(_))));
	}

	/// The CPU reference: an all-miss memory scores the mode's default cell
	/// weight everywhere (0 / 0.25 / 0.5 / empty_value), and coverage_aware
	/// makes every miss weigh 0.0 in every mode.
	#[test]
	fn cpu_miss_semantics_per_mode()
	{
		for mode in CellMode::ALL
		{
			let mut f = fixture(16, mode, 7);
			f.memory.counts.iter_mut().for_each(|c| *c = 0);
			f.memory.offsets.iter_mut().for_each(|o| *o = 0);
			f.memory.keys.clear();
			f.memory.values.clear();
			let want =
				crate::neuron_memory::cell_to_weight(mode.default_cell() as i64, mode.as_u8(), 0.37);
			let mut read = ReadParams::deterministic(mode);
			read.empty_value = 0.37;
			// A stochastic miss is a coin (PLN fair, QSR at the WEAK_FALSE weight
			// 0.25), so only its MEAN is pinned; every other mode is exact.
			let out = score(&CpuForward, &f, read);
			if mode.is_stochastic()
			{
				let mean = out.iter().sum::<f32>() / out.len() as f32;
				assert!(
					(mean - want).abs() < 0.1,
					"{mode}: stochastic miss mean {mean} far from {want}"
				);
			}
			else
			{
				assert!(
					out.iter().all(|&v| (v - want).abs() < 1e-6),
					"{mode}: expected {want}, got {:?}",
					&out[..4]
				);
			}
			read.coverage_aware = true;
			let out = score(&CpuForward, &f, read);
			assert!(
				out.iter().all(|&v| v == 0.0),
				"{mode}: coverage-aware miss must weigh 0"
			);
		}
	}

	/// The contract: every backend == CpuForward, every mode, both coverage
	/// settings, at 16 bits (direct address) and 96 bits (hashed wide address).
	#[cfg(target_os = "macos")]
	#[test]
	fn metal_matches_cpu_every_mode()
	{
		let Ok(metal) = MetalForward::new()
		else
		{
			eprintln!("skipping metal parity: no Metal device");
			return;
		};
		backend_matches_cpu_every_mode(&metal);
	}

	/// Same contract for the portable backend — including 200 bits (four
	/// hashed words) and a dispatch larger than one chunk would be, on the
	/// u64-emulation paths the Metal kernel never needed.
	#[cfg(feature = "gpu-wgpu")]
	#[test]
	fn wgpu_matches_cpu_every_mode()
	{
		let Ok(w) = WgpuForward::new()
		else
		{
			eprintln!("skipping wgpu parity: no adapter");
			return;
		};
		eprintln!("wgpu backend: {}", w.backend_name());
		backend_matches_cpu_every_mode(&w);
		let f = fixture(200, CellMode::Qsr, 0xABCDEF);
		let read = ReadParams {
			cell_mode: CellMode::Qsr,
			empty_value: 0.37,
			coverage_aware: false,
			run_seed: 0x1234_5678_9ABC_DEF0,
		};
		let cpu = score(&CpuForward, &f, read);
		let gpu = score(&w, &f, read);
		let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
		assert!(worst <= 1e-6, "wgpu 200-bit QSR: max |cpu-gpu| = {worst}");
	}

	fn backend_matches_cpu_every_mode(backend: &dyn Forward)
	{
		for bits in [16usize, 96]
		{
			for mode in CellMode::ALL
			{
				for coverage_aware in [false, true]
				{
					let f = fixture(bits, mode, 0x5EED_0000 + bits as u64);
					let read = ReadParams {
						cell_mode: mode,
						empty_value: 0.37,
						coverage_aware,
						run_seed: 0xC0FFEE,
					};
					let cpu = score(&CpuForward, &f, read);
					let gpu = score(backend, &f, read);
					assert_eq!(cpu.len(), gpu.len());
					let worst = cpu
						.iter()
						.zip(&gpu)
						.map(|(a, b)| (a - b).abs())
						.fold(0.0f32, f32::max);
					assert!(
						worst <= 1e-6,
						"{mode} bits={bits} coverage={coverage_aware}: max |cpu-gpu| = {worst}"
					);
					// Vacuity guard: the fixture guarantees hits, so at least one
					// score must differ from the all-miss baseline (except when
					// coverage-aware turns every hit-less cluster to 0 AND the
					// mode's hit weights can be 0 — BINARY with all-FALSE values
					// is the only way to get there, which the seed does not do).
					let baseline = crate::neuron_memory::cell_to_weight(
						mode.miss_cell(coverage_aware) as i64,
						mode.as_u8(),
						0.37,
					);
					if !mode.is_stochastic()
					{
						assert!(
							cpu.iter().any(|&v| (v - baseline).abs() > 1e-6),
							"{mode} bits={bits}: every score equals the miss baseline — parity was vacuous"
						);
					}
				}
			}
		}
	}
}
