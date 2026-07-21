// ============================================================================
// GenomeCells — the opaque Rust-side cell store for controller genomes.
//
// Stage B of the Rust-first cell migration: Python's MemoryPayload becomes a
// thin wrapper around THIS handle, so genome cells live in six flat Rust
// vectors (u32 neuron / u64 address / u8 value per layer, 13 B/cell) and never
// cross the FFI boundary as per-cell Python tuples on any hot path:
//
//   * clone       -> memcpy of six Vecs (was: four numpy array copies through
//                    Python, ~60-120 of them per GA generation);
//   * remaps      -> cell_remap::* in place (was: per-cell Python loops);
//   * mutation /  -> memory_ops::* in place (was: whole-layer Vec<u8> round-
//     crossover      trips plus list() re-boxing per offspring);
//   * fingerprint -> a 128-bit digest (was: tobytes() copies of every buffer,
//                    tens of MB per elite per generation);
//   * train init  -> dagger_train reads the columns directly (was: one 3-int
//                    tuple per cell per genome per generation).
//
// Python materialisation (numpy views / triples) still exists for YAML
// serialisation, tests, and diagnostics — but only on demand, never as the
// storage format.
//
// ORDER IS IDENTITY. The buffers are ordered; every remap preserves the
// first-encounter order contract of cell_remap.rs, and the digest hashes the
// buffers in order. Two payloads with the same cells in a different order are
// different genomes — exactly as with MemoryPayload's tobytes() fingerprint.
// ============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::{PyOverflowError, PyValueError};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

use crate::cell_remap;
use crate::memory_ops;

#[pyclass]
#[derive(Clone, Default)]
pub struct GenomeCells {
	// state layer: parallel columns, one row per cell
	pub sn: Vec<u32>,
	pub sa: Vec<u64>,
	pub sv: Vec<u8>,
	// output layer
	pub on_: Vec<u32>,
	pub oa: Vec<u64>,
	pub ov: Vec<u8>,
}

fn overflow_err(e: cell_remap::AddrOverflow) -> PyErr {
	PyOverflowError::new_err(format!("cell address {} exceeds u64 after remap", e.0))
}

/// FNV-1a over one buffer into two independent u64 lanes; a length separator
/// between buffers prevents concatenation ambiguity.
fn fnv_feed(lanes: &mut [u64; 2], bytes: &[u8]) {
	const P1: u64 = 0x0000_0100_0000_01B3;
	const P2: u64 = 0x9E37_79B9_7F4A_7C15;
	for &b in bytes {
		lanes[0] = (lanes[0] ^ b as u64).wrapping_mul(P1);
		lanes[1] = (lanes[1] ^ b as u64).wrapping_mul(P2);
	}
	for &b in (bytes.len() as u64).to_le_bytes().iter() {
		lanes[0] = (lanes[0] ^ b as u64).wrapping_mul(P1);
		lanes[1] = (lanes[1] ^ b as u64).wrapping_mul(P2);
	}
}

fn as_bytes_u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn as_bytes_u64(v: &[u64]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

// ---- base64 (standard alphabet, padded) — byte-compatible with Python's
// base64.b64encode/b64decode; hand-rolled to avoid a dependency for ~40 lines.

const B64_ALPHABET: &[u8; 64] =
	b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(data: &[u8]) -> String {
	let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
	for chunk in data.chunks(3) {
		let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
		let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
		out.push(B64_ALPHABET[(n >> 18) as usize & 63] as char);
		out.push(B64_ALPHABET[(n >> 12) as usize & 63] as char);
		out.push(if chunk.len() > 1 { B64_ALPHABET[(n >> 6) as usize & 63] as char } else { '=' });
		out.push(if chunk.len() > 2 { B64_ALPHABET[n as usize & 63] as char } else { '=' });
	}
	out
}

fn b64_decode(s: &str) -> Result<Vec<u8>, String> {
	let mut rev = [255u8; 256];
	for (i, &c) in B64_ALPHABET.iter().enumerate() {
		rev[c as usize] = i as u8;
	}
	let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'\n' && c != b'\r').collect();
	if bytes.len() % 4 != 0 {
		return Err(format!("base64 length {} not a multiple of 4", bytes.len()));
	}
	let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
	for chunk in bytes.chunks(4) {
		let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
		let mut n: u32 = 0;
		for (i, &c) in chunk.iter().enumerate() {
			let v = if c == b'=' { 0 } else { rev[c as usize] };
			if v == 255 {
				return Err(format!("invalid base64 byte {c}"));
			}
			n |= (v as u32) << (18 - 6 * i);
		}
		out.push((n >> 16) as u8);
		if pad < 2 { out.push((n >> 8) as u8); }
		if pad < 1 { out.push(n as u8); }
	}
	Ok(out)
}

/// One layer's triples → the pack_int_columns byte stream: row-major
/// (neuron, addr, value) as i64 LE, or 16-byte i128 LE when any address
/// exceeds i64::MAX (the >63-bit-genome case). Returns (b64, n_rows, is_i128).
fn pack_layer(n: &[u32], a: &[u64], v: &[u8]) -> (String, usize, bool) {
	let wide = a.iter().any(|&x| x > i64::MAX as u64);
	let mut bytes = Vec::with_capacity(n.len() * 3 * if wide { 16 } else { 8 });
	for i in 0..n.len() {
		if wide {
			bytes.extend((n[i] as i128).to_le_bytes());
			bytes.extend((a[i] as i128).to_le_bytes());
			bytes.extend((v[i] as i128).to_le_bytes());
		} else {
			bytes.extend((n[i] as i64).to_le_bytes());
			bytes.extend((a[i] as i64).to_le_bytes());
			bytes.extend((v[i] as i64).to_le_bytes());
		}
	}
	(b64_encode(&bytes), n.len(), wide)
}

/// Inverse of pack_layer; validates the row count and the u64/u32/QSR ranges.
fn unpack_layer(b64: &str, n_rows: usize, i128fmt: bool, what: &str)
	-> Result<(Vec<u32>, Vec<u64>, Vec<u8>), String>
{
	let raw = b64_decode(b64)?;
	let w = if i128fmt { 16 } else { 8 };
	if raw.len() != n_rows * 3 * w {
		return Err(format!("{what}: packed length {} != n*3*{w} = {}", raw.len(), n_rows * 3 * w));
	}
	let read = |i: usize| -> i128 {
		let off = i * w;
		if i128fmt {
			i128::from_le_bytes(raw[off..off + 16].try_into().unwrap())
		} else {
			i64::from_le_bytes(raw[off..off + 8].try_into().unwrap()) as i128
		}
	};
	let mut on = Vec::with_capacity(n_rows);
	let mut oa = Vec::with_capacity(n_rows);
	let mut ov = Vec::with_capacity(n_rows);
	for r in 0..n_rows {
		let (n_, a_, v_) = (read(r * 3), read(r * 3 + 1), read(r * 3 + 2));
		if !(0..=u32::MAX as i128).contains(&n_) {
			return Err(format!("{what}: neuron {n_} out of u32 range"));
		}
		if !(0..=u64::MAX as i128).contains(&a_) {
			// Cells are u64-keyed end-to-end; a legacy >2^64 relic must fail
			// loudly, exactly as from_triples does.
			return Err(format!("{what}: address {a_} out of u64 range"));
		}
		if !(0..=255).contains(&v_) {
			return Err(format!("{what}: value {v_} out of u8 range"));
		}
		on.push(n_ as u32);
		oa.push(a_ as u64);
		ov.push(v_ as u8);
	}
	Ok((on, oa, ov))
}

fn validate_layer(
	n: &[u32], a: &[u64], v: &[u8], n_neurons: u32, bits: u32, what: &str,
) -> PyResult<()> {
	use std::collections::HashSet;
	let mut seen: HashSet<(u32, u64)> = HashSet::with_capacity(n.len());
	for i in 0..n.len() {
		if !seen.insert((n[i], a[i])) {
			return Err(PyValueError::new_err(format!("duplicate {what} cell key")));
		}
		if n[i] >= n_neurons {
			return Err(PyValueError::new_err(format!("{what} cell neuron out of range")));
		}
		if bits < 64 && (a[i] as u128) >= (1u128 << bits) {
			return Err(PyValueError::new_err(format!("{what} cell address exceeds 2^bits")));
		}
		if v[i] > 3 {
			return Err(PyValueError::new_err(format!("{what} cell value not QSR 0..3")));
		}
	}
	Ok(())
}

#[pymethods]
impl GenomeCells {
	/// Drop-in for the deleted Python MemoryPayload: same 4-arg constructor
	/// (universes as (neuron, address) pair lists, values as int lists), with
	/// all-empty defaults so `GenomeCells()` is the "no cells" placeholder.
	#[new]
	#[pyo3(signature = (state_universe=Vec::new(), output_universe=Vec::new(),
	                    state_values=Vec::new(), output_values=Vec::new()))]
	pub fn py_new(
		state_universe: Vec<(u32, u64)>, output_universe: Vec<(u32, u64)>,
		state_values: Vec<u8>, output_values: Vec<u8>,
	) -> PyResult<Self> {
		let (sn, sa): (Vec<u32>, Vec<u64>) = state_universe.into_iter().unzip();
		let (on_, oa): (Vec<u32>, Vec<u64>) = output_universe.into_iter().unzip();
		Self::from_columns(sn, sa, state_values, on_, oa, output_values)
	}

	/// Inverse of to_triples(). Rows may be tuples OR lists (YAML round-trip).
	/// Addresses are u64-keyed end-to-end; >= 2^64 raises OverflowError, loudly.
	#[staticmethod]
	pub fn from_triples(
		state_triples: Vec<Vec<i128>>, output_triples: Vec<Vec<i128>>,
	) -> PyResult<Self> {
		fn cols(rows: Vec<Vec<i128>>, what: &str)
			-> PyResult<(Vec<u32>, Vec<u64>, Vec<u8>)>
		{
			let mut n = Vec::with_capacity(rows.len());
			let mut a = Vec::with_capacity(rows.len());
			let mut v = Vec::with_capacity(rows.len());
			for row in rows {
				if row.len() != 3 {
					return Err(PyValueError::new_err(format!(
						"{what} triple has {} fields, expected 3", row.len())));
				}
				if !(0..=u32::MAX as i128).contains(&row[0]) {
					return Err(PyValueError::new_err(format!("{what} neuron {} out of u32 range", row[0])));
				}
				if !(0..=u64::MAX as i128).contains(&row[1]) {
					return Err(PyOverflowError::new_err(format!(
						"{what} address {} out of u64 range (cells are u64-keyed)", row[1])));
				}
				if !(0..=255).contains(&row[2]) {
					return Err(PyValueError::new_err(format!("{what} value {} out of u8 range", row[2])));
				}
				n.push(row[0] as u32);
				a.push(row[1] as u64);
				v.push(row[2] as u8);
			}
			Ok((n, a, v))
		}
		let (sn, sa, sv) = cols(state_triples, "state")?;
		let (on_, oa, ov) = cols(output_triples, "output")?;
		Ok(Self { sn, sa, sv, on_, oa, ov })
	}

	/// Transition affordance: the payload IS the handle now, so `.handle` is
	/// identity — keeps external diagnostics written against the wrapper alive.
	#[getter]
	pub fn handle(slf: PyRef<'_, Self>) -> Py<Self> { slf.into() }

	/// Python-facing clone (the MemoryPayload method name).
	#[pyo3(name = "clone")]
	pub fn py_clone(&self) -> Self { Clone::clone(self) }

	/// MemoryPayload.fingerprint(): the dedup identity tuple (= digest()).
	pub fn fingerprint(&self) -> (u64, u64) { self.digest() }

	/// Total cells across both layers — O(1).
	pub fn cell_count(&self) -> usize { self.sn.len() + self.on_.len() }

	/// Construct from six columns. The cold ingress (deserialize / tests /
	/// universe genesis); hot paths never build cells from Python data.
	#[staticmethod]
	pub fn from_columns(
		sn: Vec<u32>, sa: Vec<u64>, sv: Vec<u8>,
		on_: Vec<u32>, oa: Vec<u64>, ov: Vec<u8>,
	) -> PyResult<Self> {
		if sn.len() != sa.len() || sn.len() != sv.len() {
			return Err(PyValueError::new_err("state values/universe misaligned"));
		}
		if on_.len() != oa.len() || on_.len() != ov.len() {
			return Err(PyValueError::new_err("output values/universe misaligned"));
		}
		Ok(Self { sn, sa, sv, on_, oa, ov })
	}

	pub fn clone_cells(&self) -> Self { self.clone() }

	/// (state_cells, output_cells) counts — O(1), no materialisation.
	pub fn counts(&self) -> (usize, usize) { (self.sn.len(), self.on_.len()) }

	/// 128-bit content digest of the ordered buffers (dedup identity).
	pub fn digest(&self) -> (u64, u64) {
		let mut lanes = [0xcbf2_9ce4_8422_2325u64, 0x51_7cc1_b727_220a_95u64];
		fnv_feed(&mut lanes, &as_bytes_u32(&self.sn));
		fnv_feed(&mut lanes, &as_bytes_u64(&self.sa));
		fnv_feed(&mut lanes, &self.sv);
		fnv_feed(&mut lanes, &as_bytes_u32(&self.on_));
		fnv_feed(&mut lanes, &as_bytes_u64(&self.oa));
		fnv_feed(&mut lanes, &self.ov);
		(lanes[0], lanes[1])
	}

	// ---- materialisation (cold paths: YAML, tests, diagnostics) -------------

	pub fn to_triples(&self) -> (Vec<(u32, u64, u8)>, Vec<(u32, u64, u8)>) {
		let st = (0..self.sn.len()).map(|i| (self.sn[i], self.sa[i], self.sv[i])).collect();
		let ot = (0..self.on_.len()).map(|i| (self.on_[i], self.oa[i], self.ov[i])).collect();
		(st, ot)
	}

	#[getter(state_universe)]
	pub fn state_universe_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u64>>> {
		let mut flat = Vec::with_capacity(self.sn.len() * 2);
		for i in 0..self.sn.len() {
			flat.push(self.sn[i] as u64);
			flat.push(self.sa[i]);
		}
		Ok(flat.into_pyarray(py).reshape([self.sn.len(), 2])?)
	}

	#[getter(output_universe)]
	pub fn output_universe_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u64>>> {
		let mut flat = Vec::with_capacity(self.on_.len() * 2);
		for i in 0..self.on_.len() {
			flat.push(self.on_[i] as u64);
			flat.push(self.oa[i]);
		}
		Ok(flat.into_pyarray(py).reshape([self.on_.len(), 2])?)
	}

	#[getter(state_values)]
	pub fn state_values_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
		self.sv.clone().into_pyarray(py)
	}

	#[getter(output_values)]
	pub fn output_values_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
		self.ov.clone().into_pyarray(py)
	}

	/// Both layers packed as pack_int_columns-compatible byte streams, entirely
	/// in Rust: ((b64, n, is_i128) state, (…) output). The Python packer built
	/// one 3-int tuple per cell first (~1.3 GB transient per big genome, per
	/// genome per stage-checkpoint save — the 12/07 OOM's neighbourhood).
	pub fn export_packed(&self) -> ((String, usize, bool), (String, usize, bool)) {
		(pack_layer(&self.sn, &self.sa, &self.sv),
		 pack_layer(&self.on_, &self.oa, &self.ov))
	}

	/// Inverse of export_packed — YAML load without materialising triples.
	#[staticmethod]
	pub fn from_packed(
		state_b64: &str, state_n: usize, state_i128: bool,
		output_b64: &str, output_n: usize, output_i128: bool,
	) -> PyResult<Self> {
		let (sn, sa, sv) = unpack_layer(state_b64, state_n, state_i128, "state")
			.map_err(PyValueError::new_err)?;
		let (on_, oa, ov) = unpack_layer(output_b64, output_n, output_i128, "output")
			.map_err(PyValueError::new_err)?;
		Ok(Self { sn, sa, sv, on_, oa, ov })
	}

	// ---- validity (mirrors recurrent_genome._check_layer) -------------------

	pub fn validate(&self, state_neurons: u32, state_bits: u32,
	                output_neurons: u32, output_bits: u32) -> PyResult<()> {
		validate_layer(&self.sn, &self.sa, &self.sv, state_neurons, state_bits, "state")?;
		validate_layer(&self.on_, &self.oa, &self.ov, output_neurons, output_bits, "output")
	}

	// ---- remaps (in place; bit-exact via cell_remap) ------------------------

	pub fn remap_bits_state(&mut self, d: i64) -> PyResult<()> {
		let (n, a, v) = if d > 0 {
			cell_remap::remap_grow(&self.sn, &self.sa, &self.sv, d as u32).map_err(overflow_err)?
		} else {
			cell_remap::remap_shrink(&self.sn, &self.sa, &self.sv, (-d) as u32)
		};
		(self.sn, self.sa, self.sv) = (n, a, v);
		Ok(())
	}

	pub fn remap_bits_output(&mut self, d: i64) -> PyResult<()> {
		let (n, a, v) = if d > 0 {
			cell_remap::remap_grow(&self.on_, &self.oa, &self.ov, d as u32).map_err(overflow_err)?
		} else {
			cell_remap::remap_shrink(&self.on_, &self.oa, &self.ov, (-d) as u32)
		};
		(self.on_, self.oa, self.ov) = (n, a, v);
		Ok(())
	}

	/// STATE neurogenesis of +k / -k neurons: prefix reshapes in BOTH layers;
	/// on shrink, removed state neurons' own cells are dropped first.
	/// Mirrors RecurrentArchGenome._remap_state_neuro exactly.
	pub fn state_neuro(&mut self, k: i64, sw: u32, ow: u32, pf: u32,
	                   removed_floor: u32) -> PyResult<()> {
		if k > 0 {
			let (n, a, v) = cell_remap::remap_prefix_grow(
				&self.sn, &self.sa, &self.sv, k as u32, sw, pf).map_err(overflow_err)?;
			(self.sn, self.sa, self.sv) = (n, a, v);
			let (n, a, v) = cell_remap::remap_prefix_grow(
				&self.on_, &self.oa, &self.ov, k as u32, ow, pf).map_err(overflow_err)?;
			(self.on_, self.oa, self.ov) = (n, a, v);
		} else {
			let (n, a, v) = cell_remap::drop_neurons_ge(&self.sn, &self.sa, &self.sv, removed_floor);
			let (n, a, v) = cell_remap::remap_prefix_shrink(&n, &a, &v, (-k) as u32, sw, pf);
			(self.sn, self.sa, self.sv) = (n, a, v);
			let (n, a, v) = cell_remap::remap_prefix_shrink(
				&self.on_, &self.oa, &self.ov, (-k) as u32, ow, pf);
			(self.on_, self.oa, self.ov) = (n, a, v);
		}
		Ok(())
	}

	/// Surgical removal of state neuron k: drop its cells, reindex higher state
	/// neurons down by one, excise its pf-bit prefix window from every address
	/// in BOTH layers. Mirrors RecurrentArchGenome.remove_state_neuron.
	pub fn remove_state_neuron(&mut self, k: u32, p_lsb_s: u32, p_lsb_o: u32, pf: u32) {
		let mut n2 = Vec::with_capacity(self.sn.len());
		let mut a2 = Vec::with_capacity(self.sa.len());
		let mut v2 = Vec::with_capacity(self.sv.len());
		for i in 0..self.sn.len() {
			if self.sn[i] == k { continue; }
			n2.push(if self.sn[i] > k { self.sn[i] - 1 } else { self.sn[i] });
			a2.push(self.sa[i]);
			v2.push(self.sv[i]);
		}
		let (n, a, v) = cell_remap::remap_delete_bit_window(&n2, &a2, &v2, p_lsb_s, pf);
		(self.sn, self.sa, self.sv) = (n, a, v);
		let (n, a, v) = cell_remap::remap_delete_bit_window(
			&self.on_, &self.oa, &self.ov, p_lsb_o, pf);
		(self.on_, self.oa, self.ov) = (n, a, v);
	}

	pub fn drop_output_neurons_ge(&mut self, limit: u32) {
		let (n, a, v) = cell_remap::drop_neurons_ge(&self.on_, &self.oa, &self.ov, limit);
		(self.on_, self.oa, self.ov) = (n, a, v);
	}

	pub fn drop_changed_state(&mut self, changed: Vec<u32>) {
		let (n, a, v) = cell_remap::drop_changed_neurons(&self.sn, &self.sa, &self.sv, &changed);
		(self.sn, self.sa, self.sv) = (n, a, v);
	}

	pub fn drop_changed_output(&mut self, changed: Vec<u32>) {
		let (n, a, v) = cell_remap::drop_changed_neurons(&self.on_, &self.oa, &self.ov, &changed);
		(self.on_, self.oa, self.ov) = (n, a, v);
	}

	/// Inheritance filter (arch_strategy._filter_inherited_cells): keep a cell iff
	/// its neuron survives in the child, its address fits the child's 2^bits space
	/// (addresses are u64 by construction, so bits >= 64 means no address check),
	/// and its neuron's wiring did not change.
	#[allow(clippy::too_many_arguments)]
	pub fn filter_inherited(
		&self,
		state_neurons: u32, state_bits: u32, changed_state: Vec<u32>,
		output_neurons: u32, output_bits: u32, changed_output: Vec<u32>,
	) -> Self {
		use std::collections::HashSet;
		let cs: HashSet<u32> = changed_state.into_iter().collect();
		let co: HashSet<u32> = changed_output.into_iter().collect();
		let keep = |n: u32, a: u64, limit: u32, bits: u32, changed: &HashSet<u32>| {
			n < limit && (bits >= 64 || (a as u128) < (1u128 << bits)) && !changed.contains(&n)
		};
		let mut out = Self::default();
		for i in 0..self.sn.len() {
			if keep(self.sn[i], self.sa[i], state_neurons, state_bits, &cs) {
				out.sn.push(self.sn[i]); out.sa.push(self.sa[i]); out.sv.push(self.sv[i]);
			}
		}
		for i in 0..self.on_.len() {
			if keep(self.on_[i], self.oa[i], output_neurons, output_bits, &co) {
				out.on_.push(self.on_[i]); out.oa.push(self.oa[i]); out.ov.push(self.ov[i]);
			}
		}
		out
	}

	// ---- GA-MEMORY value operators (same memory_ops the Python called) ------

	/// Nudge ~rate of the values one step, both layers, in place. Identical
	/// draws to the old memory_mutate_values(values, quad, rate, seed, 0, 0, L)
	/// calls — same function, same coordinates, zero FFI payload.
	pub fn mutate_values(&mut self, quad: bool, rate: f64, seed: u64) {
		memory_ops::mutate_values(&mut self.sv, quad, rate, seed, 0, 0, memory_ops::LAYER_STATE);
		memory_ops::mutate_values(&mut self.ov, quad, rate, seed, 0, 0, memory_ops::LAYER_OUTPUT);
	}

	/// Address-KEYED value crossover in place: self's universe is the base (the
	/// child cloned parent `a`, so self's values ARE a's values); where `b`
	/// holds the same (neuron, address) key, adopt b's value with p=0.5.
	pub fn crossover_values_from(&mut self, b: PyRef<GenomeCells>, seed: u64) {
		self.sv = memory_ops::crossover_values_keyed(
			&self.sn, &self.sa, &self.sv, &b.sn, &b.sa, &b.sv,
			seed, 0, 0, memory_ops::LAYER_STATE);
		self.ov = memory_ops::crossover_values_keyed(
			&self.on_, &self.oa, &self.ov, &b.on_, &b.oa, &b.ov,
			seed, 0, 0, memory_ops::LAYER_OUTPUT);
	}

	/// Indices where values differ from `other` (Tabu move tokens). Universes
	/// are index-aligned in the MEMORY phase (frozen architecture).
	pub fn diff_indices(&self, other: PyRef<GenomeCells>) -> (Vec<u32>, Vec<u32>) {
		let sd = self.sv.iter().zip(other.sv.iter()).enumerate()
			.filter(|(_, (a, b))| a != b).map(|(i, _)| i as u32).collect();
		let od = self.ov.iter().zip(other.ov.iter()).enumerate()
			.filter(|(_, (a, b))| a != b).map(|(i, _)| i as u32).collect();
		(sd, od)
	}
}
