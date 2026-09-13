// Sparse RAM forward — WGSL twin of core/shaders/sparse_forward.metal
// (accumulate_sparse + sparse_forward_pass) for the portable wgpu backend.
//
// One invocation per (example, cluster): walk the cluster's neurons, address
// each through its connectivity (raw MSB-first integer up to 64 bits, the
// splitmix64 chain above — wnn_mix64 / compute_address_wide), binary-search
// the neuron's sorted keys (miss → P.default_cell), turn the cell into a
// weight (deterministic per mode; seeded coin for QSR / PLN — the SAME
// (run_seed, neuron, address, example) key and splitmix finalizer as
// neuron_memory.rs qsr_key / common.metal wnn_qsr_key) and average.
//
// WGSL has no 64-bit integers, so a u64 is a vec2<u32> (x = low word,
// y = high word) throughout; every 64-bit op used by the Rust/Metal twins
// (xor, shift, add, wrapping multiply, compare) is spelled out below. Keys
// are stored as (lo, hi) pairs; cells are packed 4 per u32.
//
// Accumulation order is neuron-major in f32 exactly like CpuForward, so the
// parity test holds to 1e-6 (the Metal kernel's integer-count shortcuts for
// the non-graded modes are equivalent: sums of 1.0 are exact in f32).

struct Params {
	num_examples: u32,
	words_per_example: u32,
	num_neurons: u32,
	bits_per_neuron: u32,
	neurons_per_cluster: u32,
	num_clusters: u32,
	memory_mode: u32,
	default_cell: u32,
	run_seed_lo: u32,
	run_seed_hi: u32,
	empty_value: f32,
	base_idx: u32,            // first (example, cluster) index of this dispatch chunk
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> packed_input: array<u32>;   // 2 u32 per u64 word, low word first
@group(0) @binding(2) var<storage, read> connections: array<i32>;    // neuron-major, -1 = unconnected
@group(0) @binding(3) var<storage, read> keys: array<vec2<u32>>;     // sorted per neuron, (lo, hi)
@group(0) @binding(4) var<storage, read> values: array<u32>;         // 4 cells per u32, byte i = cell i
@group(0) @binding(5) var<storage, read> offsets: array<u32>;
@group(0) @binding(6) var<storage, read> counts: array<u32>;
@group(0) @binding(7) var<storage, read_write> probs_out: array<f32>;

// ---- u64 as vec2<u32> --------------------------------------------------------

fn u64_shr(a: vec2<u32>, s: u32) -> vec2<u32> {
	if (s == 0u) { return a; }
	if (s >= 32u) { return vec2<u32>(a.y >> (s - 32u), 0u); }
	return vec2<u32>((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}

fn u64_bit(p: u32) -> vec2<u32> {
	if (p >= 32u) { return vec2<u32>(0u, 1u << (p - 32u)); }
	return vec2<u32>(1u << p, 0u);
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
	let lo = a.x + b.x;
	let carry = select(0u, 1u, lo < a.x);
	return vec2<u32>(lo, a.y + b.y + carry);
}

// Full 64-bit product of two u32 via 16-bit limbs (no mul_hi in WGSL).
fn mul32x32(a: u32, b: u32) -> vec2<u32> {
	let a0 = a & 0xffffu; let a1 = a >> 16u;
	let b0 = b & 0xffffu; let b1 = b >> 16u;
	let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
	let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
	let lo = (p00 & 0xffffu) | (mid << 16u);
	let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
	return vec2<u32>(lo, hi);
}

// Low 64 bits of a*b (wrapping), = Rust wrapping_mul / Metal ulong *.
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
	let ll = mul32x32(a.x, b.x);
	return vec2<u32>(ll.x, ll.y + a.x * b.y + a.y * b.x);
}

fn u64_lt(a: vec2<u32>, b: vec2<u32>) -> bool {
	return (a.y < b.y) || (a.y == b.y && a.x < b.x);
}

// splitmix64 finalizer — wnn_mix64 / neuron_memory::mix64.
fn mix64(v: vec2<u32>) -> vec2<u32> {
	var x = v;
	x = x ^ u64_shr(x, 30u);
	x = u64_mul(x, vec2<u32>(0x1ce4e5b9u, 0xbf58476du));
	x = x ^ u64_shr(x, 27u);
	x = u64_mul(x, vec2<u32>(0x133111ebu, 0x94d049bbu));
	x = x ^ u64_shr(x, 31u);
	return x;
}

// qsr_hash: the same finalizer preceded by + 0x9E3779B97F4A7C15.
fn qsr_hash(key: vec2<u32>) -> vec2<u32> {
	var x = u64_add(key, vec2<u32>(0x7f4a7c15u, 0x9e3779b9u));
	x = x ^ u64_shr(x, 30u);
	x = u64_mul(x, vec2<u32>(0x1ce4e5b9u, 0xbf58476du));
	x = x ^ u64_shr(x, 27u);
	x = u64_mul(x, vec2<u32>(0x133111ebu, 0x94d049bbu));
	x = x ^ u64_shr(x, 31u);
	return x;
}

// qsr_key(run_seed, neuron, address, example) — same odd multipliers as the twins.
fn qsr_key(seed: vec2<u32>, neuron: u32, address: vec2<u32>, example: u32) -> vec2<u32> {
	let e = u64_mul(vec2<u32>(example, 0u), vec2<u32>(0x7f4a7c15u, 0x9e3779b9u));
	let n = u64_mul(vec2<u32>(neuron, 0u), vec2<u32>(0x27d4eb4fu, 0xc2b2ae3du));
	let a = u64_mul(address, vec2<u32>(0x9e3779f9u, 0x165667b1u));
	return qsr_hash(seed ^ e ^ n ^ a);
}

// ---- addressing ---------------------------------------------------------------

fn input_bit(ex_base: u32, conn: i32) -> bool {
	if (conn < 0) { return false; }
	let ci = u32(conn);
	let w = packed_input[ex_base + (ci / 64u) * 2u + ((ci % 64u) / 32u)];
	return ((w >> (ci % 32u)) & 1u) != 0u;
}

// wnn_compute_address_u64: raw MSB-first integer up to 64 bits; above that the
// splitmix chain over 64-slot words (slot i at bit 63 - i%64), seeded with
// 0x9E3779B97F4A7C15 ^ bits.
fn compute_address(ex_base: u32, conn_base: u32, bits: u32) -> vec2<u32> {
	let wide = bits > 64u;
	var acc = vec2<u32>(0u, 0u);
	var word = vec2<u32>(0u, 0u);
	if (wide) { acc = vec2<u32>(0x7f4a7c15u ^ bits, 0x9e3779b9u); }
	for (var i = 0u; i < bits; i = i + 1u) {
		let bit = input_bit(ex_base, connections[conn_base + i]);
		if (!wide) {
			if (bit) { acc = acc | u64_bit(bits - 1u - i); }
		} else {
			if (bit) { word = word | u64_bit(63u - (i & 63u)); }
			if ((i & 63u) == 63u || i + 1u == bits) {
				acc = mix64(acc ^ word);
				word = vec2<u32>(0u, 0u);
			}
		}
	}
	return acc;
}

// ---- memory read -----------------------------------------------------------------

fn cell_at(idx: u32) -> u32 {
	return (values[idx / 4u] >> ((idx % 4u) * 8u)) & 0xffu;
}

// binary_search_lookup: a miss reads P.default_cell (coverage-aware → 0).
fn lookup(neuron: u32, address: vec2<u32>) -> u32 {
	let start = offsets[neuron];
	var left = 0u;
	var right = counts[neuron];
	while (left < right) {
		let mid = left + (right - left) / 2u;
		let k = keys[start + mid];
		if (all(k == address)) { return cell_at(start + mid); }
		if (u64_lt(k, address)) { left = mid + 1u; } else { right = mid; }
	}
	return P.default_cell;
}

// ---- cell → weight (wnn_cell_weight / wnn_cell_weight_rng) ----------------------------

fn quad_weight(cell: u32) -> f32 {
	if (cell == 0u) { return 0.0; }
	if (cell == 1u) { return 0.25; }
	if (cell == 2u) { return 0.75; }
	return 1.0;
}

fn coin(p: f32, rng: vec2<u32>) -> f32 {
	// top 24 bits of the mixed u64 → uniform in [0, 1); = (rng >> 40) / 2^24
	let u = f32(rng.y >> 8u) / 16777216.0;
	return select(0.0, 1.0, u < p);
}

fn cell_weight(cell: u32, mode: u32, empty_value: f32) -> f32 {
	if (mode == 0u) {                       // TERNARY
		if (cell == 0u) { return 0.0; }
		if (cell == 1u) { return 1.0; }
		return empty_value;
	}
	if (mode == 5u) {                       // PLN, deterministic expectation
		if (cell == 0u) { return 0.0; }
		if (cell == 1u) { return 1.0; }
		return 0.5;
	}
	if (mode == 3u) { return select(0.0, 1.0, cell == 1u); }   // BINARY
	if (mode == 1u) { return select(0.0, 1.0, cell >= 2u); }   // QUAD_BINARY
	return quad_weight(cell);               // QUAD_WEIGHTED / QSR expectation
}

fn cell_weight_rng(cell: u32, mode: u32, empty_value: f32, rng: vec2<u32>) -> f32 {
	if (mode == 4u) { return coin(quad_weight(cell), rng); }   // QSR
	if (mode == 5u) {                                         // PLN
		if (cell == 0u) { return 0.0; }
		if (cell == 1u) { return 1.0; }
		return coin(0.5, rng);
	}
	return cell_weight(cell, mode, empty_value);
}

// ---- kernel ------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn sparse_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
	let idx = P.base_idx + gid.x;
	if (idx >= P.num_examples * P.num_clusters) { return; }
	let ex = idx / P.num_clusters;
	let c = idx % P.num_clusters;
	let ex_base = ex * P.words_per_example * 2u;
	let seed = vec2<u32>(P.run_seed_lo, P.run_seed_hi);
	var sum = 0.0;
	for (var n = 0u; n < P.neurons_per_cluster; n = n + 1u) {
		let neuron = c * P.neurons_per_cluster + n;
		let address = compute_address(ex_base, neuron * P.bits_per_neuron, P.bits_per_neuron);
		let cell = lookup(neuron, address);
		let rng = qsr_key(seed, neuron, address, ex);
		sum = sum + cell_weight_rng(cell, P.memory_mode, P.empty_value, rng);
	}
	probs_out[idx] = sum / f32(P.neurons_per_cluster);
}
