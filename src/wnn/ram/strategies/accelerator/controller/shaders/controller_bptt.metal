// controller_bptt.metal — ABI 35: the GPU-resident sn>0 EDRA-BPTT window walk.
//
// Twin of WnnController::bptt_walk_cpu (controller/bptt_walk_cpu.rs) over the SAME
// prepared windows (bptt_plan.rs). ONE THREADGROUP = ONE GENOME'S WALK over a chunk of
// consecutive windows; records are strictly sequential (record d's commits are read by
// record d-1's solves), so all parallelism is WITHIN a record: phase-1 top-k over every
// neuron (lanes per neuron), the 4 motor beams + the state beam side by side, candidate
// ranking across threads, and one writer per neuron for the commits.
//
// Prepended at compile time: common.metal (WnnAddr), marker_slots.metal (slot_hash),
// controller_rollout.metal (ctrl_nudge_distance, ctrl_fire_bit, ctrl_candidate_rank).
// Cell semantics come ONLY from those; nothing here re-defines a cell weight.
//
// MEMORY: per (layer, neuron) region an append-only ENTRY list (ulong addr<<8 | cell)
// plus an open-addressing INDEX (uint entry id, 0xFFFFFFFF = empty slot). Single writer
// per region (the commit thread of that neuron), so no atomics. A cell written to the
// layer's canonical default is marked 2 (= absent: reads EMPTY, not "trained") instead
// of being removed — the exact read semantics of SparseLayerMemory's delete-on-default.
//
// EXACTNESS (no tolerance): phase-1 ranks are the integer 7*d + h (per-neuron saturation
// is a common shift, and the CPU's f64 sums of dyadic saturations are exact for
// n_bits <= 40, enforced by the host), beam ties break on (cost, parent, address-rank)
// exactly like the CPU's stable sort, and the final argmin takes the FIRST minimum.
// Only the sn state bits of either solve are mutable (the rest are immutable inputs), so
// a beam entry is two 64-bit masks (set, value) over those bits.

constant uint BW_NONE = 0xFFFFFFFFu;
constant uint BW_MAX_KTOP = 8u;
constant uint BW_MAX_BEAM = 64u;
constant uint BW_MAX_BEAMS = 5u; // <= 4 motors + the state solve
constant uint BW_MAX_CAND = 512u; // BW_MAX_BEAM * BW_MAX_KTOP
constant uint BW_REACH_SCAN = 1000000u;

struct BpttParams
{
	uint num_windows;
	uint mode;
	uint num_motors;
	uint levels;
	uint num_out;
	uint obpn;
	uint sn;
	uint sbpn;
	uint frame_bits;
	uint sensor_window;
	uint sil; // state-layer input bits = sensor_window + sn
	uint oil; // output-layer input bits = frame_bits + sn
	uint k_out;
	uint k_st;
	uint beam_out;
	uint beam_st;
	uint protect;
	uint has_override;
	uint default_cell;
	uint sens_words;
	uint frame_words;
	uint tgt_words;
	uint si_words;
	uint oi_words;
	uint threads;
};

inline bool bw_bit(device const uint *w, uint j)
{
	return ((w[j >> 5] >> (j & 31u)) & 1u) != 0u;
}

inline bool bw_quad(uint mode)
{
	return mode == 1u || mode == 2u || mode == 4u;
}

// cell_mode::nudge_cell twin.
inline uint bw_nudge_cell(uint cur, bool t, uint mode)
{
	if (bw_quad(mode))
	{
		uint c = cur & 3u;
		return t ? min(c + 1u, 3u) : (c == 0u ? 0u : c - 1u);
	}
	return t ? WNN_CELL_TRUE : WNN_CELL_FALSE;
}

// cell_mode::nudge_cell_value twin.
inline uint bw_nudge_value(uint cur, uint target, uint mode)
{
	if (bw_quad(mode))
	{
		uint c = cur & 3u, tv = target & 3u;
		return c < tv ? c + 1u : (c > tv ? c - 1u : c);
	}
	return target;
}

inline uint bw_true_cell(uint mode)
{
	return bw_quad(mode) ? 3u : WNN_CELL_TRUE;
}

// Index probe: entry id of `addr` in the region, or BW_NONE.
inline uint bw_find(device const uint *idx, device const ulong *ent, uint4 meta, ulong addr)
{
	uint mask = meta.w - 1u;
	uint i = slot_hash(addr, mask);
	for (uint p = 0u; p < meta.w; ++p)
	{
		uint e = idx[meta.z + i];
		if (e == BW_NONE)
			return BW_NONE;
		if ((ent[meta.x + e] >> 8) == addr)
			return e;
		i = (i + 1u) & mask;
	}
	return BW_NONE;
}

inline uint bw_read(device const uint *idx, device const ulong *ent, uint4 meta, ulong addr)
{
	uint e = bw_find(idx, ent, meta, addr);
	return e == BW_NONE ? WNN_CELL_EMPTY : (uint)(ent[meta.x + e] & 0xFFul);
}

// SparseLayerMemory::write_cell(.., allow_override=true) twin, single writer.
inline void bw_write(device uint *idx, device ulong *ent, device uint *cnt, uint4 meta, ulong addr,
										 uint v, uint default_cell, device atomic_uint *overflow)
{
	uint mask = meta.w - 1u;
	uint i = slot_hash(addr, mask);
	for (uint p = 0u; p < meta.w; ++p)
	{
		uint e = idx[meta.z + i];
		if (e == BW_NONE)
			break;
		if ((ent[meta.x + e] >> 8) == addr)
		{
			uint stored = (v == default_cell) ? WNN_CELL_EMPTY : v;
			ent[meta.x + e] = (addr << 8) | (ulong)stored;
			return;
		}
		i = (i + 1u) & mask;
	}
	if (v == default_cell)
		return; // vacant + default: nothing to store
	uint n = *cnt;
	if (n >= meta.y)
	{
		atomic_store_explicit(overflow, 1u, memory_order_relaxed);
		return;
	}
	ent[meta.x + n] = (addr << 8) | (ulong)v;
	idx[meta.z + i] = n;
	*cnt = n + 1u;
}

inline ulong bw_address(device const uint *in, device const int *conn, uint bits)
{
	WnnAddr a = wnn_addr_begin(bits);
	for (uint k = 0u; k < bits; ++k)
		wnn_addr_push(a, k, bw_bit(in, (uint)conn[k]));
	return wnn_addr_finish(a);
}

inline ulong bw_project(device const uint *in, device const int *conn, uint bits)
{
	ulong p = 0ul;
	for (uint k = 0u; k < bits; ++k)
		if (bw_bit(in, (uint)conn[k]))
			p |= (1ul << (ulong)(bits - 1u - k));
	return p;
}

inline void bw_topk_insert(thread uint *r, thread ulong *a, thread uint &cnt, uint k, uint rank, ulong addr)
{
	if (cnt == k && (rank > r[cnt - 1u] || (rank == r[cnt - 1u] && addr >= a[cnt - 1u])))
		return;
	uint pos = cnt < k ? cnt : k - 1u;
	while (pos > 0u && (r[pos - 1u] > rank || (r[pos - 1u] == rank && a[pos - 1u] > addr)))
	{
		r[pos] = r[pos - 1u];
		a[pos] = a[pos - 1u];
		pos -= 1u;
	}
	r[pos] = rank;
	a[pos] = addr;
	if (cnt < k)
		cnt += 1u;
}

// One layer's phase 1 (reachable top-k) for `ntask` neurons with ALL threads
// cooperating: L lanes per neuron scan its entry list (local top-k each), lane 0 merges
// and runs the untrained Hamming hunt. Task j = layer neuron n0+j, result slot tk0+j.
// Twin of controller_training::reachable_topk_core (phase (a) + (b), same scan cap).
inline void bw_phase1(
		device const ulong *ent, device const uint *idx, device const uint4 *meta, device const uint *cnt,
		device const int *conns, uint n_bits, uint ntask, uint tk0,
		device const uint *input, device const uchar *task_tgt,
		uint k, uint mode, uint tid, uint T,
		device ulong *ln_addr, device uint *ln_rank, device uint *ln_cnt,
		device ulong *tk_addr, device uint *tk_rank, device uint *tk_cnt)
{
	uint L = max(1u, T / max(ntask, 1u));
	uint per_pass = T / L;
	for (uint base = 0u; base < ntask; base += per_pass)
	{
		uint j = base + tid / L;
		uint lane = tid % L;
		bool active = (tid / L) < per_pass && j < ntask;
		uint4 m = active ? meta[j] : uint4(0u);
		ulong proj = 0ul;
		bool target_true = false;
		if (active)
		{
			device const int *conn = conns + j * n_bits;
			proj = bw_project(input, conn, n_bits);
			target_true = task_tgt[tk0 + j] != 0u;
			thread uint r[BW_MAX_KTOP];
			thread ulong a[BW_MAX_KTOP];
			uint c = 0u;
			uint ne = cnt[j];
			for (uint e = lane; e < ne; e += L)
			{
				ulong v = ent[m.x + e];
				uint cell = (uint)(v & 0xFFul);
				if (cell == WNN_CELL_EMPTY)
					continue; // deleted (canonical default) = absent
				ulong addr = v >> 8;
				bw_topk_insert(r, a, c, k, ctrl_candidate_rank(cell, target_true, mode, addr, proj), addr);
			}
			for (uint i = 0u; i < c; ++i)
			{
				ln_addr[tid * BW_MAX_KTOP + i] = a[i];
				ln_rank[tid * BW_MAX_KTOP + i] = r[i];
			}
			ln_cnt[tid] = c;
		}
		threadgroup_barrier(mem_flags::mem_device);
		if (active && lane == 0u)
		{
			thread uint r[BW_MAX_KTOP];
			thread ulong a[BW_MAX_KTOP];
			uint c = 0u;
			for (uint l = 0u; l < L; ++l)
			{
				uint src = tid + l;
				for (uint i = 0u; i < ln_cnt[src]; ++i)
					bw_topk_insert(r, a, c, k, ln_rank[src * BW_MAX_KTOP + i], ln_addr[src * BW_MAX_KTOP + i]);
			}
			// (b) untrained cells at ascending Hamming radius (value EMPTY).
			uint def_base = ctrl_nudge_distance(WNN_CELL_EMPTY, target_true, mode) * 7u;
			uint collected = 0u, examined = 0u;
			bool stop = false;
			for (uint h = 0u; h <= n_bits && !stop; ++h)
			{
				thread uint combo[64];
				for (uint q = 0u; q < h; ++q)
					combo[q] = q;
				bool more = true;
				while (more)
				{
					ulong addr = proj;
					for (uint q = 0u; q < h; ++q)
						addr ^= (1ul << (ulong)(n_bits - 1u - combo[q]));
					uint e = bw_find(idx, ent, m, addr);
					bool trained = e != BW_NONE && (uint)(ent[m.x + e] & 0xFFul) != WNN_CELL_EMPTY;
					if (!trained)
					{
						bw_topk_insert(r, a, c, k, def_base + h, addr);
						collected += 1u;
					}
					examined += 1u;
					if (examined >= BW_REACH_SCAN)
					{
						stop = true;
						break;
					}
					if (h == 0u)
						break;
					uint i = h;
					more = false;
					while (i > 0u)
					{
						i -= 1u;
						if (combo[i] != i + n_bits - h)
						{
							combo[i] += 1u;
							for (uint q = i + 1u; q < h; ++q)
								combo[q] = combo[q - 1u] + 1u;
							more = true;
							break;
						}
					}
				}
				if (collected >= k)
					break;
			}
			tk_cnt[tk0 + j] = c;
			for (uint i = 0u; i < c; ++i)
			{
				tk_addr[(tk0 + j) * BW_MAX_KTOP + i] = a[i];
				tk_rank[(tk0 + j) * BW_MAX_KTOP + i] = r[i];
			}
		}
		threadgroup_barrier(mem_flags::mem_device);
	}
}

// Per (task, candidate) constraint summary over the solve's MUTABLE bits
// [n_imm, n_imm+sn): which positions the address demands 1 / 0, the value it leaves
// (last connection wins, like the CPU's materialise loop), and whether it agrees with
// the immutable input everywhere (otherwise every parent rejects it).
inline void bw_cand_info(device const int *conn, uint n_bits, ulong addr, device const uint *input,
												 uint n_imm, device ulong *out3, device uint *ok)
{
	ulong req0 = 0ul, req1 = 0ul, cval = 0ul;
	bool imm_ok = true;
	for (uint k = 0u; k < n_bits; ++k)
	{
		uint pos = (uint)conn[k];
		bool b = ((addr >> (ulong)(n_bits - 1u - k)) & 1ul) != 0ul;
		if (pos < n_imm)
		{
			if (bw_bit(input, pos) != b)
				imm_ok = false;
		}
		else
		{
			ulong bit = 1ul << (ulong)(pos - n_imm);
			if (b)
			{
				req1 |= bit;
				cval |= bit;
			}
			else
			{
				req0 |= bit;
				cval &= ~bit;
			}
		}
	}
	out3[0] = req0;
	out3[1] = req1;
	out3[2] = cval;
	*ok = imm_ok ? 1u : 0u;
}

kernel void controller_bptt_walk(
		device ulong *o_ent [[buffer(0)]],
		device uint *o_idx [[buffer(1)]],
		device const uint4 *o_meta [[buffer(2)]], // (ent_off, ent_cap, idx_off, idx_cap)
		device uint *o_cnt [[buffer(3)]],
		device ulong *s_ent [[buffer(4)]],
		device uint *s_idx [[buffer(5)]],
		device const uint4 *s_meta [[buffer(6)]],
		device uint *s_cnt [[buffer(7)]],
		device const int *o_conn [[buffer(8)]],
		device const int *s_conn [[buffer(9)]],
		device const uint *win_meta [[buffer(10)]], // [num_windows*3] (rec_start, n_rec, reset)
		device const uint *sens [[buffer(11)]],
		device const uint *frame [[buffer(12)]],
		device const uint *tgt [[buffer(13)]],
		device const uchar *sover [[buffer(14)]],
		device uint *prev_state [[buffer(15)]],
		device uint *si [[buffer(16)]],
		device uint *oi [[buffer(17)]],
		device ulong *tk_addr [[buffer(18)]],
		device uint *tk_rank [[buffer(19)]],
		device uint *tk_cnt [[buffer(20)]],
		device ulong *ln_addr [[buffer(21)]],
		device uint *ln_rank [[buffer(22)]],
		device uint *ln_cnt [[buffer(23)]],
		device ulong *ci_bits [[buffer(24)]],
		device uint *ci_ok [[buffer(25)]],
		device atomic_uint *stats [[buffer(26)]], // s_writes, o_writes, overflow
		constant BpttParams &P [[buffer(27)]],
		device uchar *task_tgt [[buffer(28)]], // [num_out + sn] this record's solve targets
		uint tid [[thread_index_in_threadgroup]])
{
	const uint T = P.threads;
	const uint mode = P.mode;
	const uint sn = P.sn;
	const uint nm = P.num_motors;
	const uint ntask = P.num_out + sn;

	threadgroup uint b_cost[2][BW_MAX_BEAMS][BW_MAX_BEAM];
	threadgroup ulong b_mask[2][BW_MAX_BEAMS][BW_MAX_BEAM];
	threadgroup ulong b_val[2][BW_MAX_BEAMS][BW_MAX_BEAM];
	threadgroup uint b_n[BW_MAX_BEAMS];
	threadgroup uint b_ok[BW_MAX_BEAMS];
	threadgroup uint b_cur;
	threadgroup atomic_uint b_valid[BW_MAX_BEAMS];
	threadgroup uint c_cost[BW_MAX_BEAMS][BW_MAX_CAND];
	threadgroup ulong sol_bits[BW_MAX_BEAMS];
	threadgroup uint sol_ok[BW_MAX_BEAMS];
	threadgroup uchar ds[64];
	threadgroup uint new_state[64];
	threadgroup atomic_uint w_s, w_o;

	if (tid == 0u)
	{
		atomic_store_explicit(&w_s, 0u, memory_order_relaxed);
		atomic_store_explicit(&w_o, 0u, memory_order_relaxed);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);

	const uint nbeams = nm + 1u;
	const uint TB = T / nbeams; // threads per beam
	const uint my_beam = TB > 0u ? tid / TB : 0u;
	const uint my_lane = TB > 0u ? tid % TB : 0u;

	for (uint w = 0u; w < P.num_windows; ++w)
	{
		const uint rec0 = win_meta[w * 3u];
		const uint n_rec = win_meta[w * 3u + 1u];
		if (win_meta[w * 3u + 2u] != 0u)
		{
			for (uint n = tid; n < sn; n += T)
				prev_state[n] = 0u;
		}
		threadgroup_barrier(mem_flags::mem_device);

		// ---- forward state roll (memory as of the window start) ----
		for (uint d = 0u; d < n_rec; ++d)
		{
			device uint *sid = si + d * P.si_words;
			device uint *oid = oi + d * P.oi_words;
			device const uint *sd = sens + (rec0 + d) * P.sens_words;
			device const uint *fd = frame + (rec0 + d) * P.frame_words;
			for (uint q = tid; q < P.si_words; q += T)
				sid[q] = q < P.sens_words ? sd[q] : 0u;
			for (uint q = tid; q < P.oi_words; q += T)
				oid[q] = q < P.frame_words ? fd[q] : 0u;
			threadgroup_barrier(mem_flags::mem_device);
			if (tid == 0u)
			{
				for (uint n = 0u; n < sn; ++n)
					if (ctrl_fire_bit(prev_state[n], mode))
					{
						uint j = P.sensor_window + n;
						sid[j >> 5] |= (1u << (j & 31u));
					}
			}
			threadgroup_barrier(mem_flags::mem_device);
			for (uint n = tid; n < sn; n += T)
				new_state[n] = bw_read(s_idx, s_ent, s_meta[n], bw_address(sid, s_conn + n * P.sbpn, P.sbpn));
			threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
			if (tid == 0u)
			{
				for (uint n = 0u; n < sn; ++n)
				{
					if (ctrl_fire_bit(new_state[n], mode))
					{
						uint j = P.frame_bits + n;
						oid[j >> 5] |= (1u << (j & 31u));
					}
					prev_state[n] = new_state[n];
				}
			}
			threadgroup_barrier(mem_flags::mem_device);
		}

		// ---- backward walk ----
		for (uint dd = n_rec; dd > 0u; --dd)
		{
			const uint d = dd - 1u;
			const bool do_state = (d + 1u < n_rec);
			device const uint *oid = oi + d * P.oi_words;
			device const uint *sid = si + d * P.si_words;
			device const uint *sid1 = si + (d + 1u) * P.si_words;
			device const uint *td = tgt + (rec0 + d) * P.tgt_words;
			for (uint n = tid; n < ntask; n += T)
				task_tgt[n] = n < P.num_out ? (bw_bit(td, n) ? 1u : 0u) : ds[n - P.num_out];
			threadgroup_barrier(mem_flags::mem_device);

			// phase 1: output neurons on oi[d]; state neurons on si[d+1] (if solved)
			bw_phase1(o_ent, o_idx, o_meta, o_cnt, o_conn, P.obpn, P.num_out, 0u, oid, task_tgt,
								P.k_out, mode, tid, T, ln_addr, ln_rank, ln_cnt, tk_addr, tk_rank, tk_cnt);
			if (do_state)
				bw_phase1(s_ent, s_idx, s_meta, s_cnt, s_conn, P.sbpn, sn, P.num_out, sid1, task_tgt,
									P.k_st, mode, tid, T, ln_addr, ln_rank, ln_cnt, tk_addr, tk_rank, tk_cnt);

			// candidate constraint summaries
			for (uint q = tid; q < ntask * BW_MAX_KTOP; q += T)
			{
				uint task = q / BW_MAX_KTOP, ai = q % BW_MAX_KTOP;
				bool is_out = task < P.num_out;
				if (!is_out && !do_state)
					continue;
				if (ai >= tk_cnt[task])
					continue;
				device const int *conn = is_out ? (o_conn + task * P.obpn) : (s_conn + (task - P.num_out) * P.sbpn);
				bw_cand_info(conn, is_out ? P.obpn : P.sbpn, tk_addr[q], is_out ? oid : sid1,
										 is_out ? P.frame_bits : P.sensor_window, ci_bits + q * 3u, ci_ok + q);
			}
			if (tid < nbeams)
			{
				bool live = tid < nm || do_state;
				b_n[tid] = 1u;
				b_ok[tid] = live ? 1u : 0u;
				b_cost[0][tid][0] = 0u;
				b_mask[0][tid][0] = 0ul;
				b_val[0][tid][0] = 0ul;
				atomic_store_explicit(&b_valid[tid], 0u, memory_order_relaxed);
			}
			if (tid == 0u)
				b_cur = 0u;
			threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

			// phase 2: the beams, side by side (motor m = beam m; state = beam nm)
			const uint max_depth = max(P.levels, do_state ? sn : 0u);
			for (uint s = 0u; s < max_depth; ++s)
			{
				const uint cur = b_cur;
				const uint b = my_beam;
				const bool is_state = (b == nm);
				const uint depth = is_state ? sn : P.levels;
				const bool run = b < nbeams && s < depth && b_ok[b] != 0u;
				const uint task = is_state ? (P.num_out + s) : (b * P.levels + s);
				const uint kc = run ? tk_cnt[task] : 0u;
				const uint C = run ? b_n[b] * kc : 0u;
				// A: score + conflict-check every (parent, address) pair
				for (uint c = my_lane; c < C; c += TB)
				{
					uint pi = c / kc, ai = c % kc;
					uint q = task * BW_MAX_KTOP + ai;
					ulong pm = b_mask[cur][b][pi], pv = b_val[cur][b][pi];
					ulong r0 = ci_bits[q * 3u], r1 = ci_bits[q * 3u + 1u];
					bool ok = ci_ok[q] != 0u && ((pm & pv & r0) | (pm & ~pv & r1)) == 0ul;
					c_cost[b][c] = ok ? b_cost[cur][b][pi] + tk_rank[q] : BW_NONE;
					if (ok)
						atomic_fetch_add_explicit(&b_valid[b], 1u, memory_order_relaxed);
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
				// B: stable rank by (cost, c); keep the first beam_width
				const uint W = is_state ? P.beam_st : P.beam_out;
				for (uint c = my_lane; c < C; c += TB)
				{
					uint cc = c_cost[b][c];
					if (cc == BW_NONE)
						continue;
					uint pos = 0u;
					for (uint o = 0u; o < C && pos < W; ++o)
					{
						uint oc = c_cost[b][o];
						if (oc < cc || (oc == cc && o < c))
							pos += 1u;
					}
					if (pos < W)
					{
						uint pi = c / kc, ai = c % kc;
						uint q = task * BW_MAX_KTOP + ai;
						ulong cm = ci_bits[q * 3u] | ci_bits[q * 3u + 1u];
						b_cost[1u - cur][b][pos] = cc;
						b_mask[1u - cur][b][pos] = b_mask[cur][b][pi] | cm;
						b_val[1u - cur][b][pos] = (b_val[cur][b][pi] & ~cm) | ci_bits[q * 3u + 2u];
					}
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
				// C: advance
				if (tid < nbeams)
				{
					bool r = tid < nbeams && s < (tid == nm ? sn : P.levels) && b_ok[tid] != 0u;
					if (r)
					{
						uint nv = atomic_load_explicit(&b_valid[tid], memory_order_relaxed);
						if (nv == 0u)
							b_ok[tid] = 0u;
						else
							b_n[tid] = min(nv, tid == nm ? P.beam_st : P.beam_out);
					}
					atomic_store_explicit(&b_valid[tid], 0u, memory_order_relaxed);
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
				if (tid == 0u)
				{
					// Beams that did not run this step keep their buffer: copy cur → next so
					// the swap is uniform (only beams past their depth or dead are affected).
					for (uint bb = 0u; bb < nbeams; ++bb)
					{
						bool r = s < (bb == nm ? sn : P.levels) && b_ok[bb] != 0u;
						if (!r)
						{
							for (uint i = 0u; i < b_n[bb]; ++i)
							{
								b_cost[1u - cur][bb][i] = b_cost[cur][bb][i];
								b_mask[1u - cur][bb][i] = b_mask[cur][bb][i];
								b_val[1u - cur][bb][i] = b_val[cur][bb][i];
							}
						}
					}
					b_cur = 1u - cur;
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
			}

			// argmin (FIRST minimum) per beam → the solve's mutable bits
			if (tid < nbeams)
			{
				uint cur = b_cur;
				sol_ok[tid] = b_ok[tid];
				if (b_ok[tid] != 0u)
				{
					uint best = 0u;
					for (uint i = 1u; i < b_n[tid]; ++i)
						if (b_cost[cur][tid][i] < b_cost[cur][tid][best])
							best = i;
					ulong m = b_mask[cur][tid][best], v = b_val[cur][tid][best];
					device const uint *inp = (tid == nm) ? sid1 : oid;
					uint n_imm = (tid == nm) ? P.sensor_window : P.frame_bits;
					ulong out = 0ul;
					for (uint i = 0u; i < sn; ++i)
					{
						bool bit = ((m >> i) & 1ul) != 0ul ? ((v >> i) & 1ul) != 0ul : bw_bit(inp, n_imm + i);
						if (bit)
							out |= (1ul << i);
					}
					sol_bits[tid] = out;
				}
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			// vote + transition aggregate → ds
			if (tid == 0u)
			{
				for (uint i = 0u; i < sn; ++i)
				{
					int vote = 0;
					for (uint mm = 0u; mm < nm; ++mm)
						if (sol_ok[mm] != 0u)
							vote += ((sol_bits[mm] >> i) & 1ul) != 0ul ? 1 : -1;
					bool d_out = vote > 0 ? true : (vote < 0 ? false : bw_bit(oid, P.frame_bits + i));
					bool d_s = d_out;
					if (do_state)
					{
						bool d_trans = sol_ok[nm] != 0u ? ((sol_bits[nm] >> i) & 1ul) != 0ul : d_out;
						d_s = (d_out == d_trans) ? d_out : bw_bit(sid, P.sensor_window + i);
					}
					ds[i] = d_s ? 1u : 0u;
				}
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);

			// (c) state + (d) output commits — one writer per neuron
			for (uint n = tid; n < ntask; n += T)
			{
				if (n < P.num_out)
				{
					bool t = bw_bit(td, n);
					ulong addr = bw_address(oid, o_conn + n * P.obpn, P.obpn);
					uint cur = bw_read(o_idx, o_ent, o_meta[n], addr);
					if (P.protect != 0u && cur != WNN_CELL_EMPTY && ctrl_fire_bit(cur, mode) != t)
						continue;
					uint nv = bw_nudge_cell(cur, t, mode);
					if (nv != cur)
					{
						bw_write(o_idx, o_ent, o_cnt + n, o_meta[n], addr, nv, P.default_cell, &stats[2]);
						atomic_fetch_add_explicit(&w_o, 1u, memory_order_relaxed);
					}
				}
				else
				{
					uint sn_i = n - P.num_out;
					uint ov = P.has_override != 0u ? (uint)sover[(rec0 + d) * sn + sn_i] : 255u;
					uint tv = ov != 255u ? ov : (ds[sn_i] != 0u ? bw_true_cell(mode) : WNN_CELL_FALSE);
					ulong addr = bw_address(sid, s_conn + sn_i * P.sbpn, P.sbpn);
					uint cur = bw_read(s_idx, s_ent, s_meta[sn_i], addr);
					if (P.protect != 0u && cur != WNN_CELL_EMPTY && ctrl_fire_bit(cur, mode) != ctrl_fire_bit(tv, mode))
						continue;
					uint nv = bw_nudge_value(cur, tv, mode);
					if (nv != cur)
					{
						bw_write(s_idx, s_ent, s_cnt + sn_i, s_meta[sn_i], addr, nv, P.default_cell, &stats[2]);
						atomic_fetch_add_explicit(&w_s, 1u, memory_order_relaxed);
					}
				}
			}
			threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
		}
	}
	if (tid == 0u)
	{
		atomic_fetch_add_explicit(&stats[0], atomic_load_explicit(&w_s, memory_order_relaxed), memory_order_relaxed);
		atomic_fetch_add_explicit(&stats[1], atomic_load_explicit(&w_o, memory_order_relaxed), memory_order_relaxed);
	}
}
