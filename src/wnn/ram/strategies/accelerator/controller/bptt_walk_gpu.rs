//! ABI 35 (A): host side of the GPU-resident sn>0 bptt walk (shaders/controller_bptt.metal).
//!
//! One call walks a LIST of prepared windows (a genome's whole gated round, in order)
//! on the GPU: both layers are uploaded ONCE as writable tables, the windows run in
//! chunks of whole windows (each chunk = one dispatch of one threadgroup, bounded so a
//! dispatch stays far below the GPU watchdog and the cancel flag is polled between
//! chunks), and the cells come back ONCE and are applied to the controller's memory.
//! Bit-exact against `bptt_walk_cpu` over the same plans — the acceptance gate is
//! `abi35_tests`.
//!
//! Opt-in (WNN_CONTROLLER_GPU_BPTT=1, or `--gpu-bptt` in phased_ga): the IDS worker
//! shares the Metal GPU.

use ram_core::cell_mode::{CellMode, UnwrittenLean};
use metal::*;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::bptt_plan::{BpttWindowPlan, NO_STATE_OVERRIDE};
use crate::controller::WnnController;
use crate::metal_controller::host_slot_hash;

const NONE_U32: u32 = 0xFFFF_FFFF;
const MAX_KTOP: usize = 8;
const MAX_MOTORS: usize = 4;
const MAX_SN: usize = 64;
/// f64 sums of dyadic saturations stay exact (=> integer ranks order identically).
const MAX_SOLVE_BITS: usize = 40;
const EMPTY_CELL: u8 = 2;
const QUEUES: usize = 16;

/// Records per dispatch (whole windows; at least one window). Bounds a dispatch's
/// wall time (GPU watchdog, cancel latency, the IDS worker's share of the GPU).
fn chunk_records() -> usize
{
	std::env::var("WNN_CONTROLLER_BPTT_CHUNK_RECORDS")
		.ok()
		.and_then(|v| v.parse::<usize>().ok())
		.filter(|&v| v > 0)
		.unwrap_or(256)
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BpttParams
{
	num_windows: u32,
	mode: u32,
	num_motors: u32,
	levels: u32,
	num_out: u32,
	obpn: u32,
	sn: u32,
	sbpn: u32,
	frame_bits: u32,
	sensor_window: u32,
	sil: u32,
	oil: u32,
	k_out: u32,
	k_st: u32,
	beam_out: u32,
	beam_st: u32,
	protect: u32,
	has_override: u32,
	default_cell: u32,
	sens_words: u32,
	frame_words: u32,
	tgt_words: u32,
	si_words: u32,
	oi_words: u32,
	threads: u32,
	/// CTRL-46: 1 ⇒ vote round (no writes; visits to buffers 29/30).
	vote: u32,
}

/// The compiled walk kernel + a small pool of command queues (genomes walk
/// concurrently from rayon threads; separate queues let their dispatches overlap).
pub(crate) struct BpttWalker
{
	device: Device,
	pipeline: ComputePipelineState,
	queues: Vec<CommandQueue>,
	rr: AtomicUsize,
	threads: u64,
}

static WALKER: OnceLock<Result<BpttWalker, String>> = OnceLock::new();

pub(crate) fn bptt_walker() -> Result<&'static BpttWalker, String>
{
	WALKER.get_or_init(BpttWalker::new).as_ref().map_err(|e| e.clone())
}

impl BpttWalker
{
	fn new() -> Result<Self, String>
	{
		let device = Device::system_default().ok_or("No Metal device found")?;
		let src = concat!(
			include_str!("../core/shaders/common.metal"),
			"\n",
			include_str!("../core/shaders/marker_slots.metal"),
			"\n",
			include_str!("shaders/controller_rollout.metal"),
			"\n",
			include_str!("shaders/controller_bptt.metal"),
		);
		let library = device
			.new_library_with_source(src, &CompileOptions::new())
			.map_err(|e| format!("controller_bptt shader compile failed: {e}"))?;
		let func = library
			.get_function("controller_bptt_walk", Some(FunctionConstantValues::new()))
			.map_err(|e| format!("kernel controller_bptt_walk not found: {e}"))?;
		let pipeline = device
			.new_compute_pipeline_state_with_function(&func)
			.map_err(|e| format!("controller_bptt_walk pipeline creation failed: {e}"))?;
		let threads = (pipeline.max_total_threads_per_threadgroup().min(512) / 32) * 32;
		if threads < 64
		{
			return Err(format!("controller_bptt_walk: only {threads} threads per threadgroup"));
		}
		let queues = (0..QUEUES).map(|_| device.new_command_queue()).collect();
		Ok(BpttWalker { device, pipeline, queues, rr: AtomicUsize::new(0), threads })
	}

	fn buf<T>(&self, data: &[T]) -> Buffer
	{
		let bytes = (data.len().max(1) * mem::size_of::<T>()) as u64;
		if data.is_empty()
		{
			self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared)
		}
		else
		{
			self.device.new_buffer_with_data(data.as_ptr() as *const _, bytes, MTLResourceOptions::StorageModeShared)
		}
	}

	fn zeroed<T: Copy + Default>(&self, n: usize) -> Buffer
	{
		self.buf(&vec![T::default(); n.max(1)])
	}
}

/// One layer's writable table: per neuron an append-only entry list (addr<<8 | cell,
/// cell 2 = deleted/absent) + an open-addressing index (entry id, NONE = empty).
/// Tightly coupled to the kernel layout; host-side build, readback and growth.
struct LayerTable
{
	ent: Vec<u64>,
	idx: Vec<u32>,
	meta: Vec<[u32; 4]>, // (ent_off, ent_cap, idx_off, idx_cap)
	cnt: Vec<u32>,
}

impl LayerTable
{
	fn build(cells: &[Vec<(u64, u8)>], headroom: usize) -> Self
	{
		let mut t = LayerTable { ent: Vec::new(), idx: Vec::new(), meta: Vec::new(), cnt: Vec::new() };
		for e in cells
		{
			let ent_cap = (e.len() + headroom).max(16).next_power_of_two();
			let idx_cap = ent_cap * 2;
			let (eo, io) = (t.ent.len(), t.idx.len());
			t.meta.push([eo as u32, ent_cap as u32, io as u32, idx_cap as u32]);
			t.ent.resize(eo + ent_cap, 0);
			t.idx.resize(io + idx_cap, NONE_U32);
			let mask = (idx_cap - 1) as u64;
			for (i, &(a, v)) in e.iter().enumerate()
			{
				t.ent[eo + i] = (a << 8) | v as u64;
				let mut s = host_slot_hash(a, mask);
				while t.idx[io + s] != NONE_U32
				{
					s = (s + 1) & (idx_cap - 1);
				}
				t.idx[io + s] = i as u32;
			}
			t.cnt.push(e.len() as u32);
		}
		t
	}
}

/// GPU buffers of one LayerTable.
struct LayerBufs
{
	ent: Buffer,
	idx: Buffer,
	meta: Buffer,
	cnt: Buffer,
	caps: Vec<u32>,
}

impl LayerBufs
{
	fn upload(w: &BpttWalker, t: &LayerTable) -> Self
	{
		LayerBufs {
			ent: w.buf(&t.ent),
			idx: w.buf(&t.idx),
			meta: w.buf(&t.meta),
			cnt: w.buf(&t.cnt),
			caps: t.meta.iter().map(|m| m[1]).collect(),
		}
	}

	fn counts(&self) -> Vec<u32>
	{
		let n = self.caps.len();
		unsafe { std::slice::from_raw_parts(self.cnt.contents() as *const u32, n) }.to_vec()
	}

	/// Per neuron, every entry the table holds (deleted ones as cell 2).
	fn cells(&self, meta: &[[u32; 4]]) -> Vec<Vec<(u64, u8)>>
	{
		let cnt = self.counts();
		let ent_len = meta.last().map(|m| (m[0] + m[1]) as usize).unwrap_or(0);
		let ent = unsafe { std::slice::from_raw_parts(self.ent.contents() as *const u64, ent_len.max(1)) };
		meta
			.iter()
			.zip(cnt.iter())
			.map(|(m, &c)| {
				(0..c as usize)
					.map(|i| {
						let v = ent[m[0] as usize + i];
						(v >> 8, (v & 0xFF) as u8)
					})
					.collect()
			})
			.collect()
	}
}

/// Refuse (loudly, with the reason) a controller the kernel cannot walk exactly.
fn check_supported(c: &WnnController, topk: usize) -> Result<(), String>
{
	let (nm, _levels, obpn, sn, sbpn, _mode, full) = c.bptt_dims();
	if sn == 0
	{
		return Err("GPU bptt walk is for sn>0 (sn=0 is the CPU direct-write fast path)".into());
	}
	if sn > MAX_SN
	{
		return Err(format!("GPU bptt walk supports <= {MAX_SN} state neurons, got {sn}"));
	}
	if nm > MAX_MOTORS
	{
		return Err(format!("GPU bptt walk supports <= {MAX_MOTORS} output banks, got {nm}"));
	}
	if full
	{
		return Err("GPU bptt walk: output_full_window is sn=0 only".into());
	}
	if obpn == 0 || sbpn == 0 || obpn > MAX_SOLVE_BITS || sbpn > MAX_SOLVE_BITS
	{
		return Err(format!(
			"GPU bptt walk supports 1..={MAX_SOLVE_BITS} bits per neuron (exact integer ranking), got out {obpn} / state {sbpn}"
		));
	}
	if topk == 0 || topk > MAX_KTOP
	{
		return Err(format!("GPU bptt walk supports topk_per_neuron 1..={MAX_KTOP}, got {topk}"));
	}
	Ok(())
}

fn pack_bits(dst: &mut Vec<u32>, bits: &[bool], words: usize)
{
	let base = dst.len();
	dst.resize(base + words, 0);
	for (j, &b) in bits.iter().enumerate()
	{
		if b
		{
			dst[base + (j >> 5)] |= 1u32 << (j & 31);
		}
	}
}

fn words(bits: usize) -> usize
{
	bits.div_ceil(32).max(1)
}

/// CTRL-46: fold one vote chunk's visits into the round's tally. Both walks visit the
/// same (record, task) set — every record × every task, exactly what the direct-write
/// commits touch — with the memory frozen, so the solves (and hence the state
/// targets) match record for record. Integer sums ⇒ chunk order is irrelevant.
fn add_chunk_visits(
	c: &mut WnnController,
	plans: &[BpttWindowPlan],
	b_va: &Buffer,
	b_vd: &Buffer,
	num_out: usize,
	ntask: usize,
	recs: usize,
)
{
	let addr = unsafe { std::slice::from_raw_parts(b_va.contents() as *const u64, recs * ntask) };
	let dir = unsafe { std::slice::from_raw_parts(b_vd.contents() as *const u8, recs * ntask) };
	let walked: usize = plans.iter().map(|p| p.n_rec).sum();
	debug_assert_eq!(walked, recs);
	let t = c.vote_tally_mut().expect("vote round open for the whole GPU walk");
	for slot in 0..recs * ntask
	{
		let n = slot % ntask;
		match dir[slot]
		{
			255 => {}
			d if n < num_out => t.visit(false, n, addr[slot], d != 0),
			d => t.visit(true, n - num_out, addr[slot], d != 0),
		}
	}
}

/// Walk `plans` (in order) on the GPU and apply the result to `c`. Returns
/// (state writes, output writes) — the same counts the CPU walk reports.
pub(crate) fn walk_plans_gpu(
	c: &mut WnnController,
	plans: &[BpttWindowPlan],
	topk: usize,
	protect_learned: bool,
) -> Result<(usize, usize), String>
{
	walk_plans_gpu_with(c, plans, topk, protect_learned, chunk_records())
}

/// `walk_plans_gpu` with an explicit records-per-dispatch budget (tests exercise
/// one-window chunks and table growth without touching the process env).
pub(crate) fn walk_plans_gpu_with(
	c: &mut WnnController,
	plans: &[BpttWindowPlan],
	topk: usize,
	protect_learned: bool,
	budget: usize,
) -> Result<(usize, usize), String>
{
	// CTRL-46: a vote round's tally is restored on failure, so the caller's CPU
	// fallback re-walks the same plans without double-counting finished chunks.
	let pre = c.vote_tally_mut().map(|t| t.clone());
	let r = walk_plans_gpu_inner(c, plans, topk, protect_learned, budget);
	if r.is_err()
	{
		if let (Some(t), Some(p)) = (c.vote_tally_mut(), pre)
		{
			*t = p;
		}
	}
	r
}

fn walk_plans_gpu_inner(
	c: &mut WnnController,
	plans: &[BpttWindowPlan],
	topk: usize,
	protect_learned: bool,
	budget: usize,
) -> Result<(usize, usize), String>
{
	check_supported(c, topk)?;
	let w = bptt_walker()?;
	let (nm, levels, obpn, sn, sbpn, mode, _full) = c.bptt_dims();
	let (frame_bits, sensor_window, sil, oil, num_out) = c.bptt_geometry();
	let default_cell = CellMode::of(mode).default_cell(UnwrittenLean::True);
	// Chunks of whole windows.
	let mut chunks: Vec<(usize, usize, usize)> = Vec::new(); // (first plan, end plan, records)
	let mut i = 0;
	while i < plans.len()
	{
		let (start, mut recs) = (i, 0usize);
		while i < plans.len() && (i == start || recs + plans[i].n_rec <= budget)
		{
			recs += plans[i].n_rec;
			i += 1;
		}
		chunks.push((start, i, recs));
	}
	let max_chunk = chunks.iter().map(|c| c.2).max().unwrap_or(0);
	if max_chunk == 0
	{
		return Ok((0, 0));
	}
	let layer_cells = |c: &WnnController, state: bool, n: usize| -> Vec<Vec<(u64, u8)>> {
		(0..n).map(|i| c.bptt_layer(state).neuron_entries(i)).collect()
	};
	let mut o_tab = LayerTable::build(&layer_cells(c, false, num_out), max_chunk);
	let mut s_tab = LayerTable::build(&layer_cells(c, true, sn), max_chunk);
	let mut o_buf = LayerBufs::upload(w, &o_tab);
	let mut s_buf = LayerBufs::upload(w, &s_tab);
	let (sconn, oconn) = c.bptt_conns();
	let b_oconn = w.buf(&oconn.iter().map(|&x| x as i32).collect::<Vec<i32>>());
	let b_sconn = w.buf(&sconn.iter().map(|&x| x as i32).collect::<Vec<i32>>());
	let b_prev = w.buf(&c.bptt_prev_state().iter().map(|&v| v as u32).collect::<Vec<u32>>());
	let max_win = plans.iter().map(|p| p.n_rec).max().unwrap_or(1).max(1);
	let mut p = BpttParams {
		mode: mode as u32,
		num_motors: nm as u32,
		levels: levels as u32,
		num_out: num_out as u32,
		obpn: obpn as u32,
		sn: sn as u32,
		sbpn: sbpn as u32,
		frame_bits: frame_bits as u32,
		sensor_window: sensor_window as u32,
		sil: sil as u32,
		oil: oil as u32,
		k_out: topk.min(1usize << obpn) as u32,
		k_st: topk.min(1usize << sbpn) as u32,
		beam_out: 64.min((8 * levels).max(16)) as u32,
		beam_st: 64.min((8 * sn).max(16)) as u32,
		protect: protect_learned as u32,
		has_override: plans.iter().any(|p| p.state_target.is_some()) as u32,
		default_cell: default_cell as u32,
		sens_words: words(sensor_window) as u32,
		frame_words: words(frame_bits) as u32,
		tgt_words: words(num_out) as u32,
		si_words: words(sil) as u32,
		oi_words: words(oil) as u32,
		threads: w.threads as u32,
		vote: c.vote_tally_mut().is_some() as u32,
		..Default::default()
	};
	let ntask = num_out + sn;
	// CTRL-46: per-chunk visit buffers (one slot per record × task); 1-element dummies
	// when the round is not a vote.
	let vote_slots = if p.vote != 0 { max_chunk * ntask } else { 1 };
	let b_va = w.zeroed::<u64>(vote_slots);
	let b_vd = w.zeroed::<u8>(vote_slots);
	let t = w.threads as usize;
	let b_si = w.zeroed::<u32>(max_win * p.si_words as usize);
	let b_oi = w.zeroed::<u32>(max_win * p.oi_words as usize);
	let b_tka = w.zeroed::<u64>(ntask * MAX_KTOP);
	let b_tkr = w.zeroed::<u32>(ntask * MAX_KTOP);
	let b_tkc = w.zeroed::<u32>(ntask);
	let b_lna = w.zeroed::<u64>(t * MAX_KTOP);
	let b_lnr = w.zeroed::<u32>(t * MAX_KTOP);
	let b_lnc = w.zeroed::<u32>(t);
	let b_cib = w.zeroed::<u64>(ntask * MAX_KTOP * 3);
	let b_cio = w.zeroed::<u32>(ntask * MAX_KTOP);
	let b_stats = w.zeroed::<u32>(3);
	let b_tt = w.zeroed::<u8>(ntask);
	for &(c0, c1, recs) in &chunks
	{
		if ram_core::cancel::check_cancel()
		{
			break;
		}
		// Capacity: each record claims at most one new entry per neuron.
		let grow = |buf: &LayerBufs| buf.counts().iter().zip(buf.caps.iter()).any(|(&n, &cap)| n as usize + recs > cap as usize);
		if grow(&o_buf)
		{
			o_tab = LayerTable::build(&o_buf.cells(&o_tab.meta), max_chunk.max(recs) * 2);
			o_buf = LayerBufs::upload(w, &o_tab);
		}
		if grow(&s_buf)
		{
			s_tab = LayerTable::build(&s_buf.cells(&s_tab.meta), max_chunk.max(recs) * 2);
			s_buf = LayerBufs::upload(w, &s_tab);
		}
		let (mut wm, mut sens, mut frame, mut tgt, mut sover) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
		let mut rec0 = 0u32;
		for pl in &plans[c0..c1]
		{
			wm.extend_from_slice(&[rec0, pl.n_rec as u32, pl.reset as u32]);
			rec0 += pl.n_rec as u32;
			for d in 0..pl.n_rec
			{
				pack_bits(&mut sens, &pl.sensor[d * sensor_window..(d + 1) * sensor_window], p.sens_words as usize);
				pack_bits(&mut frame, &pl.frame[d * frame_bits..(d + 1) * frame_bits], p.frame_words as usize);
				pack_bits(&mut tgt, &pl.out_target[d * num_out..(d + 1) * num_out], p.tgt_words as usize);
			}
			match &pl.state_target
			{
				Some(v) => sover.extend_from_slice(v),
				None => sover.extend(std::iter::repeat(NO_STATE_OVERRIDE).take(pl.n_rec * sn)),
			}
		}
		p.num_windows = (c1 - c0) as u32;
		let (b_wm, b_sens, b_fr, b_tg, b_so) = (w.buf(&wm), w.buf(&sens), w.buf(&frame), w.buf(&tgt), w.buf(&sover));
		let b_p = w.buf(std::slice::from_ref(&p));
		let q = &w.queues[w.rr.fetch_add(1, Ordering::Relaxed) % QUEUES];
		let cmd = q.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(&w.pipeline);
		let bufs: [&Buffer; 31] = [
			&o_buf.ent, &o_buf.idx, &o_buf.meta, &o_buf.cnt, &s_buf.ent, &s_buf.idx, &s_buf.meta, &s_buf.cnt,
			&b_oconn, &b_sconn, &b_wm, &b_sens, &b_fr, &b_tg, &b_so, &b_prev, &b_si, &b_oi, &b_tka, &b_tkr,
			&b_tkc, &b_lna, &b_lnr, &b_lnc, &b_cib, &b_cio, &b_stats, &b_p, &b_tt, &b_va, &b_vd,
		];
		for (k, b) in bufs.iter().enumerate()
		{
			enc.set_buffer(k as u64, Some(b), 0);
		}
		enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(w.threads, 1, 1));
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();
		if cmd.status() == MTLCommandBufferStatus::Error
		{
			return Err("controller_bptt_walk command buffer failed (GPU error / watchdog?)".into());
		}
		let st = unsafe { std::slice::from_raw_parts(b_stats.contents() as *const u32, 3) };
		if st[2] != 0
		{
			return Err("controller_bptt_walk: a cell table overflowed (host capacity bug)".into());
		}
		if p.vote != 0
		{
			add_chunk_visits(c, &plans[c0..c1], &b_va, &b_vd, num_out, ntask, recs);
		}
	}
	// Apply back: every entry the tables hold, diffed against the host memory.
	let st = unsafe { std::slice::from_raw_parts(b_stats.contents() as *const u32, 3) }.to_vec();
	for (state, buf, tab, n) in [(false, &o_buf, &o_tab, num_out), (true, &s_buf, &s_tab, sn)]
	{
		let cells = buf.cells(&tab.meta);
		let mem = c.bptt_layer(state);
		for (ni, e) in cells.iter().enumerate().take(n)
		{
			for &(a, v) in e
			{
				if mem.read_cell(ni, a) != v
				{
					let wv = if v == EMPTY_CELL { default_cell } else { v };
					mem.write_cell(ni, a, wv, true);
				}
			}
		}
	}
	let prev: Vec<u8> = unsafe { std::slice::from_raw_parts(b_prev.contents() as *const u32, sn) }
		.iter()
		.map(|&v| v as u8)
		.collect();
	c.bptt_set_prev_state(&prev);
	Ok((st[0] as usize, st[1] as usize))
}
