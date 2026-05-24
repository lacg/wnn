"""Python port of fpga/tb/dram_contention_tb.sv — single-DDR-port arbiter.

Faithful cycle-step reimplementation of the SystemVerilog arbiter testbench
(validated there: N=1, DEPTH=16, SERVICE=4, LATENCY=30 -> 481 cycles;
GRANTS = N*DEPTH). Kept in Python so the paper's Table-5 latency numbers are
reproducible without a simulator. We assert the N=1 baseline matches the tb
before trusting the multi-neuron numbers.

Model: N neurons each issue DEPTH dependent binary-search read probes through
ONE shared DDR port. The port grants one pending request every SERVICE cycles
(DDR command throughput); granted data returns LATENCY cycles later (DDR
random-access read latency). A neuron stalls on its outstanding read before
issuing the next (dependent) probe. We count cycles until all neurons finish
all DEPTH probes — the contention-bound classification latency.
"""

from __future__ import annotations
from math import ceil, log2


def contention_cycles(N: int, DEPTH: int, SERVICE: int = 4, LATENCY: int = 30,
                      max_cyc: int = 50_000_000) -> tuple[int, int]:
	"""Return (cycles, grants) until all N neurons complete DEPTH probes."""
	probes_done = [0] * N
	ready_at = [-1] * N          # cycle the outstanding read returns; -1 = idle
	last_grant = -SERVICE        # port free at start
	rr = 0                       # round-robin pointer
	grants = 0
	cyc = 0
	while cyc < max_cyc:
		cyc += 1
		# 1) complete reads whose latency elapsed -> neuron advances
		for i in range(N):
			if ready_at[i] >= 0 and cyc >= ready_at[i]:
				ready_at[i] = -1
				probes_done[i] += 1
		# 2) arbiter: if port free, grant one pending request (round-robin)
		if cyc - last_grant >= SERVICE:
			cand = -1
			for k in range(N):
				idx = (rr + k) % N
				if ready_at[idx] < 0 and probes_done[idx] < DEPTH:
					cand = idx
					break
			if cand >= 0:
				ready_at[cand] = cyc + LATENCY
				last_grant = cyc
				rr = (cand + 1) % N
				grants += 1
		# 3) done when every neuron finished all probes
		if all(p >= DEPTH for p in probes_done):
			return cyc, grants
	return cyc, grants


def main():
	# --- validate against the SystemVerilog tb baseline ---
	c, g = contention_cycles(N=1, DEPTH=16, SERVICE=4, LATENCY=30)
	assert g == 1 * 16, f"grants {g} != N*DEPTH"
	assert c == 481, f"N=1 baseline {c} != 481 (tb-validated)"
	print(f"[parity] N=1 DEPTH=16 SERVICE=4 LATENCY=30 -> {c} cyc, {g} grants (matches tb)\n")

	# --- Table 5 genomes (depth 17 binary-search probes each) ---
	# DDR3 single-port model @ f_mc = 200 MHz (5 ns/cyc):
	#   SERVICE=4 cyc (back-to-back read command issue, ~20 ns)
	#   LATENCY=30 cyc (DDR3 random-access read latency incl. controller, ~150 ns)
	F_MC_MHZ = 200.0
	ns_per_cyc = 1000.0 / F_MC_MHZ
	genomes = [
		("Best F1 / Best Acc", 211, 17, 108.7),
		("Best F1 (FPR<6%)",   247, 17, 114.6),
		("Best F1 (FPR<5%/<4%)",245, 17, 111.4),
	]
	print(f"DDR3 single-port arbiter @ {F_MC_MHZ:.0f} MHz (SERVICE=4, LATENCY=30 cyc):")
	print(f"{'Row':22s} {'N':>4s} {'depth':>5s} {'cycles':>8s} {'latency':>10s}  {'Size':>9s}")
	for name, N, depth, mb in genomes:
		cyc, grants = contention_cycles(N=N, DEPTH=depth)
		us = cyc * ns_per_cyc / 1000.0
		print(f"{name:22s} {N:>4d} {depth:>5d} {cyc:>8d} {us:>8.1f} us  {mb:>7.1f}MB")

	# sensitivity: faster/slower DDR assumptions bracket the number
	print("\nsensitivity (Best F1, N=211, depth=17):")
	for svc, lat, fmhz in [(2, 20, 300), (4, 30, 200), (8, 45, 150)]:
		cyc, _ = contention_cycles(N=211, DEPTH=17, SERVICE=svc, LATENCY=lat)
		us = cyc * (1000.0 / fmhz) / 1000.0
		print(f"  SERVICE={svc} LATENCY={lat} @ {fmhz}MHz -> {cyc} cyc = {us:.1f} us")


if __name__ == "__main__":
	main()
