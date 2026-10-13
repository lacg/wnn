"""The unit-weighted adder contract shared by both RTL generators (IDS-37 stage 7).

Score (Rust) = sum over LOGICAL neurons of weight(cell) / N. Identical neurons
are one physical unit, so the hardware instantiates one neuron per unit and
adds count_u x quarters(cell_u): sum_q = 4 N x score, exactly, for the
deterministic modes. ACC_BITS is sized from the LOGICAL N (max sum = 4 N).
The cell -> quarters LUT comes from the accelerator (cell_to_weight x 4), so
the RTL never hard-codes cell semantics.
"""

from math import ceil, log2


def acc_bits(meta: dict) -> int:
	return max(1, ceil(log2(meta["score_denominator_quarters"] + 1)))


def require_integer_weights(meta: dict) -> list[int]:
	weights = meta.get("cell_weight_quarters")
	if weights is None:
		raise ValueError(f"memory_mode {meta.get('memory_mode')} reads a stochastic coin per example: "
			f"no combinational adder reproduces it — refusing to generate RTL")
	return weights


def cell_quarters_function(weights: list[int]) -> list[str]:
	"""SystemVerilog LUT: cell code -> adder weight in quarters."""
	lines = ["\tfunction automatic logic [2:0] cell_quarters(input logic [7:0] cell);", "\t\tcase (cell)"]
	lines += [f"\t\t\t8'd{code}: cell_quarters = 3'd{w};" for code, w in enumerate(weights)]
	lines += ["\t\t\tdefault: cell_quarters = 3'd0;", "\t\tendcase", "\tendfunction", ""]
	return lines


def weighted_sum_block(results: str, counts: list[int]) -> list[str]:
	"""always_comb sum of count_u x cell_quarters(results[u]) over units."""
	lines = ["\tlogic [ACC_BITS-1:0] weighted_sum;", "\talways_comb begin", "\t\tweighted_sum = '0;"]
	lines += [f"\t\tweighted_sum = weighted_sum + ACC_BITS'({c}) * ACC_BITS'(cell_quarters({results}[{u}]));"
		for u, c in enumerate(counts)]
	lines += ["\tend", ""]
	return lines


def header_lines(meta: dict, kind: str) -> list[str]:
	bits = meta["unit_bits"]
	b = f"{min(bits)}b" if min(bits) == max(bits) else f"{min(bits)}-{max(bits)}b"
	return [
		f"// Auto-generated {kind} WNN classifier — one neuron per PHYSICAL unit",
		f"// Genome: {meta['genome_hash']} from flow {meta['flow_id']} ({meta['memory_mode']}, rate {meta['neuron_sample_rate']})",
		f"// {meta['logical_neurons']} logical neurons x {b} -> {meta['physical_neurons']} units (adder weight = unit count)",
		f"// Dataset: {meta['dataset']} ({meta['split']} split), {meta['input_bits']} input bits",
		f"// score_out = sum count_u x quarters(cell_u) = Rust score x {meta['score_denominator_quarters']}",
		"",
	]
