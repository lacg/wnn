#!/usr/bin/env python3
"""Generate dense FPGA RTL from exported genome.

For small address widths — direct ROM lookup, O(1) inference.
"""

import json
from pathlib import Path
from math import ceil, log2
import argparse


def load_export(export_dir):
	with open(export_dir / "metadata.json") as f:
		meta = json.load(f)
	with open(export_dir / "connections.json") as f:
		connections = json.load(f)
	return meta, connections


def generate_dense_impl(output_dir, meta, connections):
	num_neurons = meta["total_neurons"]
	addr_bits = meta["addr_bits"]
	input_bits = meta["input_bits"]
	n_bits_therm = meta["n_bits_thermometer"]

	lines = []
	lines.append(f"// Auto-generated dense WNN classifier (O(1) lookup)")
	lines.append(f"// Genome: {meta['genome_hash']} from flow {meta['flow_id']}")
	lines.append(f"// {num_neurons} neurons × {addr_bits}-bit addresses (dense)")
	lines.append(f"// Dataset: {meta['dataset']} ({meta['split']} split)")
	lines.append(f"// Thermometer: {n_bits_therm}-bit ({input_bits} input bits)")
	lines.append("")
	acc_bits = max(1, ceil(log2(num_neurons * 3 + 1)))
	lines.append(f"module wnn_classifier_dense #(")
	lines.append(f"\tparameter int THRESHOLD = 0")
	lines.append(f") (")
	lines.append(f"\tinput  logic                     clk,")
	lines.append(f"\tinput  logic                     rst_n,")
	lines.append(f"\tinput  logic                     input_valid,")
	lines.append(f"\tinput  logic [{input_bits-1}:0]  input_vec,")
	lines.append(f"")
	lines.append(f"\toutput logic                     class_out,")
	lines.append(f"\toutput logic [{acc_bits}-1:0]    score_out,")
	lines.append(f"\toutput logic                     output_valid,")
	lines.append(f"\toutput logic                     busy")
	lines.append(f");")
	lines.append("")
	lines.append(f"\tlocalparam int NUM_NEURONS = {num_neurons};")
	lines.append(f"\tlocalparam int ADDR_BITS   = {addr_bits};")
	lines.append(f"\tlocalparam int INPUT_BITS  = {input_bits};")
	lines.append(f"\tlocalparam int ACC_BITS    = {acc_bits};")
	lines.append("")

	# Per-neuron signals
	lines.append(f"\tlogic [7:0]  neuron_result [{num_neurons}];")
	lines.append(f"\tlogic [{num_neurons}-1:0] neuron_valid;")
	lines.append(f"\tlogic neuron_start;")
	lines.append(f"\tassign neuron_start = input_valid;")
	lines.append(f"\tassign busy = 1'b0;  // Dense: always ready")
	lines.append("")

	# Address formation
	lines.append(f"\t// --- Per-neuron address formation (evolved connections) ---")
	for n_idx in range(num_neurons):
		conn = connections[n_idx]
		lines.append(f"\tlogic [{addr_bits-1}:0] addr_{n_idx};")
		assignments = []
		for input_idx in conn:
			verilog_idx = input_bits - 1 - input_idx
			assignments.append(f"input_vec[{verilog_idx}]")
		lines.append(f"\tassign addr_{n_idx} = {{{', '.join(assignments)}}};")
	lines.append("")

	# Neuron instances
	lines.append(f"\t// --- Dense neuron instances (O(1) lookup) ---")
	for n_idx in range(num_neurons):
		lines.append(f"\twnn_neuron_dense #(")
		lines.append(f"\t\t.ADDR_BITS({addr_bits}),")
		lines.append(f"\t\t.INPUT_BITS({input_bits}),")
		# Pass the .mem file as a param so the $readmemh lives inside the leaf
		# module (Vivado can't resolve hierarchical readmemh at synth time —
		# same fix as the sparse wnn_neuron.sv). The old commented-out
		# hierarchical readmemh left memory uninitialized → optimized away →
		# logic-only synth with 0 BRAM.
		lines.append(f"\t\t.MEM_FILE(\"neuron_{n_idx:03d}.mem\")")
		lines.append(f"\t) neuron_{n_idx} (")
		lines.append(f"\t\t.clk(clk),")
		lines.append(f"\t\t.rst_n(rst_n),")
		lines.append(f"\t\t.start(neuron_start),")
		lines.append(f"\t\t.input_vec(input_vec),")
		lines.append(f"\t\t.address(addr_{n_idx}),")
		lines.append(f"\t\t.result(neuron_result[{n_idx}]),")
		lines.append(f"\t\t.result_valid(neuron_valid[{n_idx}]),")
		lines.append(f"\t\t.busy()")
		lines.append(f"\t);")
		lines.append("")

	# Accumulation — all neurons fire simultaneously (1 cycle)
	lines.append(f"\t// --- Weighted accumulation (all neurons valid same cycle) ---")
	lines.append(f"\tlogic all_valid;")
	lines.append(f"\tassign all_valid = neuron_valid[0];  // All fire on same cycle")
	lines.append("")
	lines.append(f"\tlogic [ACC_BITS-1:0] weighted_sum;")
	lines.append(f"\talways_comb begin")
	lines.append(f"\t\tweighted_sum = '0;")
	for n_idx in range(num_neurons):
		lines.append(f"\t\tweighted_sum = weighted_sum + ACC_BITS'(neuron_result[{n_idx}]);")
	lines.append(f"\tend")
	lines.append("")

	# Output
	lines.append(f"\t// --- Output ---")
	lines.append(f"\talways_ff @(posedge clk or negedge rst_n) begin")
	lines.append(f"\t\tif (!rst_n) begin")
	lines.append(f"\t\t\tclass_out    <= 1'b0;")
	lines.append(f"\t\t\tscore_out    <= '0;")
	lines.append(f"\t\t\toutput_valid <= 1'b0;")
	lines.append(f"\t\tend else begin")
	lines.append(f"\t\t\toutput_valid <= 1'b0;")
	lines.append(f"\t\t\tif (all_valid) begin")
	lines.append(f"\t\t\t\tscore_out    <= weighted_sum;")
	lines.append(f"\t\t\t\tclass_out    <= (weighted_sum > ACC_BITS'(THRESHOLD));")
	lines.append(f"\t\t\t\toutput_valid <= 1'b1;")
	lines.append(f"\t\t\tend")
	lines.append(f"\t\tend")
	lines.append(f"\tend")
	lines.append("")
	lines.append(f"endmodule")

	sv_path = output_dir / "wnn_classifier_dense.sv"
	with open(sv_path, "w") as f:
		f.write("\n".join(lines) + "\n")
	print(f"  Generated: {sv_path}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--export-dir", required=True)
	args = parser.parse_args()

	export_dir = Path(args.export_dir)
	meta, connections = load_export(export_dir)

	print(f"Generating dense RTL: {meta['total_neurons']}n × {meta['addr_bits']}b")
	generate_dense_impl(export_dir, meta, connections)

	print(f"\n  Latency: 1 cycle (O(1) direct lookup)")
	print(f"  To synthesize: add wnn_neuron_dense.sv + wnn_classifier_dense.sv")


if __name__ == "__main__":
	main()
