#!/usr/bin/env python3
"""Generate FPGA RTL from exported WNN genome.

Reads the exported sparse memory and connections, generates:
1. Per-neuron BRAM initialization files (.mem)
2. wnn_classifier_impl.sv with actual neuron instances and connection wiring
3. Vivado project TCL script for synthesis

Usage:
    python generate_rtl.py --export-dir fpga/export/flow_973
"""

import argparse
import json
import struct
from pathlib import Path
from math import ceil, log2


def load_export(export_dir: Path):
	"""Load exported genome data."""
	with open(export_dir / "metadata.json") as f:
		meta = json.load(f)
	with open(export_dir / "connections.json") as f:
		connections = json.load(f)

	# Load sparse memory binary
	sparse_neurons = []
	with open(export_dir / "sparse_memory.bin", "rb") as f:
		num_neurons = struct.unpack("<I", f.read(4))[0]
		counts = [struct.unpack("<I", f.read(4))[0] for _ in range(num_neurons)]
		for count in counts:
			keys = []
			values = []
			for _ in range(count):
				k, v = struct.unpack("<QB", f.read(9))
				keys.append(k)
				values.append(v)
			sparse_neurons.append((keys, values))

	return meta, connections, sparse_neurons


def write_mem_files(output_dir: Path, sparse_neurons, key_bits=64, value_bits=8):
	"""Write Vivado-compatible .mem files for BRAM initialization."""
	mem_dir = output_dir / "mem"
	mem_dir.mkdir(parents=True, exist_ok=True)

	for n_idx, (keys, values) in enumerate(sparse_neurons):
		# Keys memory file (hex format, one entry per line)
		with open(mem_dir / f"neuron_{n_idx:03d}_keys.mem", "w") as f:
			for k in keys:
				f.write(f"{k:016x}\n")

		# Values memory file
		with open(mem_dir / f"neuron_{n_idx:03d}_values.mem", "w") as f:
			for v in values:
				f.write(f"{v:02x}\n")

	print(f"  Wrote {len(sparse_neurons)} neuron memory files to {mem_dir}/")


def generate_classifier_impl(output_dir: Path, meta, connections, sparse_neurons):
	"""Generate the wnn_classifier_impl.sv with neuron instances."""
	num_neurons = meta["total_neurons"]
	bits_per_neuron = meta["bits_per_neuron"]
	input_bits = meta["input_bits"]
	n_features = meta["n_features"]
	n_bits_therm = meta["n_bits_thermometer"]

	lines = []
	lines.append("// Auto-generated WNN classifier implementation")
	lines.append(f"// Genome: {meta['genome_hash']} from flow {meta['flow_id']}")
	lines.append(f"// {num_neurons} neurons × {bits_per_neuron}-bit addresses")
	lines.append(f"// {meta['total_sparse_entries']:,} total sparse entries")
	lines.append(f"// Dataset: {meta['dataset']} ({meta['split']} split)")
	lines.append(f"// Thermometer: {n_bits_therm}-bit ({input_bits} input bits)")
	lines.append("")
	lines.append(f"module wnn_classifier_impl #(")
	lines.append(f"\tparameter int THRESHOLD = 0")
	lines.append(f") (")
	lines.append(f"\tinput  logic                     clk,")
	lines.append(f"\tinput  logic                     rst_n,")
	lines.append(f"\tinput  logic                     input_valid,")
	lines.append(f"\tinput  logic [{input_bits-1}:0]  input_vec,")
	lines.append(f"")
	lines.append(f"\toutput logic                     class_out,")
	lines.append(f"\toutput logic [{ceil(log2(num_neurons * 3 + 1))}-1:0] score_out,")
	lines.append(f"\toutput logic                     output_valid,")
	lines.append(f"\toutput logic                     busy")
	lines.append(f");")
	lines.append("")
	lines.append(f"\tlocalparam int NUM_NEURONS = {num_neurons};")
	lines.append(f"\tlocalparam int ADDR_BITS   = {bits_per_neuron};")
	lines.append(f"\tlocalparam int INPUT_BITS  = {input_bits};")
	lines.append(f"\tlocalparam int ACC_BITS    = {ceil(log2(num_neurons * 3 + 1))};")
	lines.append("")

	# Per-neuron signals
	lines.append(f"\t// Per-neuron signals")
	lines.append(f"\tlogic [7:0]  neuron_result [{num_neurons}];")
	lines.append(f"\tlogic [{num_neurons}-1:0] neuron_valid;")
	lines.append(f"\tlogic [{num_neurons}-1:0] neuron_busy;")
	lines.append(f"\tlogic neuron_start;")
	lines.append("")
	lines.append(f"\tassign neuron_start = input_valid & ~(|neuron_busy);")
	lines.append(f"\tassign busy = |neuron_busy;")
	lines.append("")

	# Per-neuron address formation (connection wiring)
	lines.append(f"\t// --- Per-neuron address formation (evolved connections) ---")
	for n_idx in range(num_neurons):
		conn = connections[n_idx]
		num_entries = len(sparse_neurons[n_idx][0])
		lines.append(f"\t// Neuron {n_idx}: {num_entries} entries, "
					 f"bits from features {sorted(set(c // n_bits_therm for c in conn))}")
		lines.append(f"\tlogic [{bits_per_neuron-1}:0] addr_{n_idx};")

		# Wire each address bit to the corresponding input bit.
		# Python bit index k maps to Verilog input_vec[INPUT_BITS-1-k]
		# because $readmemh loads hex MSB-first into the highest bit.
		assignments = []
		for bit_pos, input_idx in enumerate(conn):
			verilog_idx = input_bits - 1 - input_idx
			assignments.append(f"input_vec[{verilog_idx}]")

		# Concatenate: MSB first
		lines.append(f"\tassign addr_{n_idx} = {{{', '.join(assignments)}}};")
		lines.append("")

	# Neuron instances
	lines.append(f"\t// --- Neuron instances ---")
	for n_idx in range(num_neurons):
		num_entries = len(sparse_neurons[n_idx][0])
		search_depth = max(1, ceil(log2(max(num_entries, 2))))
		lines.append(f"\twnn_neuron #(")
		lines.append(f"\t\t.NUM_ENTRIES({num_entries}),")
		lines.append(f"\t\t.ADDR_BITS({bits_per_neuron}),")
		lines.append(f"\t\t.INPUT_BITS({input_bits}),")
		lines.append(f"\t\t.SEARCH_DEPTH({search_depth})")
		lines.append(f"\t) neuron_{n_idx} (")
		lines.append(f"\t\t.clk(clk),")
		lines.append(f"\t\t.rst_n(rst_n),")
		lines.append(f"\t\t.start(neuron_start),")
		lines.append(f"\t\t.input_vec(input_vec),")
		lines.append(f"\t\t.address(addr_{n_idx}),")
		lines.append(f"\t\t.result(neuron_result[{n_idx}]),")
		lines.append(f"\t\t.result_valid(neuron_valid[{n_idx}]),")
		lines.append(f"\t\t.busy(neuron_busy[{n_idx}])")
		lines.append(f"\t);")
		lines.append("")

	# `initial $readmemh` block to bind .mem files into each neuron's BRAM
	# arrays. Without this, Vivado synthesizes the memory as uninitialized
	# storage with no driver and optimizes the entire array away — leaving
	# us with the FSM scaffolding (~7K LUTs) but ZERO memory entries
	# synthesized. The hierarchical paths neuron_<n>.key_mem / .value_mem
	# work because wnn_classifier_impl is the parent module.
	lines.append(f"\t// --- BRAM initialization from .mem files ---")
	lines.append(f"\t// Each $readmemh binds the sparse keys/values exported by")
	lines.append(f"\t// fpga/scripts/export_genome.py into the neuron's BRAM array.")
	lines.append(f"\tinitial begin")
	for n_idx in range(num_neurons):
		lines.append(f"\t\t$readmemh(\"neuron_{n_idx:03d}_keys.mem\",   neuron_{n_idx}.key_mem);")
		lines.append(f"\t\t$readmemh(\"neuron_{n_idx:03d}_values.mem\", neuron_{n_idx}.value_mem);")
	lines.append(f"\tend")
	lines.append("")

	# Accumulation — use latching registers since neurons finish at different cycles
	acc_bits = ceil(log2(num_neurons * 3 + 1))
	lines.append(f"\t// --- Neuron completion latching ---")
	lines.append(f"\t// Neurons finish at different cycles (different entry counts).")
	lines.append(f"\t// Latch each result when valid; fire output when all latched.")
	lines.append(f"\tlogic [{num_neurons}-1:0] neuron_done;")
	lines.append(f"\tlogic [7:0] neuron_latched [{num_neurons}];")
	lines.append(f"\tlogic all_done;")
	lines.append(f"\tassign all_done = &neuron_done;")
	lines.append("")
	lines.append(f"\talways_ff @(posedge clk or negedge rst_n) begin")
	lines.append(f"\t\tif (!rst_n) begin")
	lines.append(f"\t\t\tneuron_done <= '0;")
	lines.append(f"\t\tend else if (neuron_start) begin")
	lines.append(f"\t\t\t// Reset latches on new classification")
	lines.append(f"\t\t\tneuron_done <= '0;")
	lines.append(f"\t\tend else begin")
	for n_idx in range(num_neurons):
		lines.append(f"\t\t\tif (neuron_valid[{n_idx}]) begin")
		lines.append(f"\t\t\t\tneuron_done[{n_idx}] <= 1'b1;")
		lines.append(f"\t\t\t\tneuron_latched[{n_idx}] <= neuron_result[{n_idx}];")
		lines.append(f"\t\t\tend")
	lines.append(f"\t\tend")
	lines.append(f"\tend")
	lines.append("")

	lines.append(f"\t// --- Weighted accumulation ---")
	lines.append(f"\tlogic [ACC_BITS-1:0] weighted_sum;")
	lines.append(f"\talways_comb begin")
	lines.append(f"\t\tweighted_sum = '0;")
	for n_idx in range(num_neurons):
		lines.append(f"\t\tweighted_sum = weighted_sum + ACC_BITS'(neuron_latched[{n_idx}]);")
	lines.append(f"\tend")
	lines.append("")

	# Output
	lines.append(f"\t// --- Output (fires once when all neurons done) ---")
	lines.append(f"\tlogic output_fired;")
	lines.append(f"\talways_ff @(posedge clk or negedge rst_n) begin")
	lines.append(f"\t\tif (!rst_n) begin")
	lines.append(f"\t\t\tclass_out    <= 1'b0;")
	lines.append(f"\t\t\tscore_out    <= '0;")
	lines.append(f"\t\t\toutput_valid <= 1'b0;")
	lines.append(f"\t\t\toutput_fired <= 1'b0;")
	lines.append(f"\t\tend else begin")
	lines.append(f"\t\t\toutput_valid <= 1'b0;")
	lines.append(f"\t\t\tif (neuron_start) begin")
	lines.append(f"\t\t\t\toutput_fired <= 1'b0;")
	lines.append(f"\t\t\tend else if (all_done && !output_fired) begin")
	lines.append(f"\t\t\t\tscore_out    <= weighted_sum;")
	lines.append(f"\t\t\t\tclass_out    <= (weighted_sum > ACC_BITS'(THRESHOLD));")
	lines.append(f"\t\t\t\toutput_valid <= 1'b1;")
	lines.append(f"\t\t\t\toutput_fired <= 1'b1;")
	lines.append(f"\t\t\tend")
	lines.append(f"\t\tend")
	lines.append(f"\tend")
	lines.append("")
	lines.append(f"endmodule")

	# Add address port to wnn_neuron (needs to be an input, not internal)
	sv_path = output_dir / "wnn_classifier_impl.sv"
	with open(sv_path, "w") as f:
		f.write("\n".join(lines) + "\n")
	print(f"  Generated: {sv_path}")
	return sv_path


def generate_vivado_tcl(output_dir: Path, meta, target_fpga="xc7z020clg400-1", clock_period_ns=5.0):
	"""Generate Vivado TCL script for synthesis.

	Adds an explicit `create_clock` so the timing and power reports
	are constrained to a real target frequency (default 200 MHz =
	5.0 ns). Without this, all timing slacks come back NA and the
	power estimate runs away to unconstrained-frequency values.

	OOC (out-of-context) synthesis is also enabled because the
	classifier module has no top-level pads — wrapping it for OOC
	prevents Vivado from inferring IBUFs/OBUFs that don't exist in
	the design intent.
	"""
	tcl_path = output_dir / "synth.tcl"
	rtl_dir = output_dir.parent / "rtl"

	lines = []
	lines.append(f"# Vivado synthesis script for WNN classifier")
	lines.append(f"# Genome: {meta['genome_hash']} ({meta['total_neurons']}n × {meta['bits_per_neuron']}b)")
	lines.append(f"# Target: {target_fpga}  Clock: {clock_period_ns} ns ({1000.0/clock_period_ns:.0f} MHz)")
	lines.append("")
	lines.append(f"create_project wnn_synth {output_dir}/vivado_project -part {target_fpga} -force")
	lines.append("")
	lines.append(f"# Add RTL sources")
	lines.append(f"add_files {rtl_dir}/wnn_neuron.sv")
	lines.append(f"add_files {output_dir}/wnn_classifier_impl.sv")
	lines.append("")
	lines.append(f"# Add BRAM initialization files")
	lines.append(f"add_files {output_dir}/mem/")
	lines.append("")
	lines.append(f"# Set top module")
	lines.append(f"set_property top wnn_classifier_impl [current_fileset]")
	lines.append("")
	lines.append(f"# Out-of-context synthesis — module has no top-level pads, so")
	lines.append(f"# OOC prevents Vivado from inferring nonexistent IO buffers.")
	lines.append(f"synth_design -top wnn_classifier_impl -part {target_fpga} -mode out_of_context")
	lines.append("")
	lines.append(f"# Clock constraint — required for non-NA timing/power reports.")
	lines.append(f"create_clock -period {clock_period_ns} -name clk [get_ports clk]")
	lines.append("")
	lines.append(f"# Re-run timing analysis with the clock applied (synth_design")
	lines.append(f"# above ran before the clock was created).")
	lines.append(f"report_utilization -file {output_dir}/utilization.rpt")
	lines.append(f"report_timing_summary -file {output_dir}/timing.rpt")
	lines.append(f"report_power -file {output_dir}/power.rpt")
	lines.append("")
	lines.append(f"# Capture Fmax from the worst-case path for the summary.")
	lines.append(f"set timing_paths [get_timing_paths -max_paths 1 -quiet]")
	lines.append(f"set fp [open {output_dir}/summary.txt w]")
	lines.append(f"puts $fp \"Genome: {meta['genome_hash']}\"")
	lines.append(f"puts $fp \"Architecture: {meta['total_neurons']}n x {meta['bits_per_neuron']}b\"")
	lines.append(f"puts $fp \"Clock period (target): {clock_period_ns} ns\"")
	lines.append(f"if {{[llength $timing_paths] > 0}} {{")
	lines.append(f"\tset wns [get_property SLACK [lindex $timing_paths 0]]")
	lines.append(f"\tset fmax_mhz [expr {{1000.0 / ({clock_period_ns} - $wns)}}]")
	lines.append(f"\tputs $fp \"WNS: $wns ns\"")
	lines.append(f"\tputs $fp \"Fmax: $fmax_mhz MHz\"")
	lines.append(f"}}")
	lines.append(f"close $fp")
	lines.append("")
	lines.append(f"# Write checkpoint")
	lines.append(f"write_checkpoint -force {output_dir}/post_synth.dcp")
	lines.append("")
	lines.append(f"puts \"Synthesis complete!\"")
	lines.append(f"puts \"Utilization: {output_dir}/utilization.rpt\"")
	lines.append(f"puts \"Timing:      {output_dir}/timing.rpt\"")
	lines.append(f"puts \"Power:       {output_dir}/power.rpt\"")
	lines.append(f"puts \"Summary:     {output_dir}/summary.txt\"")

	with open(tcl_path, "w") as f:
		f.write("\n".join(lines) + "\n")
	print(f"  Generated: {tcl_path}")


def main():
	parser = argparse.ArgumentParser(description="Generate FPGA RTL from exported genome")
	parser.add_argument("--export-dir", required=True, help="Path to exported genome directory")
	parser.add_argument("--target", default="xc7z020clg400-1", help="FPGA part number")
	parser.add_argument("--clock-period-ns", type=float, default=5.0,
		help="Clock period in ns for create_clock (default 5.0 = 200 MHz)")
	args = parser.parse_args()

	export_dir = Path(args.export_dir)
	output_dir = export_dir

	print(f"Loading export from {export_dir}...")
	meta, connections, sparse_neurons = load_export(export_dir)

	print(f"\nGenome: {meta['total_neurons']}n × {meta['bits_per_neuron']}b, "
		  f"{meta['total_sparse_entries']:,} entries")
	print(f"Target FPGA: {args.target}")

	# 1. Write BRAM initialization files
	print(f"\n1. Writing BRAM .mem files...")
	write_mem_files(output_dir, sparse_neurons)

	# 2. Generate classifier implementation
	print(f"\n2. Generating classifier RTL...")
	generate_classifier_impl(output_dir, meta, connections, sparse_neurons)

	# 3. Generate Vivado TCL
	print(f"\n3. Generating Vivado synthesis script...")
	generate_vivado_tcl(output_dir, meta, args.target, args.clock_period_ns)

	# Summary
	total_bytes = meta["sparse_bytes_u64_u8"]
	max_entries = max(meta["per_neuron_entries"])
	search_cycles = ceil(log2(max(max_entries, 2)))
	print(f"\n{'='*50}")
	print(f"RTL Generation Complete")
	print(f"{'='*50}")
	print(f"  Neurons: {meta['total_neurons']}")
	print(f"  Max entries/neuron: {max_entries:,}")
	print(f"  Binary search depth: {search_cycles} cycles")
	print(f"  Total BRAM: {total_bytes/1024:.1f} KB")
	print(f"  Estimated latency: {search_cycles + 2} cycles")
	print(f"  At 100 MHz: {(search_cycles + 2) * 10} ns")
	print(f"  At 200 MHz: {(search_cycles + 2) * 5} ns")
	print(f"\n  Files:")
	print(f"    RTL:  {output_dir}/wnn_classifier_impl.sv")
	print(f"    BRAM: {output_dir}/mem/neuron_*.mem")
	print(f"    TCL:  {output_dir}/synth.tcl")
	print(f"\n  To synthesize: vivado -mode batch -source {output_dir}/synth.tcl")


if __name__ == "__main__":
	main()
