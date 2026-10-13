#!/usr/bin/env python3
"""Generate FPGA RTL from exported WNN genome.

Reads the per-UNIT export (export_genome.py) and generates:
1. Per-unit table initialization files (.mem)
2. wnn_classifier_impl.sv: one neuron instance per physical unit, the adder
   weighting each unit by its logical count (rtl_adder.py)
3. Vivado project TCL script for synthesis

Usage:
    python generate_rtl.py --export-dir fpga/export/flow_973
"""

import argparse
import json
import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math import ceil, log2

from rtl_adder import acc_bits, cell_quarters_function, header_lines, require_integer_weights, weighted_sum_block


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
		with open(mem_dir / f"unit_{n_idx:03d}_keys.mem", "w") as f:
			for k in keys:
				f.write(f"{k:016x}\n")

		# Values memory file
		with open(mem_dir / f"unit_{n_idx:03d}_values.mem", "w") as f:
			for v in values:
				f.write(f"{v:02x}\n")

	print(f"  Wrote {len(sparse_neurons)} unit memory files to {mem_dir}/")


def generate_classifier_impl(output_dir: Path, meta, connections, sparse_neurons):
	"""wnn_classifier_impl.sv: one wnn_neuron per unit + the unit-weighted adder."""
	weights = require_integer_weights(meta)
	lines = header_lines(meta, "sparse") + _ports(meta) + cell_quarters_function(weights)
	lines += _unit_signals(len(connections))
	for u, conn in enumerate(connections):
		lines += _address_lines(u, conn, meta) + _neuron_lines(u, conn, len(sparse_neurons[u][0]), meta)
	lines += _latching(len(connections)) + weighted_sum_block("neuron_latched", meta["counts"]) + _output_block()
	sv_path = output_dir / "wnn_classifier_impl.sv"
	sv_path.write_text("\n".join(lines) + "\n")
	print(f"  Generated: {sv_path}")
	return sv_path


def _ports(meta) -> list[str]:
	n = acc_bits(meta)
	return [
		"module wnn_classifier_impl #(",
		"\tparameter int THRESHOLD = 0,  // class_out = (score_out >= THRESHOLD); metadata threshold_quarters.sum_ge",
		"\t// MEM_READ_LATENCY threads down to every unit's binary-search probe (1 = on-chip).",
		"\tparameter int MEM_READ_LATENCY = 1",
		") (",
		"\tinput  logic                     clk,",
		"\tinput  logic                     rst_n,",
		"\tinput  logic                     input_valid,",
		f"\tinput  logic [{meta['input_bits']-1}:0]  input_vec,",
		"",
		"\toutput logic                     class_out,",
		f"\toutput logic [{n}-1:0] score_out,",
		"\toutput logic                     output_valid,",
		"\toutput logic                     busy",
		");",
		"",
		f"\tlocalparam int NUM_UNITS  = {meta['physical_neurons']};",
		f"\tlocalparam int INPUT_BITS = {meta['input_bits']};",
		f"\tlocalparam int ACC_BITS   = {n};",
		"",
	]


def _unit_signals(num_units: int) -> list[str]:
	return [
		f"\tlogic [7:0]  neuron_result [{num_units}];",
		f"\tlogic [{num_units}-1:0] neuron_valid;",
		f"\tlogic [{num_units}-1:0] neuron_busy;",
		"\tlogic neuron_start;",
		"\tassign neuron_start = input_valid & ~(|neuron_busy);",
		"\tassign busy = |neuron_busy;",
		"",
	]


def _address_lines(u: int, conn, meta) -> list[str]:
	"""Unit u's address: its connections MSB-first (input bit k = input_vec[INPUT_BITS-1-k])."""
	wires = ", ".join(f"input_vec[{meta['input_bits'] - 1 - c}]" for c in conn)
	return [f"\tlogic [{len(conn)-1}:0] addr_{u};", f"\tassign addr_{u} = {{{wires}}};", ""]


def _neuron_lines(u: int, conn, entries: int, meta) -> list[str]:
	return [
		"\twnn_neuron #(",
		f"\t\t.NUM_ENTRIES({entries}),",
		f"\t\t.ADDR_BITS({len(conn)}),",
		f"\t\t.INPUT_BITS({meta['input_bits']}),",
		f"\t\t.SEARCH_DEPTH({max(1, ceil(log2(max(entries, 2))))}),",
		f"\t\t.DEFAULT_VALUE(8'd{meta['miss_cell']}),",
		"\t\t.MEM_READ_LATENCY(MEM_READ_LATENCY),",
		# $readmemh lives in the leaf (hierarchical readmemh does not synthesize: Vivado 8-6086)
		f"\t\t.KEY_MEM_FILE(\"unit_{u:03d}_keys.mem\"),",
		f"\t\t.VALUE_MEM_FILE(\"unit_{u:03d}_values.mem\")",
		f"\t) unit_{u} (",
		"\t\t.clk(clk), .rst_n(rst_n), .start(neuron_start), .input_vec(input_vec),",
		f"\t\t.address(addr_{u}), .result(neuron_result[{u}]), .result_valid(neuron_valid[{u}]), .busy(neuron_busy[{u}])",
		"\t);",
		"",
	]


def _latching(num_units: int) -> list[str]:
	"""Units finish at different cycles (different entry counts): latch each result."""
	lines = [
		f"\tlogic [{num_units}-1:0] neuron_done;",
		f"\tlogic [7:0] neuron_latched [{num_units}];",
		"\tlogic all_done;",
		"\tassign all_done = &neuron_done;",
		"\talways_ff @(posedge clk or negedge rst_n) begin",
		"\t\tif (!rst_n) neuron_done <= '0;",
		"\t\telse if (neuron_start) neuron_done <= '0;",
		"\t\telse begin",
	]
	for u in range(num_units):
		lines += [f"\t\t\tif (neuron_valid[{u}]) begin neuron_done[{u}] <= 1'b1; neuron_latched[{u}] <= neuron_result[{u}]; end"]
	return lines + ["\t\tend", "\tend", ""]


def _output_block() -> list[str]:
	return [
		"\tlogic output_fired;",
		"\talways_ff @(posedge clk or negedge rst_n) begin",
		"\t\tif (!rst_n) begin",
		"\t\t\tclass_out <= 1'b0; score_out <= '0; output_valid <= 1'b0; output_fired <= 1'b0;",
		"\t\tend else begin",
		"\t\t\toutput_valid <= 1'b0;",
		"\t\t\tif (neuron_start) output_fired <= 1'b0;",
		"\t\t\telse if (all_done && !output_fired) begin",
		"\t\t\t\tscore_out <= weighted_sum;",
		"\t\t\t\tclass_out <= (weighted_sum >= ACC_BITS'(THRESHOLD));",
		"\t\t\t\toutput_valid <= 1'b1;",
		"\t\t\t\toutput_fired <= 1'b1;",
		"\t\t\tend",
		"\t\tend",
		"\tend",
		"",
		"endmodule",
	]


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
	lines.append(f"# Genome: {meta['genome_hash']} ({meta['logical_neurons']} logical neurons -> {meta['physical_neurons']} units)")
	lines.append(f"# Target: {target_fpga}  Clock: {clock_period_ns} ns ({1000.0/clock_period_ns:.0f} MHz)")
	lines.append("")
	lines.append(f"create_project wnn_synth {output_dir}/vivado_project -part {target_fpga} -force")
	lines.append("")
	lines.append(f"# Add RTL sources")
	lines.append(f"add_files {rtl_dir}/wnn_neuron.sv")
	lines.append(f"add_files {output_dir}/wnn_classifier_impl.sv")
	lines.append("")
	lines.append(f"# Add BRAM initialization files — must be marked file_type")
	lines.append(f"# {{Memory File}} for Vivado synth to put them on the $readmemh")
	lines.append(f"# search path (otherwise the readmemh calls silently fail with")
	lines.append(f"# CRITICAL WARNING [Synth 8-4445] and the memory stays empty).")
	lines.append(f"set_property file_type {{Memory File}} [add_files [glob {output_dir}/mem/*.mem]]")
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
	lines.append(f"puts $fp \"Architecture: {meta['logical_neurons']} logical neurons, {meta['physical_neurons']} units\"")
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

	from export_metadata import footprint_line
	print(f"\nGenome: {footprint_line(meta)}")
	print(f"Target FPGA: {args.target}")

	# 1. Write BRAM initialization files
	print(f"\n1. Writing BRAM .mem files...")
	write_mem_files(output_dir, sparse_neurons)

	# 2. Generate classifier implementation
	print(f"\n2. Generating classifier RTL...")
	generate_classifier_impl(output_dir, meta, connections, sparse_neurons)

	if (output_dir / "testbench" / "tb_metadata.json").exists():
		from rtl_testbench import generate_testbench
		generate_testbench(output_dir, meta, "wnn_classifier_impl")

	# 3. Generate Vivado TCL
	print(f"\n3. Generating Vivado synthesis script...")
	generate_vivado_tcl(output_dir, meta, args.target, args.clock_period_ns)

	# Summary
	max_entries = max(meta["unit_entries"])
	search_cycles = ceil(log2(max(max_entries, 2)))
	print(f"\n{'='*50}\nRTL Generation Complete\n{'='*50}")
	print(f"  Units: {meta['physical_neurons']} (logical neurons {meta['logical_neurons']}; adder weights = counts)")
	print(f"  Max entries/unit: {max_entries:,}, binary search depth {search_cycles} cycles")
	print(f"  Estimated latency: {search_cycles + 2} cycles")
	print(f"  Files: {output_dir}/wnn_classifier_impl.sv, {output_dir}/mem/unit_*.mem, {output_dir}/synth.tcl")
	print(f"  To synthesize: vivado -mode batch -source {output_dir}/synth.tcl")


if __name__ == "__main__":
	main()
