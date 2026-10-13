#!/usr/bin/env python3
"""Generate dense FPGA RTL from a per-UNIT dense export (export_genome_dense.py).

One wnn_neuron_dense ROM per physical unit (direct O(1) lookup), the adder
weighting each unit by its logical count (rtl_adder.py), so
score_out = Rust score x 4 x logical_neurons. Combinational LUT logic (BRAM 0).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtl_adder import acc_bits, cell_quarters_function, header_lines, require_integer_weights, weighted_sum_block  # noqa: E402


def load_export(export_dir: Path):
	meta = json.loads((export_dir / "metadata.json").read_text())
	connections = json.loads((export_dir / "connections.json").read_text())
	return meta, connections


def generate_dense_impl(output_dir: Path, meta: dict, connections) -> Path:
	weights = require_integer_weights(meta)
	lines = header_lines(meta, "dense") + _ports(meta) + cell_quarters_function(weights)
	lines += _signals(len(connections))
	for u, conn in enumerate(connections):
		lines += _address_lines(u, conn, meta) + _unit_lines(u, conn, meta)
	lines += weighted_sum_block("neuron_result", meta["counts"]) + _output_block()
	sv_path = output_dir / "wnn_classifier_dense.sv"
	sv_path.write_text("\n".join(lines) + "\n")
	print(f"  Generated: {sv_path}")
	return sv_path


def _ports(meta: dict) -> list[str]:
	n = acc_bits(meta)
	return [
		"module wnn_classifier_dense #(",
		"\tparameter int THRESHOLD = 0  // class_out = (score_out >= THRESHOLD); metadata threshold_quarters.sum_ge",
		") (",
		"\tinput  logic                     clk,",
		"\tinput  logic                     rst_n,",
		"\tinput  logic                     input_valid,",
		f"\tinput  logic [{meta['input_bits']-1}:0]  input_vec,",
		"",
		"\toutput logic                     class_out,",
		f"\toutput logic [{n}-1:0]    score_out,",
		"\toutput logic                     output_valid,",
		"\toutput logic                     busy",
		");",
		"",
		f"\tlocalparam int NUM_UNITS  = {meta['physical_neurons']};",
		f"\tlocalparam int INPUT_BITS = {meta['input_bits']};",
		f"\tlocalparam int ACC_BITS   = {n};",
		"",
	]


def _signals(num_units: int) -> list[str]:
	return [
		f"\tlogic [7:0]  neuron_result [{num_units}];",
		f"\tlogic [{num_units}-1:0] neuron_valid;",
		"\tlogic neuron_start;",
		"\tassign neuron_start = input_valid;",
		"\tassign busy = 1'b0;  // Dense: always ready",
		"",
	]


def _address_lines(u: int, conn, meta: dict) -> list[str]:
	wires = ", ".join(f"input_vec[{meta['input_bits'] - 1 - c}]" for c in conn)
	return [f"\tlogic [{len(conn)-1}:0] addr_{u};", f"\tassign addr_{u} = {{{wires}}};", ""]


def _unit_lines(u: int, conn, meta: dict) -> list[str]:
	return [
		"\twnn_neuron_dense #(",
		f"\t\t.ADDR_BITS({len(conn)}),",
		f"\t\t.INPUT_BITS({meta['input_bits']}),",
		f"\t\t.DEFAULT_VALUE(8'd{meta['miss_cell']}),",
		# $readmemh inside the leaf (hierarchical readmemh does not synthesize)
		f"\t\t.MEM_FILE(\"unit_{u:03d}.mem\")",
		f"\t) unit_{u} (",
		"\t\t.clk(clk), .rst_n(rst_n), .start(neuron_start), .input_vec(input_vec),",
		f"\t\t.address(addr_{u}), .result(neuron_result[{u}]), .result_valid(neuron_valid[{u}]), .busy()",
		"\t);",
		"",
	]


def _output_block() -> list[str]:
	return [
		"\talways_ff @(posedge clk or negedge rst_n) begin",
		"\t\tif (!rst_n) begin",
		"\t\t\tclass_out <= 1'b0; score_out <= '0; output_valid <= 1'b0;",
		"\t\tend else begin",
		"\t\t\toutput_valid <= 1'b0;",
		"\t\t\tif (neuron_valid[0]) begin  // all units fire on the same cycle",
		"\t\t\t\tscore_out <= weighted_sum;",
		"\t\t\t\tclass_out <= (weighted_sum >= ACC_BITS'(THRESHOLD));",
		"\t\t\t\toutput_valid <= 1'b1;",
		"\t\t\tend",
		"\t\tend",
		"\tend",
		"",
		"endmodule",
	]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--export-dir", required=True)
	args = parser.parse_args()
	export_dir = Path(args.export_dir)
	meta, connections = load_export(export_dir)
	from export_metadata import footprint_line
	print(f"Generating dense RTL: {footprint_line(meta)}")
	generate_dense_impl(export_dir, meta, connections)
	if (export_dir / "testbench" / "tb_metadata.json").exists():
		from rtl_testbench import generate_testbench
		generate_testbench(export_dir, meta, "wnn_classifier_dense")
	print("\n  Latency: 1 cycle (O(1) direct lookup)")
	print("  To synthesize: add wnn_neuron_dense.sv + wnn_classifier_dense.sv")


if __name__ == "__main__":
	main()
