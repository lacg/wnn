"""Generated HW-vs-Rust testbench for a unit-emitted export (IDS-37 stage 7).

Replaces the hand-written, F973-specific tb/wnn_tb.sv (hard-coded 11 neurons,
360 input bits, 6-bit accumulator, hierarchical $readmemh). Parameters come from
metadata.json; memories initialise inside the leaf neurons (MEM_FILE params);
expected values are the RUST scores x 4 x logical_neurons (export_test_vectors).
PASS = every vector's score_out equals the Rust score exactly.
"""

from pathlib import Path

from rtl_adder import acc_bits

TIMEOUT_CYCLES = 200


def generate_testbench(export_dir: Path, meta: dict, dut_module: str) -> Path:
	tb_meta_path = export_dir / "testbench" / "tb_metadata.json"
	import json
	num_vectors = json.loads(tb_meta_path.read_text())["num_vectors"]
	n = acc_bits(meta)
	lines = _declarations(meta, num_vectors, n, dut_module) + _loop(meta, num_vectors) + ["endmodule"]
	path = export_dir / "testbench" / "wnn_tb_generated.sv"
	path.write_text("\n".join(lines) + "\n")
	print(f"  Generated: {path}")
	return path


def _declarations(meta: dict, num_vectors: int, n: int, dut: str) -> list[str]:
	return [
		"`timescale 1ns / 1ps",
		f"// HW-vs-Rust parity: {dut} for flow {meta['flow_id']} genome {meta['genome_hash']}",
		"module wnn_tb;",
		f"\tlocalparam int INPUT_BITS = {meta['input_bits']};",
		f"\tlocalparam int NUM_VECTORS = {num_vectors};",
		f"\tlocalparam int ACC_BITS = {n};",
		f"\tlocalparam int TIMEOUT_CYCLES = {TIMEOUT_CYCLES};",
		"\tlogic clk = 0, rst_n = 0, input_valid, class_out, output_valid, busy;",
		"\talways #2.5 clk = ~clk;",
		"\tlogic [INPUT_BITS-1:0] input_vec;",
		"\tlogic [ACC_BITS-1:0] score_out;",
		f"\t{dut} #(.THRESHOLD(0)) dut (.clk(clk), .rst_n(rst_n), .input_valid(input_valid), .input_vec(input_vec),",
		"\t\t.class_out(class_out), .score_out(score_out), .output_valid(output_valid), .busy(busy));",
		"\tlogic [INPUT_BITS-1:0] test_inputs [0:NUM_VECTORS-1];",
		"\tlogic [31:0] expected [0:NUM_VECTORS-1];",
		"\tint matched = 0, mismatched = 0, cycles;",
		"\tinitial begin",
		"\t\t$readmemh(\"test_inputs.mem\", test_inputs);",
		"\t\t$readmemh(\"test_expected.mem\", expected);",
		"\tend",
	]


def _loop(meta: dict, num_vectors: int) -> list[str]:
	return [
		"\tinitial begin",
		"\t\tinput_valid = 0; input_vec = '0; repeat (10) @(posedge clk); rst_n = 1; repeat (5) @(posedge clk);",
		"\t\tfor (int i = 0; i < NUM_VECTORS; i++) begin",
		"\t\t\twhile (busy) @(posedge clk);",
		"\t\t\tinput_vec = test_inputs[i]; input_valid = 1; @(posedge clk); input_valid = 0;",
		"\t\t\tcycles = 0;",
		"\t\t\twhile (!output_valid && cycles < TIMEOUT_CYCLES) begin @(posedge clk); cycles++; end",
		"\t\t\tif (output_valid && 32'(score_out) == expected[i]) matched++;",
		"\t\t\telse begin",
		"\t\t\t\tmismatched++;",
		"\t\t\t\tif (mismatched <= 20) $display(\"MISMATCH %0d: RTL=%0d Rust=%0d\", i, score_out, expected[i]);",
		"\t\t\tend",
		"\t\tend",
		f"\t\t$display(\"HW-vs-Rust: %0d/%0d match ({meta['physical_neurons']} units for {meta['logical_neurons']} logical neurons)\", matched, NUM_VECTORS);",
		"\t\tif (mismatched == 0) $display(\"RESULT: PASS\"); else $display(\"RESULT: FAIL (%0d)\", mismatched);",
		"\t\t$finish;",
		"\tend",
	]
