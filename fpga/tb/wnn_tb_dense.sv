// Dense WNN Classifier Testbench — Functional Verification
// For O(1) direct-lookup dense configurations

`timescale 1ns / 1ps

module wnn_tb_dense;

	localparam int INPUT_BITS  = 176;  // UNSW 16-bit therm, 13 features (F1024)
	localparam int NUM_VECTORS = 1000;
	localparam int ACC_BITS    = 8;
	localparam int NUM_NEURONS = 83;

	logic clk = 0;
	logic rst_n = 0;
	always #2.5 clk = ~clk;

	logic                    input_valid;
	logic [INPUT_BITS-1:0]   input_vec;
	logic                    class_out;
	logic [ACC_BITS-1:0]     score_out;
	logic                    output_valid;
	logic                    busy;

	wnn_classifier_dense #(.THRESHOLD(0)) dut (
		.clk(clk), .rst_n(rst_n),
		.input_valid(input_valid), .input_vec(input_vec),
		.class_out(class_out), .score_out(score_out),
		.output_valid(output_valid), .busy(busy)
	);

	logic [INPUT_BITS-1:0] test_inputs [0:NUM_VECTORS-1];
	logic [7:0]            expected_scores [0:NUM_VECTORS-1];

	int total_tested = 0, total_matched = 0, total_mismatched = 0;

	initial begin
		$readmemh("test_inputs.mem", test_inputs);
		$readmemh("test_expected.mem", expected_scores);
		$display("Loaded %0d test vectors (%0d input bits)", NUM_VECTORS, INPUT_BITS);
	end

	// Load dense neuron memories
	initial begin
		`include "mem_init.svh"
	end

	initial begin
		input_valid = 0;
		input_vec = '0;
		rst_n = 0;
		repeat (10) @(posedge clk);
		rst_n = 1;
		repeat (5) @(posedge clk);

		$display("");
		$display("========================================");
		$display("  Dense WNN FPGA Verification (UNSW)");
		$display("  Genome: F1024 best_ce (83 neurons)");
		$display("  Test vectors: %0d", NUM_VECTORS);
		$display("========================================");
		$display("");

		for (int i = 0; i < NUM_VECTORS; i++) begin
			input_vec = test_inputs[i];
			input_valid = 1;
			@(posedge clk);
			input_valid = 0;
			@(posedge clk);  // Neuron registered output
			@(posedge clk);  // Classifier registered output

			if (output_valid) begin
				if (score_out == expected_scores[i][ACC_BITS-1:0]) begin
					total_matched++;
				end else begin
					total_mismatched++;
					if (total_mismatched <= 20)
						$display("MISMATCH vector %0d: RTL=%0d expected=%0d", i, score_out, expected_scores[i]);
				end
			end else begin
				total_mismatched++;
				if (total_mismatched <= 5)
					$display("NO OUTPUT for vector %0d", i);
			end

			total_tested++;
			if (i % 100 == 99)
				$display("  Progress: %0d/%0d, %0d matched, %0d mismatched",
						 total_tested, NUM_VECTORS, total_matched, total_mismatched);
		end

		$display("");
		$display("========================================");
		$display("  VERIFICATION COMPLETE");
		$display("========================================");
		$display("  Total: %0d, Matched: %0d, Mismatched: %0d", total_tested, total_matched, total_mismatched);
		$display("  Match rate: %0.2f%%", 100.0 * total_matched / total_tested);
		if (total_mismatched == 0) $display("  RESULT: PASS");
		else $display("  RESULT: FAIL (%0d mismatches)", total_mismatched);
		$display("");
		$finish;
	end

endmodule
