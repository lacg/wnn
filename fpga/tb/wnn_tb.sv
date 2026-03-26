// WNN Classifier Testbench — Functional Verification
//
// Reads test vectors from .mem files, feeds them to the DUT,
// and compares output scores against Python-computed expected values.
//
// Usage (Vivado xsim):
//   xvlog --sv ../rtl/wnn_neuron.sv ../export/flow_973/wnn_classifier_impl.sv wnn_tb.sv
//   xelab wnn_tb -s sim
//   xsim sim -runall

`timescale 1ns / 1ps

module wnn_tb;

	// Parameters matching F973 genome
	localparam int INPUT_BITS  = 360;
	localparam int NUM_VECTORS = 1000;
	localparam int ACC_BITS    = 6;
	localparam int TIMEOUT_CYCLES = 100;  // Max cycles per classification

	// Clock and reset
	logic clk = 0;
	logic rst_n = 0;
	always #2.5 clk = ~clk;  // 200 MHz

	// DUT signals
	logic                    input_valid;
	logic [INPUT_BITS-1:0]   input_vec;
	logic                    class_out;
	logic [ACC_BITS-1:0]     score_out;
	logic                    output_valid;
	logic                    busy;

	// Instantiate DUT
	wnn_classifier_impl #(
		.THRESHOLD(0)  // We'll compare scores directly
	) dut (
		.clk(clk),
		.rst_n(rst_n),
		.input_valid(input_valid),
		.input_vec(input_vec),
		.class_out(class_out),
		.score_out(score_out),
		.output_valid(output_valid),
		.busy(busy)
	);

	// Test vector storage
	logic [INPUT_BITS-1:0] test_inputs [0:NUM_VECTORS-1];
	logic [7:0]            expected_scores [0:NUM_VECTORS-1];

	// Counters
	int total_tested = 0;
	int total_matched = 0;
	int total_mismatched = 0;
	int cycle_count = 0;

	// Load test data
	initial begin
		$readmemh("test_inputs.mem", test_inputs);
		$readmemh("test_expected.mem", expected_scores);
		$display("Loaded %0d test vectors", NUM_VECTORS);
	end

	// BRAM initialization for each neuron
	// The .mem files are loaded via $readmemh in initial blocks
	initial begin
		$readmemh("mem/neuron_000_keys.mem", dut.neuron_0.key_mem);
		$readmemh("mem/neuron_000_values.mem", dut.neuron_0.value_mem);
		$readmemh("mem/neuron_001_keys.mem", dut.neuron_1.key_mem);
		$readmemh("mem/neuron_001_values.mem", dut.neuron_1.value_mem);
		$readmemh("mem/neuron_002_keys.mem", dut.neuron_2.key_mem);
		$readmemh("mem/neuron_002_values.mem", dut.neuron_2.value_mem);
		$readmemh("mem/neuron_003_keys.mem", dut.neuron_3.key_mem);
		$readmemh("mem/neuron_003_values.mem", dut.neuron_3.value_mem);
		$readmemh("mem/neuron_004_keys.mem", dut.neuron_4.key_mem);
		$readmemh("mem/neuron_004_values.mem", dut.neuron_4.value_mem);
		$readmemh("mem/neuron_005_keys.mem", dut.neuron_5.key_mem);
		$readmemh("mem/neuron_005_values.mem", dut.neuron_5.value_mem);
		$readmemh("mem/neuron_006_keys.mem", dut.neuron_6.key_mem);
		$readmemh("mem/neuron_006_values.mem", dut.neuron_6.value_mem);
		$readmemh("mem/neuron_007_keys.mem", dut.neuron_7.key_mem);
		$readmemh("mem/neuron_007_values.mem", dut.neuron_7.value_mem);
		$readmemh("mem/neuron_008_keys.mem", dut.neuron_8.key_mem);
		$readmemh("mem/neuron_008_values.mem", dut.neuron_8.value_mem);
		$readmemh("mem/neuron_009_keys.mem", dut.neuron_9.key_mem);
		$readmemh("mem/neuron_009_values.mem", dut.neuron_9.value_mem);
		$readmemh("mem/neuron_010_keys.mem", dut.neuron_10.key_mem);
		$readmemh("mem/neuron_010_values.mem", dut.neuron_10.value_mem);
	end

	// Main test sequence
	initial begin
		// Reset
		input_valid = 0;
		input_vec = '0;
		rst_n = 0;
		repeat (10) @(posedge clk);
		rst_n = 1;
		repeat (5) @(posedge clk);

		$display("");
		$display("========================================");
		$display("  WNN FPGA Functional Verification");
		$display("  Genome: F973 (11 neurons, 34-bit)");
		$display("  Test vectors: %0d", NUM_VECTORS);
		$display("========================================");
		$display("");

		// Feed test vectors
		for (int i = 0; i < NUM_VECTORS; i++) begin
			// Wait until not busy
			while (busy) @(posedge clk);

			// Present input
			input_vec = test_inputs[i];
			input_valid = 1;
			@(posedge clk);
			input_valid = 0;

			// Wait for result
			cycle_count = 0;
			while (!output_valid && cycle_count < TIMEOUT_CYCLES) begin
				@(posedge clk);
				cycle_count++;
			end

			if (cycle_count >= TIMEOUT_CYCLES) begin
				$display("TIMEOUT on vector %0d after %0d cycles", i, TIMEOUT_CYCLES);
				total_mismatched++;
			end else begin
				// Compare score
				if (score_out == expected_scores[i][ACC_BITS-1:0]) begin
					total_matched++;
				end else begin
					total_mismatched++;
					if (total_mismatched <= 20) begin  // Print first 20 mismatches
						$display("MISMATCH vector %0d: RTL score=%0d, expected=%0d",
								 i, score_out, expected_scores[i]);
					end
				end
			end

			total_tested++;

			// Progress
			if (i % 100 == 99) begin
				$display("  Progress: %0d/%0d tested, %0d matched, %0d mismatched",
						 total_tested, NUM_VECTORS, total_matched, total_mismatched);
			end
		end

		// Final report
		$display("");
		$display("========================================");
		$display("  VERIFICATION COMPLETE");
		$display("========================================");
		$display("  Total tested:     %0d", total_tested);
		$display("  Matched:          %0d", total_matched);
		$display("  Mismatched:       %0d", total_mismatched);
		$display("  Match rate:       %0.2f%%", 100.0 * total_matched / total_tested);
		$display("========================================");

		if (total_mismatched == 0)
			$display("  RESULT: PASS (100%% match with Python)");
		else
			$display("  RESULT: FAIL (%0d mismatches)", total_mismatched);

		$display("");
		$finish;
	end

	// Watchdog timer
	initial begin
		#(NUM_VECTORS * TIMEOUT_CYCLES * 10 * 1ns);
		$display("GLOBAL TIMEOUT - simulation took too long");
		$finish;
	end

endmodule
