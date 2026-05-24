`timescale 1ns/1ps
// Generic latency testbench: reset, pulse input_valid for one classification,
// count clock cycles until output_valid. Reports LATENCY_CYCLES (Fmax-
// independent — multiply by 1/Fmax for seconds). INPUT_BITS is overridden
// per-genome at elaboration via -generic_top.
module lat_tb #(
  parameter int INPUT_BITS = 160,
  parameter int MEM_READ_LATENCY = 1   // 1 = on-chip BRAM; 30 ≈ DRAM @300MHz
);
  logic clk = 0, rst_n = 0, input_valid = 0;
  logic [INPUT_BITS-1:0] input_vec = '0;
  logic class_out, output_valid, busy;
  logic [10:0] score_out;
  integer cyc = 0, start_cyc = 0, lat = 0;

  always #1666 clk = ~clk;  // ~300 MHz period; we report CYCLES not ns

  wnn_classifier_impl #(.MEM_READ_LATENCY(MEM_READ_LATENCY)) dut (
    .clk(clk), .rst_n(rst_n), .input_valid(input_valid),
    .input_vec(input_vec), .class_out(class_out), .score_out(score_out),
    .output_valid(output_valid), .busy(busy)
  );

  always @(posedge clk) if (rst_n) cyc <= cyc + 1;

  initial begin
    rst_n = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    // Drive a sample input (alternating 10 pattern across the vector).
    for (int i = 0; i < INPUT_BITS; i++) input_vec[i] = (i % 2 == 0);
    input_valid = 1;
    start_cyc = cyc;
    @(posedge clk);
    input_valid = 0;
    // Wait for output_valid.
    while (!output_valid) begin
      @(posedge clk);
      if (cyc - start_cyc > 200000) begin
        $display("LATENCY_TIMEOUT after %0d cycles", cyc - start_cyc);
        $finish;
      end
    end
    lat = cyc - start_cyc;
    $display("LATENCY_CYCLES=%0d", lat);
    $display("RESULT class_out=%0d score_out=%0d", class_out, score_out);
    $finish;
  end
endmodule
