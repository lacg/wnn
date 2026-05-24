`timescale 1ns/1ps
// =============================================================================
// DDR contention timing model for the sparse WNN classifier.
//
// The on-chip design gives each neuron its OWN BRAM → all N neurons binary-
// search in parallel, so latency = one neuron's search (the lat_tb result).
// But a DRAM-backed deployment has ONE DDR port shared by all N neurons.
// Their binary-search probes are DEPENDENT within a neuron (next address
// needs the previous comparison) but INDEPENDENT across neurons, so they
// queue at the single port.
//
// This model captures exactly that: N neurons each issue DEPTH sequential
// read requests; a shared port grants ONE request every SERVICE cycles
// (DDR command throughput) and returns its data LATENCY cycles later (DDR
// random-access latency). A neuron stalls until its outstanding read
// returns before issuing its next probe. We count cycles until ALL neurons
// finish ALL probes — the realistic contention-bound classification latency.
//
// This is the "route all neuron probes through one AXI-like port with a
// service-rate + dependent-read stalls" model. SERVICE and LATENCY are the
// DDR controller parameters (swept by the runner); DEPTH = binary-search
// depth (from the genome), N = neuron count.
// =============================================================================
module dram_contention_tb #(
  parameter int N        = 211,  // neurons sharing the DDR port
  parameter int DEPTH    = 16,   // binary-search probes per neuron (worst case)
  parameter int SERVICE  = 4,    // cycles between successive grants (DDR throughput)
  parameter int LATENCY  = 30,   // cycles from grant to data (DDR random-access latency)
  parameter int MAX_CYC  = 5000000
);
  logic clk = 0;
  always #1666 clk = ~clk;  // period arbitrary; we report CYCLES

  integer cyc = 0;
  integer probes_done [N];      // how many probes this neuron has completed
  integer ready_at    [N];      // cycle its outstanding read returns (-1 = idle)
  integer last_grant_cyc;       // last cycle the port granted a request
  integer rr;                   // round-robin pointer
  integer done_count;
  integer i, cand;
  integer total_grants;

  initial begin
    for (i = 0; i < N; i++) begin probes_done[i] = 0; ready_at[i] = -1; end
    last_grant_cyc = -SERVICE;  // port free at start
    rr = 0; total_grants = 0;

    forever begin
      @(posedge clk);
      cyc = cyc + 1;

      // 1) Complete any reads whose latency has elapsed → neuron advances.
      for (i = 0; i < N; i++) begin
        if (ready_at[i] >= 0 && cyc >= ready_at[i]) begin
          ready_at[i] = -1;
          probes_done[i] = probes_done[i] + 1;
        end
      end

      // 2) Arbiter: if the port is free (>= SERVICE since last grant), grant
      //    ONE pending request, round-robin. A neuron has a pending request
      //    iff it's not waiting on a read AND hasn't finished all probes.
      if (cyc - last_grant_cyc >= SERVICE) begin
        cand = -1;
        for (i = 0; i < N; i++) begin
          integer idx;
          idx = (rr + i) % N;
          if (ready_at[idx] < 0 && probes_done[idx] < DEPTH) begin
            cand = idx;
            break;
          end
        end
        if (cand >= 0) begin
          ready_at[cand]  = cyc + LATENCY;   // data returns LATENCY cycles later
          last_grant_cyc  = cyc;
          rr              = (cand + 1) % N;
          total_grants    = total_grants + 1;
        end
      end

      // 3) Done when every neuron has completed all DEPTH probes.
      done_count = 0;
      for (i = 0; i < N; i++) if (probes_done[i] >= DEPTH) done_count++;
      if (done_count == N) begin
        $display("CONTENTION N=%0d DEPTH=%0d SERVICE=%0d LATENCY=%0d -> CYCLES=%0d GRANTS=%0d",
                 N, DEPTH, SERVICE, LATENCY, cyc, total_grants);
        $finish;
      end
      if (cyc > MAX_CYC) begin $display("TIMEOUT"); $finish; end
    end
  end
endmodule
