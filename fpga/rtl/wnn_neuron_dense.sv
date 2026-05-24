// WNN Dense ROM Neuron — Direct LUT lookup (O(1))
//
// For small address widths (8-12 bits), the full 2^b lookup table
// fits in distributed LUTs. Single-cycle inference, no search.

module wnn_neuron_dense #(
	parameter int ADDR_BITS    = 12,
	parameter int INPUT_BITS   = 320,
	parameter int VALUE_BITS   = 8,
	parameter int NUM_ENTRIES  = 2**ADDR_BITS,
	parameter bit [7:0] DEFAULT_VALUE = 8'd1,  // QUAD_WEAK_FALSE
	// .mem file path for $readmemh init. Empty = leave uninitialized (only
	// safe for testbenches that drive the memory). Vivado synthesis requires
	// a LOCAL (non-hierarchical) path because $readmemh is resolved at
	// synth-time relative to the source-file search dirs. See the sparse
	// wnn_neuron.sv for the same fix + rationale.
	parameter string MEM_FILE = ""
) (
	input  logic                    clk,
	input  logic                    rst_n,
	input  logic                    start,
	input  logic [INPUT_BITS-1:0]   input_vec,
	input  logic [ADDR_BITS-1:0]    address,

	output logic [VALUE_BITS-1:0]   result,
	output logic                    result_valid,
	output logic                    busy
);

	// Dense lookup table — full 2^ADDR_BITS entries
	// Initialized from .mem file, synthesized as distributed LUTs/BRAM.
	logic [VALUE_BITS-1:0] mem [0:NUM_ENTRIES-1];

	// Synth-time init from the .mem file (skipped when MEM_FILE empty).
	initial begin
		if (MEM_FILE != "") $readmemh(MEM_FILE, mem);
	end

	// Single-cycle lookup
	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			result       <= DEFAULT_VALUE;
			result_valid <= 1'b0;
		end else begin
			result_valid <= 1'b0;
			if (start) begin
				result       <= mem[address];
				result_valid <= 1'b1;
			end
		end
	end

	assign busy = 1'b0;  // Never busy — always ready

endmodule
