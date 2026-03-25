// WNN Sparse RAM Neuron — BRAM-based binary search lookup
//
// Each neuron:
//   1. Gathers selected input bits (evolved connections) to form address
//   2. Binary searches sorted sparse memory for the address
//   3. Returns QUAD_WEIGHTED value (2-bit: 0=FALSE, 1=WEAK_FALSE, 2=WEAK_TRUE, 3=TRUE)
//   4. Default value for untrained addresses: WEAK_FALSE (1 = 0.25 weight)
//
// Memory layout in BRAM (sorted arrays):
//   keys[0..N-1]:   64-bit addresses, sorted ascending
//   values[0..N-1]: 8-bit cell values (only 2 bits used)
//
// Binary search takes ceil(log2(N)) cycles per lookup.

module wnn_neuron #(
	parameter int NUM_ENTRIES     = 4096,   // Number of sparse entries
	parameter int ADDR_BITS       = 34,     // Address bits per neuron
	parameter int INPUT_BITS      = 360,    // Total input vector width
	parameter int KEY_BITS        = 64,     // Key width in BRAM (u64)
	parameter int VALUE_BITS      = 8,      // Value width in BRAM (u8)
	parameter int SEARCH_DEPTH    = $clog2(NUM_ENTRIES) + 1,  // Binary search cycles
	parameter bit [7:0] DEFAULT_VALUE = 8'd1  // QUAD_WEAK_FALSE (untrained)
) (
	input  logic                    clk,
	input  logic                    rst_n,
	input  logic                    start,       // Pulse to begin lookup
	input  logic [INPUT_BITS-1:0]   input_vec,   // Full thermometer-encoded input
	input  logic [ADDR_BITS-1:0]    address,     // Pre-gathered address from connection map

	output logic [VALUE_BITS-1:0]   result,      // Neuron output value
	output logic                    result_valid, // Result ready
	output logic                    busy          // Lookup in progress
);

	// --- BRAM for sparse keys and values ---
	// Initialized from exported genome data
	(* ram_style = "block" *)
	logic [KEY_BITS-1:0]   key_mem   [0:NUM_ENTRIES-1];
	(* ram_style = "block" *)
	logic [VALUE_BITS-1:0] value_mem [0:NUM_ENTRIES-1];

	// --- Binary search FSM ---
	typedef enum logic [1:0] {
		IDLE    = 2'b00,
		SEARCH  = 2'b01,
		DONE    = 2'b10
	} state_t;

	state_t state;
	logic [$clog2(NUM_ENTRIES):0] lo, hi, mid;
	logic [KEY_BITS-1:0] mid_key;
	logic found;
	logic [VALUE_BITS-1:0] found_value;

	// Extend address to KEY_BITS for comparison
	logic [KEY_BITS-1:0] search_key;
	assign search_key = {{(KEY_BITS - ADDR_BITS){1'b0}}, address};

	assign busy = (state != IDLE);

	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			state        <= IDLE;
			result       <= DEFAULT_VALUE;
			result_valid <= 1'b0;
			lo           <= '0;
			hi           <= '0;
			found        <= 1'b0;
			found_value  <= DEFAULT_VALUE;
		end else begin
			result_valid <= 1'b0;  // Default: pulse

			case (state)
				IDLE: begin
					if (start) begin
						lo    <= '0;
						hi    <= NUM_ENTRIES[$clog2(NUM_ENTRIES):0];
						found <= 1'b0;
						found_value <= DEFAULT_VALUE;
						state <= SEARCH;
					end
				end

				SEARCH: begin
					if (lo >= hi) begin
						// Search complete
						result       <= found ? found_value : DEFAULT_VALUE;
						result_valid <= 1'b1;
						state        <= IDLE;
					end else begin
						mid = (lo + hi) >> 1;
						mid_key = key_mem[mid[$clog2(NUM_ENTRIES)-1:0]];

						if (mid_key == search_key) begin
							// Found!
							found       <= 1'b1;
							found_value <= value_mem[mid[$clog2(NUM_ENTRIES)-1:0]];
							result      <= value_mem[mid[$clog2(NUM_ENTRIES)-1:0]];
							result_valid <= 1'b1;
							state       <= IDLE;
						end else if (mid_key < search_key) begin
							lo <= mid + 1;
						end else begin
							hi <= mid;
						end
					end
				end

				default: state <= IDLE;
			endcase
		end
	end

endmodule
