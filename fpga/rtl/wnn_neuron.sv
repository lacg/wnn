// WNN Sparse RAM Neuron — BRAM-based binary search lookup
//
// Two-phase binary search: ADDR phase computes midpoint and issues BRAM read,
// COMPARE phase checks the result and narrows the range.
// Total latency: 2 * ceil(log2(NUM_ENTRIES)) + 2 cycles.

module wnn_neuron #(
	parameter int NUM_ENTRIES     = 4096,
	parameter int ADDR_BITS       = 34,
	parameter int INPUT_BITS      = 360,
	parameter int KEY_BITS        = 64,
	parameter int VALUE_BITS      = 8,
	parameter int SEARCH_DEPTH    = $clog2(NUM_ENTRIES) + 1,
	parameter bit [7:0] DEFAULT_VALUE = 8'd1,  // QUAD_WEAK_FALSE
	// Memory-read latency in cycles for each binary-search probe. Default 1 =
	// on-chip BRAM (registered single-cycle read). Set higher to MODEL the
	// per-probe latency of EXTERNAL DRAM (e.g. ~30 cycles ≈ 100 ns at 300 MHz)
	// for the genomes whose tables don't fit on-chip. The binary-search probes
	// are dependent (each address needs the previous comparison), so they can't
	// be pipelined — the search latency is SEARCH_DEPTH × MEM_READ_LATENCY.
	parameter int MEM_READ_LATENCY = 1,
	// .mem file paths for $readmemh initialization. Empty = leave
	// memory uninitialized (only safe for simulation testbenches that
	// drive the memory themselves). Vivado synthesis requires these
	// strings to be local (not hierarchical) paths because the
	// $readmemh task is resolved at synth-time relative to the source
	// file search dirs.
	parameter string KEY_MEM_FILE   = "",
	parameter string VALUE_MEM_FILE = ""
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

	// BRAM for sparse keys and values.
	//
	// ⚠️ rom_style, NOT ram_style (fixed 31/08/2026). These arrays are WRITTEN
	// NOWHERE — they are initialized by $readmemh and only ever read, which makes
	// them ROMs. Vivado's `ram_style` attribute applies to inferred RAM and is
	// silently IGNORED on a ROM; the attribute that steers ROM inference is
	// `rom_style`. With the wrong attribute the arrays fell back to whatever the
	// tool chose, and every synthesis we have reports 0 BRAM.
	(* rom_style = "block" *)
	logic [KEY_BITS-1:0]   key_mem   [0:NUM_ENTRIES-1];
	(* rom_style = "block" *)
	logic [VALUE_BITS-1:0] value_mem [0:NUM_ENTRIES-1];

	// Synth-time memory initialization.
	//
	// ⚠️ The $readmemh calls MUST be UNCONDITIONAL inside the initial block
	// (fixed 31/08/2026). They used to sit behind `if (KEY_MEM_FILE != "")`, and
	// Vivado does not infer an initialized memory from a CONDITIONAL $readmemh —
	// it treats the array as uninitialized, sees that nothing ever writes it, and
	// optimizes the whole memory away. The design then synthesizes to just the
	// search FSM, reports 0 BRAM, and produces a LUT count that DOES NOT CONTAIN
	// THE MODEL. That is exactly what happened to every number in fpga/results/.
	//
	// A generate-if keeps the "no mem file" path for testbenches: the branch is
	// resolved at ELABORATION time, so the surviving $readmemh is unconditional.
	generate
		if (KEY_MEM_FILE != "" && VALUE_MEM_FILE != "") begin : g_mem_init
			initial $readmemh(KEY_MEM_FILE,   key_mem);
			initial $readmemh(VALUE_MEM_FILE, value_mem);
		end
	endgenerate

	// FSM states
	typedef enum logic [1:0] {
		IDLE         = 2'b00,
		SEARCH_ADDR  = 2'b01,  // Issue BRAM read at mid
		SEARCH_WAIT  = 2'b11,  // Wait for BRAM read to complete
		SEARCH_CMP   = 2'b10   // Compare key, update range
	} state_t;

	state_t state;

	// Binary search registers
	localparam int IDX_BITS = $clog2(NUM_ENTRIES) + 1;
	logic [IDX_BITS-1:0] lo, hi, mid;
	logic [KEY_BITS-1:0] search_key;
	logic [KEY_BITS-1:0] read_key;
	logic [VALUE_BITS-1:0] read_value;
	logic found;
	logic [15:0] wait_ctr;  // counts memory-read wait cycles (DRAM model)

	// Extend address to KEY_BITS
	assign search_key = {{(KEY_BITS - ADDR_BITS){1'b0}}, address};

	assign busy = (state != IDLE);

	// BRAM read (registered, 1-cycle latency)
	logic [$clog2(NUM_ENTRIES)-1:0] read_addr;
	always_ff @(posedge clk) begin
		read_key   <= key_mem[read_addr];
		read_value <= value_mem[read_addr];
	end

	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			state        <= IDLE;
			result       <= DEFAULT_VALUE;
			result_valid <= 1'b0;
			lo           <= '0;
			hi           <= '0;
			mid          <= '0;
			found        <= 1'b0;
			read_addr    <= '0;
			wait_ctr     <= '0;
		end else begin
			result_valid <= 1'b0;

			case (state)
				IDLE: begin
					if (start) begin
						lo    <= '0;
						hi    <= IDX_BITS'(NUM_ENTRIES);
						found <= 1'b0;
						// Compute first mid and issue BRAM read
						mid       <= IDX_BITS'(NUM_ENTRIES) >> 1;
						read_addr <= (NUM_ENTRIES >> 1);
						state     <= SEARCH_ADDR;  // Go to ADDR first to let BRAM read settle
					end
				end

				SEARCH_ADDR: begin
					// Issue BRAM read at mid
					read_addr <= mid[$clog2(NUM_ENTRIES)-1:0];
					state     <= SEARCH_WAIT;
				end

				SEARCH_WAIT: begin
					// Wait for the memory read to complete. 1 cycle for on-chip
					// BRAM; MEM_READ_LATENCY cycles to model external DRAM.
					if (wait_ctr + 1 >= MEM_READ_LATENCY[15:0]) begin
						wait_ctr <= '0;
						state    <= SEARCH_CMP;
					end else begin
						wait_ctr <= wait_ctr + 1;
					end
				end

				SEARCH_CMP: begin
					// BRAM data is now available in read_key/read_value
					if (lo >= hi) begin
						// Search exhausted
						result       <= found ? result : DEFAULT_VALUE;
						result_valid <= 1'b1;
						state        <= IDLE;
					end else if (read_key == search_key) begin
						// Found the key
						result       <= read_value;
						result_valid <= 1'b1;
						found        <= 1'b1;
						state        <= IDLE;
					end else if (read_key < search_key) begin
						lo  <= mid + 1;
						mid <= (mid + 1 + hi) >> 1;
						state <= SEARCH_ADDR;
					end else begin
						hi  <= mid;
						mid <= (lo + mid) >> 1;
						state <= SEARCH_ADDR;
					end
				end

				default: state <= IDLE;
			endcase
		end
	end

endmodule
