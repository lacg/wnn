// Auto-generated WNN classifier implementation
// Genome: 46cbd295dd2796f5 from flow 973
// 11 neurons × 34-bit addresses
// 41,215 total sparse entries
// Dataset: cicids2017 (random split)
// Thermometer: 18-bit (360 input bits)

module wnn_classifier_impl #(
	parameter int THRESHOLD = 0
) (
	input  logic                     clk,
	input  logic                     rst_n,
	input  logic                     input_valid,
	input  logic [359:0]  input_vec,

	output logic                     class_out,
	output logic [6-1:0] score_out,
	output logic                     output_valid,
	output logic                     busy
);

	localparam int NUM_NEURONS = 11;
	localparam int ADDR_BITS   = 34;
	localparam int INPUT_BITS  = 360;
	localparam int ACC_BITS    = 6;

	// Per-neuron signals
	logic [7:0]  neuron_result [11];
	logic [11-1:0] neuron_valid;
	logic [11-1:0] neuron_busy;
	logic neuron_start;

	assign neuron_start = input_valid & ~(|neuron_busy);
	assign busy = |neuron_busy;

	// --- Per-neuron address formation (evolved connections) ---
	// Neuron 0: 3825 entries, bits from features [0, 1, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [33:0] addr_0;
	assign addr_0 = {input_vec[123], input_vec[30], input_vec[239], input_vec[208], input_vec[0], input_vec[292], input_vec[95], input_vec[199], input_vec[229], input_vec[130], input_vec[64], input_vec[33], input_vec[67], input_vec[296], input_vec[100], input_vec[221], input_vec[41], input_vec[331], input_vec[251], input_vec[334], input_vec[338], input_vec[51], input_vec[280], input_vec[243], input_vec[102], input_vec[147], input_vec[4], input_vec[155], input_vec[55], input_vec[213], input_vec[358], input_vec[130], input_vec[4], input_vec[243]};

	// Neuron 1: 3601 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
	logic [33:0] addr_1;
	assign addr_1 = {input_vec[211], input_vec[88], input_vec[217], input_vec[104], input_vec[85], input_vec[118], input_vec[188], input_vec[123], input_vec[225], input_vec[267], input_vec[8], input_vec[248], input_vec[342], input_vec[223], input_vec[158], input_vec[146], input_vec[10], input_vec[76], input_vec[333], input_vec[1], input_vec[57], input_vec[20], input_vec[212], input_vec[171], input_vec[331], input_vec[226], input_vec[225], input_vec[345], input_vec[66], input_vec[300], input_vec[3], input_vec[127], input_vec[336], input_vec[271]};

	// Neuron 2: 3887 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_2;
	assign addr_2 = {input_vec[252], input_vec[184], input_vec[30], input_vec[65], input_vec[332], input_vec[189], input_vec[145], input_vec[304], input_vec[341], input_vec[223], input_vec[67], input_vec[92], input_vec[108], input_vec[41], input_vec[28], input_vec[232], input_vec[70], input_vec[314], input_vec[10], input_vec[153], input_vec[253], input_vec[81], input_vec[165], input_vec[2], input_vec[48], input_vec[57], input_vec[7], input_vec[207], input_vec[286], input_vec[341], input_vec[112], input_vec[143], input_vec[232], input_vec[50]};

	// Neuron 3: 2018 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
	logic [33:0] addr_3;
	assign addr_3 = {input_vec[222], input_vec[79], input_vec[254], input_vec[303], input_vec[267], input_vec[140], input_vec[277], input_vec[159], input_vec[200], input_vec[94], input_vec[249], input_vec[336], input_vec[204], input_vec[210], input_vec[240], input_vec[273], input_vec[135], input_vec[124], input_vec[189], input_vec[238], input_vec[78], input_vec[51], input_vec[185], input_vec[174], input_vec[117], input_vec[25], input_vec[46], input_vec[24], input_vec[200], input_vec[322], input_vec[88], input_vec[134], input_vec[126], input_vec[330]};

	// Neuron 4: 5003 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]
	logic [33:0] addr_4;
	assign addr_4 = {input_vec[10], input_vec[307], input_vec[186], input_vec[289], input_vec[168], input_vec[94], input_vec[302], input_vec[149], input_vec[292], input_vec[53], input_vec[162], input_vec[114], input_vec[230], input_vec[265], input_vec[181], input_vec[333], input_vec[269], input_vec[238], input_vec[334], input_vec[151], input_vec[314], input_vec[145], input_vec[200], input_vec[328], input_vec[92], input_vec[316], input_vec[84], input_vec[28], input_vec[22], input_vec[105], input_vec[339], input_vec[346], input_vec[22], input_vec[120]};

	// Neuron 5: 5003 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]
	logic [33:0] addr_5;
	assign addr_5 = {input_vec[10], input_vec[307], input_vec[186], input_vec[289], input_vec[168], input_vec[94], input_vec[302], input_vec[149], input_vec[292], input_vec[53], input_vec[162], input_vec[114], input_vec[230], input_vec[265], input_vec[181], input_vec[333], input_vec[269], input_vec[238], input_vec[334], input_vec[151], input_vec[314], input_vec[145], input_vec[200], input_vec[328], input_vec[92], input_vec[316], input_vec[84], input_vec[28], input_vec[22], input_vec[105], input_vec[339], input_vec[346], input_vec[22], input_vec[120]};

	// Neuron 6: 3887 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_6;
	assign addr_6 = {input_vec[252], input_vec[184], input_vec[30], input_vec[65], input_vec[332], input_vec[189], input_vec[145], input_vec[304], input_vec[341], input_vec[223], input_vec[67], input_vec[92], input_vec[108], input_vec[41], input_vec[28], input_vec[232], input_vec[70], input_vec[314], input_vec[10], input_vec[153], input_vec[253], input_vec[81], input_vec[165], input_vec[2], input_vec[48], input_vec[57], input_vec[7], input_vec[207], input_vec[286], input_vec[341], input_vec[112], input_vec[143], input_vec[232], input_vec[50]};

	// Neuron 7: 3992 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_7;
	assign addr_7 = {input_vec[254], input_vec[177], input_vec[69], input_vec[9], input_vec[200], input_vec[345], input_vec[321], input_vec[46], input_vec[105], input_vec[316], input_vec[49], input_vec[7], input_vec[146], input_vec[334], input_vec[146], input_vec[357], input_vec[193], input_vec[279], input_vec[172], input_vec[177], input_vec[81], input_vec[138], input_vec[295], input_vec[162], input_vec[68], input_vec[330], input_vec[156], input_vec[340], input_vec[324], input_vec[323], input_vec[93], input_vec[13], input_vec[33], input_vec[67]};

	// Neuron 8: 3144 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19]
	logic [33:0] addr_8;
	assign addr_8 = {input_vec[15], input_vec[132], input_vec[354], input_vec[142], input_vec[329], input_vec[306], input_vec[42], input_vec[230], input_vec[339], input_vec[329], input_vec[335], input_vec[84], input_vec[187], input_vec[84], input_vec[284], input_vec[45], input_vec[345], input_vec[229], input_vec[265], input_vec[154], input_vec[144], input_vec[27], input_vec[217], input_vec[276], input_vec[12], input_vec[293], input_vec[255], input_vec[127], input_vec[187], input_vec[236], input_vec[333], input_vec[119], input_vec[260], input_vec[43]};

	// Neuron 9: 3383 entries, bits from features [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_9;
	assign addr_9 = {input_vec[102], input_vec[285], input_vec[24], input_vec[299], input_vec[217], input_vec[289], input_vec[349], input_vec[74], input_vec[107], input_vec[87], input_vec[42], input_vec[355], input_vec[19], input_vec[51], input_vec[58], input_vec[284], input_vec[8], input_vec[147], input_vec[113], input_vec[224], input_vec[338], input_vec[216], input_vec[293], input_vec[333], input_vec[70], input_vec[280], input_vec[210], input_vec[337], input_vec[135], input_vec[311], input_vec[307], input_vec[89], input_vec[168], input_vec[296]};

	// Neuron 10: 3472 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]
	logic [33:0] addr_10;
	assign addr_10 = {input_vec[20], input_vec[347], input_vec[288], input_vec[40], input_vec[215], input_vec[27], input_vec[299], input_vec[146], input_vec[183], input_vec[293], input_vec[5], input_vec[333], input_vec[348], input_vec[297], input_vec[281], input_vec[179], input_vec[69], input_vec[283], input_vec[146], input_vec[176], input_vec[113], input_vec[81], input_vec[337], input_vec[192], input_vec[258], input_vec[189], input_vec[245], input_vec[149], input_vec[290], input_vec[281], input_vec[187], input_vec[9], input_vec[306], input_vec[193]};

	// --- Neuron instances ---
	wnn_neuron #(
		.NUM_ENTRIES(3825),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_0 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_0),
		.result(neuron_result[0]),
		.result_valid(neuron_valid[0]),
		.busy(neuron_busy[0])
	);
	// BRAM init: $readmemh("mem/neuron_000_keys.mem", neuron_0.key_mem);
	// BRAM init: $readmemh("mem/neuron_000_values.mem", neuron_0.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(3601),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_1 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_1),
		.result(neuron_result[1]),
		.result_valid(neuron_valid[1]),
		.busy(neuron_busy[1])
	);
	// BRAM init: $readmemh("mem/neuron_001_keys.mem", neuron_1.key_mem);
	// BRAM init: $readmemh("mem/neuron_001_values.mem", neuron_1.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(3887),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_2 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_2),
		.result(neuron_result[2]),
		.result_valid(neuron_valid[2]),
		.busy(neuron_busy[2])
	);
	// BRAM init: $readmemh("mem/neuron_002_keys.mem", neuron_2.key_mem);
	// BRAM init: $readmemh("mem/neuron_002_values.mem", neuron_2.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(2018),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(11)
	) neuron_3 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_3),
		.result(neuron_result[3]),
		.result_valid(neuron_valid[3]),
		.busy(neuron_busy[3])
	);
	// BRAM init: $readmemh("mem/neuron_003_keys.mem", neuron_3.key_mem);
	// BRAM init: $readmemh("mem/neuron_003_values.mem", neuron_3.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(5003),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(13)
	) neuron_4 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_4),
		.result(neuron_result[4]),
		.result_valid(neuron_valid[4]),
		.busy(neuron_busy[4])
	);
	// BRAM init: $readmemh("mem/neuron_004_keys.mem", neuron_4.key_mem);
	// BRAM init: $readmemh("mem/neuron_004_values.mem", neuron_4.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(5003),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(13)
	) neuron_5 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_5),
		.result(neuron_result[5]),
		.result_valid(neuron_valid[5]),
		.busy(neuron_busy[5])
	);
	// BRAM init: $readmemh("mem/neuron_005_keys.mem", neuron_5.key_mem);
	// BRAM init: $readmemh("mem/neuron_005_values.mem", neuron_5.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(3887),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_6 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_6),
		.result(neuron_result[6]),
		.result_valid(neuron_valid[6]),
		.busy(neuron_busy[6])
	);
	// BRAM init: $readmemh("mem/neuron_006_keys.mem", neuron_6.key_mem);
	// BRAM init: $readmemh("mem/neuron_006_values.mem", neuron_6.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(3992),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_7 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_7),
		.result(neuron_result[7]),
		.result_valid(neuron_valid[7]),
		.busy(neuron_busy[7])
	);
	// BRAM init: $readmemh("mem/neuron_007_keys.mem", neuron_7.key_mem);
	// BRAM init: $readmemh("mem/neuron_007_values.mem", neuron_7.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(3144),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_8 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_8),
		.result(neuron_result[8]),
		.result_valid(neuron_valid[8]),
		.busy(neuron_busy[8])
	);
	// BRAM init: $readmemh("mem/neuron_008_keys.mem", neuron_8.key_mem);
	// BRAM init: $readmemh("mem/neuron_008_values.mem", neuron_8.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(3383),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_9 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_9),
		.result(neuron_result[9]),
		.result_valid(neuron_valid[9]),
		.busy(neuron_busy[9])
	);
	// BRAM init: $readmemh("mem/neuron_009_keys.mem", neuron_9.key_mem);
	// BRAM init: $readmemh("mem/neuron_009_values.mem", neuron_9.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(3472),
		.ADDR_BITS(34),
		.INPUT_BITS(360),
		.SEARCH_DEPTH(12)
	) neuron_10 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_10),
		.result(neuron_result[10]),
		.result_valid(neuron_valid[10]),
		.busy(neuron_busy[10])
	);
	// BRAM init: $readmemh("mem/neuron_010_keys.mem", neuron_10.key_mem);
	// BRAM init: $readmemh("mem/neuron_010_values.mem", neuron_10.value_mem);

	// --- Neuron completion latching ---
	// Neurons finish at different cycles (different entry counts).
	// Latch each result when valid; fire output when all latched.
	logic [11-1:0] neuron_done;
	logic [7:0] neuron_latched [11];
	logic all_done;
	assign all_done = &neuron_done;

	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			neuron_done <= '0;
		end else if (neuron_start) begin
			// Reset latches on new classification
			neuron_done <= '0;
		end else begin
			if (neuron_valid[0]) begin
				neuron_done[0] <= 1'b1;
				neuron_latched[0] <= neuron_result[0];
			end
			if (neuron_valid[1]) begin
				neuron_done[1] <= 1'b1;
				neuron_latched[1] <= neuron_result[1];
			end
			if (neuron_valid[2]) begin
				neuron_done[2] <= 1'b1;
				neuron_latched[2] <= neuron_result[2];
			end
			if (neuron_valid[3]) begin
				neuron_done[3] <= 1'b1;
				neuron_latched[3] <= neuron_result[3];
			end
			if (neuron_valid[4]) begin
				neuron_done[4] <= 1'b1;
				neuron_latched[4] <= neuron_result[4];
			end
			if (neuron_valid[5]) begin
				neuron_done[5] <= 1'b1;
				neuron_latched[5] <= neuron_result[5];
			end
			if (neuron_valid[6]) begin
				neuron_done[6] <= 1'b1;
				neuron_latched[6] <= neuron_result[6];
			end
			if (neuron_valid[7]) begin
				neuron_done[7] <= 1'b1;
				neuron_latched[7] <= neuron_result[7];
			end
			if (neuron_valid[8]) begin
				neuron_done[8] <= 1'b1;
				neuron_latched[8] <= neuron_result[8];
			end
			if (neuron_valid[9]) begin
				neuron_done[9] <= 1'b1;
				neuron_latched[9] <= neuron_result[9];
			end
			if (neuron_valid[10]) begin
				neuron_done[10] <= 1'b1;
				neuron_latched[10] <= neuron_result[10];
			end
		end
	end

	// --- Weighted accumulation ---
	logic [ACC_BITS-1:0] weighted_sum;
	always_comb begin
		weighted_sum = '0;
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[0]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[1]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[2]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[3]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[4]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[5]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[6]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[7]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[8]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[9]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_latched[10]);
	end

	// --- Output (fires once when all neurons done) ---
	logic output_fired;
	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			class_out    <= 1'b0;
			score_out    <= '0;
			output_valid <= 1'b0;
			output_fired <= 1'b0;
		end else begin
			output_valid <= 1'b0;
			if (neuron_start) begin
				output_fired <= 1'b0;
			end else if (all_done && !output_fired) begin
				score_out    <= weighted_sum;
				class_out    <= (weighted_sum > ACC_BITS'(THRESHOLD));
				output_valid <= 1'b1;
				output_fired <= 1'b1;
			end
		end
	end

endmodule
