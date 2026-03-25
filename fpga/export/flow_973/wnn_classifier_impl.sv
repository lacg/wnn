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
	assign addr_0 = {input_vec[236], input_vec[329], input_vec[120], input_vec[151], input_vec[359], input_vec[67], input_vec[264], input_vec[160], input_vec[130], input_vec[229], input_vec[295], input_vec[326], input_vec[292], input_vec[63], input_vec[259], input_vec[138], input_vec[318], input_vec[28], input_vec[108], input_vec[25], input_vec[21], input_vec[308], input_vec[79], input_vec[116], input_vec[257], input_vec[212], input_vec[355], input_vec[204], input_vec[304], input_vec[146], input_vec[1], input_vec[229], input_vec[355], input_vec[116]};

	// Neuron 1: 3601 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
	logic [33:0] addr_1;
	assign addr_1 = {input_vec[148], input_vec[271], input_vec[142], input_vec[255], input_vec[274], input_vec[241], input_vec[171], input_vec[236], input_vec[134], input_vec[92], input_vec[351], input_vec[111], input_vec[17], input_vec[136], input_vec[201], input_vec[213], input_vec[349], input_vec[283], input_vec[26], input_vec[358], input_vec[302], input_vec[339], input_vec[147], input_vec[188], input_vec[28], input_vec[133], input_vec[134], input_vec[14], input_vec[293], input_vec[59], input_vec[356], input_vec[232], input_vec[23], input_vec[88]};

	// Neuron 2: 3887 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_2;
	assign addr_2 = {input_vec[107], input_vec[175], input_vec[329], input_vec[294], input_vec[27], input_vec[170], input_vec[214], input_vec[55], input_vec[18], input_vec[136], input_vec[292], input_vec[267], input_vec[251], input_vec[318], input_vec[331], input_vec[127], input_vec[289], input_vec[45], input_vec[349], input_vec[206], input_vec[106], input_vec[278], input_vec[194], input_vec[357], input_vec[311], input_vec[302], input_vec[352], input_vec[152], input_vec[73], input_vec[18], input_vec[247], input_vec[216], input_vec[127], input_vec[309]};

	// Neuron 3: 2018 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
	logic [33:0] addr_3;
	assign addr_3 = {input_vec[137], input_vec[280], input_vec[105], input_vec[56], input_vec[92], input_vec[219], input_vec[82], input_vec[200], input_vec[159], input_vec[265], input_vec[110], input_vec[23], input_vec[155], input_vec[149], input_vec[119], input_vec[86], input_vec[224], input_vec[235], input_vec[170], input_vec[121], input_vec[281], input_vec[308], input_vec[174], input_vec[185], input_vec[242], input_vec[334], input_vec[313], input_vec[335], input_vec[159], input_vec[37], input_vec[271], input_vec[225], input_vec[233], input_vec[29]};

	// Neuron 4: 5003 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]
	logic [33:0] addr_4;
	assign addr_4 = {input_vec[349], input_vec[52], input_vec[173], input_vec[70], input_vec[191], input_vec[265], input_vec[57], input_vec[210], input_vec[67], input_vec[306], input_vec[197], input_vec[245], input_vec[129], input_vec[94], input_vec[178], input_vec[26], input_vec[90], input_vec[121], input_vec[25], input_vec[208], input_vec[45], input_vec[214], input_vec[159], input_vec[31], input_vec[267], input_vec[43], input_vec[275], input_vec[331], input_vec[337], input_vec[254], input_vec[20], input_vec[13], input_vec[337], input_vec[239]};

	// Neuron 5: 5003 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]
	logic [33:0] addr_5;
	assign addr_5 = {input_vec[349], input_vec[52], input_vec[173], input_vec[70], input_vec[191], input_vec[265], input_vec[57], input_vec[210], input_vec[67], input_vec[306], input_vec[197], input_vec[245], input_vec[129], input_vec[94], input_vec[178], input_vec[26], input_vec[90], input_vec[121], input_vec[25], input_vec[208], input_vec[45], input_vec[214], input_vec[159], input_vec[31], input_vec[267], input_vec[43], input_vec[275], input_vec[331], input_vec[337], input_vec[254], input_vec[20], input_vec[13], input_vec[337], input_vec[239]};

	// Neuron 6: 3887 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_6;
	assign addr_6 = {input_vec[107], input_vec[175], input_vec[329], input_vec[294], input_vec[27], input_vec[170], input_vec[214], input_vec[55], input_vec[18], input_vec[136], input_vec[292], input_vec[267], input_vec[251], input_vec[318], input_vec[331], input_vec[127], input_vec[289], input_vec[45], input_vec[349], input_vec[206], input_vec[106], input_vec[278], input_vec[194], input_vec[357], input_vec[311], input_vec[302], input_vec[352], input_vec[152], input_vec[73], input_vec[18], input_vec[247], input_vec[216], input_vec[127], input_vec[309]};

	// Neuron 7: 3992 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_7;
	assign addr_7 = {input_vec[105], input_vec[182], input_vec[290], input_vec[350], input_vec[159], input_vec[14], input_vec[38], input_vec[313], input_vec[254], input_vec[43], input_vec[310], input_vec[352], input_vec[213], input_vec[25], input_vec[213], input_vec[2], input_vec[166], input_vec[80], input_vec[187], input_vec[182], input_vec[278], input_vec[221], input_vec[64], input_vec[197], input_vec[291], input_vec[29], input_vec[203], input_vec[19], input_vec[35], input_vec[36], input_vec[266], input_vec[346], input_vec[326], input_vec[292]};

	// Neuron 8: 3144 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19]
	logic [33:0] addr_8;
	assign addr_8 = {input_vec[344], input_vec[227], input_vec[5], input_vec[217], input_vec[30], input_vec[53], input_vec[317], input_vec[129], input_vec[20], input_vec[30], input_vec[24], input_vec[275], input_vec[172], input_vec[275], input_vec[75], input_vec[314], input_vec[14], input_vec[130], input_vec[94], input_vec[205], input_vec[215], input_vec[332], input_vec[142], input_vec[83], input_vec[347], input_vec[66], input_vec[104], input_vec[232], input_vec[172], input_vec[123], input_vec[26], input_vec[240], input_vec[99], input_vec[316]};

	// Neuron 9: 3383 entries, bits from features [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [33:0] addr_9;
	assign addr_9 = {input_vec[257], input_vec[74], input_vec[335], input_vec[60], input_vec[142], input_vec[70], input_vec[10], input_vec[285], input_vec[252], input_vec[272], input_vec[317], input_vec[4], input_vec[340], input_vec[308], input_vec[301], input_vec[75], input_vec[351], input_vec[212], input_vec[246], input_vec[135], input_vec[21], input_vec[143], input_vec[66], input_vec[26], input_vec[289], input_vec[79], input_vec[149], input_vec[22], input_vec[224], input_vec[48], input_vec[52], input_vec[270], input_vec[191], input_vec[63]};

	// Neuron 10: 3472 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19]
	logic [33:0] addr_10;
	assign addr_10 = {input_vec[339], input_vec[12], input_vec[71], input_vec[319], input_vec[144], input_vec[332], input_vec[60], input_vec[213], input_vec[176], input_vec[66], input_vec[354], input_vec[26], input_vec[11], input_vec[62], input_vec[78], input_vec[180], input_vec[290], input_vec[76], input_vec[213], input_vec[183], input_vec[246], input_vec[278], input_vec[22], input_vec[167], input_vec[101], input_vec[170], input_vec[114], input_vec[210], input_vec[69], input_vec[78], input_vec[172], input_vec[350], input_vec[53], input_vec[166]};

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

	// --- Weighted accumulation ---
	logic all_valid;
	assign all_valid = &neuron_valid;

	logic [ACC_BITS-1:0] weighted_sum;
	always_comb begin
		weighted_sum = '0;
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[0]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[1]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[2]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[3]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[4]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[5]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[6]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[7]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[8]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[9]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[10]);
	end

	// --- Output ---
	always_ff @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			class_out    <= 1'b0;
			score_out    <= '0;
			output_valid <= 1'b0;
		end else begin
			output_valid <= 1'b0;
			if (all_valid) begin
				score_out    <= weighted_sum;
				class_out    <= (weighted_sum > ACC_BITS'(THRESHOLD));
				output_valid <= 1'b1;
			end
		end
	end

endmodule
