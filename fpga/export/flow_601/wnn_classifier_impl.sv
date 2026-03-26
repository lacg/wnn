// Auto-generated WNN classifier implementation
// Genome: 8363c08423c3df1d from flow 601
// 369 neurons × 32-bit addresses
// 293,047 total sparse entries
// Dataset: unsw-nb15 (temporal split)
// Thermometer: 8-bit (152 input bits)

module wnn_classifier_impl #(
	parameter int THRESHOLD = 0
) (
	input  logic                     clk,
	input  logic                     rst_n,
	input  logic                     input_valid,
	input  logic [151:0]  input_vec,

	output logic                     class_out,
	output logic [11-1:0] score_out,
	output logic                     output_valid,
	output logic                     busy
);

	localparam int NUM_NEURONS = 369;
	localparam int ADDR_BITS   = 32;
	localparam int INPUT_BITS  = 152;
	localparam int ACC_BITS    = 11;

	// Per-neuron signals
	logic [7:0]  neuron_result [369];
	logic [369-1:0] neuron_valid;
	logic [369-1:0] neuron_busy;
	logic neuron_start;

	assign neuron_start = input_valid & ~(|neuron_busy);
	assign busy = |neuron_busy;

	// --- Per-neuron address formation (evolved connections) ---
	// Neuron 0: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_0;
	assign addr_0 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 1: 656 entries, bits from features [1, 2, 3, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_1;
	assign addr_1 = {input_vec[111], input_vec[131], input_vec[15], input_vec[114], input_vec[30], input_vec[124], input_vec[130], input_vec[17], input_vec[143], input_vec[147], input_vec[144], input_vec[75], input_vec[44], input_vec[46], input_vec[23], input_vec[43], input_vec[125], input_vec[99], input_vec[122], input_vec[86], input_vec[111], input_vec[139], input_vec[130], input_vec[145], input_vec[143], input_vec[105], input_vec[82], input_vec[118], input_vec[102], input_vec[21], input_vec[122], input_vec[26]};

	// Neuron 2: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_2;
	assign addr_2 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 3: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_3;
	assign addr_3 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 4: 1102 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_4;
	assign addr_4 = {input_vec[112], input_vec[72], input_vec[102], input_vec[71], input_vec[89], input_vec[96], input_vec[79], input_vec[134], input_vec[64], input_vec[78], input_vec[9], input_vec[150], input_vec[48], input_vec[54], input_vec[2], input_vec[126], input_vec[88], input_vec[94], input_vec[113], input_vec[116], input_vec[10], input_vec[132], input_vec[141], input_vec[114], input_vec[67], input_vec[125], input_vec[136], input_vec[37], input_vec[19], input_vec[26], input_vec[134], input_vec[19]};

	// Neuron 5: 378 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18]
	logic [31:0] addr_5;
	assign addr_5 = {input_vec[21], input_vec[78], input_vec[25], input_vec[28], input_vec[149], input_vec[83], input_vec[99], input_vec[112], input_vec[130], input_vec[125], input_vec[15], input_vec[59], input_vec[37], input_vec[97], input_vec[32], input_vec[120], input_vec[75], input_vec[99], input_vec[133], input_vec[59], input_vec[30], input_vec[15], input_vec[126], input_vec[32], input_vec[40], input_vec[64], input_vec[15], input_vec[53], input_vec[92], input_vec[63], input_vec[74], input_vec[69]};

	// Neuron 6: 630 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_6;
	assign addr_6 = {input_vec[31], input_vec[77], input_vec[55], input_vec[90], input_vec[48], input_vec[41], input_vec[5], input_vec[75], input_vec[114], input_vec[44], input_vec[55], input_vec[36], input_vec[25], input_vec[151], input_vec[63], input_vec[100], input_vec[110], input_vec[103], input_vec[14], input_vec[106], input_vec[71], input_vec[138], input_vec[72], input_vec[145], input_vec[58], input_vec[149], input_vec[73], input_vec[31], input_vec[91], input_vec[85], input_vec[92], input_vec[135]};

	// Neuron 7: 601 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18]
	logic [31:0] addr_7;
	assign addr_7 = {input_vec[41], input_vec[97], input_vec[64], input_vec[141], input_vec[4], input_vec[143], input_vec[102], input_vec[114], input_vec[39], input_vec[15], input_vec[96], input_vec[59], input_vec[31], input_vec[53], input_vec[103], input_vec[32], input_vec[78], input_vec[8], input_vec[60], input_vec[79], input_vec[148], input_vec[136], input_vec[137], input_vec[113], input_vec[13], input_vec[84], input_vec[130], input_vec[29], input_vec[26], input_vec[79], input_vec[142], input_vec[13]};

	// Neuron 8: 326 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18]
	logic [31:0] addr_8;
	assign addr_8 = {input_vec[141], input_vec[76], input_vec[50], input_vec[111], input_vec[64], input_vec[40], input_vec[17], input_vec[68], input_vec[147], input_vec[73], input_vec[123], input_vec[18], input_vec[8], input_vec[77], input_vec[0], input_vec[113], input_vec[0], input_vec[35], input_vec[41], input_vec[141], input_vec[57], input_vec[19], input_vec[34], input_vec[57], input_vec[63], input_vec[58], input_vec[35], input_vec[147], input_vec[20], input_vec[32], input_vec[83], input_vec[86]};

	// Neuron 9: 814 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_9;
	assign addr_9 = {input_vec[66], input_vec[3], input_vec[31], input_vec[80], input_vec[83], input_vec[68], input_vec[141], input_vec[33], input_vec[37], input_vec[51], input_vec[20], input_vec[119], input_vec[62], input_vec[112], input_vec[120], input_vec[79], input_vec[99], input_vec[151], input_vec[13], input_vec[66], input_vec[3], input_vec[101], input_vec[61], input_vec[8], input_vec[74], input_vec[64], input_vec[138], input_vec[59], input_vec[64], input_vec[28], input_vec[109], input_vec[144]};

	// Neuron 10: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_10;
	assign addr_10 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 11: 522 entries, bits from features [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_11;
	assign addr_11 = {input_vec[52], input_vec[143], input_vec[103], input_vec[131], input_vec[115], input_vec[97], input_vec[28], input_vec[124], input_vec[99], input_vec[32], input_vec[139], input_vec[86], input_vec[12], input_vec[131], input_vec[114], input_vec[37], input_vec[96], input_vec[67], input_vec[74], input_vec[90], input_vec[150], input_vec[99], input_vec[33], input_vec[120], input_vec[147], input_vec[91], input_vec[78], input_vec[124], input_vec[112], input_vec[42], input_vec[124], input_vec[100]};

	// Neuron 12: 897 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_12;
	assign addr_12 = {input_vec[120], input_vec[25], input_vec[15], input_vec[115], input_vec[71], input_vec[59], input_vec[132], input_vec[125], input_vec[132], input_vec[26], input_vec[100], input_vec[90], input_vec[99], input_vec[135], input_vec[64], input_vec[36], input_vec[106], input_vec[88], input_vec[115], input_vec[53], input_vec[45], input_vec[120], input_vec[113], input_vec[94], input_vec[45], input_vec[78], input_vec[39], input_vec[57], input_vec[17], input_vec[131], input_vec[137], input_vec[117]};

	// Neuron 13: 1102 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_13;
	assign addr_13 = {input_vec[112], input_vec[72], input_vec[102], input_vec[71], input_vec[89], input_vec[96], input_vec[79], input_vec[134], input_vec[64], input_vec[78], input_vec[9], input_vec[150], input_vec[48], input_vec[54], input_vec[2], input_vec[126], input_vec[88], input_vec[94], input_vec[113], input_vec[116], input_vec[10], input_vec[132], input_vec[141], input_vec[114], input_vec[67], input_vec[125], input_vec[136], input_vec[37], input_vec[19], input_vec[26], input_vec[134], input_vec[19]};

	// Neuron 14: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_14;
	assign addr_14 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 15: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_15;
	assign addr_15 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 16: 917 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_16;
	assign addr_16 = {input_vec[43], input_vec[65], input_vec[138], input_vec[20], input_vec[45], input_vec[77], input_vec[0], input_vec[148], input_vec[89], input_vec[100], input_vec[1], input_vec[26], input_vec[16], input_vec[59], input_vec[147], input_vec[58], input_vec[69], input_vec[4], input_vec[52], input_vec[73], input_vec[41], input_vec[104], input_vec[87], input_vec[86], input_vec[114], input_vec[52], input_vec[31], input_vec[97], input_vec[51], input_vec[9], input_vec[80], input_vec[111]};

	// Neuron 17: 343 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_17;
	assign addr_17 = {input_vec[96], input_vec[0], input_vec[56], input_vec[74], input_vec[16], input_vec[52], input_vec[81], input_vec[39], input_vec[40], input_vec[84], input_vec[141], input_vec[87], input_vec[64], input_vec[127], input_vec[121], input_vec[113], input_vec[75], input_vec[113], input_vec[127], input_vec[146], input_vec[31], input_vec[115], input_vec[100], input_vec[31], input_vec[9], input_vec[143], input_vec[8], input_vec[48], input_vec[107], input_vec[72], input_vec[30], input_vec[12]};

	// Neuron 18: 386 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18]
	logic [31:0] addr_18;
	assign addr_18 = {input_vec[4], input_vec[47], input_vec[33], input_vec[117], input_vec[96], input_vec[52], input_vec[90], input_vec[10], input_vec[117], input_vec[102], input_vec[137], input_vec[12], input_vec[112], input_vec[28], input_vec[134], input_vec[128], input_vec[91], input_vec[83], input_vec[76], input_vec[32], input_vec[11], input_vec[100], input_vec[25], input_vec[29], input_vec[128], input_vec[23], input_vec[61], input_vec[151], input_vec[136], input_vec[88], input_vec[40], input_vec[63]};

	// Neuron 19: 438 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18]
	logic [31:0] addr_19;
	assign addr_19 = {input_vec[67], input_vec[88], input_vec[0], input_vec[105], input_vec[30], input_vec[62], input_vec[45], input_vec[33], input_vec[68], input_vec[42], input_vec[11], input_vec[65], input_vec[84], input_vec[27], input_vec[23], input_vec[88], input_vec[2], input_vec[90], input_vec[112], input_vec[36], input_vec[104], input_vec[53], input_vec[78], input_vec[3], input_vec[23], input_vec[122], input_vec[44], input_vec[100], input_vec[93], input_vec[148], input_vec[29], input_vec[148]};

	// Neuron 20: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_20;
	assign addr_20 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 21: 730 entries, bits from features [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17]
	logic [31:0] addr_21;
	assign addr_21 = {input_vec[47], input_vec[65], input_vec[42], input_vec[83], input_vec[103], input_vec[136], input_vec[46], input_vec[12], input_vec[103], input_vec[125], input_vec[138], input_vec[121], input_vec[80], input_vec[100], input_vec[101], input_vec[36], input_vec[22], input_vec[9], input_vec[81], input_vec[24], input_vec[129], input_vec[30], input_vec[48], input_vec[78], input_vec[9], input_vec[10], input_vec[33], input_vec[67], input_vec[139], input_vec[31], input_vec[17], input_vec[25]};

	// Neuron 22: 546 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18]
	logic [31:0] addr_22;
	assign addr_22 = {input_vec[127], input_vec[115], input_vec[11], input_vec[122], input_vec[109], input_vec[115], input_vec[49], input_vec[6], input_vec[103], input_vec[150], input_vec[31], input_vec[117], input_vec[89], input_vec[43], input_vec[91], input_vec[74], input_vec[19], input_vec[68], input_vec[15], input_vec[123], input_vec[101], input_vec[89], input_vec[61], input_vec[78], input_vec[72], input_vec[55], input_vec[31], input_vec[97], input_vec[80], input_vec[34], input_vec[114], input_vec[126]};

	// Neuron 23: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_23;
	assign addr_23 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 24: 481 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_24;
	assign addr_24 = {input_vec[29], input_vec[112], input_vec[24], input_vec[60], input_vec[90], input_vec[138], input_vec[91], input_vec[105], input_vec[130], input_vec[20], input_vec[96], input_vec[108], input_vec[61], input_vec[83], input_vec[6], input_vec[55], input_vec[119], input_vec[134], input_vec[122], input_vec[138], input_vec[98], input_vec[115], input_vec[138], input_vec[96], input_vec[47], input_vec[38], input_vec[22], input_vec[126], input_vec[72], input_vec[0], input_vec[145], input_vec[89]};

	// Neuron 25: 545 entries, bits from features [1, 2, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_25;
	assign addr_25 = {input_vec[132], input_vec[97], input_vec[34], input_vec[118], input_vec[96], input_vec[18], input_vec[82], input_vec[34], input_vec[134], input_vec[147], input_vec[132], input_vec[137], input_vec[80], input_vec[41], input_vec[56], input_vec[55], input_vec[116], input_vec[133], input_vec[138], input_vec[107], input_vec[8], input_vec[146], input_vec[57], input_vec[86], input_vec[19], input_vec[133], input_vec[124], input_vec[129], input_vec[48], input_vec[123], input_vec[145], input_vec[57]};

	// Neuron 26: 429 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18]
	logic [31:0] addr_26;
	assign addr_26 = {input_vec[71], input_vec[151], input_vec[90], input_vec[32], input_vec[141], input_vec[24], input_vec[44], input_vec[4], input_vec[3], input_vec[110], input_vec[118], input_vec[56], input_vec[150], input_vec[105], input_vec[114], input_vec[105], input_vec[136], input_vec[122], input_vec[126], input_vec[54], input_vec[78], input_vec[118], input_vec[56], input_vec[18], input_vec[86], input_vec[68], input_vec[66], input_vec[40], input_vec[51], input_vec[48], input_vec[16], input_vec[30]};

	// Neuron 27: 730 entries, bits from features [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17]
	logic [31:0] addr_27;
	assign addr_27 = {input_vec[47], input_vec[65], input_vec[42], input_vec[83], input_vec[103], input_vec[136], input_vec[46], input_vec[12], input_vec[103], input_vec[125], input_vec[138], input_vec[121], input_vec[80], input_vec[100], input_vec[101], input_vec[36], input_vec[22], input_vec[9], input_vec[81], input_vec[24], input_vec[129], input_vec[30], input_vec[48], input_vec[78], input_vec[9], input_vec[10], input_vec[33], input_vec[67], input_vec[139], input_vec[31], input_vec[17], input_vec[25]};

	// Neuron 28: 844 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17]
	logic [31:0] addr_28;
	assign addr_28 = {input_vec[14], input_vec[126], input_vec[61], input_vec[128], input_vec[51], input_vec[3], input_vec[86], input_vec[20], input_vec[132], input_vec[97], input_vec[41], input_vec[79], input_vec[63], input_vec[13], input_vec[36], input_vec[50], input_vec[9], input_vec[45], input_vec[139], input_vec[64], input_vec[23], input_vec[48], input_vec[86], input_vec[97], input_vec[2], input_vec[70], input_vec[130], input_vec[41], input_vec[135], input_vec[56], input_vec[48], input_vec[117]};

	// Neuron 29: 522 entries, bits from features [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_29;
	assign addr_29 = {input_vec[52], input_vec[143], input_vec[103], input_vec[131], input_vec[115], input_vec[97], input_vec[28], input_vec[124], input_vec[99], input_vec[32], input_vec[139], input_vec[86], input_vec[12], input_vec[131], input_vec[114], input_vec[37], input_vec[96], input_vec[67], input_vec[74], input_vec[90], input_vec[150], input_vec[99], input_vec[33], input_vec[120], input_vec[147], input_vec[91], input_vec[78], input_vec[124], input_vec[112], input_vec[42], input_vec[124], input_vec[100]};

	// Neuron 30: 326 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18]
	logic [31:0] addr_30;
	assign addr_30 = {input_vec[141], input_vec[76], input_vec[50], input_vec[111], input_vec[64], input_vec[40], input_vec[17], input_vec[68], input_vec[147], input_vec[73], input_vec[123], input_vec[18], input_vec[8], input_vec[77], input_vec[0], input_vec[113], input_vec[0], input_vec[35], input_vec[41], input_vec[141], input_vec[57], input_vec[19], input_vec[34], input_vec[57], input_vec[63], input_vec[58], input_vec[35], input_vec[147], input_vec[20], input_vec[32], input_vec[83], input_vec[86]};

	// Neuron 31: 386 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18]
	logic [31:0] addr_31;
	assign addr_31 = {input_vec[4], input_vec[47], input_vec[33], input_vec[117], input_vec[96], input_vec[52], input_vec[90], input_vec[10], input_vec[117], input_vec[102], input_vec[137], input_vec[12], input_vec[112], input_vec[28], input_vec[134], input_vec[128], input_vec[91], input_vec[83], input_vec[76], input_vec[32], input_vec[11], input_vec[100], input_vec[25], input_vec[29], input_vec[128], input_vec[23], input_vec[61], input_vec[151], input_vec[136], input_vec[88], input_vec[40], input_vec[63]};

	// Neuron 32: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_32;
	assign addr_32 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 33: 545 entries, bits from features [1, 2, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_33;
	assign addr_33 = {input_vec[132], input_vec[97], input_vec[34], input_vec[118], input_vec[96], input_vec[18], input_vec[82], input_vec[34], input_vec[134], input_vec[147], input_vec[132], input_vec[137], input_vec[80], input_vec[41], input_vec[56], input_vec[55], input_vec[116], input_vec[133], input_vec[138], input_vec[107], input_vec[8], input_vec[146], input_vec[57], input_vec[86], input_vec[19], input_vec[133], input_vec[124], input_vec[129], input_vec[48], input_vec[123], input_vec[145], input_vec[57]};

	// Neuron 34: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_34;
	assign addr_34 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 35: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_35;
	assign addr_35 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 36: 898 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_36;
	assign addr_36 = {input_vec[79], input_vec[18], input_vec[107], input_vec[98], input_vec[100], input_vec[34], input_vec[65], input_vec[54], input_vec[93], input_vec[0], input_vec[38], input_vec[49], input_vec[130], input_vec[34], input_vec[60], input_vec[27], input_vec[115], input_vec[142], input_vec[146], input_vec[82], input_vec[88], input_vec[105], input_vec[60], input_vec[100], input_vec[47], input_vec[18], input_vec[150], input_vec[97], input_vec[4], input_vec[111], input_vec[29], input_vec[26]};

	// Neuron 37: 943 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]
	logic [31:0] addr_37;
	assign addr_37 = {input_vec[103], input_vec[19], input_vec[93], input_vec[70], input_vec[33], input_vec[123], input_vec[99], input_vec[11], input_vec[48], input_vec[132], input_vec[130], input_vec[112], input_vec[113], input_vec[70], input_vec[56], input_vec[81], input_vec[20], input_vec[63], input_vec[83], input_vec[56], input_vec[52], input_vec[25], input_vec[85], input_vec[119], input_vec[86], input_vec[74], input_vec[7], input_vec[70], input_vec[101], input_vec[57], input_vec[98], input_vec[142]};

	// Neuron 38: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_38;
	assign addr_38 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 39: 1785 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_39;
	assign addr_39 = {input_vec[3], input_vec[68], input_vec[93], input_vec[103], input_vec[37], input_vec[64], input_vec[90], input_vec[67], input_vec[151], input_vec[94], input_vec[22], input_vec[60], input_vec[11], input_vec[18], input_vec[81], input_vec[140], input_vec[114], input_vec[100], input_vec[144], input_vec[135], input_vec[71], input_vec[12], input_vec[88], input_vec[148], input_vec[30], input_vec[111], input_vec[0], input_vec[129], input_vec[43], input_vec[64], input_vec[76], input_vec[19]};

	// Neuron 40: 892 entries, bits from features [0, 1, 2, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_40;
	assign addr_40 = {input_vec[100], input_vec[2], input_vec[130], input_vec[139], input_vec[129], input_vec[144], input_vec[114], input_vec[127], input_vec[133], input_vec[88], input_vec[17], input_vec[45], input_vec[48], input_vec[60], input_vec[93], input_vec[13], input_vec[131], input_vec[103], input_vec[146], input_vec[139], input_vec[113], input_vec[55], input_vec[2], input_vec[61], input_vec[111], input_vec[104], input_vec[123], input_vec[149], input_vec[21], input_vec[88], input_vec[14], input_vec[120]};

	// Neuron 41: 730 entries, bits from features [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17]
	logic [31:0] addr_41;
	assign addr_41 = {input_vec[47], input_vec[65], input_vec[42], input_vec[83], input_vec[103], input_vec[136], input_vec[46], input_vec[12], input_vec[103], input_vec[125], input_vec[138], input_vec[121], input_vec[80], input_vec[100], input_vec[101], input_vec[36], input_vec[22], input_vec[9], input_vec[81], input_vec[24], input_vec[129], input_vec[30], input_vec[48], input_vec[78], input_vec[9], input_vec[10], input_vec[33], input_vec[67], input_vec[139], input_vec[31], input_vec[17], input_vec[25]};

	// Neuron 42: 897 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_42;
	assign addr_42 = {input_vec[120], input_vec[25], input_vec[15], input_vec[115], input_vec[71], input_vec[59], input_vec[132], input_vec[125], input_vec[132], input_vec[26], input_vec[100], input_vec[90], input_vec[99], input_vec[135], input_vec[64], input_vec[36], input_vec[106], input_vec[88], input_vec[115], input_vec[53], input_vec[45], input_vec[120], input_vec[113], input_vec[94], input_vec[45], input_vec[78], input_vec[39], input_vec[57], input_vec[17], input_vec[131], input_vec[137], input_vec[117]};

	// Neuron 43: 254 entries, bits from features [0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_43;
	assign addr_43 = {input_vec[65], input_vec[72], input_vec[87], input_vec[2], input_vec[4], input_vec[103], input_vec[54], input_vec[60], input_vec[5], input_vec[47], input_vec[65], input_vec[139], input_vec[25], input_vec[59], input_vec[1], input_vec[130], input_vec[61], input_vec[59], input_vec[1], input_vec[130], input_vec[124], input_vec[46], input_vec[116], input_vec[27], input_vec[16], input_vec[87], input_vec[105], input_vec[118], input_vec[80], input_vec[107], input_vec[61], input_vec[119]};

	// Neuron 44: 284 entries, bits from features [0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
	logic [31:0] addr_44;
	assign addr_44 = {input_vec[128], input_vec[54], input_vec[68], input_vec[73], input_vec[71], input_vec[136], input_vec[122], input_vec[94], input_vec[74], input_vec[39], input_vec[133], input_vec[141], input_vec[0], input_vec[68], input_vec[58], input_vec[113], input_vec[34], input_vec[63], input_vec[69], input_vec[128], input_vec[73], input_vec[90], input_vec[119], input_vec[114], input_vec[53], input_vec[42], input_vec[66], input_vec[137], input_vec[68], input_vec[86], input_vec[43], input_vec[112]};

	// Neuron 45: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_45;
	assign addr_45 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 46: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_46;
	assign addr_46 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 47: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_47;
	assign addr_47 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 48: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_48;
	assign addr_48 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 49: 819 entries, bits from features [0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_49;
	assign addr_49 = {input_vec[121], input_vec[43], input_vec[139], input_vec[149], input_vec[116], input_vec[140], input_vec[3], input_vec[141], input_vec[121], input_vec[111], input_vec[131], input_vec[124], input_vec[55], input_vec[109], input_vec[97], input_vec[89], input_vec[42], input_vec[31], input_vec[28], input_vec[11], input_vec[147], input_vec[79], input_vec[8], input_vec[18], input_vec[51], input_vec[98], input_vec[94], input_vec[82], input_vec[132], input_vec[79], input_vec[96], input_vec[19]};

	// Neuron 50: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_50;
	assign addr_50 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 51: 420 entries, bits from features [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_51;
	assign addr_51 = {input_vec[81], input_vec[17], input_vec[65], input_vec[122], input_vec[56], input_vec[92], input_vec[46], input_vec[109], input_vec[42], input_vec[35], input_vec[41], input_vec[66], input_vec[25], input_vec[82], input_vec[23], input_vec[63], input_vec[43], input_vec[27], input_vec[19], input_vec[149], input_vec[48], input_vec[42], input_vec[103], input_vec[110], input_vec[24], input_vec[40], input_vec[96], input_vec[48], input_vec[143], input_vec[74], input_vec[63], input_vec[22]};

	// Neuron 52: 368 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17]
	logic [31:0] addr_52;
	assign addr_52 = {input_vec[109], input_vec[19], input_vec[55], input_vec[85], input_vec[118], input_vec[0], input_vec[24], input_vec[24], input_vec[41], input_vec[119], input_vec[24], input_vec[111], input_vec[82], input_vec[92], input_vec[31], input_vec[87], input_vec[26], input_vec[15], input_vec[142], input_vec[40], input_vec[139], input_vec[57], input_vec[109], input_vec[90], input_vec[5], input_vec[63], input_vec[113], input_vec[126], input_vec[121], input_vec[57], input_vec[126], input_vec[128]};

	// Neuron 53: 735 entries, bits from features [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_53;
	assign addr_53 = {input_vec[150], input_vec[102], input_vec[11], input_vec[13], input_vec[62], input_vec[36], input_vec[38], input_vec[35], input_vec[116], input_vec[80], input_vec[38], input_vec[54], input_vec[39], input_vec[115], input_vec[123], input_vec[15], input_vec[22], input_vec[26], input_vec[9], input_vec[71], input_vec[103], input_vec[38], input_vec[59], input_vec[142], input_vec[53], input_vec[125], input_vec[113], input_vec[138], input_vec[35], input_vec[131], input_vec[79], input_vec[15]};

	// Neuron 54: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_54;
	assign addr_54 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 55: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_55;
	assign addr_55 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 56: 1124 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_56;
	assign addr_56 = {input_vec[86], input_vec[117], input_vec[31], input_vec[80], input_vec[129], input_vec[11], input_vec[25], input_vec[15], input_vec[91], input_vec[59], input_vec[118], input_vec[112], input_vec[118], input_vec[91], input_vec[1], input_vec[53], input_vec[22], input_vec[39], input_vec[36], input_vec[146], input_vec[94], input_vec[78], input_vec[0], input_vec[24], input_vec[87], input_vec[72], input_vec[139], input_vec[104], input_vec[54], input_vec[2], input_vec[111], input_vec[17]};

	// Neuron 57: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_57;
	assign addr_57 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 58: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_58;
	assign addr_58 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 59: 656 entries, bits from features [1, 2, 3, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_59;
	assign addr_59 = {input_vec[111], input_vec[131], input_vec[15], input_vec[114], input_vec[30], input_vec[124], input_vec[130], input_vec[17], input_vec[143], input_vec[147], input_vec[144], input_vec[75], input_vec[44], input_vec[46], input_vec[23], input_vec[43], input_vec[125], input_vec[99], input_vec[122], input_vec[86], input_vec[111], input_vec[139], input_vec[130], input_vec[145], input_vec[143], input_vec[105], input_vec[82], input_vec[118], input_vec[102], input_vec[21], input_vec[122], input_vec[26]};

	// Neuron 60: 601 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18]
	logic [31:0] addr_60;
	assign addr_60 = {input_vec[41], input_vec[97], input_vec[64], input_vec[141], input_vec[4], input_vec[143], input_vec[102], input_vec[114], input_vec[39], input_vec[15], input_vec[96], input_vec[59], input_vec[31], input_vec[53], input_vec[103], input_vec[32], input_vec[78], input_vec[8], input_vec[60], input_vec[79], input_vec[148], input_vec[136], input_vec[137], input_vec[113], input_vec[13], input_vec[84], input_vec[130], input_vec[29], input_vec[26], input_vec[79], input_vec[142], input_vec[13]};

	// Neuron 61: 481 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_61;
	assign addr_61 = {input_vec[29], input_vec[112], input_vec[24], input_vec[60], input_vec[90], input_vec[138], input_vec[91], input_vec[105], input_vec[130], input_vec[20], input_vec[96], input_vec[108], input_vec[61], input_vec[83], input_vec[6], input_vec[55], input_vec[119], input_vec[134], input_vec[122], input_vec[138], input_vec[98], input_vec[115], input_vec[138], input_vec[96], input_vec[47], input_vec[38], input_vec[22], input_vec[126], input_vec[72], input_vec[0], input_vec[145], input_vec[89]};

	// Neuron 62: 898 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_62;
	assign addr_62 = {input_vec[79], input_vec[18], input_vec[107], input_vec[98], input_vec[100], input_vec[34], input_vec[65], input_vec[54], input_vec[93], input_vec[0], input_vec[38], input_vec[49], input_vec[130], input_vec[34], input_vec[60], input_vec[27], input_vec[115], input_vec[142], input_vec[146], input_vec[82], input_vec[88], input_vec[105], input_vec[60], input_vec[100], input_vec[47], input_vec[18], input_vec[150], input_vec[97], input_vec[4], input_vec[111], input_vec[29], input_vec[26]};

	// Neuron 63: 232 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_63;
	assign addr_63 = {input_vec[90], input_vec[113], input_vec[48], input_vec[151], input_vec[117], input_vec[74], input_vec[15], input_vec[143], input_vec[96], input_vec[106], input_vec[44], input_vec[63], input_vec[104], input_vec[149], input_vec[19], input_vec[149], input_vec[91], input_vec[90], input_vec[47], input_vec[74], input_vec[110], input_vec[28], input_vec[112], input_vec[99], input_vec[112], input_vec[65], input_vec[112], input_vec[118], input_vec[99], input_vec[97], input_vec[149], input_vec[32]};

	// Neuron 64: 447 entries, bits from features [0, 1, 2, 3, 4, 8, 9, 10, 11, 13, 14, 15, 17]
	logic [31:0] addr_64;
	assign addr_64 = {input_vec[6], input_vec[141], input_vec[80], input_vec[70], input_vec[68], input_vec[30], input_vec[71], input_vec[122], input_vec[16], input_vec[91], input_vec[33], input_vec[7], input_vec[10], input_vec[117], input_vec[104], input_vec[35], input_vec[30], input_vec[91], input_vec[38], input_vec[85], input_vec[22], input_vec[36], input_vec[113], input_vec[68], input_vec[121], input_vec[90], input_vec[77], input_vec[64], input_vec[33], input_vec[89], input_vec[26], input_vec[82]};

	// Neuron 65: 1034 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18]
	logic [31:0] addr_65;
	assign addr_65 = {input_vec[68], input_vec[80], input_vec[76], input_vec[1], input_vec[114], input_vec[146], input_vec[112], input_vec[103], input_vec[120], input_vec[149], input_vec[33], input_vec[51], input_vec[17], input_vec[25], input_vec[82], input_vec[62], input_vec[116], input_vec[77], input_vec[86], input_vec[75], input_vec[95], input_vec[134], input_vec[145], input_vec[130], input_vec[120], input_vec[131], input_vec[115], input_vec[74], input_vec[11], input_vec[4], input_vec[88], input_vec[131]};

	// Neuron 66: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_66;
	assign addr_66 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 67: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_67;
	assign addr_67 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 68: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_68;
	assign addr_68 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 69: 326 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18]
	logic [31:0] addr_69;
	assign addr_69 = {input_vec[141], input_vec[76], input_vec[50], input_vec[111], input_vec[64], input_vec[40], input_vec[17], input_vec[68], input_vec[147], input_vec[73], input_vec[123], input_vec[18], input_vec[8], input_vec[77], input_vec[0], input_vec[113], input_vec[0], input_vec[35], input_vec[41], input_vec[141], input_vec[57], input_vec[19], input_vec[34], input_vec[57], input_vec[63], input_vec[58], input_vec[35], input_vec[147], input_vec[20], input_vec[32], input_vec[83], input_vec[86]};

	// Neuron 70: 796 entries, bits from features [0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_70;
	assign addr_70 = {input_vec[11], input_vec[28], input_vec[100], input_vec[137], input_vec[92], input_vec[3], input_vec[91], input_vec[71], input_vec[120], input_vec[68], input_vec[114], input_vec[102], input_vec[148], input_vec[31], input_vec[13], input_vec[135], input_vec[12], input_vec[139], input_vec[30], input_vec[103], input_vec[32], input_vec[72], input_vec[15], input_vec[69], input_vec[2], input_vec[6], input_vec[80], input_vec[41], input_vec[97], input_vec[122], input_vec[145], input_vec[139]};

	// Neuron 71: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_71;
	assign addr_71 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 72: 772 entries, bits from features [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_72;
	assign addr_72 = {input_vec[34], input_vec[148], input_vec[124], input_vec[9], input_vec[84], input_vec[96], input_vec[101], input_vec[10], input_vec[106], input_vec[146], input_vec[106], input_vec[17], input_vec[102], input_vec[20], input_vec[63], input_vec[69], input_vec[1], input_vec[136], input_vec[69], input_vec[102], input_vec[56], input_vec[138], input_vec[28], input_vec[37], input_vec[2], input_vec[76], input_vec[28], input_vec[116], input_vec[58], input_vec[65], input_vec[66], input_vec[89]};

	// Neuron 73: 718 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_73;
	assign addr_73 = {input_vec[61], input_vec[65], input_vec[64], input_vec[108], input_vec[90], input_vec[86], input_vec[90], input_vec[73], input_vec[99], input_vec[24], input_vec[142], input_vec[36], input_vec[151], input_vec[149], input_vec[114], input_vec[4], input_vec[9], input_vec[73], input_vec[145], input_vec[115], input_vec[97], input_vec[139], input_vec[110], input_vec[96], input_vec[29], input_vec[41], input_vec[10], input_vec[130], input_vec[110], input_vec[129], input_vec[32], input_vec[18]};

	// Neuron 74: 386 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18]
	logic [31:0] addr_74;
	assign addr_74 = {input_vec[4], input_vec[47], input_vec[33], input_vec[117], input_vec[96], input_vec[52], input_vec[90], input_vec[10], input_vec[117], input_vec[102], input_vec[137], input_vec[12], input_vec[112], input_vec[28], input_vec[134], input_vec[128], input_vec[91], input_vec[83], input_vec[76], input_vec[32], input_vec[11], input_vec[100], input_vec[25], input_vec[29], input_vec[128], input_vec[23], input_vec[61], input_vec[151], input_vec[136], input_vec[88], input_vec[40], input_vec[63]};

	// Neuron 75: 1881 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_75;
	assign addr_75 = {input_vec[140], input_vec[54], input_vec[97], input_vec[49], input_vec[115], input_vec[134], input_vec[149], input_vec[41], input_vec[85], input_vec[125], input_vec[19], input_vec[56], input_vec[101], input_vec[64], input_vec[129], input_vec[147], input_vec[70], input_vec[95], input_vec[31], input_vec[0], input_vec[33], input_vec[63], input_vec[114], input_vec[146], input_vec[84], input_vec[79], input_vec[18], input_vec[16], input_vec[12], input_vec[2], input_vec[147], input_vec[52]};

	// Neuron 76: 927 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_76;
	assign addr_76 = {input_vec[94], input_vec[147], input_vec[106], input_vec[76], input_vec[101], input_vec[18], input_vec[50], input_vec[10], input_vec[149], input_vec[46], input_vec[147], input_vec[73], input_vec[109], input_vec[0], input_vec[7], input_vec[63], input_vec[102], input_vec[52], input_vec[78], input_vec[64], input_vec[126], input_vec[77], input_vec[148], input_vec[143], input_vec[11], input_vec[47], input_vec[115], input_vec[53], input_vec[12], input_vec[25], input_vec[96], input_vec[10]};

	// Neuron 77: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_77;
	assign addr_77 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 78: 955 entries, bits from features [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_78;
	assign addr_78 = {input_vec[64], input_vec[52], input_vec[49], input_vec[144], input_vec[2], input_vec[147], input_vec[17], input_vec[43], input_vec[116], input_vec[100], input_vec[75], input_vec[99], input_vec[45], input_vec[139], input_vec[15], input_vec[18], input_vec[32], input_vec[110], input_vec[72], input_vec[69], input_vec[23], input_vec[140], input_vec[125], input_vec[146], input_vec[19], input_vec[91], input_vec[113], input_vec[12], input_vec[91], input_vec[144], input_vec[116], input_vec[151]};

	// Neuron 79: 1756 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]
	logic [31:0] addr_79;
	assign addr_79 = {input_vec[44], input_vec[101], input_vec[56], input_vec[69], input_vec[46], input_vec[93], input_vec[128], input_vec[29], input_vec[96], input_vec[41], input_vec[103], input_vec[88], input_vec[84], input_vec[139], input_vec[71], input_vec[135], input_vec[3], input_vec[146], input_vec[33], input_vec[62], input_vec[76], input_vec[131], input_vec[87], input_vec[133], input_vec[100], input_vec[86], input_vec[12], input_vec[7], input_vec[21], input_vec[49], input_vec[72], input_vec[58]};

	// Neuron 80: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_80;
	assign addr_80 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 81: 481 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_81;
	assign addr_81 = {input_vec[29], input_vec[112], input_vec[24], input_vec[60], input_vec[90], input_vec[138], input_vec[91], input_vec[105], input_vec[130], input_vec[20], input_vec[96], input_vec[108], input_vec[61], input_vec[83], input_vec[6], input_vec[55], input_vec[119], input_vec[134], input_vec[122], input_vec[138], input_vec[98], input_vec[115], input_vec[138], input_vec[96], input_vec[47], input_vec[38], input_vec[22], input_vec[126], input_vec[72], input_vec[0], input_vec[145], input_vec[89]};

	// Neuron 82: 420 entries, bits from features [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_82;
	assign addr_82 = {input_vec[81], input_vec[17], input_vec[65], input_vec[122], input_vec[56], input_vec[92], input_vec[46], input_vec[109], input_vec[42], input_vec[35], input_vec[41], input_vec[66], input_vec[25], input_vec[82], input_vec[23], input_vec[63], input_vec[43], input_vec[27], input_vec[19], input_vec[149], input_vec[48], input_vec[42], input_vec[103], input_vec[110], input_vec[24], input_vec[40], input_vec[96], input_vec[48], input_vec[143], input_vec[74], input_vec[63], input_vec[22]};

	// Neuron 83: 840 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_83;
	assign addr_83 = {input_vec[103], input_vec[24], input_vec[81], input_vec[12], input_vec[64], input_vec[26], input_vec[106], input_vec[144], input_vec[80], input_vec[1], input_vec[120], input_vec[121], input_vec[31], input_vec[86], input_vec[44], input_vec[2], input_vec[77], input_vec[29], input_vec[22], input_vec[16], input_vec[32], input_vec[150], input_vec[150], input_vec[142], input_vec[23], input_vec[98], input_vec[79], input_vec[37], input_vec[126], input_vec[129], input_vec[134], input_vec[101]};

	// Neuron 84: 1588 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_84;
	assign addr_84 = {input_vec[110], input_vec[40], input_vec[17], input_vec[106], input_vec[137], input_vec[135], input_vec[129], input_vec[33], input_vec[3], input_vec[13], input_vec[130], input_vec[70], input_vec[144], input_vec[49], input_vec[59], input_vec[78], input_vec[21], input_vec[142], input_vec[14], input_vec[145], input_vec[103], input_vec[109], input_vec[134], input_vec[114], input_vec[24], input_vec[83], input_vec[141], input_vec[37], input_vec[105], input_vec[46], input_vec[84], input_vec[66]};

	// Neuron 85: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_85;
	assign addr_85 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 86: 814 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_86;
	assign addr_86 = {input_vec[66], input_vec[3], input_vec[31], input_vec[80], input_vec[83], input_vec[68], input_vec[141], input_vec[33], input_vec[37], input_vec[51], input_vec[20], input_vec[119], input_vec[62], input_vec[112], input_vec[120], input_vec[79], input_vec[99], input_vec[151], input_vec[13], input_vec[66], input_vec[3], input_vec[101], input_vec[61], input_vec[8], input_vec[74], input_vec[64], input_vec[138], input_vec[59], input_vec[64], input_vec[28], input_vec[109], input_vec[144]};

	// Neuron 87: 927 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_87;
	assign addr_87 = {input_vec[94], input_vec[147], input_vec[106], input_vec[76], input_vec[101], input_vec[18], input_vec[50], input_vec[10], input_vec[149], input_vec[46], input_vec[147], input_vec[73], input_vec[109], input_vec[0], input_vec[7], input_vec[63], input_vec[102], input_vec[52], input_vec[78], input_vec[64], input_vec[126], input_vec[77], input_vec[148], input_vec[143], input_vec[11], input_vec[47], input_vec[115], input_vec[53], input_vec[12], input_vec[25], input_vec[96], input_vec[10]};

	// Neuron 88: 910 entries, bits from features [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_88;
	assign addr_88 = {input_vec[95], input_vec[33], input_vec[37], input_vec[77], input_vec[102], input_vec[131], input_vec[29], input_vec[57], input_vec[36], input_vec[36], input_vec[137], input_vec[135], input_vec[62], input_vec[71], input_vec[55], input_vec[1], input_vec[80], input_vec[9], input_vec[148], input_vec[134], input_vec[118], input_vec[9], input_vec[119], input_vec[145], input_vec[89], input_vec[53], input_vec[124], input_vec[145], input_vec[114], input_vec[106], input_vec[135], input_vec[87]};

	// Neuron 89: 927 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_89;
	assign addr_89 = {input_vec[94], input_vec[147], input_vec[106], input_vec[76], input_vec[101], input_vec[18], input_vec[50], input_vec[10], input_vec[149], input_vec[46], input_vec[147], input_vec[73], input_vec[109], input_vec[0], input_vec[7], input_vec[63], input_vec[102], input_vec[52], input_vec[78], input_vec[64], input_vec[126], input_vec[77], input_vec[148], input_vec[143], input_vec[11], input_vec[47], input_vec[115], input_vec[53], input_vec[12], input_vec[25], input_vec[96], input_vec[10]};

	// Neuron 90: 1291 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_90;
	assign addr_90 = {input_vec[102], input_vec[108], input_vec[96], input_vec[133], input_vec[111], input_vec[9], input_vec[151], input_vec[78], input_vec[145], input_vec[40], input_vec[39], input_vec[92], input_vec[76], input_vec[6], input_vec[35], input_vec[104], input_vec[127], input_vec[151], input_vec[146], input_vec[64], input_vec[19], input_vec[30], input_vec[139], input_vec[65], input_vec[84], input_vec[98], input_vec[111], input_vec[102], input_vec[18], input_vec[116], input_vec[119], input_vec[37]};

	// Neuron 91: 763 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_91;
	assign addr_91 = {input_vec[123], input_vec[109], input_vec[24], input_vec[17], input_vec[35], input_vec[56], input_vec[78], input_vec[103], input_vec[19], input_vec[41], input_vec[10], input_vec[36], input_vec[79], input_vec[24], input_vec[150], input_vec[96], input_vec[114], input_vec[31], input_vec[147], input_vec[0], input_vec[140], input_vec[148], input_vec[117], input_vec[42], input_vec[103], input_vec[69], input_vec[67], input_vec[146], input_vec[96], input_vec[67], input_vec[29], input_vec[84]};

	// Neuron 92: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_92;
	assign addr_92 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 93: 814 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_93;
	assign addr_93 = {input_vec[66], input_vec[3], input_vec[31], input_vec[80], input_vec[83], input_vec[68], input_vec[141], input_vec[33], input_vec[37], input_vec[51], input_vec[20], input_vec[119], input_vec[62], input_vec[112], input_vec[120], input_vec[79], input_vec[99], input_vec[151], input_vec[13], input_vec[66], input_vec[3], input_vec[101], input_vec[61], input_vec[8], input_vec[74], input_vec[64], input_vec[138], input_vec[59], input_vec[64], input_vec[28], input_vec[109], input_vec[144]};

	// Neuron 94: 656 entries, bits from features [1, 2, 3, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_94;
	assign addr_94 = {input_vec[111], input_vec[131], input_vec[15], input_vec[114], input_vec[30], input_vec[124], input_vec[130], input_vec[17], input_vec[143], input_vec[147], input_vec[144], input_vec[75], input_vec[44], input_vec[46], input_vec[23], input_vec[43], input_vec[125], input_vec[99], input_vec[122], input_vec[86], input_vec[111], input_vec[139], input_vec[130], input_vec[145], input_vec[143], input_vec[105], input_vec[82], input_vec[118], input_vec[102], input_vec[21], input_vec[122], input_vec[26]};

	// Neuron 95: 1066 entries, bits from features [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_95;
	assign addr_95 = {input_vec[50], input_vec[63], input_vec[34], input_vec[133], input_vec[70], input_vec[67], input_vec[55], input_vec[19], input_vec[66], input_vec[10], input_vec[61], input_vec[83], input_vec[138], input_vec[54], input_vec[110], input_vec[123], input_vec[16], input_vec[69], input_vec[133], input_vec[94], input_vec[8], input_vec[98], input_vec[10], input_vec[47], input_vec[130], input_vec[40], input_vec[106], input_vec[95], input_vec[123], input_vec[8], input_vec[135], input_vec[64]};

	// Neuron 96: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_96;
	assign addr_96 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 97: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_97;
	assign addr_97 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 98: 897 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_98;
	assign addr_98 = {input_vec[120], input_vec[25], input_vec[15], input_vec[115], input_vec[71], input_vec[59], input_vec[132], input_vec[125], input_vec[132], input_vec[26], input_vec[100], input_vec[90], input_vec[99], input_vec[135], input_vec[64], input_vec[36], input_vec[106], input_vec[88], input_vec[115], input_vec[53], input_vec[45], input_vec[120], input_vec[113], input_vec[94], input_vec[45], input_vec[78], input_vec[39], input_vec[57], input_vec[17], input_vec[131], input_vec[137], input_vec[117]};

	// Neuron 99: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_99;
	assign addr_99 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 100: 915 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_100;
	assign addr_100 = {input_vec[87], input_vec[76], input_vec[11], input_vec[45], input_vec[23], input_vec[19], input_vec[57], input_vec[31], input_vec[101], input_vec[29], input_vec[137], input_vec[75], input_vec[129], input_vec[62], input_vec[26], input_vec[27], input_vec[17], input_vec[70], input_vec[150], input_vec[35], input_vec[142], input_vec[98], input_vec[32], input_vec[21], input_vec[107], input_vec[17], input_vec[130], input_vec[136], input_vec[136], input_vec[38], input_vec[60], input_vec[119]};

	// Neuron 101: 735 entries, bits from features [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_101;
	assign addr_101 = {input_vec[150], input_vec[102], input_vec[11], input_vec[13], input_vec[62], input_vec[36], input_vec[38], input_vec[35], input_vec[116], input_vec[80], input_vec[38], input_vec[54], input_vec[39], input_vec[115], input_vec[123], input_vec[15], input_vec[22], input_vec[26], input_vec[9], input_vec[71], input_vec[103], input_vec[38], input_vec[59], input_vec[142], input_vec[53], input_vec[125], input_vec[113], input_vec[138], input_vec[35], input_vec[131], input_vec[79], input_vec[15]};

	// Neuron 102: 1330 entries, bits from features [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_102;
	assign addr_102 = {input_vec[76], input_vec[22], input_vec[104], input_vec[101], input_vec[151], input_vec[94], input_vec[60], input_vec[63], input_vec[35], input_vec[116], input_vec[129], input_vec[41], input_vec[123], input_vec[136], input_vec[84], input_vec[4], input_vec[85], input_vec[129], input_vec[17], input_vec[114], input_vec[71], input_vec[19], input_vec[134], input_vec[87], input_vec[72], input_vec[25], input_vec[22], input_vec[67], input_vec[135], input_vec[142], input_vec[146], input_vec[20]};

	// Neuron 103: 621 entries, bits from features [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_103;
	assign addr_103 = {input_vec[64], input_vec[145], input_vec[39], input_vec[83], input_vec[84], input_vec[73], input_vec[57], input_vec[5], input_vec[30], input_vec[80], input_vec[58], input_vec[149], input_vec[151], input_vec[67], input_vec[46], input_vec[105], input_vec[116], input_vec[101], input_vec[134], input_vec[38], input_vec[3], input_vec[126], input_vec[114], input_vec[10], input_vec[102], input_vec[120], input_vec[74], input_vec[79], input_vec[88], input_vec[125], input_vec[61], input_vec[15]};

	// Neuron 104: 823 entries, bits from features [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17]
	logic [31:0] addr_104;
	assign addr_104 = {input_vec[73], input_vec[3], input_vec[56], input_vec[128], input_vec[102], input_vec[93], input_vec[112], input_vec[102], input_vec[76], input_vec[95], input_vec[88], input_vec[87], input_vec[141], input_vec[55], input_vec[93], input_vec[42], input_vec[44], input_vec[69], input_vec[47], input_vec[60], input_vec[138], input_vec[114], input_vec[141], input_vec[14], input_vec[119], input_vec[22], input_vec[9], input_vec[0], input_vec[85], input_vec[67], input_vec[46], input_vec[46]};

	// Neuron 105: 438 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18]
	logic [31:0] addr_105;
	assign addr_105 = {input_vec[67], input_vec[88], input_vec[0], input_vec[105], input_vec[30], input_vec[62], input_vec[45], input_vec[33], input_vec[68], input_vec[42], input_vec[11], input_vec[65], input_vec[84], input_vec[27], input_vec[23], input_vec[88], input_vec[2], input_vec[90], input_vec[112], input_vec[36], input_vec[104], input_vec[53], input_vec[78], input_vec[3], input_vec[23], input_vec[122], input_vec[44], input_vec[100], input_vec[93], input_vec[148], input_vec[29], input_vec[148]};

	// Neuron 106: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_106;
	assign addr_106 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 107: 1352 entries, bits from features [1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_107;
	assign addr_107 = {input_vec[132], input_vec[115], input_vec[87], input_vec[86], input_vec[98], input_vec[28], input_vec[68], input_vec[143], input_vec[65], input_vec[117], input_vec[91], input_vec[103], input_vec[65], input_vec[124], input_vec[146], input_vec[12], input_vec[103], input_vec[13], input_vec[61], input_vec[18], input_vec[101], input_vec[67], input_vec[18], input_vec[90], input_vec[41], input_vec[107], input_vec[107], input_vec[151], input_vec[135], input_vec[22], input_vec[119], input_vec[21]};

	// Neuron 108: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_108;
	assign addr_108 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 109: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_109;
	assign addr_109 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 110: 1744 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_110;
	assign addr_110 = {input_vec[47], input_vec[21], input_vec[141], input_vec[140], input_vec[121], input_vec[9], input_vec[97], input_vec[71], input_vec[74], input_vec[111], input_vec[44], input_vec[28], input_vec[6], input_vec[18], input_vec[84], input_vec[37], input_vec[116], input_vec[102], input_vec[20], input_vec[36], input_vec[134], input_vec[78], input_vec[129], input_vec[82], input_vec[17], input_vec[6], input_vec[2], input_vec[110], input_vec[150], input_vec[98], input_vec[48], input_vec[54]};

	// Neuron 111: 254 entries, bits from features [0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_111;
	assign addr_111 = {input_vec[65], input_vec[72], input_vec[87], input_vec[2], input_vec[4], input_vec[103], input_vec[54], input_vec[60], input_vec[5], input_vec[47], input_vec[65], input_vec[139], input_vec[25], input_vec[59], input_vec[1], input_vec[130], input_vec[61], input_vec[59], input_vec[1], input_vec[130], input_vec[124], input_vec[46], input_vec[116], input_vec[27], input_vec[16], input_vec[87], input_vec[105], input_vec[118], input_vec[80], input_vec[107], input_vec[61], input_vec[119]};

	// Neuron 112: 840 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_112;
	assign addr_112 = {input_vec[103], input_vec[24], input_vec[81], input_vec[12], input_vec[64], input_vec[26], input_vec[106], input_vec[144], input_vec[80], input_vec[1], input_vec[120], input_vec[121], input_vec[31], input_vec[86], input_vec[44], input_vec[2], input_vec[77], input_vec[29], input_vec[22], input_vec[16], input_vec[32], input_vec[150], input_vec[150], input_vec[142], input_vec[23], input_vec[98], input_vec[79], input_vec[37], input_vec[126], input_vec[129], input_vec[134], input_vec[101]};

	// Neuron 113: 917 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_113;
	assign addr_113 = {input_vec[43], input_vec[65], input_vec[138], input_vec[20], input_vec[45], input_vec[77], input_vec[0], input_vec[148], input_vec[89], input_vec[100], input_vec[1], input_vec[26], input_vec[16], input_vec[59], input_vec[147], input_vec[58], input_vec[69], input_vec[4], input_vec[52], input_vec[73], input_vec[41], input_vec[104], input_vec[87], input_vec[86], input_vec[114], input_vec[52], input_vec[31], input_vec[97], input_vec[51], input_vec[9], input_vec[80], input_vec[111]};

	// Neuron 114: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_114;
	assign addr_114 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 115: 511 entries, bits from features [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_115;
	assign addr_115 = {input_vec[15], input_vec[146], input_vec[98], input_vec[128], input_vec[66], input_vec[139], input_vec[87], input_vec[113], input_vec[92], input_vec[151], input_vec[110], input_vec[60], input_vec[106], input_vec[81], input_vec[64], input_vec[74], input_vec[146], input_vec[8], input_vec[119], input_vec[95], input_vec[100], input_vec[49], input_vec[122], input_vec[145], input_vec[52], input_vec[122], input_vec[17], input_vec[120], input_vec[143], input_vec[112], input_vec[50], input_vec[122]};

	// Neuron 116: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_116;
	assign addr_116 = {input_vec[58], input_vec[81], input_vec[41], input_vec[144], input_vec[96], input_vec[14], input_vec[67], input_vec[38], input_vec[21], input_vec[137], input_vec[48], input_vec[59], input_vec[126], input_vec[75], input_vec[28], input_vec[18], input_vec[21], input_vec[48], input_vec[4], input_vec[70], input_vec[117], input_vec[27], input_vec[20], input_vec[39], input_vec[9], input_vec[77], input_vec[74], input_vec[126], input_vec[149], input_vec[113], input_vec[39], input_vec[111]};

	// Neuron 117: 232 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_117;
	assign addr_117 = {input_vec[90], input_vec[113], input_vec[48], input_vec[151], input_vec[117], input_vec[74], input_vec[15], input_vec[143], input_vec[96], input_vec[106], input_vec[44], input_vec[63], input_vec[104], input_vec[149], input_vec[19], input_vec[149], input_vec[91], input_vec[90], input_vec[47], input_vec[74], input_vec[110], input_vec[28], input_vec[112], input_vec[99], input_vec[112], input_vec[65], input_vec[112], input_vec[118], input_vec[99], input_vec[97], input_vec[149], input_vec[32]};

	// Neuron 118: 326 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18]
	logic [31:0] addr_118;
	assign addr_118 = {input_vec[141], input_vec[76], input_vec[50], input_vec[111], input_vec[64], input_vec[40], input_vec[17], input_vec[68], input_vec[147], input_vec[73], input_vec[123], input_vec[18], input_vec[8], input_vec[77], input_vec[0], input_vec[113], input_vec[0], input_vec[35], input_vec[41], input_vec[141], input_vec[57], input_vec[19], input_vec[34], input_vec[57], input_vec[63], input_vec[58], input_vec[35], input_vec[147], input_vec[20], input_vec[32], input_vec[83], input_vec[86]};

	// Neuron 119: 284 entries, bits from features [0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
	logic [31:0] addr_119;
	assign addr_119 = {input_vec[128], input_vec[54], input_vec[68], input_vec[73], input_vec[71], input_vec[136], input_vec[122], input_vec[94], input_vec[74], input_vec[39], input_vec[133], input_vec[141], input_vec[0], input_vec[68], input_vec[58], input_vec[113], input_vec[34], input_vec[63], input_vec[69], input_vec[128], input_vec[73], input_vec[90], input_vec[119], input_vec[114], input_vec[53], input_vec[42], input_vec[66], input_vec[137], input_vec[68], input_vec[86], input_vec[43], input_vec[112]};

	// Neuron 120: 460 entries, bits from features [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_120;
	assign addr_120 = {input_vec[36], input_vec[121], input_vec[35], input_vec[32], input_vec[4], input_vec[104], input_vec[102], input_vec[126], input_vec[66], input_vec[45], input_vec[111], input_vec[37], input_vec[113], input_vec[115], input_vec[148], input_vec[142], input_vec[117], input_vec[59], input_vec[80], input_vec[40], input_vec[145], input_vec[103], input_vec[2], input_vec[10], input_vec[2], input_vec[91], input_vec[118], input_vec[59], input_vec[12], input_vec[28], input_vec[79], input_vec[9]};

	// Neuron 121: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 16, 18]
	logic [31:0] addr_121;
	assign addr_121 = {input_vec[107], input_vec[111], input_vec[87], input_vec[119], input_vec[28], input_vec[91], input_vec[2], input_vec[89], input_vec[100], input_vec[132], input_vec[19], input_vec[47], input_vec[89], input_vec[25], input_vec[25], input_vec[94], input_vec[109], input_vec[48], input_vec[37], input_vec[31], input_vec[42], input_vec[38], input_vec[115], input_vec[117], input_vec[10], input_vec[24], input_vec[149], input_vec[81], input_vec[45], input_vec[81], input_vec[99], input_vec[151]};

	// Neuron 122: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_122;
	assign addr_122 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 123: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_123;
	assign addr_123 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 124: 740 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_124;
	assign addr_124 = {input_vec[73], input_vec[96], input_vec[31], input_vec[67], input_vec[140], input_vec[56], input_vec[9], input_vec[3], input_vec[28], input_vec[0], input_vec[13], input_vec[56], input_vec[126], input_vec[113], input_vec[123], input_vec[54], input_vec[94], input_vec[114], input_vec[100], input_vec[72], input_vec[22], input_vec[34], input_vec[146], input_vec[94], input_vec[106], input_vec[12], input_vec[129], input_vec[76], input_vec[13], input_vec[132], input_vec[91], input_vec[45]};

	// Neuron 125: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_125;
	assign addr_125 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 126: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_126;
	assign addr_126 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 127: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_127;
	assign addr_127 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 128: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_128;
	assign addr_128 = {input_vec[58], input_vec[81], input_vec[41], input_vec[144], input_vec[96], input_vec[14], input_vec[67], input_vec[38], input_vec[21], input_vec[137], input_vec[48], input_vec[59], input_vec[126], input_vec[75], input_vec[28], input_vec[18], input_vec[21], input_vec[48], input_vec[4], input_vec[70], input_vec[117], input_vec[27], input_vec[20], input_vec[39], input_vec[9], input_vec[77], input_vec[74], input_vec[126], input_vec[149], input_vec[113], input_vec[39], input_vec[111]};

	// Neuron 129: 917 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_129;
	assign addr_129 = {input_vec[43], input_vec[65], input_vec[138], input_vec[20], input_vec[45], input_vec[77], input_vec[0], input_vec[148], input_vec[89], input_vec[100], input_vec[1], input_vec[26], input_vec[16], input_vec[59], input_vec[147], input_vec[58], input_vec[69], input_vec[4], input_vec[52], input_vec[73], input_vec[41], input_vec[104], input_vec[87], input_vec[86], input_vec[114], input_vec[52], input_vec[31], input_vec[97], input_vec[51], input_vec[9], input_vec[80], input_vec[111]};

	// Neuron 130: 1066 entries, bits from features [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_130;
	assign addr_130 = {input_vec[50], input_vec[63], input_vec[34], input_vec[133], input_vec[70], input_vec[67], input_vec[55], input_vec[19], input_vec[66], input_vec[10], input_vec[61], input_vec[83], input_vec[138], input_vec[54], input_vec[110], input_vec[123], input_vec[16], input_vec[69], input_vec[133], input_vec[94], input_vec[8], input_vec[98], input_vec[10], input_vec[47], input_vec[130], input_vec[40], input_vec[106], input_vec[95], input_vec[123], input_vec[8], input_vec[135], input_vec[64]};

	// Neuron 131: 718 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_131;
	assign addr_131 = {input_vec[61], input_vec[65], input_vec[64], input_vec[108], input_vec[90], input_vec[86], input_vec[90], input_vec[73], input_vec[99], input_vec[24], input_vec[142], input_vec[36], input_vec[151], input_vec[149], input_vec[114], input_vec[4], input_vec[9], input_vec[73], input_vec[145], input_vec[115], input_vec[97], input_vec[139], input_vec[110], input_vec[96], input_vec[29], input_vec[41], input_vec[10], input_vec[130], input_vec[110], input_vec[129], input_vec[32], input_vec[18]};

	// Neuron 132: 546 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18]
	logic [31:0] addr_132;
	assign addr_132 = {input_vec[127], input_vec[115], input_vec[11], input_vec[122], input_vec[109], input_vec[115], input_vec[49], input_vec[6], input_vec[103], input_vec[150], input_vec[31], input_vec[117], input_vec[89], input_vec[43], input_vec[91], input_vec[74], input_vec[19], input_vec[68], input_vec[15], input_vec[123], input_vec[101], input_vec[89], input_vec[61], input_vec[78], input_vec[72], input_vec[55], input_vec[31], input_vec[97], input_vec[80], input_vec[34], input_vec[114], input_vec[126]};

	// Neuron 133: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_133;
	assign addr_133 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 134: 718 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_134;
	assign addr_134 = {input_vec[61], input_vec[65], input_vec[64], input_vec[108], input_vec[90], input_vec[86], input_vec[90], input_vec[73], input_vec[99], input_vec[24], input_vec[142], input_vec[36], input_vec[151], input_vec[149], input_vec[114], input_vec[4], input_vec[9], input_vec[73], input_vec[145], input_vec[115], input_vec[97], input_vec[139], input_vec[110], input_vec[96], input_vec[29], input_vec[41], input_vec[10], input_vec[130], input_vec[110], input_vec[129], input_vec[32], input_vec[18]};

	// Neuron 135: 957 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18]
	logic [31:0] addr_135;
	assign addr_135 = {input_vec[133], input_vec[26], input_vec[62], input_vec[147], input_vec[79], input_vec[114], input_vec[139], input_vec[144], input_vec[142], input_vec[36], input_vec[65], input_vec[7], input_vec[3], input_vec[151], input_vec[21], input_vec[50], input_vec[123], input_vec[55], input_vec[9], input_vec[62], input_vec[82], input_vec[74], input_vec[63], input_vec[89], input_vec[6], input_vec[122], input_vec[46], input_vec[14], input_vec[140], input_vec[27], input_vec[115], input_vec[70]};

	// Neuron 136: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_136;
	assign addr_136 = {input_vec[58], input_vec[81], input_vec[41], input_vec[144], input_vec[96], input_vec[14], input_vec[67], input_vec[38], input_vec[21], input_vec[137], input_vec[48], input_vec[59], input_vec[126], input_vec[75], input_vec[28], input_vec[18], input_vec[21], input_vec[48], input_vec[4], input_vec[70], input_vec[117], input_vec[27], input_vec[20], input_vec[39], input_vec[9], input_vec[77], input_vec[74], input_vec[126], input_vec[149], input_vec[113], input_vec[39], input_vec[111]};

	// Neuron 137: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_137;
	assign addr_137 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 138: 447 entries, bits from features [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_138;
	assign addr_138 = {input_vec[104], input_vec[35], input_vec[31], input_vec[62], input_vec[63], input_vec[115], input_vec[7], input_vec[82], input_vec[128], input_vec[118], input_vec[126], input_vec[17], input_vec[22], input_vec[61], input_vec[71], input_vec[94], input_vec[5], input_vec[149], input_vec[138], input_vec[81], input_vec[81], input_vec[127], input_vec[45], input_vec[115], input_vec[59], input_vec[104], input_vec[100], input_vec[116], input_vec[4], input_vec[5], input_vec[17], input_vec[43]};

	// Neuron 139: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_139;
	assign addr_139 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 140: 447 entries, bits from features [0, 1, 2, 3, 4, 8, 9, 10, 11, 13, 14, 15, 17]
	logic [31:0] addr_140;
	assign addr_140 = {input_vec[6], input_vec[141], input_vec[80], input_vec[70], input_vec[68], input_vec[30], input_vec[71], input_vec[122], input_vec[16], input_vec[91], input_vec[33], input_vec[7], input_vec[10], input_vec[117], input_vec[104], input_vec[35], input_vec[30], input_vec[91], input_vec[38], input_vec[85], input_vec[22], input_vec[36], input_vec[113], input_vec[68], input_vec[121], input_vec[90], input_vec[77], input_vec[64], input_vec[33], input_vec[89], input_vec[26], input_vec[82]};

	// Neuron 141: 420 entries, bits from features [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_141;
	assign addr_141 = {input_vec[81], input_vec[17], input_vec[65], input_vec[122], input_vec[56], input_vec[92], input_vec[46], input_vec[109], input_vec[42], input_vec[35], input_vec[41], input_vec[66], input_vec[25], input_vec[82], input_vec[23], input_vec[63], input_vec[43], input_vec[27], input_vec[19], input_vec[149], input_vec[48], input_vec[42], input_vec[103], input_vec[110], input_vec[24], input_vec[40], input_vec[96], input_vec[48], input_vec[143], input_vec[74], input_vec[63], input_vec[22]};

	// Neuron 142: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_142;
	assign addr_142 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 143: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_143;
	assign addr_143 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 144: 511 entries, bits from features [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_144;
	assign addr_144 = {input_vec[15], input_vec[146], input_vec[98], input_vec[128], input_vec[66], input_vec[139], input_vec[87], input_vec[113], input_vec[92], input_vec[151], input_vec[110], input_vec[60], input_vec[106], input_vec[81], input_vec[64], input_vec[74], input_vec[146], input_vec[8], input_vec[119], input_vec[95], input_vec[100], input_vec[49], input_vec[122], input_vec[145], input_vec[52], input_vec[122], input_vec[17], input_vec[120], input_vec[143], input_vec[112], input_vec[50], input_vec[122]};

	// Neuron 145: 232 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_145;
	assign addr_145 = {input_vec[90], input_vec[113], input_vec[48], input_vec[151], input_vec[117], input_vec[74], input_vec[15], input_vec[143], input_vec[96], input_vec[106], input_vec[44], input_vec[63], input_vec[104], input_vec[149], input_vec[19], input_vec[149], input_vec[91], input_vec[90], input_vec[47], input_vec[74], input_vec[110], input_vec[28], input_vec[112], input_vec[99], input_vec[112], input_vec[65], input_vec[112], input_vec[118], input_vec[99], input_vec[97], input_vec[149], input_vec[32]};

	// Neuron 146: 284 entries, bits from features [0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
	logic [31:0] addr_146;
	assign addr_146 = {input_vec[128], input_vec[54], input_vec[68], input_vec[73], input_vec[71], input_vec[136], input_vec[122], input_vec[94], input_vec[74], input_vec[39], input_vec[133], input_vec[141], input_vec[0], input_vec[68], input_vec[58], input_vec[113], input_vec[34], input_vec[63], input_vec[69], input_vec[128], input_vec[73], input_vec[90], input_vec[119], input_vec[114], input_vec[53], input_vec[42], input_vec[66], input_vec[137], input_vec[68], input_vec[86], input_vec[43], input_vec[112]};

	// Neuron 147: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_147;
	assign addr_147 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 148: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_148;
	assign addr_148 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 149: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_149;
	assign addr_149 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 150: 764 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18]
	logic [31:0] addr_150;
	assign addr_150 = {input_vec[7], input_vec[29], input_vec[113], input_vec[86], input_vec[116], input_vec[109], input_vec[58], input_vec[22], input_vec[86], input_vec[76], input_vec[65], input_vec[2], input_vec[106], input_vec[98], input_vec[70], input_vec[37], input_vec[115], input_vec[51], input_vec[66], input_vec[1], input_vec[108], input_vec[144], input_vec[83], input_vec[12], input_vec[29], input_vec[28], input_vec[145], input_vec[43], input_vec[68], input_vec[127], input_vec[105], input_vec[76]};

	// Neuron 151: 522 entries, bits from features [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_151;
	assign addr_151 = {input_vec[52], input_vec[143], input_vec[103], input_vec[131], input_vec[115], input_vec[97], input_vec[28], input_vec[124], input_vec[99], input_vec[32], input_vec[139], input_vec[86], input_vec[12], input_vec[131], input_vec[114], input_vec[37], input_vec[96], input_vec[67], input_vec[74], input_vec[90], input_vec[150], input_vec[99], input_vec[33], input_vec[120], input_vec[147], input_vec[91], input_vec[78], input_vec[124], input_vec[112], input_vec[42], input_vec[124], input_vec[100]};

	// Neuron 152: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_152;
	assign addr_152 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 153: 460 entries, bits from features [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_153;
	assign addr_153 = {input_vec[36], input_vec[121], input_vec[35], input_vec[32], input_vec[4], input_vec[104], input_vec[102], input_vec[126], input_vec[66], input_vec[45], input_vec[111], input_vec[37], input_vec[113], input_vec[115], input_vec[148], input_vec[142], input_vec[117], input_vec[59], input_vec[80], input_vec[40], input_vec[145], input_vec[103], input_vec[2], input_vec[10], input_vec[2], input_vec[91], input_vec[118], input_vec[59], input_vec[12], input_vec[28], input_vec[79], input_vec[9]};

	// Neuron 154: 438 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18]
	logic [31:0] addr_154;
	assign addr_154 = {input_vec[67], input_vec[88], input_vec[0], input_vec[105], input_vec[30], input_vec[62], input_vec[45], input_vec[33], input_vec[68], input_vec[42], input_vec[11], input_vec[65], input_vec[84], input_vec[27], input_vec[23], input_vec[88], input_vec[2], input_vec[90], input_vec[112], input_vec[36], input_vec[104], input_vec[53], input_vec[78], input_vec[3], input_vec[23], input_vec[122], input_vec[44], input_vec[100], input_vec[93], input_vec[148], input_vec[29], input_vec[148]};

	// Neuron 155: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_155;
	assign addr_155 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 156: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_156;
	assign addr_156 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 157: 271 entries, bits from features [3, 5, 6, 7, 10, 11, 13, 15, 16, 17, 18]
	logic [31:0] addr_157;
	assign addr_157 = {input_vec[48], input_vec[138], input_vec[108], input_vec[139], input_vec[27], input_vec[55], input_vec[123], input_vec[124], input_vec[93], input_vec[136], input_vec[130], input_vec[110], input_vec[121], input_vec[46], input_vec[149], input_vec[140], input_vec[46], input_vec[91], input_vec[62], input_vec[92], input_vec[52], input_vec[140], input_vec[139], input_vec[93], input_vec[29], input_vec[27], input_vec[126], input_vec[144], input_vec[53], input_vec[106], input_vec[93], input_vec[81]};

	// Neuron 158: 840 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_158;
	assign addr_158 = {input_vec[103], input_vec[24], input_vec[81], input_vec[12], input_vec[64], input_vec[26], input_vec[106], input_vec[144], input_vec[80], input_vec[1], input_vec[120], input_vec[121], input_vec[31], input_vec[86], input_vec[44], input_vec[2], input_vec[77], input_vec[29], input_vec[22], input_vec[16], input_vec[32], input_vec[150], input_vec[150], input_vec[142], input_vec[23], input_vec[98], input_vec[79], input_vec[37], input_vec[126], input_vec[129], input_vec[134], input_vec[101]};

	// Neuron 159: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_159;
	assign addr_159 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 160: 763 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_160;
	assign addr_160 = {input_vec[123], input_vec[109], input_vec[24], input_vec[17], input_vec[35], input_vec[56], input_vec[78], input_vec[103], input_vec[19], input_vec[41], input_vec[10], input_vec[36], input_vec[79], input_vec[24], input_vec[150], input_vec[96], input_vec[114], input_vec[31], input_vec[147], input_vec[0], input_vec[140], input_vec[148], input_vec[117], input_vec[42], input_vec[103], input_vec[69], input_vec[67], input_vec[146], input_vec[96], input_vec[67], input_vec[29], input_vec[84]};

	// Neuron 161: 764 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18]
	logic [31:0] addr_161;
	assign addr_161 = {input_vec[7], input_vec[29], input_vec[113], input_vec[86], input_vec[116], input_vec[109], input_vec[58], input_vec[22], input_vec[86], input_vec[76], input_vec[65], input_vec[2], input_vec[106], input_vec[98], input_vec[70], input_vec[37], input_vec[115], input_vec[51], input_vec[66], input_vec[1], input_vec[108], input_vec[144], input_vec[83], input_vec[12], input_vec[29], input_vec[28], input_vec[145], input_vec[43], input_vec[68], input_vec[127], input_vec[105], input_vec[76]};

	// Neuron 162: 383 entries, bits from features [0, 2, 3, 5, 6, 7, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_162;
	assign addr_162 = {input_vec[120], input_vec[142], input_vec[144], input_vec[81], input_vec[148], input_vec[52], input_vec[30], input_vec[44], input_vec[58], input_vec[118], input_vec[62], input_vec[4], input_vec[124], input_vec[107], input_vec[145], input_vec[63], input_vec[19], input_vec[5], input_vec[22], input_vec[59], input_vec[2], input_vec[148], input_vec[23], input_vec[144], input_vec[58], input_vec[138], input_vec[81], input_vec[41], input_vec[124], input_vec[22], input_vec[102], input_vec[54]};

	// Neuron 163: 943 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]
	logic [31:0] addr_163;
	assign addr_163 = {input_vec[103], input_vec[19], input_vec[93], input_vec[70], input_vec[33], input_vec[123], input_vec[99], input_vec[11], input_vec[48], input_vec[132], input_vec[130], input_vec[112], input_vec[113], input_vec[70], input_vec[56], input_vec[81], input_vec[20], input_vec[63], input_vec[83], input_vec[56], input_vec[52], input_vec[25], input_vec[85], input_vec[119], input_vec[86], input_vec[74], input_vec[7], input_vec[70], input_vec[101], input_vec[57], input_vec[98], input_vec[142]};

	// Neuron 164: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_164;
	assign addr_164 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 165: 840 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_165;
	assign addr_165 = {input_vec[103], input_vec[24], input_vec[81], input_vec[12], input_vec[64], input_vec[26], input_vec[106], input_vec[144], input_vec[80], input_vec[1], input_vec[120], input_vec[121], input_vec[31], input_vec[86], input_vec[44], input_vec[2], input_vec[77], input_vec[29], input_vec[22], input_vec[16], input_vec[32], input_vec[150], input_vec[150], input_vec[142], input_vec[23], input_vec[98], input_vec[79], input_vec[37], input_vec[126], input_vec[129], input_vec[134], input_vec[101]};

	// Neuron 166: 546 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18]
	logic [31:0] addr_166;
	assign addr_166 = {input_vec[127], input_vec[115], input_vec[11], input_vec[122], input_vec[109], input_vec[115], input_vec[49], input_vec[6], input_vec[103], input_vec[150], input_vec[31], input_vec[117], input_vec[89], input_vec[43], input_vec[91], input_vec[74], input_vec[19], input_vec[68], input_vec[15], input_vec[123], input_vec[101], input_vec[89], input_vec[61], input_vec[78], input_vec[72], input_vec[55], input_vec[31], input_vec[97], input_vec[80], input_vec[34], input_vec[114], input_vec[126]};

	// Neuron 167: 1102 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_167;
	assign addr_167 = {input_vec[112], input_vec[72], input_vec[102], input_vec[71], input_vec[89], input_vec[96], input_vec[79], input_vec[134], input_vec[64], input_vec[78], input_vec[9], input_vec[150], input_vec[48], input_vec[54], input_vec[2], input_vec[126], input_vec[88], input_vec[94], input_vec[113], input_vec[116], input_vec[10], input_vec[132], input_vec[141], input_vec[114], input_vec[67], input_vec[125], input_vec[136], input_vec[37], input_vec[19], input_vec[26], input_vec[134], input_vec[19]};

	// Neuron 168: 254 entries, bits from features [0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_168;
	assign addr_168 = {input_vec[65], input_vec[72], input_vec[87], input_vec[2], input_vec[4], input_vec[103], input_vec[54], input_vec[60], input_vec[5], input_vec[47], input_vec[65], input_vec[139], input_vec[25], input_vec[59], input_vec[1], input_vec[130], input_vec[61], input_vec[59], input_vec[1], input_vec[130], input_vec[124], input_vec[46], input_vec[116], input_vec[27], input_vec[16], input_vec[87], input_vec[105], input_vec[118], input_vec[80], input_vec[107], input_vec[61], input_vec[119]};

	// Neuron 169: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_169;
	assign addr_169 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 170: 669 entries, bits from features [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_170;
	assign addr_170 = {input_vec[5], input_vec[114], input_vec[56], input_vec[30], input_vec[100], input_vec[55], input_vec[11], input_vec[67], input_vec[37], input_vec[97], input_vec[108], input_vec[12], input_vec[126], input_vec[8], input_vec[139], input_vec[145], input_vec[91], input_vec[129], input_vec[84], input_vec[52], input_vec[0], input_vec[74], input_vec[2], input_vec[95], input_vec[11], input_vec[99], input_vec[39], input_vec[121], input_vec[146], input_vec[60], input_vec[88], input_vec[104]};

	// Neuron 171: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_171;
	assign addr_171 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 172: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_172;
	assign addr_172 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 173: 621 entries, bits from features [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_173;
	assign addr_173 = {input_vec[64], input_vec[145], input_vec[39], input_vec[83], input_vec[84], input_vec[73], input_vec[57], input_vec[5], input_vec[30], input_vec[80], input_vec[58], input_vec[149], input_vec[151], input_vec[67], input_vec[46], input_vec[105], input_vec[116], input_vec[101], input_vec[134], input_vec[38], input_vec[3], input_vec[126], input_vec[114], input_vec[10], input_vec[102], input_vec[120], input_vec[74], input_vec[79], input_vec[88], input_vec[125], input_vec[61], input_vec[15]};

	// Neuron 174: 897 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_174;
	assign addr_174 = {input_vec[120], input_vec[25], input_vec[15], input_vec[115], input_vec[71], input_vec[59], input_vec[132], input_vec[125], input_vec[132], input_vec[26], input_vec[100], input_vec[90], input_vec[99], input_vec[135], input_vec[64], input_vec[36], input_vec[106], input_vec[88], input_vec[115], input_vec[53], input_vec[45], input_vec[120], input_vec[113], input_vec[94], input_vec[45], input_vec[78], input_vec[39], input_vec[57], input_vec[17], input_vec[131], input_vec[137], input_vec[117]};

	// Neuron 175: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 16, 18]
	logic [31:0] addr_175;
	assign addr_175 = {input_vec[107], input_vec[111], input_vec[87], input_vec[119], input_vec[28], input_vec[91], input_vec[2], input_vec[89], input_vec[100], input_vec[132], input_vec[19], input_vec[47], input_vec[89], input_vec[25], input_vec[25], input_vec[94], input_vec[109], input_vec[48], input_vec[37], input_vec[31], input_vec[42], input_vec[38], input_vec[115], input_vec[117], input_vec[10], input_vec[24], input_vec[149], input_vec[81], input_vec[45], input_vec[81], input_vec[99], input_vec[151]};

	// Neuron 176: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_176;
	assign addr_176 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 177: 730 entries, bits from features [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17]
	logic [31:0] addr_177;
	assign addr_177 = {input_vec[47], input_vec[65], input_vec[42], input_vec[83], input_vec[103], input_vec[136], input_vec[46], input_vec[12], input_vec[103], input_vec[125], input_vec[138], input_vec[121], input_vec[80], input_vec[100], input_vec[101], input_vec[36], input_vec[22], input_vec[9], input_vec[81], input_vec[24], input_vec[129], input_vec[30], input_vec[48], input_vec[78], input_vec[9], input_vec[10], input_vec[33], input_vec[67], input_vec[139], input_vec[31], input_vec[17], input_vec[25]};

	// Neuron 178: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_178;
	assign addr_178 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 179: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_179;
	assign addr_179 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 180: 698 entries, bits from features [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 18]
	logic [31:0] addr_180;
	assign addr_180 = {input_vec[83], input_vec[54], input_vec[151], input_vec[26], input_vec[76], input_vec[31], input_vec[44], input_vec[68], input_vec[107], input_vec[114], input_vec[108], input_vec[116], input_vec[5], input_vec[23], input_vec[19], input_vec[53], input_vec[54], input_vec[74], input_vec[51], input_vec[90], input_vec[21], input_vec[78], input_vec[87], input_vec[145], input_vec[145], input_vec[56], input_vec[3], input_vec[3], input_vec[104], input_vec[56], input_vec[150], input_vec[55]};

	// Neuron 181: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_181;
	assign addr_181 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 182: 479 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_182;
	assign addr_182 = {input_vec[57], input_vec[12], input_vec[65], input_vec[65], input_vec[120], input_vec[69], input_vec[82], input_vec[139], input_vec[131], input_vec[82], input_vec[130], input_vec[37], input_vec[19], input_vec[35], input_vec[48], input_vec[28], input_vec[43], input_vec[74], input_vec[38], input_vec[20], input_vec[150], input_vec[127], input_vec[42], input_vec[61], input_vec[105], input_vec[145], input_vec[116], input_vec[68], input_vec[52], input_vec[118], input_vec[29], input_vec[129]};

	// Neuron 183: 1352 entries, bits from features [1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_183;
	assign addr_183 = {input_vec[132], input_vec[115], input_vec[87], input_vec[86], input_vec[98], input_vec[28], input_vec[68], input_vec[143], input_vec[65], input_vec[117], input_vec[91], input_vec[103], input_vec[65], input_vec[124], input_vec[146], input_vec[12], input_vec[103], input_vec[13], input_vec[61], input_vec[18], input_vec[101], input_vec[67], input_vec[18], input_vec[90], input_vec[41], input_vec[107], input_vec[107], input_vec[151], input_vec[135], input_vec[22], input_vec[119], input_vec[21]};

	// Neuron 184: 326 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18]
	logic [31:0] addr_184;
	assign addr_184 = {input_vec[141], input_vec[76], input_vec[50], input_vec[111], input_vec[64], input_vec[40], input_vec[17], input_vec[68], input_vec[147], input_vec[73], input_vec[123], input_vec[18], input_vec[8], input_vec[77], input_vec[0], input_vec[113], input_vec[0], input_vec[35], input_vec[41], input_vec[141], input_vec[57], input_vec[19], input_vec[34], input_vec[57], input_vec[63], input_vec[58], input_vec[35], input_vec[147], input_vec[20], input_vec[32], input_vec[83], input_vec[86]};

	// Neuron 185: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_185;
	assign addr_185 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 186: 1291 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_186;
	assign addr_186 = {input_vec[102], input_vec[108], input_vec[96], input_vec[133], input_vec[111], input_vec[9], input_vec[151], input_vec[78], input_vec[145], input_vec[40], input_vec[39], input_vec[92], input_vec[76], input_vec[6], input_vec[35], input_vec[104], input_vec[127], input_vec[151], input_vec[146], input_vec[64], input_vec[19], input_vec[30], input_vec[139], input_vec[65], input_vec[84], input_vec[98], input_vec[111], input_vec[102], input_vec[18], input_vec[116], input_vec[119], input_vec[37]};

	// Neuron 187: 730 entries, bits from features [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17]
	logic [31:0] addr_187;
	assign addr_187 = {input_vec[47], input_vec[65], input_vec[42], input_vec[83], input_vec[103], input_vec[136], input_vec[46], input_vec[12], input_vec[103], input_vec[125], input_vec[138], input_vec[121], input_vec[80], input_vec[100], input_vec[101], input_vec[36], input_vec[22], input_vec[9], input_vec[81], input_vec[24], input_vec[129], input_vec[30], input_vec[48], input_vec[78], input_vec[9], input_vec[10], input_vec[33], input_vec[67], input_vec[139], input_vec[31], input_vec[17], input_vec[25]};

	// Neuron 188: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_188;
	assign addr_188 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 189: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_189;
	assign addr_189 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 190: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_190;
	assign addr_190 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 191: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_191;
	assign addr_191 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 192: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_192;
	assign addr_192 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 193: 511 entries, bits from features [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_193;
	assign addr_193 = {input_vec[15], input_vec[146], input_vec[98], input_vec[128], input_vec[66], input_vec[139], input_vec[87], input_vec[113], input_vec[92], input_vec[151], input_vec[110], input_vec[60], input_vec[106], input_vec[81], input_vec[64], input_vec[74], input_vec[146], input_vec[8], input_vec[119], input_vec[95], input_vec[100], input_vec[49], input_vec[122], input_vec[145], input_vec[52], input_vec[122], input_vec[17], input_vec[120], input_vec[143], input_vec[112], input_vec[50], input_vec[122]};

	// Neuron 194: 1352 entries, bits from features [1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_194;
	assign addr_194 = {input_vec[132], input_vec[115], input_vec[87], input_vec[86], input_vec[98], input_vec[28], input_vec[68], input_vec[143], input_vec[65], input_vec[117], input_vec[91], input_vec[103], input_vec[65], input_vec[124], input_vec[146], input_vec[12], input_vec[103], input_vec[13], input_vec[61], input_vec[18], input_vec[101], input_vec[67], input_vec[18], input_vec[90], input_vec[41], input_vec[107], input_vec[107], input_vec[151], input_vec[135], input_vec[22], input_vec[119], input_vec[21]};

	// Neuron 195: 630 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_195;
	assign addr_195 = {input_vec[31], input_vec[77], input_vec[55], input_vec[90], input_vec[48], input_vec[41], input_vec[5], input_vec[75], input_vec[114], input_vec[44], input_vec[55], input_vec[36], input_vec[25], input_vec[151], input_vec[63], input_vec[100], input_vec[110], input_vec[103], input_vec[14], input_vec[106], input_vec[71], input_vec[138], input_vec[72], input_vec[145], input_vec[58], input_vec[149], input_vec[73], input_vec[31], input_vec[91], input_vec[85], input_vec[92], input_vec[135]};

	// Neuron 196: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_196;
	assign addr_196 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 197: 420 entries, bits from features [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_197;
	assign addr_197 = {input_vec[81], input_vec[17], input_vec[65], input_vec[122], input_vec[56], input_vec[92], input_vec[46], input_vec[109], input_vec[42], input_vec[35], input_vec[41], input_vec[66], input_vec[25], input_vec[82], input_vec[23], input_vec[63], input_vec[43], input_vec[27], input_vec[19], input_vec[149], input_vec[48], input_vec[42], input_vec[103], input_vec[110], input_vec[24], input_vec[40], input_vec[96], input_vec[48], input_vec[143], input_vec[74], input_vec[63], input_vec[22]};

	// Neuron 198: 740 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_198;
	assign addr_198 = {input_vec[73], input_vec[96], input_vec[31], input_vec[67], input_vec[140], input_vec[56], input_vec[9], input_vec[3], input_vec[28], input_vec[0], input_vec[13], input_vec[56], input_vec[126], input_vec[113], input_vec[123], input_vec[54], input_vec[94], input_vec[114], input_vec[100], input_vec[72], input_vec[22], input_vec[34], input_vec[146], input_vec[94], input_vec[106], input_vec[12], input_vec[129], input_vec[76], input_vec[13], input_vec[132], input_vec[91], input_vec[45]};

	// Neuron 199: 415 entries, bits from features [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_199;
	assign addr_199 = {input_vec[96], input_vec[105], input_vec[105], input_vec[75], input_vec[63], input_vec[112], input_vec[63], input_vec[20], input_vec[149], input_vec[133], input_vec[26], input_vec[73], input_vec[67], input_vec[60], input_vec[122], input_vec[121], input_vec[52], input_vec[89], input_vec[81], input_vec[99], input_vec[29], input_vec[129], input_vec[16], input_vec[146], input_vec[106], input_vec[76], input_vec[129], input_vec[143], input_vec[129], input_vec[3], input_vec[61], input_vec[137]};

	// Neuron 200: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_200;
	assign addr_200 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 201: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_201;
	assign addr_201 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 202: 892 entries, bits from features [0, 1, 2, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_202;
	assign addr_202 = {input_vec[100], input_vec[2], input_vec[130], input_vec[139], input_vec[129], input_vec[144], input_vec[114], input_vec[127], input_vec[133], input_vec[88], input_vec[17], input_vec[45], input_vec[48], input_vec[60], input_vec[93], input_vec[13], input_vec[131], input_vec[103], input_vec[146], input_vec[139], input_vec[113], input_vec[55], input_vec[2], input_vec[61], input_vec[111], input_vec[104], input_vec[123], input_vec[149], input_vec[21], input_vec[88], input_vec[14], input_vec[120]};

	// Neuron 203: 631 entries, bits from features [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_203;
	assign addr_203 = {input_vec[144], input_vec[23], input_vec[105], input_vec[95], input_vec[87], input_vec[143], input_vec[122], input_vec[101], input_vec[96], input_vec[1], input_vec[147], input_vec[118], input_vec[36], input_vec[2], input_vec[0], input_vec[29], input_vec[142], input_vec[91], input_vec[88], input_vec[113], input_vec[89], input_vec[122], input_vec[79], input_vec[47], input_vec[53], input_vec[99], input_vec[111], input_vec[99], input_vec[73], input_vec[44], input_vec[128], input_vec[33]};

	// Neuron 204: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_204;
	assign addr_204 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 205: 407 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_205;
	assign addr_205 = {input_vec[67], input_vec[67], input_vec[146], input_vec[92], input_vec[84], input_vec[36], input_vec[130], input_vec[124], input_vec[8], input_vec[141], input_vec[133], input_vec[81], input_vec[41], input_vec[87], input_vec[49], input_vec[148], input_vec[56], input_vec[33], input_vec[56], input_vec[130], input_vec[129], input_vec[141], input_vec[44], input_vec[80], input_vec[113], input_vec[30], input_vec[42], input_vec[20], input_vec[96], input_vec[34], input_vec[45], input_vec[70]};

	// Neuron 206: 553 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18]
	logic [31:0] addr_206;
	assign addr_206 = {input_vec[12], input_vec[56], input_vec[48], input_vec[23], input_vec[15], input_vec[43], input_vec[72], input_vec[10], input_vec[77], input_vec[128], input_vec[127], input_vec[79], input_vec[148], input_vec[102], input_vec[13], input_vec[58], input_vec[131], input_vec[62], input_vec[145], input_vec[10], input_vec[39], input_vec[17], input_vec[58], input_vec[58], input_vec[27], input_vec[64], input_vec[147], input_vec[24], input_vec[79], input_vec[27], input_vec[100], input_vec[150]};

	// Neuron 207: 1756 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]
	logic [31:0] addr_207;
	assign addr_207 = {input_vec[44], input_vec[101], input_vec[56], input_vec[69], input_vec[46], input_vec[93], input_vec[128], input_vec[29], input_vec[96], input_vec[41], input_vec[103], input_vec[88], input_vec[84], input_vec[139], input_vec[71], input_vec[135], input_vec[3], input_vec[146], input_vec[33], input_vec[62], input_vec[76], input_vec[131], input_vec[87], input_vec[133], input_vec[100], input_vec[86], input_vec[12], input_vec[7], input_vec[21], input_vec[49], input_vec[72], input_vec[58]};

	// Neuron 208: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_208;
	assign addr_208 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 209: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_209;
	assign addr_209 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 210: 479 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_210;
	assign addr_210 = {input_vec[57], input_vec[12], input_vec[65], input_vec[65], input_vec[120], input_vec[69], input_vec[82], input_vec[139], input_vec[131], input_vec[82], input_vec[130], input_vec[37], input_vec[19], input_vec[35], input_vec[48], input_vec[28], input_vec[43], input_vec[74], input_vec[38], input_vec[20], input_vec[150], input_vec[127], input_vec[42], input_vec[61], input_vec[105], input_vec[145], input_vec[116], input_vec[68], input_vec[52], input_vec[118], input_vec[29], input_vec[129]};

	// Neuron 211: 787 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 14, 15, 16, 17]
	logic [31:0] addr_211;
	assign addr_211 = {input_vec[59], input_vec[140], input_vec[98], input_vec[60], input_vec[64], input_vec[38], input_vec[5], input_vec[133], input_vec[98], input_vec[46], input_vec[4], input_vec[31], input_vec[124], input_vec[18], input_vec[129], input_vec[27], input_vec[116], input_vec[35], input_vec[132], input_vec[19], input_vec[77], input_vec[64], input_vec[59], input_vec[9], input_vec[98], input_vec[42], input_vec[138], input_vec[35], input_vec[132], input_vec[69], input_vec[33], input_vec[100]};

	// Neuron 212: 447 entries, bits from features [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_212;
	assign addr_212 = {input_vec[104], input_vec[35], input_vec[31], input_vec[62], input_vec[63], input_vec[115], input_vec[7], input_vec[82], input_vec[128], input_vec[118], input_vec[126], input_vec[17], input_vec[22], input_vec[61], input_vec[71], input_vec[94], input_vec[5], input_vec[149], input_vec[138], input_vec[81], input_vec[81], input_vec[127], input_vec[45], input_vec[115], input_vec[59], input_vec[104], input_vec[100], input_vec[116], input_vec[4], input_vec[5], input_vec[17], input_vec[43]};

	// Neuron 213: 630 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_213;
	assign addr_213 = {input_vec[31], input_vec[77], input_vec[55], input_vec[90], input_vec[48], input_vec[41], input_vec[5], input_vec[75], input_vec[114], input_vec[44], input_vec[55], input_vec[36], input_vec[25], input_vec[151], input_vec[63], input_vec[100], input_vec[110], input_vec[103], input_vec[14], input_vec[106], input_vec[71], input_vec[138], input_vec[72], input_vec[145], input_vec[58], input_vec[149], input_vec[73], input_vec[31], input_vec[91], input_vec[85], input_vec[92], input_vec[135]};

	// Neuron 214: 917 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_214;
	assign addr_214 = {input_vec[43], input_vec[65], input_vec[138], input_vec[20], input_vec[45], input_vec[77], input_vec[0], input_vec[148], input_vec[89], input_vec[100], input_vec[1], input_vec[26], input_vec[16], input_vec[59], input_vec[147], input_vec[58], input_vec[69], input_vec[4], input_vec[52], input_vec[73], input_vec[41], input_vec[104], input_vec[87], input_vec[86], input_vec[114], input_vec[52], input_vec[31], input_vec[97], input_vec[51], input_vec[9], input_vec[80], input_vec[111]};

	// Neuron 215: 1109 entries, bits from features [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_215;
	assign addr_215 = {input_vec[96], input_vec[48], input_vec[114], input_vec[116], input_vec[1], input_vec[88], input_vec[23], input_vec[144], input_vec[13], input_vec[94], input_vec[31], input_vec[11], input_vec[100], input_vec[73], input_vec[128], input_vec[123], input_vec[83], input_vec[64], input_vec[135], input_vec[106], input_vec[20], input_vec[40], input_vec[136], input_vec[116], input_vec[10], input_vec[21], input_vec[6], input_vec[69], input_vec[120], input_vec[148], input_vec[151], input_vec[41]};

	// Neuron 216: 897 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_216;
	assign addr_216 = {input_vec[120], input_vec[25], input_vec[15], input_vec[115], input_vec[71], input_vec[59], input_vec[132], input_vec[125], input_vec[132], input_vec[26], input_vec[100], input_vec[90], input_vec[99], input_vec[135], input_vec[64], input_vec[36], input_vec[106], input_vec[88], input_vec[115], input_vec[53], input_vec[45], input_vec[120], input_vec[113], input_vec[94], input_vec[45], input_vec[78], input_vec[39], input_vec[57], input_vec[17], input_vec[131], input_vec[137], input_vec[117]};

	// Neuron 217: 545 entries, bits from features [1, 2, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_217;
	assign addr_217 = {input_vec[132], input_vec[97], input_vec[34], input_vec[118], input_vec[96], input_vec[18], input_vec[82], input_vec[34], input_vec[134], input_vec[147], input_vec[132], input_vec[137], input_vec[80], input_vec[41], input_vec[56], input_vec[55], input_vec[116], input_vec[133], input_vec[138], input_vec[107], input_vec[8], input_vec[146], input_vec[57], input_vec[86], input_vec[19], input_vec[133], input_vec[124], input_vec[129], input_vec[48], input_vec[123], input_vec[145], input_vec[57]};

	// Neuron 218: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_218;
	assign addr_218 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 219: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_219;
	assign addr_219 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 220: 840 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_220;
	assign addr_220 = {input_vec[103], input_vec[24], input_vec[81], input_vec[12], input_vec[64], input_vec[26], input_vec[106], input_vec[144], input_vec[80], input_vec[1], input_vec[120], input_vec[121], input_vec[31], input_vec[86], input_vec[44], input_vec[2], input_vec[77], input_vec[29], input_vec[22], input_vec[16], input_vec[32], input_vec[150], input_vec[150], input_vec[142], input_vec[23], input_vec[98], input_vec[79], input_vec[37], input_vec[126], input_vec[129], input_vec[134], input_vec[101]};

	// Neuron 221: 1588 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_221;
	assign addr_221 = {input_vec[110], input_vec[40], input_vec[17], input_vec[106], input_vec[137], input_vec[135], input_vec[129], input_vec[33], input_vec[3], input_vec[13], input_vec[130], input_vec[70], input_vec[144], input_vec[49], input_vec[59], input_vec[78], input_vec[21], input_vec[142], input_vec[14], input_vec[145], input_vec[103], input_vec[109], input_vec[134], input_vec[114], input_vec[24], input_vec[83], input_vec[141], input_vec[37], input_vec[105], input_vec[46], input_vec[84], input_vec[66]};

	// Neuron 222: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_222;
	assign addr_222 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 223: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_223;
	assign addr_223 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 224: 955 entries, bits from features [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_224;
	assign addr_224 = {input_vec[64], input_vec[52], input_vec[49], input_vec[144], input_vec[2], input_vec[147], input_vec[17], input_vec[43], input_vec[116], input_vec[100], input_vec[75], input_vec[99], input_vec[45], input_vec[139], input_vec[15], input_vec[18], input_vec[32], input_vec[110], input_vec[72], input_vec[69], input_vec[23], input_vec[140], input_vec[125], input_vec[146], input_vec[19], input_vec[91], input_vec[113], input_vec[12], input_vec[91], input_vec[144], input_vec[116], input_vec[151]};

	// Neuron 225: 840 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_225;
	assign addr_225 = {input_vec[103], input_vec[24], input_vec[81], input_vec[12], input_vec[64], input_vec[26], input_vec[106], input_vec[144], input_vec[80], input_vec[1], input_vec[120], input_vec[121], input_vec[31], input_vec[86], input_vec[44], input_vec[2], input_vec[77], input_vec[29], input_vec[22], input_vec[16], input_vec[32], input_vec[150], input_vec[150], input_vec[142], input_vec[23], input_vec[98], input_vec[79], input_vec[37], input_vec[126], input_vec[129], input_vec[134], input_vec[101]};

	// Neuron 226: 383 entries, bits from features [0, 2, 3, 5, 6, 7, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_226;
	assign addr_226 = {input_vec[120], input_vec[142], input_vec[144], input_vec[81], input_vec[148], input_vec[52], input_vec[30], input_vec[44], input_vec[58], input_vec[118], input_vec[62], input_vec[4], input_vec[124], input_vec[107], input_vec[145], input_vec[63], input_vec[19], input_vec[5], input_vec[22], input_vec[59], input_vec[2], input_vec[148], input_vec[23], input_vec[144], input_vec[58], input_vec[138], input_vec[81], input_vec[41], input_vec[124], input_vec[22], input_vec[102], input_vec[54]};

	// Neuron 227: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_227;
	assign addr_227 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 228: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_228;
	assign addr_228 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 229: 917 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_229;
	assign addr_229 = {input_vec[43], input_vec[65], input_vec[138], input_vec[20], input_vec[45], input_vec[77], input_vec[0], input_vec[148], input_vec[89], input_vec[100], input_vec[1], input_vec[26], input_vec[16], input_vec[59], input_vec[147], input_vec[58], input_vec[69], input_vec[4], input_vec[52], input_vec[73], input_vec[41], input_vec[104], input_vec[87], input_vec[86], input_vec[114], input_vec[52], input_vec[31], input_vec[97], input_vec[51], input_vec[9], input_vec[80], input_vec[111]};

	// Neuron 230: 1352 entries, bits from features [1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_230;
	assign addr_230 = {input_vec[132], input_vec[115], input_vec[87], input_vec[86], input_vec[98], input_vec[28], input_vec[68], input_vec[143], input_vec[65], input_vec[117], input_vec[91], input_vec[103], input_vec[65], input_vec[124], input_vec[146], input_vec[12], input_vec[103], input_vec[13], input_vec[61], input_vec[18], input_vec[101], input_vec[67], input_vec[18], input_vec[90], input_vec[41], input_vec[107], input_vec[107], input_vec[151], input_vec[135], input_vec[22], input_vec[119], input_vec[21]};

	// Neuron 231: 303 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15, 17, 18]
	logic [31:0] addr_231;
	assign addr_231 = {input_vec[113], input_vec[115], input_vec[149], input_vec[0], input_vec[35], input_vec[84], input_vec[112], input_vec[124], input_vec[146], input_vec[39], input_vec[85], input_vec[145], input_vec[16], input_vec[24], input_vec[41], input_vec[86], input_vec[84], input_vec[150], input_vec[151], input_vec[112], input_vec[0], input_vec[46], input_vec[119], input_vec[75], input_vec[43], input_vec[62], input_vec[55], input_vec[142], input_vec[108], input_vec[35], input_vec[51], input_vec[127]};

	// Neuron 232: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_232;
	assign addr_232 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 233: 915 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_233;
	assign addr_233 = {input_vec[87], input_vec[76], input_vec[11], input_vec[45], input_vec[23], input_vec[19], input_vec[57], input_vec[31], input_vec[101], input_vec[29], input_vec[137], input_vec[75], input_vec[129], input_vec[62], input_vec[26], input_vec[27], input_vec[17], input_vec[70], input_vec[150], input_vec[35], input_vec[142], input_vec[98], input_vec[32], input_vec[21], input_vec[107], input_vec[17], input_vec[130], input_vec[136], input_vec[136], input_vec[38], input_vec[60], input_vec[119]};

	// Neuron 234: 892 entries, bits from features [0, 1, 2, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_234;
	assign addr_234 = {input_vec[100], input_vec[2], input_vec[130], input_vec[139], input_vec[129], input_vec[144], input_vec[114], input_vec[127], input_vec[133], input_vec[88], input_vec[17], input_vec[45], input_vec[48], input_vec[60], input_vec[93], input_vec[13], input_vec[131], input_vec[103], input_vec[146], input_vec[139], input_vec[113], input_vec[55], input_vec[2], input_vec[61], input_vec[111], input_vec[104], input_vec[123], input_vec[149], input_vec[21], input_vec[88], input_vec[14], input_vec[120]};

	// Neuron 235: 1265 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18]
	logic [31:0] addr_235;
	assign addr_235 = {input_vec[150], input_vec[146], input_vec[70], input_vec[2], input_vec[137], input_vec[144], input_vec[126], input_vec[45], input_vec[77], input_vec[119], input_vec[122], input_vec[81], input_vec[35], input_vec[0], input_vec[18], input_vec[129], input_vec[114], input_vec[120], input_vec[87], input_vec[133], input_vec[55], input_vec[12], input_vec[13], input_vec[117], input_vec[47], input_vec[148], input_vec[116], input_vec[14], input_vec[30], input_vec[89], input_vec[115], input_vec[24]};

	// Neuron 236: 532 entries, bits from features [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_236;
	assign addr_236 = {input_vec[42], input_vec[64], input_vec[58], input_vec[98], input_vec[39], input_vec[98], input_vec[129], input_vec[61], input_vec[103], input_vec[43], input_vec[69], input_vec[106], input_vec[42], input_vec[136], input_vec[137], input_vec[38], input_vec[89], input_vec[14], input_vec[151], input_vec[9], input_vec[10], input_vec[88], input_vec[20], input_vec[146], input_vec[151], input_vec[35], input_vec[117], input_vec[64], input_vec[127], input_vec[76], input_vec[43], input_vec[35]};

	// Neuron 237: 927 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_237;
	assign addr_237 = {input_vec[94], input_vec[147], input_vec[106], input_vec[76], input_vec[101], input_vec[18], input_vec[50], input_vec[10], input_vec[149], input_vec[46], input_vec[147], input_vec[73], input_vec[109], input_vec[0], input_vec[7], input_vec[63], input_vec[102], input_vec[52], input_vec[78], input_vec[64], input_vec[126], input_vec[77], input_vec[148], input_vec[143], input_vec[11], input_vec[47], input_vec[115], input_vec[53], input_vec[12], input_vec[25], input_vec[96], input_vec[10]};

	// Neuron 238: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_238;
	assign addr_238 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 239: 814 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_239;
	assign addr_239 = {input_vec[66], input_vec[3], input_vec[31], input_vec[80], input_vec[83], input_vec[68], input_vec[141], input_vec[33], input_vec[37], input_vec[51], input_vec[20], input_vec[119], input_vec[62], input_vec[112], input_vec[120], input_vec[79], input_vec[99], input_vec[151], input_vec[13], input_vec[66], input_vec[3], input_vec[101], input_vec[61], input_vec[8], input_vec[74], input_vec[64], input_vec[138], input_vec[59], input_vec[64], input_vec[28], input_vec[109], input_vec[144]};

	// Neuron 240: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_240;
	assign addr_240 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 241: 796 entries, bits from features [0, 1, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_241;
	assign addr_241 = {input_vec[11], input_vec[28], input_vec[100], input_vec[137], input_vec[92], input_vec[3], input_vec[91], input_vec[71], input_vec[120], input_vec[68], input_vec[114], input_vec[102], input_vec[148], input_vec[31], input_vec[13], input_vec[135], input_vec[12], input_vec[139], input_vec[30], input_vec[103], input_vec[32], input_vec[72], input_vec[15], input_vec[69], input_vec[2], input_vec[6], input_vec[80], input_vec[41], input_vec[97], input_vec[122], input_vec[145], input_vec[139]};

	// Neuron 242: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_242;
	assign addr_242 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 243: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_243;
	assign addr_243 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 244: 326 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18]
	logic [31:0] addr_244;
	assign addr_244 = {input_vec[141], input_vec[76], input_vec[50], input_vec[111], input_vec[64], input_vec[40], input_vec[17], input_vec[68], input_vec[147], input_vec[73], input_vec[123], input_vec[18], input_vec[8], input_vec[77], input_vec[0], input_vec[113], input_vec[0], input_vec[35], input_vec[41], input_vec[141], input_vec[57], input_vec[19], input_vec[34], input_vec[57], input_vec[63], input_vec[58], input_vec[35], input_vec[147], input_vec[20], input_vec[32], input_vec[83], input_vec[86]};

	// Neuron 245: 343 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_245;
	assign addr_245 = {input_vec[96], input_vec[0], input_vec[56], input_vec[74], input_vec[16], input_vec[52], input_vec[81], input_vec[39], input_vec[40], input_vec[84], input_vec[141], input_vec[87], input_vec[64], input_vec[127], input_vec[121], input_vec[113], input_vec[75], input_vec[113], input_vec[127], input_vec[146], input_vec[31], input_vec[115], input_vec[100], input_vec[31], input_vec[9], input_vec[143], input_vec[8], input_vec[48], input_vec[107], input_vec[72], input_vec[30], input_vec[12]};

	// Neuron 246: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_246;
	assign addr_246 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 247: 381 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 17, 18]
	logic [31:0] addr_247;
	assign addr_247 = {input_vec[98], input_vec[19], input_vec[11], input_vec[140], input_vec[7], input_vec[91], input_vec[25], input_vec[63], input_vec[59], input_vec[34], input_vec[61], input_vec[139], input_vec[122], input_vec[54], input_vec[10], input_vec[26], input_vec[14], input_vec[4], input_vec[54], input_vec[40], input_vec[88], input_vec[90], input_vec[6], input_vec[149], input_vec[140], input_vec[118], input_vec[114], input_vec[121], input_vec[14], input_vec[125], input_vec[99], input_vec[66]};

	// Neuron 248: 630 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_248;
	assign addr_248 = {input_vec[31], input_vec[77], input_vec[55], input_vec[90], input_vec[48], input_vec[41], input_vec[5], input_vec[75], input_vec[114], input_vec[44], input_vec[55], input_vec[36], input_vec[25], input_vec[151], input_vec[63], input_vec[100], input_vec[110], input_vec[103], input_vec[14], input_vec[106], input_vec[71], input_vec[138], input_vec[72], input_vec[145], input_vec[58], input_vec[149], input_vec[73], input_vec[31], input_vec[91], input_vec[85], input_vec[92], input_vec[135]};

	// Neuron 249: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_249;
	assign addr_249 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 250: 1265 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18]
	logic [31:0] addr_250;
	assign addr_250 = {input_vec[150], input_vec[146], input_vec[70], input_vec[2], input_vec[137], input_vec[144], input_vec[126], input_vec[45], input_vec[77], input_vec[119], input_vec[122], input_vec[81], input_vec[35], input_vec[0], input_vec[18], input_vec[129], input_vec[114], input_vec[120], input_vec[87], input_vec[133], input_vec[55], input_vec[12], input_vec[13], input_vec[117], input_vec[47], input_vec[148], input_vec[116], input_vec[14], input_vec[30], input_vec[89], input_vec[115], input_vec[24]};

	// Neuron 251: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_251;
	assign addr_251 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 252: 718 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_252;
	assign addr_252 = {input_vec[61], input_vec[65], input_vec[64], input_vec[108], input_vec[90], input_vec[86], input_vec[90], input_vec[73], input_vec[99], input_vec[24], input_vec[142], input_vec[36], input_vec[151], input_vec[149], input_vec[114], input_vec[4], input_vec[9], input_vec[73], input_vec[145], input_vec[115], input_vec[97], input_vec[139], input_vec[110], input_vec[96], input_vec[29], input_vec[41], input_vec[10], input_vec[130], input_vec[110], input_vec[129], input_vec[32], input_vec[18]};

	// Neuron 253: 981 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_253;
	assign addr_253 = {input_vec[35], input_vec[126], input_vec[105], input_vec[39], input_vec[17], input_vec[70], input_vec[33], input_vec[85], input_vec[30], input_vec[147], input_vec[104], input_vec[35], input_vec[140], input_vec[127], input_vec[135], input_vec[102], input_vec[104], input_vec[97], input_vec[50], input_vec[127], input_vec[136], input_vec[6], input_vec[78], input_vec[151], input_vec[87], input_vec[123], input_vec[22], input_vec[104], input_vec[89], input_vec[29], input_vec[9], input_vec[38]};

	// Neuron 254: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_254;
	assign addr_254 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 255: 631 entries, bits from features [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_255;
	assign addr_255 = {input_vec[144], input_vec[23], input_vec[105], input_vec[95], input_vec[87], input_vec[143], input_vec[122], input_vec[101], input_vec[96], input_vec[1], input_vec[147], input_vec[118], input_vec[36], input_vec[2], input_vec[0], input_vec[29], input_vec[142], input_vec[91], input_vec[88], input_vec[113], input_vec[89], input_vec[122], input_vec[79], input_vec[47], input_vec[53], input_vec[99], input_vec[111], input_vec[99], input_vec[73], input_vec[44], input_vec[128], input_vec[33]};

	// Neuron 256: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_256;
	assign addr_256 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 257: 1066 entries, bits from features [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_257;
	assign addr_257 = {input_vec[50], input_vec[63], input_vec[34], input_vec[133], input_vec[70], input_vec[67], input_vec[55], input_vec[19], input_vec[66], input_vec[10], input_vec[61], input_vec[83], input_vec[138], input_vec[54], input_vec[110], input_vec[123], input_vec[16], input_vec[69], input_vec[133], input_vec[94], input_vec[8], input_vec[98], input_vec[10], input_vec[47], input_vec[130], input_vec[40], input_vec[106], input_vec[95], input_vec[123], input_vec[8], input_vec[135], input_vec[64]};

	// Neuron 258: 522 entries, bits from features [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_258;
	assign addr_258 = {input_vec[52], input_vec[143], input_vec[103], input_vec[131], input_vec[115], input_vec[97], input_vec[28], input_vec[124], input_vec[99], input_vec[32], input_vec[139], input_vec[86], input_vec[12], input_vec[131], input_vec[114], input_vec[37], input_vec[96], input_vec[67], input_vec[74], input_vec[90], input_vec[150], input_vec[99], input_vec[33], input_vec[120], input_vec[147], input_vec[91], input_vec[78], input_vec[124], input_vec[112], input_vec[42], input_vec[124], input_vec[100]};

	// Neuron 259: 730 entries, bits from features [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17]
	logic [31:0] addr_259;
	assign addr_259 = {input_vec[47], input_vec[65], input_vec[42], input_vec[83], input_vec[103], input_vec[136], input_vec[46], input_vec[12], input_vec[103], input_vec[125], input_vec[138], input_vec[121], input_vec[80], input_vec[100], input_vec[101], input_vec[36], input_vec[22], input_vec[9], input_vec[81], input_vec[24], input_vec[129], input_vec[30], input_vec[48], input_vec[78], input_vec[9], input_vec[10], input_vec[33], input_vec[67], input_vec[139], input_vec[31], input_vec[17], input_vec[25]};

	// Neuron 260: 284 entries, bits from features [0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
	logic [31:0] addr_260;
	assign addr_260 = {input_vec[128], input_vec[54], input_vec[68], input_vec[73], input_vec[71], input_vec[136], input_vec[122], input_vec[94], input_vec[74], input_vec[39], input_vec[133], input_vec[141], input_vec[0], input_vec[68], input_vec[58], input_vec[113], input_vec[34], input_vec[63], input_vec[69], input_vec[128], input_vec[73], input_vec[90], input_vec[119], input_vec[114], input_vec[53], input_vec[42], input_vec[66], input_vec[137], input_vec[68], input_vec[86], input_vec[43], input_vec[112]};

	// Neuron 261: 1352 entries, bits from features [1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_261;
	assign addr_261 = {input_vec[132], input_vec[115], input_vec[87], input_vec[86], input_vec[98], input_vec[28], input_vec[68], input_vec[143], input_vec[65], input_vec[117], input_vec[91], input_vec[103], input_vec[65], input_vec[124], input_vec[146], input_vec[12], input_vec[103], input_vec[13], input_vec[61], input_vec[18], input_vec[101], input_vec[67], input_vec[18], input_vec[90], input_vec[41], input_vec[107], input_vec[107], input_vec[151], input_vec[135], input_vec[22], input_vec[119], input_vec[21]};

	// Neuron 262: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_262;
	assign addr_262 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 263: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_263;
	assign addr_263 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 264: 453 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
	logic [31:0] addr_264;
	assign addr_264 = {input_vec[84], input_vec[4], input_vec[80], input_vec[42], input_vec[73], input_vec[35], input_vec[65], input_vec[33], input_vec[113], input_vec[132], input_vec[15], input_vec[137], input_vec[1], input_vec[107], input_vec[114], input_vec[18], input_vec[41], input_vec[93], input_vec[123], input_vec[94], input_vec[61], input_vec[16], input_vec[83], input_vec[76], input_vec[21], input_vec[50], input_vec[122], input_vec[42], input_vec[112], input_vec[1], input_vec[142], input_vec[57]};

	// Neuron 265: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_265;
	assign addr_265 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 266: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 16, 18]
	logic [31:0] addr_266;
	assign addr_266 = {input_vec[107], input_vec[111], input_vec[87], input_vec[119], input_vec[28], input_vec[91], input_vec[2], input_vec[89], input_vec[100], input_vec[132], input_vec[19], input_vec[47], input_vec[89], input_vec[25], input_vec[25], input_vec[94], input_vec[109], input_vec[48], input_vec[37], input_vec[31], input_vec[42], input_vec[38], input_vec[115], input_vec[117], input_vec[10], input_vec[24], input_vec[149], input_vec[81], input_vec[45], input_vec[81], input_vec[99], input_vec[151]};

	// Neuron 267: 352 entries, bits from features [1, 2, 4, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_267;
	assign addr_267 = {input_vec[124], input_vec[49], input_vec[54], input_vec[113], input_vec[21], input_vec[56], input_vec[36], input_vec[142], input_vec[12], input_vec[59], input_vec[133], input_vec[37], input_vec[122], input_vec[90], input_vec[33], input_vec[80], input_vec[134], input_vec[104], input_vec[40], input_vec[12], input_vec[37], input_vec[124], input_vec[141], input_vec[149], input_vec[151], input_vec[57], input_vec[53], input_vec[127], input_vec[125], input_vec[115], input_vec[61], input_vec[82]};

	// Neuron 268: 553 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18]
	logic [31:0] addr_268;
	assign addr_268 = {input_vec[12], input_vec[56], input_vec[48], input_vec[23], input_vec[15], input_vec[43], input_vec[72], input_vec[10], input_vec[77], input_vec[128], input_vec[127], input_vec[79], input_vec[148], input_vec[102], input_vec[13], input_vec[58], input_vec[131], input_vec[62], input_vec[145], input_vec[10], input_vec[39], input_vec[17], input_vec[58], input_vec[58], input_vec[27], input_vec[64], input_vec[147], input_vec[24], input_vec[79], input_vec[27], input_vec[100], input_vec[150]};

	// Neuron 269: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 16, 18]
	logic [31:0] addr_269;
	assign addr_269 = {input_vec[107], input_vec[111], input_vec[87], input_vec[119], input_vec[28], input_vec[91], input_vec[2], input_vec[89], input_vec[100], input_vec[132], input_vec[19], input_vec[47], input_vec[89], input_vec[25], input_vec[25], input_vec[94], input_vec[109], input_vec[48], input_vec[37], input_vec[31], input_vec[42], input_vec[38], input_vec[115], input_vec[117], input_vec[10], input_vec[24], input_vec[149], input_vec[81], input_vec[45], input_vec[81], input_vec[99], input_vec[151]};

	// Neuron 270: 460 entries, bits from features [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_270;
	assign addr_270 = {input_vec[36], input_vec[121], input_vec[35], input_vec[32], input_vec[4], input_vec[104], input_vec[102], input_vec[126], input_vec[66], input_vec[45], input_vec[111], input_vec[37], input_vec[113], input_vec[115], input_vec[148], input_vec[142], input_vec[117], input_vec[59], input_vec[80], input_vec[40], input_vec[145], input_vec[103], input_vec[2], input_vec[10], input_vec[2], input_vec[91], input_vec[118], input_vec[59], input_vec[12], input_vec[28], input_vec[79], input_vec[9]};

	// Neuron 271: 814 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_271;
	assign addr_271 = {input_vec[66], input_vec[3], input_vec[31], input_vec[80], input_vec[83], input_vec[68], input_vec[141], input_vec[33], input_vec[37], input_vec[51], input_vec[20], input_vec[119], input_vec[62], input_vec[112], input_vec[120], input_vec[79], input_vec[99], input_vec[151], input_vec[13], input_vec[66], input_vec[3], input_vec[101], input_vec[61], input_vec[8], input_vec[74], input_vec[64], input_vec[138], input_vec[59], input_vec[64], input_vec[28], input_vec[109], input_vec[144]};

	// Neuron 272: 1744 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_272;
	assign addr_272 = {input_vec[47], input_vec[21], input_vec[141], input_vec[140], input_vec[121], input_vec[9], input_vec[97], input_vec[71], input_vec[74], input_vec[111], input_vec[44], input_vec[28], input_vec[6], input_vec[18], input_vec[84], input_vec[37], input_vec[116], input_vec[102], input_vec[20], input_vec[36], input_vec[134], input_vec[78], input_vec[129], input_vec[82], input_vec[17], input_vec[6], input_vec[2], input_vec[110], input_vec[150], input_vec[98], input_vec[48], input_vec[54]};

	// Neuron 273: 576 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
	logic [31:0] addr_273;
	assign addr_273 = {input_vec[21], input_vec[62], input_vec[25], input_vec[16], input_vec[106], input_vec[48], input_vec[21], input_vec[149], input_vec[98], input_vec[141], input_vec[144], input_vec[149], input_vec[26], input_vec[46], input_vec[54], input_vec[70], input_vec[81], input_vec[54], input_vec[82], input_vec[85], input_vec[126], input_vec[15], input_vec[76], input_vec[109], input_vec[64], input_vec[32], input_vec[150], input_vec[55], input_vec[89], input_vec[24], input_vec[77], input_vec[44]};

	// Neuron 274: 656 entries, bits from features [1, 2, 3, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_274;
	assign addr_274 = {input_vec[111], input_vec[131], input_vec[15], input_vec[114], input_vec[30], input_vec[124], input_vec[130], input_vec[17], input_vec[143], input_vec[147], input_vec[144], input_vec[75], input_vec[44], input_vec[46], input_vec[23], input_vec[43], input_vec[125], input_vec[99], input_vec[122], input_vec[86], input_vec[111], input_vec[139], input_vec[130], input_vec[145], input_vec[143], input_vec[105], input_vec[82], input_vec[118], input_vec[102], input_vec[21], input_vec[122], input_vec[26]};

	// Neuron 275: 383 entries, bits from features [0, 2, 3, 5, 6, 7, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_275;
	assign addr_275 = {input_vec[120], input_vec[142], input_vec[144], input_vec[81], input_vec[148], input_vec[52], input_vec[30], input_vec[44], input_vec[58], input_vec[118], input_vec[62], input_vec[4], input_vec[124], input_vec[107], input_vec[145], input_vec[63], input_vec[19], input_vec[5], input_vec[22], input_vec[59], input_vec[2], input_vec[148], input_vec[23], input_vec[144], input_vec[58], input_vec[138], input_vec[81], input_vec[41], input_vec[124], input_vec[22], input_vec[102], input_vec[54]};

	// Neuron 276: 981 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_276;
	assign addr_276 = {input_vec[35], input_vec[126], input_vec[105], input_vec[39], input_vec[17], input_vec[70], input_vec[33], input_vec[85], input_vec[30], input_vec[147], input_vec[104], input_vec[35], input_vec[140], input_vec[127], input_vec[135], input_vec[102], input_vec[104], input_vec[97], input_vec[50], input_vec[127], input_vec[136], input_vec[6], input_vec[78], input_vec[151], input_vec[87], input_vec[123], input_vec[22], input_vec[104], input_vec[89], input_vec[29], input_vec[9], input_vec[38]};

	// Neuron 277: 507 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_277;
	assign addr_277 = {input_vec[33], input_vec[10], input_vec[133], input_vec[59], input_vec[104], input_vec[1], input_vec[130], input_vec[51], input_vec[43], input_vec[107], input_vec[1], input_vec[143], input_vec[85], input_vec[140], input_vec[111], input_vec[135], input_vec[46], input_vec[89], input_vec[33], input_vec[54], input_vec[76], input_vec[30], input_vec[63], input_vec[67], input_vec[130], input_vec[151], input_vec[25], input_vec[14], input_vec[118], input_vec[98], input_vec[89], input_vec[90]};

	// Neuron 278: 514 entries, bits from features [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_278;
	assign addr_278 = {input_vec[61], input_vec[140], input_vec[84], input_vec[106], input_vec[78], input_vec[130], input_vec[54], input_vec[137], input_vec[110], input_vec[59], input_vec[9], input_vec[2], input_vec[25], input_vec[11], input_vec[105], input_vec[31], input_vec[127], input_vec[150], input_vec[75], input_vec[103], input_vec[129], input_vec[138], input_vec[47], input_vec[79], input_vec[132], input_vec[58], input_vec[47], input_vec[40], input_vec[85], input_vec[1], input_vec[91], input_vec[11]};

	// Neuron 279: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_279;
	assign addr_279 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 280: 656 entries, bits from features [1, 2, 3, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_280;
	assign addr_280 = {input_vec[111], input_vec[131], input_vec[15], input_vec[114], input_vec[30], input_vec[124], input_vec[130], input_vec[17], input_vec[143], input_vec[147], input_vec[144], input_vec[75], input_vec[44], input_vec[46], input_vec[23], input_vec[43], input_vec[125], input_vec[99], input_vec[122], input_vec[86], input_vec[111], input_vec[139], input_vec[130], input_vec[145], input_vec[143], input_vec[105], input_vec[82], input_vec[118], input_vec[102], input_vec[21], input_vec[122], input_vec[26]};

	// Neuron 281: 1756 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]
	logic [31:0] addr_281;
	assign addr_281 = {input_vec[44], input_vec[101], input_vec[56], input_vec[69], input_vec[46], input_vec[93], input_vec[128], input_vec[29], input_vec[96], input_vec[41], input_vec[103], input_vec[88], input_vec[84], input_vec[139], input_vec[71], input_vec[135], input_vec[3], input_vec[146], input_vec[33], input_vec[62], input_vec[76], input_vec[131], input_vec[87], input_vec[133], input_vec[100], input_vec[86], input_vec[12], input_vec[7], input_vec[21], input_vec[49], input_vec[72], input_vec[58]};

	// Neuron 282: 553 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18]
	logic [31:0] addr_282;
	assign addr_282 = {input_vec[12], input_vec[56], input_vec[48], input_vec[23], input_vec[15], input_vec[43], input_vec[72], input_vec[10], input_vec[77], input_vec[128], input_vec[127], input_vec[79], input_vec[148], input_vec[102], input_vec[13], input_vec[58], input_vec[131], input_vec[62], input_vec[145], input_vec[10], input_vec[39], input_vec[17], input_vec[58], input_vec[58], input_vec[27], input_vec[64], input_vec[147], input_vec[24], input_vec[79], input_vec[27], input_vec[100], input_vec[150]};

	// Neuron 283: 927 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_283;
	assign addr_283 = {input_vec[94], input_vec[147], input_vec[106], input_vec[76], input_vec[101], input_vec[18], input_vec[50], input_vec[10], input_vec[149], input_vec[46], input_vec[147], input_vec[73], input_vec[109], input_vec[0], input_vec[7], input_vec[63], input_vec[102], input_vec[52], input_vec[78], input_vec[64], input_vec[126], input_vec[77], input_vec[148], input_vec[143], input_vec[11], input_vec[47], input_vec[115], input_vec[53], input_vec[12], input_vec[25], input_vec[96], input_vec[10]};

	// Neuron 284: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_284;
	assign addr_284 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 285: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_285;
	assign addr_285 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 286: 772 entries, bits from features [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_286;
	assign addr_286 = {input_vec[34], input_vec[148], input_vec[124], input_vec[9], input_vec[84], input_vec[96], input_vec[101], input_vec[10], input_vec[106], input_vec[146], input_vec[106], input_vec[17], input_vec[102], input_vec[20], input_vec[63], input_vec[69], input_vec[1], input_vec[136], input_vec[69], input_vec[102], input_vec[56], input_vec[138], input_vec[28], input_vec[37], input_vec[2], input_vec[76], input_vec[28], input_vec[116], input_vec[58], input_vec[65], input_vec[66], input_vec[89]};

	// Neuron 287: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_287;
	assign addr_287 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 288: 718 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_288;
	assign addr_288 = {input_vec[61], input_vec[65], input_vec[64], input_vec[108], input_vec[90], input_vec[86], input_vec[90], input_vec[73], input_vec[99], input_vec[24], input_vec[142], input_vec[36], input_vec[151], input_vec[149], input_vec[114], input_vec[4], input_vec[9], input_vec[73], input_vec[145], input_vec[115], input_vec[97], input_vec[139], input_vec[110], input_vec[96], input_vec[29], input_vec[41], input_vec[10], input_vec[130], input_vec[110], input_vec[129], input_vec[32], input_vec[18]};

	// Neuron 289: 917 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18]
	logic [31:0] addr_289;
	assign addr_289 = {input_vec[43], input_vec[65], input_vec[138], input_vec[20], input_vec[45], input_vec[77], input_vec[0], input_vec[148], input_vec[89], input_vec[100], input_vec[1], input_vec[26], input_vec[16], input_vec[59], input_vec[147], input_vec[58], input_vec[69], input_vec[4], input_vec[52], input_vec[73], input_vec[41], input_vec[104], input_vec[87], input_vec[86], input_vec[114], input_vec[52], input_vec[31], input_vec[97], input_vec[51], input_vec[9], input_vec[80], input_vec[111]};

	// Neuron 290: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_290;
	assign addr_290 = {input_vec[58], input_vec[81], input_vec[41], input_vec[144], input_vec[96], input_vec[14], input_vec[67], input_vec[38], input_vec[21], input_vec[137], input_vec[48], input_vec[59], input_vec[126], input_vec[75], input_vec[28], input_vec[18], input_vec[21], input_vec[48], input_vec[4], input_vec[70], input_vec[117], input_vec[27], input_vec[20], input_vec[39], input_vec[9], input_vec[77], input_vec[74], input_vec[126], input_vec[149], input_vec[113], input_vec[39], input_vec[111]};

	// Neuron 291: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_291;
	assign addr_291 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 292: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_292;
	assign addr_292 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 293: 248 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17]
	logic [31:0] addr_293;
	assign addr_293 = {input_vec[51], input_vec[64], input_vec[137], input_vec[34], input_vec[125], input_vec[93], input_vec[24], input_vec[16], input_vec[25], input_vec[51], input_vec[143], input_vec[97], input_vec[134], input_vec[99], input_vec[45], input_vec[91], input_vec[125], input_vec[42], input_vec[80], input_vec[113], input_vec[132], input_vec[53], input_vec[26], input_vec[91], input_vec[113], input_vec[113], input_vec[7], input_vec[6], input_vec[140], input_vec[27], input_vec[140], input_vec[52]};

	// Neuron 294: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_294;
	assign addr_294 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 295: 729 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_295;
	assign addr_295 = {input_vec[93], input_vec[150], input_vec[78], input_vec[134], input_vec[39], input_vec[59], input_vec[56], input_vec[85], input_vec[23], input_vec[117], input_vec[121], input_vec[107], input_vec[63], input_vec[64], input_vec[75], input_vec[61], input_vec[124], input_vec[145], input_vec[146], input_vec[100], input_vec[33], input_vec[144], input_vec[108], input_vec[62], input_vec[40], input_vec[26], input_vec[56], input_vec[77], input_vec[23], input_vec[40], input_vec[11], input_vec[57]};

	// Neuron 296: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_296;
	assign addr_296 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 297: 897 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_297;
	assign addr_297 = {input_vec[120], input_vec[25], input_vec[15], input_vec[115], input_vec[71], input_vec[59], input_vec[132], input_vec[125], input_vec[132], input_vec[26], input_vec[100], input_vec[90], input_vec[99], input_vec[135], input_vec[64], input_vec[36], input_vec[106], input_vec[88], input_vec[115], input_vec[53], input_vec[45], input_vec[120], input_vec[113], input_vec[94], input_vec[45], input_vec[78], input_vec[39], input_vec[57], input_vec[17], input_vec[131], input_vec[137], input_vec[117]};

	// Neuron 298: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_298;
	assign addr_298 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 299: 669 entries, bits from features [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_299;
	assign addr_299 = {input_vec[5], input_vec[114], input_vec[56], input_vec[30], input_vec[100], input_vec[55], input_vec[11], input_vec[67], input_vec[37], input_vec[97], input_vec[108], input_vec[12], input_vec[126], input_vec[8], input_vec[139], input_vec[145], input_vec[91], input_vec[129], input_vec[84], input_vec[52], input_vec[0], input_vec[74], input_vec[2], input_vec[95], input_vec[11], input_vec[99], input_vec[39], input_vec[121], input_vec[146], input_vec[60], input_vec[88], input_vec[104]};

	// Neuron 300: 981 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_300;
	assign addr_300 = {input_vec[35], input_vec[126], input_vec[105], input_vec[39], input_vec[17], input_vec[70], input_vec[33], input_vec[85], input_vec[30], input_vec[147], input_vec[104], input_vec[35], input_vec[140], input_vec[127], input_vec[135], input_vec[102], input_vec[104], input_vec[97], input_vec[50], input_vec[127], input_vec[136], input_vec[6], input_vec[78], input_vec[151], input_vec[87], input_vec[123], input_vec[22], input_vec[104], input_vec[89], input_vec[29], input_vec[9], input_vec[38]};

	// Neuron 301: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_301;
	assign addr_301 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 302: 981 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_302;
	assign addr_302 = {input_vec[35], input_vec[126], input_vec[105], input_vec[39], input_vec[17], input_vec[70], input_vec[33], input_vec[85], input_vec[30], input_vec[147], input_vec[104], input_vec[35], input_vec[140], input_vec[127], input_vec[135], input_vec[102], input_vec[104], input_vec[97], input_vec[50], input_vec[127], input_vec[136], input_vec[6], input_vec[78], input_vec[151], input_vec[87], input_vec[123], input_vec[22], input_vec[104], input_vec[89], input_vec[29], input_vec[9], input_vec[38]};

	// Neuron 303: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_303;
	assign addr_303 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 304: 630 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_304;
	assign addr_304 = {input_vec[31], input_vec[77], input_vec[55], input_vec[90], input_vec[48], input_vec[41], input_vec[5], input_vec[75], input_vec[114], input_vec[44], input_vec[55], input_vec[36], input_vec[25], input_vec[151], input_vec[63], input_vec[100], input_vec[110], input_vec[103], input_vec[14], input_vec[106], input_vec[71], input_vec[138], input_vec[72], input_vec[145], input_vec[58], input_vec[149], input_vec[73], input_vec[31], input_vec[91], input_vec[85], input_vec[92], input_vec[135]};

	// Neuron 305: 499 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_305;
	assign addr_305 = {input_vec[16], input_vec[58], input_vec[25], input_vec[30], input_vec[65], input_vec[61], input_vec[32], input_vec[106], input_vec[73], input_vec[119], input_vec[3], input_vec[99], input_vec[94], input_vec[0], input_vec[66], input_vec[0], input_vec[133], input_vec[6], input_vec[140], input_vec[118], input_vec[14], input_vec[43], input_vec[100], input_vec[140], input_vec[123], input_vec[33], input_vec[39], input_vec[33], input_vec[123], input_vec[20], input_vec[97], input_vec[143]};

	// Neuron 306: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_306;
	assign addr_306 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 307: 740 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_307;
	assign addr_307 = {input_vec[73], input_vec[96], input_vec[31], input_vec[67], input_vec[140], input_vec[56], input_vec[9], input_vec[3], input_vec[28], input_vec[0], input_vec[13], input_vec[56], input_vec[126], input_vec[113], input_vec[123], input_vec[54], input_vec[94], input_vec[114], input_vec[100], input_vec[72], input_vec[22], input_vec[34], input_vec[146], input_vec[94], input_vec[106], input_vec[12], input_vec[129], input_vec[76], input_vec[13], input_vec[132], input_vec[91], input_vec[45]};

	// Neuron 308: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_308;
	assign addr_308 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 309: 1744 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_309;
	assign addr_309 = {input_vec[47], input_vec[21], input_vec[141], input_vec[140], input_vec[121], input_vec[9], input_vec[97], input_vec[71], input_vec[74], input_vec[111], input_vec[44], input_vec[28], input_vec[6], input_vec[18], input_vec[84], input_vec[37], input_vec[116], input_vec[102], input_vec[20], input_vec[36], input_vec[134], input_vec[78], input_vec[129], input_vec[82], input_vec[17], input_vec[6], input_vec[2], input_vec[110], input_vec[150], input_vec[98], input_vec[48], input_vec[54]};

	// Neuron 310: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_310;
	assign addr_310 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 311: 815 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_311;
	assign addr_311 = {input_vec[78], input_vec[2], input_vec[141], input_vec[39], input_vec[70], input_vec[125], input_vec[142], input_vec[117], input_vec[87], input_vec[68], input_vec[17], input_vec[135], input_vec[98], input_vec[151], input_vec[138], input_vec[18], input_vec[74], input_vec[113], input_vec[5], input_vec[49], input_vec[0], input_vec[52], input_vec[18], input_vec[31], input_vec[15], input_vec[110], input_vec[150], input_vec[99], input_vec[14], input_vec[122], input_vec[65], input_vec[63]};

	// Neuron 312: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_312;
	assign addr_312 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 313: 718 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_313;
	assign addr_313 = {input_vec[61], input_vec[65], input_vec[64], input_vec[108], input_vec[90], input_vec[86], input_vec[90], input_vec[73], input_vec[99], input_vec[24], input_vec[142], input_vec[36], input_vec[151], input_vec[149], input_vec[114], input_vec[4], input_vec[9], input_vec[73], input_vec[145], input_vec[115], input_vec[97], input_vec[139], input_vec[110], input_vec[96], input_vec[29], input_vec[41], input_vec[10], input_vec[130], input_vec[110], input_vec[129], input_vec[32], input_vec[18]};

	// Neuron 314: 1615 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_314;
	assign addr_314 = {input_vec[73], input_vec[55], input_vec[79], input_vec[131], input_vec[66], input_vec[21], input_vec[102], input_vec[63], input_vec[17], input_vec[13], input_vec[120], input_vec[13], input_vec[142], input_vec[13], input_vec[30], input_vec[145], input_vec[139], input_vec[20], input_vec[70], input_vec[108], input_vec[4], input_vec[137], input_vec[21], input_vec[38], input_vec[23], input_vec[72], input_vec[28], input_vec[9], input_vec[83], input_vec[85], input_vec[110], input_vec[109]};

	// Neuron 315: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_315;
	assign addr_315 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 316: 764 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18]
	logic [31:0] addr_316;
	assign addr_316 = {input_vec[7], input_vec[29], input_vec[113], input_vec[86], input_vec[116], input_vec[109], input_vec[58], input_vec[22], input_vec[86], input_vec[76], input_vec[65], input_vec[2], input_vec[106], input_vec[98], input_vec[70], input_vec[37], input_vec[115], input_vec[51], input_vec[66], input_vec[1], input_vec[108], input_vec[144], input_vec[83], input_vec[12], input_vec[29], input_vec[28], input_vec[145], input_vec[43], input_vec[68], input_vec[127], input_vec[105], input_vec[76]};

	// Neuron 317: 901 entries, bits from features [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_317;
	assign addr_317 = {input_vec[13], input_vec[101], input_vec[48], input_vec[114], input_vec[96], input_vec[75], input_vec[46], input_vec[124], input_vec[95], input_vec[70], input_vec[130], input_vec[81], input_vec[4], input_vec[97], input_vec[113], input_vec[2], input_vec[17], input_vec[49], input_vec[134], input_vec[148], input_vec[112], input_vec[129], input_vec[3], input_vec[17], input_vec[133], input_vec[57], input_vec[121], input_vec[139], input_vec[41], input_vec[69], input_vec[47], input_vec[49]};

	// Neuron 318: 1020 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_318;
	assign addr_318 = {input_vec[102], input_vec[118], input_vec[85], input_vec[26], input_vec[138], input_vec[99], input_vec[0], input_vec[76], input_vec[129], input_vec[114], input_vec[106], input_vec[57], input_vec[114], input_vec[20], input_vec[117], input_vec[10], input_vec[61], input_vec[136], input_vec[113], input_vec[115], input_vec[56], input_vec[5], input_vec[118], input_vec[21], input_vec[18], input_vec[51], input_vec[142], input_vec[19], input_vec[144], input_vec[92], input_vec[40], input_vec[84]};

	// Neuron 319: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_319;
	assign addr_319 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 320: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_320;
	assign addr_320 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 321: 927 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_321;
	assign addr_321 = {input_vec[94], input_vec[147], input_vec[106], input_vec[76], input_vec[101], input_vec[18], input_vec[50], input_vec[10], input_vec[149], input_vec[46], input_vec[147], input_vec[73], input_vec[109], input_vec[0], input_vec[7], input_vec[63], input_vec[102], input_vec[52], input_vec[78], input_vec[64], input_vec[126], input_vec[77], input_vec[148], input_vec[143], input_vec[11], input_vec[47], input_vec[115], input_vec[53], input_vec[12], input_vec[25], input_vec[96], input_vec[10]};

	// Neuron 322: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_322;
	assign addr_322 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 323: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_323;
	assign addr_323 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 324: 1881 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_324;
	assign addr_324 = {input_vec[140], input_vec[54], input_vec[97], input_vec[49], input_vec[115], input_vec[134], input_vec[149], input_vec[41], input_vec[85], input_vec[125], input_vec[19], input_vec[56], input_vec[101], input_vec[64], input_vec[129], input_vec[147], input_vec[70], input_vec[95], input_vec[31], input_vec[0], input_vec[33], input_vec[63], input_vec[114], input_vec[146], input_vec[84], input_vec[79], input_vec[18], input_vec[16], input_vec[12], input_vec[2], input_vec[147], input_vec[52]};

	// Neuron 325: 248 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17]
	logic [31:0] addr_325;
	assign addr_325 = {input_vec[51], input_vec[64], input_vec[137], input_vec[34], input_vec[125], input_vec[93], input_vec[24], input_vec[16], input_vec[25], input_vec[51], input_vec[143], input_vec[97], input_vec[134], input_vec[99], input_vec[45], input_vec[91], input_vec[125], input_vec[42], input_vec[80], input_vec[113], input_vec[132], input_vec[53], input_vec[26], input_vec[91], input_vec[113], input_vec[113], input_vec[7], input_vec[6], input_vec[140], input_vec[27], input_vec[140], input_vec[52]};

	// Neuron 326: 271 entries, bits from features [3, 5, 6, 7, 10, 11, 13, 15, 16, 17, 18]
	logic [31:0] addr_326;
	assign addr_326 = {input_vec[48], input_vec[138], input_vec[108], input_vec[139], input_vec[27], input_vec[55], input_vec[123], input_vec[124], input_vec[93], input_vec[136], input_vec[130], input_vec[110], input_vec[121], input_vec[46], input_vec[149], input_vec[140], input_vec[46], input_vec[91], input_vec[62], input_vec[92], input_vec[52], input_vec[140], input_vec[139], input_vec[93], input_vec[29], input_vec[27], input_vec[126], input_vec[144], input_vec[53], input_vec[106], input_vec[93], input_vec[81]};

	// Neuron 327: 1881 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_327;
	assign addr_327 = {input_vec[140], input_vec[54], input_vec[97], input_vec[49], input_vec[115], input_vec[134], input_vec[149], input_vec[41], input_vec[85], input_vec[125], input_vec[19], input_vec[56], input_vec[101], input_vec[64], input_vec[129], input_vec[147], input_vec[70], input_vec[95], input_vec[31], input_vec[0], input_vec[33], input_vec[63], input_vec[114], input_vec[146], input_vec[84], input_vec[79], input_vec[18], input_vec[16], input_vec[12], input_vec[2], input_vec[147], input_vec[52]};

	// Neuron 328: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_328;
	assign addr_328 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 329: 981 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_329;
	assign addr_329 = {input_vec[35], input_vec[126], input_vec[105], input_vec[39], input_vec[17], input_vec[70], input_vec[33], input_vec[85], input_vec[30], input_vec[147], input_vec[104], input_vec[35], input_vec[140], input_vec[127], input_vec[135], input_vec[102], input_vec[104], input_vec[97], input_vec[50], input_vec[127], input_vec[136], input_vec[6], input_vec[78], input_vec[151], input_vec[87], input_vec[123], input_vec[22], input_vec[104], input_vec[89], input_vec[29], input_vec[9], input_vec[38]};

	// Neuron 330: 943 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]
	logic [31:0] addr_330;
	assign addr_330 = {input_vec[103], input_vec[19], input_vec[93], input_vec[70], input_vec[33], input_vec[123], input_vec[99], input_vec[11], input_vec[48], input_vec[132], input_vec[130], input_vec[112], input_vec[113], input_vec[70], input_vec[56], input_vec[81], input_vec[20], input_vec[63], input_vec[83], input_vec[56], input_vec[52], input_vec[25], input_vec[85], input_vec[119], input_vec[86], input_vec[74], input_vec[7], input_vec[70], input_vec[101], input_vec[57], input_vec[98], input_vec[142]};

	// Neuron 331: 840 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_331;
	assign addr_331 = {input_vec[103], input_vec[24], input_vec[81], input_vec[12], input_vec[64], input_vec[26], input_vec[106], input_vec[144], input_vec[80], input_vec[1], input_vec[120], input_vec[121], input_vec[31], input_vec[86], input_vec[44], input_vec[2], input_vec[77], input_vec[29], input_vec[22], input_vec[16], input_vec[32], input_vec[150], input_vec[150], input_vec[142], input_vec[23], input_vec[98], input_vec[79], input_vec[37], input_vec[126], input_vec[129], input_vec[134], input_vec[101]};

	// Neuron 332: 403 entries, bits from features [0, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18]
	logic [31:0] addr_332;
	assign addr_332 = {input_vec[52], input_vec[31], input_vec[33], input_vec[45], input_vec[56], input_vec[35], input_vec[128], input_vec[87], input_vec[27], input_vec[66], input_vec[73], input_vec[50], input_vec[78], input_vec[86], input_vec[120], input_vec[78], input_vec[133], input_vec[63], input_vec[146], input_vec[47], input_vec[48], input_vec[107], input_vec[64], input_vec[5], input_vec[136], input_vec[61], input_vec[34], input_vec[58], input_vec[128], input_vec[54], input_vec[34], input_vec[124]};

	// Neuron 333: 563 entries, bits from features [1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18]
	logic [31:0] addr_333;
	assign addr_333 = {input_vec[151], input_vec[74], input_vec[146], input_vec[95], input_vec[24], input_vec[97], input_vec[92], input_vec[134], input_vec[118], input_vec[126], input_vec[84], input_vec[115], input_vec[109], input_vec[29], input_vec[29], input_vec[42], input_vec[117], input_vec[48], input_vec[76], input_vec[108], input_vec[8], input_vec[135], input_vec[79], input_vec[41], input_vec[43], input_vec[44], input_vec[125], input_vec[134], input_vec[87], input_vec[94], input_vec[83], input_vec[113]};

	// Neuron 334: 978 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18]
	logic [31:0] addr_334;
	assign addr_334 = {input_vec[99], input_vec[18], input_vec[11], input_vec[69], input_vec[148], input_vec[26], input_vec[45], input_vec[110], input_vec[109], input_vec[67], input_vec[100], input_vec[49], input_vec[57], input_vec[36], input_vec[66], input_vec[76], input_vec[32], input_vec[31], input_vec[1], input_vec[78], input_vec[8], input_vec[103], input_vec[4], input_vec[36], input_vec[58], input_vec[115], input_vec[151], input_vec[30], input_vec[60], input_vec[140], input_vec[111], input_vec[147]};

	// Neuron 335: 368 entries, bits from features [0, 1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17]
	logic [31:0] addr_335;
	assign addr_335 = {input_vec[109], input_vec[19], input_vec[55], input_vec[85], input_vec[118], input_vec[0], input_vec[24], input_vec[24], input_vec[41], input_vec[119], input_vec[24], input_vec[111], input_vec[82], input_vec[92], input_vec[31], input_vec[87], input_vec[26], input_vec[15], input_vec[142], input_vec[40], input_vec[139], input_vec[57], input_vec[109], input_vec[90], input_vec[5], input_vec[63], input_vec[113], input_vec[126], input_vec[121], input_vec[57], input_vec[126], input_vec[128]};

	// Neuron 336: 284 entries, bits from features [0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17]
	logic [31:0] addr_336;
	assign addr_336 = {input_vec[128], input_vec[54], input_vec[68], input_vec[73], input_vec[71], input_vec[136], input_vec[122], input_vec[94], input_vec[74], input_vec[39], input_vec[133], input_vec[141], input_vec[0], input_vec[68], input_vec[58], input_vec[113], input_vec[34], input_vec[63], input_vec[69], input_vec[128], input_vec[73], input_vec[90], input_vec[119], input_vec[114], input_vec[53], input_vec[42], input_vec[66], input_vec[137], input_vec[68], input_vec[86], input_vec[43], input_vec[112]};

	// Neuron 337: 1066 entries, bits from features [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_337;
	assign addr_337 = {input_vec[50], input_vec[63], input_vec[34], input_vec[133], input_vec[70], input_vec[67], input_vec[55], input_vec[19], input_vec[66], input_vec[10], input_vec[61], input_vec[83], input_vec[138], input_vec[54], input_vec[110], input_vec[123], input_vec[16], input_vec[69], input_vec[133], input_vec[94], input_vec[8], input_vec[98], input_vec[10], input_vec[47], input_vec[130], input_vec[40], input_vec[106], input_vec[95], input_vec[123], input_vec[8], input_vec[135], input_vec[64]};

	// Neuron 338: 563 entries, bits from features [0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_338;
	assign addr_338 = {input_vec[11], input_vec[80], input_vec[125], input_vec[97], input_vec[122], input_vec[12], input_vec[48], input_vec[131], input_vec[28], input_vec[133], input_vec[68], input_vec[21], input_vec[12], input_vec[84], input_vec[117], input_vec[82], input_vec[110], input_vec[20], input_vec[77], input_vec[27], input_vec[143], input_vec[110], input_vec[141], input_vec[121], input_vec[5], input_vec[57], input_vec[114], input_vec[8], input_vec[20], input_vec[114], input_vec[12], input_vec[124]};

	// Neuron 339: 787 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 14, 15, 16, 17]
	logic [31:0] addr_339;
	assign addr_339 = {input_vec[59], input_vec[140], input_vec[98], input_vec[60], input_vec[64], input_vec[38], input_vec[5], input_vec[133], input_vec[98], input_vec[46], input_vec[4], input_vec[31], input_vec[124], input_vec[18], input_vec[129], input_vec[27], input_vec[116], input_vec[35], input_vec[132], input_vec[19], input_vec[77], input_vec[64], input_vec[59], input_vec[9], input_vec[98], input_vec[42], input_vec[138], input_vec[35], input_vec[132], input_vec[69], input_vec[33], input_vec[100]};

	// Neuron 340: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_340;
	assign addr_340 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 341: 718 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_341;
	assign addr_341 = {input_vec[61], input_vec[65], input_vec[64], input_vec[108], input_vec[90], input_vec[86], input_vec[90], input_vec[73], input_vec[99], input_vec[24], input_vec[142], input_vec[36], input_vec[151], input_vec[149], input_vec[114], input_vec[4], input_vec[9], input_vec[73], input_vec[145], input_vec[115], input_vec[97], input_vec[139], input_vec[110], input_vec[96], input_vec[29], input_vec[41], input_vec[10], input_vec[130], input_vec[110], input_vec[129], input_vec[32], input_vec[18]};

	// Neuron 342: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_342;
	assign addr_342 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 343: 553 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18]
	logic [31:0] addr_343;
	assign addr_343 = {input_vec[12], input_vec[56], input_vec[48], input_vec[23], input_vec[15], input_vec[43], input_vec[72], input_vec[10], input_vec[77], input_vec[128], input_vec[127], input_vec[79], input_vec[148], input_vec[102], input_vec[13], input_vec[58], input_vec[131], input_vec[62], input_vec[145], input_vec[10], input_vec[39], input_vec[17], input_vec[58], input_vec[58], input_vec[27], input_vec[64], input_vec[147], input_vec[24], input_vec[79], input_vec[27], input_vec[100], input_vec[150]};

	// Neuron 344: 656 entries, bits from features [1, 2, 3, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_344;
	assign addr_344 = {input_vec[111], input_vec[131], input_vec[15], input_vec[114], input_vec[30], input_vec[124], input_vec[130], input_vec[17], input_vec[143], input_vec[147], input_vec[144], input_vec[75], input_vec[44], input_vec[46], input_vec[23], input_vec[43], input_vec[125], input_vec[99], input_vec[122], input_vec[86], input_vec[111], input_vec[139], input_vec[130], input_vec[145], input_vec[143], input_vec[105], input_vec[82], input_vec[118], input_vec[102], input_vec[21], input_vec[122], input_vec[26]};

	// Neuron 345: 630 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_345;
	assign addr_345 = {input_vec[31], input_vec[77], input_vec[55], input_vec[90], input_vec[48], input_vec[41], input_vec[5], input_vec[75], input_vec[114], input_vec[44], input_vec[55], input_vec[36], input_vec[25], input_vec[151], input_vec[63], input_vec[100], input_vec[110], input_vec[103], input_vec[14], input_vec[106], input_vec[71], input_vec[138], input_vec[72], input_vec[145], input_vec[58], input_vec[149], input_vec[73], input_vec[31], input_vec[91], input_vec[85], input_vec[92], input_vec[135]};

	// Neuron 346: 522 entries, bits from features [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_346;
	assign addr_346 = {input_vec[52], input_vec[143], input_vec[103], input_vec[131], input_vec[115], input_vec[97], input_vec[28], input_vec[124], input_vec[99], input_vec[32], input_vec[139], input_vec[86], input_vec[12], input_vec[131], input_vec[114], input_vec[37], input_vec[96], input_vec[67], input_vec[74], input_vec[90], input_vec[150], input_vec[99], input_vec[33], input_vec[120], input_vec[147], input_vec[91], input_vec[78], input_vec[124], input_vec[112], input_vec[42], input_vec[124], input_vec[100]};

	// Neuron 347: 952 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18]
	logic [31:0] addr_347;
	assign addr_347 = {input_vec[4], input_vec[87], input_vec[136], input_vec[149], input_vec[126], input_vec[123], input_vec[52], input_vec[36], input_vec[146], input_vec[59], input_vec[0], input_vec[86], input_vec[132], input_vec[67], input_vec[76], input_vec[103], input_vec[97], input_vec[19], input_vec[149], input_vec[96], input_vec[62], input_vec[49], input_vec[138], input_vec[45], input_vec[34], input_vec[76], input_vec[55], input_vec[47], input_vec[17], input_vec[59], input_vec[41], input_vec[7]};

	// Neuron 348: 915 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_348;
	assign addr_348 = {input_vec[87], input_vec[76], input_vec[11], input_vec[45], input_vec[23], input_vec[19], input_vec[57], input_vec[31], input_vec[101], input_vec[29], input_vec[137], input_vec[75], input_vec[129], input_vec[62], input_vec[26], input_vec[27], input_vec[17], input_vec[70], input_vec[150], input_vec[35], input_vec[142], input_vec[98], input_vec[32], input_vec[21], input_vec[107], input_vec[17], input_vec[130], input_vec[136], input_vec[136], input_vec[38], input_vec[60], input_vec[119]};

	// Neuron 349: 271 entries, bits from features [3, 5, 6, 7, 10, 11, 13, 15, 16, 17, 18]
	logic [31:0] addr_349;
	assign addr_349 = {input_vec[48], input_vec[138], input_vec[108], input_vec[139], input_vec[27], input_vec[55], input_vec[123], input_vec[124], input_vec[93], input_vec[136], input_vec[130], input_vec[110], input_vec[121], input_vec[46], input_vec[149], input_vec[140], input_vec[46], input_vec[91], input_vec[62], input_vec[92], input_vec[52], input_vec[140], input_vec[139], input_vec[93], input_vec[29], input_vec[27], input_vec[126], input_vec[144], input_vec[53], input_vec[106], input_vec[93], input_vec[81]};

	// Neuron 350: 981 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_350;
	assign addr_350 = {input_vec[35], input_vec[126], input_vec[105], input_vec[39], input_vec[17], input_vec[70], input_vec[33], input_vec[85], input_vec[30], input_vec[147], input_vec[104], input_vec[35], input_vec[140], input_vec[127], input_vec[135], input_vec[102], input_vec[104], input_vec[97], input_vec[50], input_vec[127], input_vec[136], input_vec[6], input_vec[78], input_vec[151], input_vec[87], input_vec[123], input_vec[22], input_vec[104], input_vec[89], input_vec[29], input_vec[9], input_vec[38]};

	// Neuron 351: 955 entries, bits from features [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 17, 18]
	logic [31:0] addr_351;
	assign addr_351 = {input_vec[64], input_vec[52], input_vec[49], input_vec[144], input_vec[2], input_vec[147], input_vec[17], input_vec[43], input_vec[116], input_vec[100], input_vec[75], input_vec[99], input_vec[45], input_vec[139], input_vec[15], input_vec[18], input_vec[32], input_vec[110], input_vec[72], input_vec[69], input_vec[23], input_vec[140], input_vec[125], input_vec[146], input_vec[19], input_vec[91], input_vec[113], input_vec[12], input_vec[91], input_vec[144], input_vec[116], input_vec[151]};

	// Neuron 352: 1352 entries, bits from features [1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_352;
	assign addr_352 = {input_vec[132], input_vec[115], input_vec[87], input_vec[86], input_vec[98], input_vec[28], input_vec[68], input_vec[143], input_vec[65], input_vec[117], input_vec[91], input_vec[103], input_vec[65], input_vec[124], input_vec[146], input_vec[12], input_vec[103], input_vec[13], input_vec[61], input_vec[18], input_vec[101], input_vec[67], input_vec[18], input_vec[90], input_vec[41], input_vec[107], input_vec[107], input_vec[151], input_vec[135], input_vec[22], input_vec[119], input_vec[21]};

	// Neuron 353: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_353;
	assign addr_353 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 354: 481 entries, bits from features [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_354;
	assign addr_354 = {input_vec[29], input_vec[112], input_vec[24], input_vec[60], input_vec[90], input_vec[138], input_vec[91], input_vec[105], input_vec[130], input_vec[20], input_vec[96], input_vec[108], input_vec[61], input_vec[83], input_vec[6], input_vec[55], input_vec[119], input_vec[134], input_vec[122], input_vec[138], input_vec[98], input_vec[115], input_vec[138], input_vec[96], input_vec[47], input_vec[38], input_vec[22], input_vec[126], input_vec[72], input_vec[0], input_vec[145], input_vec[89]};

	// Neuron 355: 1099 entries, bits from features [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	logic [31:0] addr_355;
	assign addr_355 = {input_vec[130], input_vec[69], input_vec[53], input_vec[112], input_vec[71], input_vec[100], input_vec[1], input_vec[110], input_vec[27], input_vec[18], input_vec[68], input_vec[107], input_vec[92], input_vec[136], input_vec[22], input_vec[7], input_vec[65], input_vec[141], input_vec[81], input_vec[95], input_vec[83], input_vec[48], input_vec[77], input_vec[47], input_vec[109], input_vec[1], input_vec[127], input_vec[38], input_vec[98], input_vec[113], input_vec[32], input_vec[22]};

	// Neuron 356: 784 entries, bits from features [0, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 17, 18]
	logic [31:0] addr_356;
	assign addr_356 = {input_vec[147], input_vec[43], input_vec[111], input_vec[147], input_vec[61], input_vec[82], input_vec[20], input_vec[112], input_vec[92], input_vec[6], input_vec[133], input_vec[65], input_vec[45], input_vec[89], input_vec[53], input_vec[45], input_vec[49], input_vec[49], input_vec[81], input_vec[40], input_vec[21], input_vec[141], input_vec[40], input_vec[143], input_vec[94], input_vec[39], input_vec[57], input_vec[139], input_vec[3], input_vec[18], input_vec[5], input_vec[93]};

	// Neuron 357: 1756 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]
	logic [31:0] addr_357;
	assign addr_357 = {input_vec[44], input_vec[101], input_vec[56], input_vec[69], input_vec[46], input_vec[93], input_vec[128], input_vec[29], input_vec[96], input_vec[41], input_vec[103], input_vec[88], input_vec[84], input_vec[139], input_vec[71], input_vec[135], input_vec[3], input_vec[146], input_vec[33], input_vec[62], input_vec[76], input_vec[131], input_vec[87], input_vec[133], input_vec[100], input_vec[86], input_vec[12], input_vec[7], input_vec[21], input_vec[49], input_vec[72], input_vec[58]};

	// Neuron 358: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18]
	logic [31:0] addr_358;
	assign addr_358 = {input_vec[7], input_vec[22], input_vec[39], input_vec[47], input_vec[114], input_vec[58], input_vec[95], input_vec[40], input_vec[69], input_vec[85], input_vec[20], input_vec[144], input_vec[90], input_vec[11], input_vec[45], input_vec[43], input_vec[137], input_vec[6], input_vec[33], input_vec[77], input_vec[55], input_vec[141], input_vec[26], input_vec[16], input_vec[40], input_vec[102], input_vec[28], input_vec[96], input_vec[36], input_vec[67], input_vec[146], input_vec[86]};

	// Neuron 359: 833 entries, bits from features [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
	logic [31:0] addr_359;
	assign addr_359 = {input_vec[89], input_vec[31], input_vec[66], input_vec[15], input_vec[13], input_vec[41], input_vec[44], input_vec[44], input_vec[96], input_vec[116], input_vec[105], input_vec[30], input_vec[13], input_vec[140], input_vec[45], input_vec[146], input_vec[86], input_vec[33], input_vec[10], input_vec[28], input_vec[47], input_vec[107], input_vec[20], input_vec[32], input_vec[136], input_vec[33], input_vec[62], input_vec[24], input_vec[19], input_vec[79], input_vec[130], input_vec[130]};

	// Neuron 360: 369 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
	logic [31:0] addr_360;
	assign addr_360 = {input_vec[107], input_vec[109], input_vec[51], input_vec[34], input_vec[121], input_vec[133], input_vec[81], input_vec[66], input_vec[22], input_vec[25], input_vec[126], input_vec[103], input_vec[133], input_vec[85], input_vec[74], input_vec[127], input_vec[76], input_vec[97], input_vec[41], input_vec[34], input_vec[61], input_vec[36], input_vec[81], input_vec[140], input_vec[56], input_vec[122], input_vec[42], input_vec[24], input_vec[78], input_vec[65], input_vec[88], input_vec[14]};

	// Neuron 361: 1201 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 16, 18]
	logic [31:0] addr_361;
	assign addr_361 = {input_vec[107], input_vec[111], input_vec[87], input_vec[119], input_vec[28], input_vec[91], input_vec[2], input_vec[89], input_vec[100], input_vec[132], input_vec[19], input_vec[47], input_vec[89], input_vec[25], input_vec[25], input_vec[94], input_vec[109], input_vec[48], input_vec[37], input_vec[31], input_vec[42], input_vec[38], input_vec[115], input_vec[117], input_vec[10], input_vec[24], input_vec[149], input_vec[81], input_vec[45], input_vec[81], input_vec[99], input_vec[151]};

	// Neuron 362: 514 entries, bits from features [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_362;
	assign addr_362 = {input_vec[61], input_vec[140], input_vec[84], input_vec[106], input_vec[78], input_vec[130], input_vec[54], input_vec[137], input_vec[110], input_vec[59], input_vec[9], input_vec[2], input_vec[25], input_vec[11], input_vec[105], input_vec[31], input_vec[127], input_vec[150], input_vec[75], input_vec[103], input_vec[129], input_vec[138], input_vec[47], input_vec[79], input_vec[132], input_vec[58], input_vec[47], input_vec[40], input_vec[85], input_vec[1], input_vec[91], input_vec[11]};

	// Neuron 363: 754 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_363;
	assign addr_363 = {input_vec[84], input_vec[24], input_vec[3], input_vec[151], input_vec[41], input_vec[34], input_vec[116], input_vec[118], input_vec[95], input_vec[54], input_vec[19], input_vec[136], input_vec[61], input_vec[120], input_vec[130], input_vec[17], input_vec[33], input_vec[141], input_vec[69], input_vec[112], input_vec[108], input_vec[108], input_vec[53], input_vec[9], input_vec[110], input_vec[81], input_vec[137], input_vec[25], input_vec[6], input_vec[118], input_vec[76], input_vec[124]};

	// Neuron 364: 1265 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18]
	logic [31:0] addr_364;
	assign addr_364 = {input_vec[150], input_vec[146], input_vec[70], input_vec[2], input_vec[137], input_vec[144], input_vec[126], input_vec[45], input_vec[77], input_vec[119], input_vec[122], input_vec[81], input_vec[35], input_vec[0], input_vec[18], input_vec[129], input_vec[114], input_vec[120], input_vec[87], input_vec[133], input_vec[55], input_vec[12], input_vec[13], input_vec[117], input_vec[47], input_vec[148], input_vec[116], input_vec[14], input_vec[30], input_vec[89], input_vec[115], input_vec[24]};

	// Neuron 365: 1109 entries, bits from features [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_365;
	assign addr_365 = {input_vec[96], input_vec[48], input_vec[114], input_vec[116], input_vec[1], input_vec[88], input_vec[23], input_vec[144], input_vec[13], input_vec[94], input_vec[31], input_vec[11], input_vec[100], input_vec[73], input_vec[128], input_vec[123], input_vec[83], input_vec[64], input_vec[135], input_vec[106], input_vec[20], input_vec[40], input_vec[136], input_vec[116], input_vec[10], input_vec[21], input_vec[6], input_vec[69], input_vec[120], input_vec[148], input_vec[151], input_vec[41]};

	// Neuron 366: 1291 entries, bits from features [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_366;
	assign addr_366 = {input_vec[102], input_vec[108], input_vec[96], input_vec[133], input_vec[111], input_vec[9], input_vec[151], input_vec[78], input_vec[145], input_vec[40], input_vec[39], input_vec[92], input_vec[76], input_vec[6], input_vec[35], input_vec[104], input_vec[127], input_vec[151], input_vec[146], input_vec[64], input_vec[19], input_vec[30], input_vec[139], input_vec[65], input_vec[84], input_vec[98], input_vec[111], input_vec[102], input_vec[18], input_vec[116], input_vec[119], input_vec[37]};

	// Neuron 367: 924 entries, bits from features [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_367;
	assign addr_367 = {input_vec[74], input_vec[68], input_vec[144], input_vec[20], input_vec[82], input_vec[130], input_vec[111], input_vec[125], input_vec[109], input_vec[17], input_vec[149], input_vec[94], input_vec[90], input_vec[130], input_vec[26], input_vec[85], input_vec[149], input_vec[33], input_vec[12], input_vec[114], input_vec[117], input_vec[136], input_vec[115], input_vec[18], input_vec[143], input_vec[83], input_vec[68], input_vec[122], input_vec[4], input_vec[50], input_vec[147], input_vec[130]};

	// Neuron 368: 698 entries, bits from features [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 18]
	logic [31:0] addr_368;
	assign addr_368 = {input_vec[83], input_vec[54], input_vec[151], input_vec[26], input_vec[76], input_vec[31], input_vec[44], input_vec[68], input_vec[107], input_vec[114], input_vec[108], input_vec[116], input_vec[5], input_vec[23], input_vec[19], input_vec[53], input_vec[54], input_vec[74], input_vec[51], input_vec[90], input_vec[21], input_vec[78], input_vec[87], input_vec[145], input_vec[145], input_vec[56], input_vec[3], input_vec[3], input_vec[104], input_vec[56], input_vec[150], input_vec[55]};

	// --- Neuron instances ---
	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
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
		.NUM_ENTRIES(656),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
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
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
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
		.NUM_ENTRIES(1102),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(378),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
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
		.NUM_ENTRIES(630),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
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
		.NUM_ENTRIES(601),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
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
		.NUM_ENTRIES(326),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
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
		.NUM_ENTRIES(814),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
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
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
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

	wnn_neuron #(
		.NUM_ENTRIES(522),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_11 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_11),
		.result(neuron_result[11]),
		.result_valid(neuron_valid[11]),
		.busy(neuron_busy[11])
	);
	// BRAM init: $readmemh("mem/neuron_011_keys.mem", neuron_11.key_mem);
	// BRAM init: $readmemh("mem/neuron_011_values.mem", neuron_11.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(897),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_12 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_12),
		.result(neuron_result[12]),
		.result_valid(neuron_valid[12]),
		.busy(neuron_busy[12])
	);
	// BRAM init: $readmemh("mem/neuron_012_keys.mem", neuron_12.key_mem);
	// BRAM init: $readmemh("mem/neuron_012_values.mem", neuron_12.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1102),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_13 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_13),
		.result(neuron_result[13]),
		.result_valid(neuron_valid[13]),
		.busy(neuron_busy[13])
	);
	// BRAM init: $readmemh("mem/neuron_013_keys.mem", neuron_13.key_mem);
	// BRAM init: $readmemh("mem/neuron_013_values.mem", neuron_13.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_14 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_14),
		.result(neuron_result[14]),
		.result_valid(neuron_valid[14]),
		.busy(neuron_busy[14])
	);
	// BRAM init: $readmemh("mem/neuron_014_keys.mem", neuron_14.key_mem);
	// BRAM init: $readmemh("mem/neuron_014_values.mem", neuron_14.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_15 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_15),
		.result(neuron_result[15]),
		.result_valid(neuron_valid[15]),
		.busy(neuron_busy[15])
	);
	// BRAM init: $readmemh("mem/neuron_015_keys.mem", neuron_15.key_mem);
	// BRAM init: $readmemh("mem/neuron_015_values.mem", neuron_15.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(917),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_16 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_16),
		.result(neuron_result[16]),
		.result_valid(neuron_valid[16]),
		.busy(neuron_busy[16])
	);
	// BRAM init: $readmemh("mem/neuron_016_keys.mem", neuron_16.key_mem);
	// BRAM init: $readmemh("mem/neuron_016_values.mem", neuron_16.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(343),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_17 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_17),
		.result(neuron_result[17]),
		.result_valid(neuron_valid[17]),
		.busy(neuron_busy[17])
	);
	// BRAM init: $readmemh("mem/neuron_017_keys.mem", neuron_17.key_mem);
	// BRAM init: $readmemh("mem/neuron_017_values.mem", neuron_17.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(386),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_18 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_18),
		.result(neuron_result[18]),
		.result_valid(neuron_valid[18]),
		.busy(neuron_busy[18])
	);
	// BRAM init: $readmemh("mem/neuron_018_keys.mem", neuron_18.key_mem);
	// BRAM init: $readmemh("mem/neuron_018_values.mem", neuron_18.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(438),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_19 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_19),
		.result(neuron_result[19]),
		.result_valid(neuron_valid[19]),
		.busy(neuron_busy[19])
	);
	// BRAM init: $readmemh("mem/neuron_019_keys.mem", neuron_19.key_mem);
	// BRAM init: $readmemh("mem/neuron_019_values.mem", neuron_19.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_20 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_20),
		.result(neuron_result[20]),
		.result_valid(neuron_valid[20]),
		.busy(neuron_busy[20])
	);
	// BRAM init: $readmemh("mem/neuron_020_keys.mem", neuron_20.key_mem);
	// BRAM init: $readmemh("mem/neuron_020_values.mem", neuron_20.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(730),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_21 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_21),
		.result(neuron_result[21]),
		.result_valid(neuron_valid[21]),
		.busy(neuron_busy[21])
	);
	// BRAM init: $readmemh("mem/neuron_021_keys.mem", neuron_21.key_mem);
	// BRAM init: $readmemh("mem/neuron_021_values.mem", neuron_21.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(546),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_22 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_22),
		.result(neuron_result[22]),
		.result_valid(neuron_valid[22]),
		.busy(neuron_busy[22])
	);
	// BRAM init: $readmemh("mem/neuron_022_keys.mem", neuron_22.key_mem);
	// BRAM init: $readmemh("mem/neuron_022_values.mem", neuron_22.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_23 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_23),
		.result(neuron_result[23]),
		.result_valid(neuron_valid[23]),
		.busy(neuron_busy[23])
	);
	// BRAM init: $readmemh("mem/neuron_023_keys.mem", neuron_23.key_mem);
	// BRAM init: $readmemh("mem/neuron_023_values.mem", neuron_23.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(481),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_24 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_24),
		.result(neuron_result[24]),
		.result_valid(neuron_valid[24]),
		.busy(neuron_busy[24])
	);
	// BRAM init: $readmemh("mem/neuron_024_keys.mem", neuron_24.key_mem);
	// BRAM init: $readmemh("mem/neuron_024_values.mem", neuron_24.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(545),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_25 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_25),
		.result(neuron_result[25]),
		.result_valid(neuron_valid[25]),
		.busy(neuron_busy[25])
	);
	// BRAM init: $readmemh("mem/neuron_025_keys.mem", neuron_25.key_mem);
	// BRAM init: $readmemh("mem/neuron_025_values.mem", neuron_25.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(429),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_26 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_26),
		.result(neuron_result[26]),
		.result_valid(neuron_valid[26]),
		.busy(neuron_busy[26])
	);
	// BRAM init: $readmemh("mem/neuron_026_keys.mem", neuron_26.key_mem);
	// BRAM init: $readmemh("mem/neuron_026_values.mem", neuron_26.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(730),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_27 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_27),
		.result(neuron_result[27]),
		.result_valid(neuron_valid[27]),
		.busy(neuron_busy[27])
	);
	// BRAM init: $readmemh("mem/neuron_027_keys.mem", neuron_27.key_mem);
	// BRAM init: $readmemh("mem/neuron_027_values.mem", neuron_27.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(844),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_28 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_28),
		.result(neuron_result[28]),
		.result_valid(neuron_valid[28]),
		.busy(neuron_busy[28])
	);
	// BRAM init: $readmemh("mem/neuron_028_keys.mem", neuron_28.key_mem);
	// BRAM init: $readmemh("mem/neuron_028_values.mem", neuron_28.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(522),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_29 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_29),
		.result(neuron_result[29]),
		.result_valid(neuron_valid[29]),
		.busy(neuron_busy[29])
	);
	// BRAM init: $readmemh("mem/neuron_029_keys.mem", neuron_29.key_mem);
	// BRAM init: $readmemh("mem/neuron_029_values.mem", neuron_29.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(326),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_30 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_30),
		.result(neuron_result[30]),
		.result_valid(neuron_valid[30]),
		.busy(neuron_busy[30])
	);
	// BRAM init: $readmemh("mem/neuron_030_keys.mem", neuron_30.key_mem);
	// BRAM init: $readmemh("mem/neuron_030_values.mem", neuron_30.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(386),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_31 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_31),
		.result(neuron_result[31]),
		.result_valid(neuron_valid[31]),
		.busy(neuron_busy[31])
	);
	// BRAM init: $readmemh("mem/neuron_031_keys.mem", neuron_31.key_mem);
	// BRAM init: $readmemh("mem/neuron_031_values.mem", neuron_31.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_32 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_32),
		.result(neuron_result[32]),
		.result_valid(neuron_valid[32]),
		.busy(neuron_busy[32])
	);
	// BRAM init: $readmemh("mem/neuron_032_keys.mem", neuron_32.key_mem);
	// BRAM init: $readmemh("mem/neuron_032_values.mem", neuron_32.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(545),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_33 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_33),
		.result(neuron_result[33]),
		.result_valid(neuron_valid[33]),
		.busy(neuron_busy[33])
	);
	// BRAM init: $readmemh("mem/neuron_033_keys.mem", neuron_33.key_mem);
	// BRAM init: $readmemh("mem/neuron_033_values.mem", neuron_33.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_34 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_34),
		.result(neuron_result[34]),
		.result_valid(neuron_valid[34]),
		.busy(neuron_busy[34])
	);
	// BRAM init: $readmemh("mem/neuron_034_keys.mem", neuron_34.key_mem);
	// BRAM init: $readmemh("mem/neuron_034_values.mem", neuron_34.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_35 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_35),
		.result(neuron_result[35]),
		.result_valid(neuron_valid[35]),
		.busy(neuron_busy[35])
	);
	// BRAM init: $readmemh("mem/neuron_035_keys.mem", neuron_35.key_mem);
	// BRAM init: $readmemh("mem/neuron_035_values.mem", neuron_35.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(898),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_36 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_36),
		.result(neuron_result[36]),
		.result_valid(neuron_valid[36]),
		.busy(neuron_busy[36])
	);
	// BRAM init: $readmemh("mem/neuron_036_keys.mem", neuron_36.key_mem);
	// BRAM init: $readmemh("mem/neuron_036_values.mem", neuron_36.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(943),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_37 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_37),
		.result(neuron_result[37]),
		.result_valid(neuron_valid[37]),
		.busy(neuron_busy[37])
	);
	// BRAM init: $readmemh("mem/neuron_037_keys.mem", neuron_37.key_mem);
	// BRAM init: $readmemh("mem/neuron_037_values.mem", neuron_37.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_38 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_38),
		.result(neuron_result[38]),
		.result_valid(neuron_valid[38]),
		.busy(neuron_busy[38])
	);
	// BRAM init: $readmemh("mem/neuron_038_keys.mem", neuron_38.key_mem);
	// BRAM init: $readmemh("mem/neuron_038_values.mem", neuron_38.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1785),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_39 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_39),
		.result(neuron_result[39]),
		.result_valid(neuron_valid[39]),
		.busy(neuron_busy[39])
	);
	// BRAM init: $readmemh("mem/neuron_039_keys.mem", neuron_39.key_mem);
	// BRAM init: $readmemh("mem/neuron_039_values.mem", neuron_39.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(892),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_40 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_40),
		.result(neuron_result[40]),
		.result_valid(neuron_valid[40]),
		.busy(neuron_busy[40])
	);
	// BRAM init: $readmemh("mem/neuron_040_keys.mem", neuron_40.key_mem);
	// BRAM init: $readmemh("mem/neuron_040_values.mem", neuron_40.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(730),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_41 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_41),
		.result(neuron_result[41]),
		.result_valid(neuron_valid[41]),
		.busy(neuron_busy[41])
	);
	// BRAM init: $readmemh("mem/neuron_041_keys.mem", neuron_41.key_mem);
	// BRAM init: $readmemh("mem/neuron_041_values.mem", neuron_41.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(897),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_42 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_42),
		.result(neuron_result[42]),
		.result_valid(neuron_valid[42]),
		.busy(neuron_busy[42])
	);
	// BRAM init: $readmemh("mem/neuron_042_keys.mem", neuron_42.key_mem);
	// BRAM init: $readmemh("mem/neuron_042_values.mem", neuron_42.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(254),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_43 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_43),
		.result(neuron_result[43]),
		.result_valid(neuron_valid[43]),
		.busy(neuron_busy[43])
	);
	// BRAM init: $readmemh("mem/neuron_043_keys.mem", neuron_43.key_mem);
	// BRAM init: $readmemh("mem/neuron_043_values.mem", neuron_43.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(284),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_44 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_44),
		.result(neuron_result[44]),
		.result_valid(neuron_valid[44]),
		.busy(neuron_busy[44])
	);
	// BRAM init: $readmemh("mem/neuron_044_keys.mem", neuron_44.key_mem);
	// BRAM init: $readmemh("mem/neuron_044_values.mem", neuron_44.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_45 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_45),
		.result(neuron_result[45]),
		.result_valid(neuron_valid[45]),
		.busy(neuron_busy[45])
	);
	// BRAM init: $readmemh("mem/neuron_045_keys.mem", neuron_45.key_mem);
	// BRAM init: $readmemh("mem/neuron_045_values.mem", neuron_45.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_46 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_46),
		.result(neuron_result[46]),
		.result_valid(neuron_valid[46]),
		.busy(neuron_busy[46])
	);
	// BRAM init: $readmemh("mem/neuron_046_keys.mem", neuron_46.key_mem);
	// BRAM init: $readmemh("mem/neuron_046_values.mem", neuron_46.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_47 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_47),
		.result(neuron_result[47]),
		.result_valid(neuron_valid[47]),
		.busy(neuron_busy[47])
	);
	// BRAM init: $readmemh("mem/neuron_047_keys.mem", neuron_47.key_mem);
	// BRAM init: $readmemh("mem/neuron_047_values.mem", neuron_47.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_48 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_48),
		.result(neuron_result[48]),
		.result_valid(neuron_valid[48]),
		.busy(neuron_busy[48])
	);
	// BRAM init: $readmemh("mem/neuron_048_keys.mem", neuron_48.key_mem);
	// BRAM init: $readmemh("mem/neuron_048_values.mem", neuron_48.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(819),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_49 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_49),
		.result(neuron_result[49]),
		.result_valid(neuron_valid[49]),
		.busy(neuron_busy[49])
	);
	// BRAM init: $readmemh("mem/neuron_049_keys.mem", neuron_49.key_mem);
	// BRAM init: $readmemh("mem/neuron_049_values.mem", neuron_49.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_50 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_50),
		.result(neuron_result[50]),
		.result_valid(neuron_valid[50]),
		.busy(neuron_busy[50])
	);
	// BRAM init: $readmemh("mem/neuron_050_keys.mem", neuron_50.key_mem);
	// BRAM init: $readmemh("mem/neuron_050_values.mem", neuron_50.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(420),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_51 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_51),
		.result(neuron_result[51]),
		.result_valid(neuron_valid[51]),
		.busy(neuron_busy[51])
	);
	// BRAM init: $readmemh("mem/neuron_051_keys.mem", neuron_51.key_mem);
	// BRAM init: $readmemh("mem/neuron_051_values.mem", neuron_51.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(368),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_52 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_52),
		.result(neuron_result[52]),
		.result_valid(neuron_valid[52]),
		.busy(neuron_busy[52])
	);
	// BRAM init: $readmemh("mem/neuron_052_keys.mem", neuron_52.key_mem);
	// BRAM init: $readmemh("mem/neuron_052_values.mem", neuron_52.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(735),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_53 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_53),
		.result(neuron_result[53]),
		.result_valid(neuron_valid[53]),
		.busy(neuron_busy[53])
	);
	// BRAM init: $readmemh("mem/neuron_053_keys.mem", neuron_53.key_mem);
	// BRAM init: $readmemh("mem/neuron_053_values.mem", neuron_53.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_54 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_54),
		.result(neuron_result[54]),
		.result_valid(neuron_valid[54]),
		.busy(neuron_busy[54])
	);
	// BRAM init: $readmemh("mem/neuron_054_keys.mem", neuron_54.key_mem);
	// BRAM init: $readmemh("mem/neuron_054_values.mem", neuron_54.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_55 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_55),
		.result(neuron_result[55]),
		.result_valid(neuron_valid[55]),
		.busy(neuron_busy[55])
	);
	// BRAM init: $readmemh("mem/neuron_055_keys.mem", neuron_55.key_mem);
	// BRAM init: $readmemh("mem/neuron_055_values.mem", neuron_55.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1124),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_56 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_56),
		.result(neuron_result[56]),
		.result_valid(neuron_valid[56]),
		.busy(neuron_busy[56])
	);
	// BRAM init: $readmemh("mem/neuron_056_keys.mem", neuron_56.key_mem);
	// BRAM init: $readmemh("mem/neuron_056_values.mem", neuron_56.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_57 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_57),
		.result(neuron_result[57]),
		.result_valid(neuron_valid[57]),
		.busy(neuron_busy[57])
	);
	// BRAM init: $readmemh("mem/neuron_057_keys.mem", neuron_57.key_mem);
	// BRAM init: $readmemh("mem/neuron_057_values.mem", neuron_57.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_58 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_58),
		.result(neuron_result[58]),
		.result_valid(neuron_valid[58]),
		.busy(neuron_busy[58])
	);
	// BRAM init: $readmemh("mem/neuron_058_keys.mem", neuron_58.key_mem);
	// BRAM init: $readmemh("mem/neuron_058_values.mem", neuron_58.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(656),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_59 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_59),
		.result(neuron_result[59]),
		.result_valid(neuron_valid[59]),
		.busy(neuron_busy[59])
	);
	// BRAM init: $readmemh("mem/neuron_059_keys.mem", neuron_59.key_mem);
	// BRAM init: $readmemh("mem/neuron_059_values.mem", neuron_59.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(601),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_60 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_60),
		.result(neuron_result[60]),
		.result_valid(neuron_valid[60]),
		.busy(neuron_busy[60])
	);
	// BRAM init: $readmemh("mem/neuron_060_keys.mem", neuron_60.key_mem);
	// BRAM init: $readmemh("mem/neuron_060_values.mem", neuron_60.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(481),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_61 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_61),
		.result(neuron_result[61]),
		.result_valid(neuron_valid[61]),
		.busy(neuron_busy[61])
	);
	// BRAM init: $readmemh("mem/neuron_061_keys.mem", neuron_61.key_mem);
	// BRAM init: $readmemh("mem/neuron_061_values.mem", neuron_61.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(898),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_62 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_62),
		.result(neuron_result[62]),
		.result_valid(neuron_valid[62]),
		.busy(neuron_busy[62])
	);
	// BRAM init: $readmemh("mem/neuron_062_keys.mem", neuron_62.key_mem);
	// BRAM init: $readmemh("mem/neuron_062_values.mem", neuron_62.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(232),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_63 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_63),
		.result(neuron_result[63]),
		.result_valid(neuron_valid[63]),
		.busy(neuron_busy[63])
	);
	// BRAM init: $readmemh("mem/neuron_063_keys.mem", neuron_63.key_mem);
	// BRAM init: $readmemh("mem/neuron_063_values.mem", neuron_63.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(447),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_64 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_64),
		.result(neuron_result[64]),
		.result_valid(neuron_valid[64]),
		.busy(neuron_busy[64])
	);
	// BRAM init: $readmemh("mem/neuron_064_keys.mem", neuron_64.key_mem);
	// BRAM init: $readmemh("mem/neuron_064_values.mem", neuron_64.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1034),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_65 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_65),
		.result(neuron_result[65]),
		.result_valid(neuron_valid[65]),
		.busy(neuron_busy[65])
	);
	// BRAM init: $readmemh("mem/neuron_065_keys.mem", neuron_65.key_mem);
	// BRAM init: $readmemh("mem/neuron_065_values.mem", neuron_65.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_66 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_66),
		.result(neuron_result[66]),
		.result_valid(neuron_valid[66]),
		.busy(neuron_busy[66])
	);
	// BRAM init: $readmemh("mem/neuron_066_keys.mem", neuron_66.key_mem);
	// BRAM init: $readmemh("mem/neuron_066_values.mem", neuron_66.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_67 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_67),
		.result(neuron_result[67]),
		.result_valid(neuron_valid[67]),
		.busy(neuron_busy[67])
	);
	// BRAM init: $readmemh("mem/neuron_067_keys.mem", neuron_67.key_mem);
	// BRAM init: $readmemh("mem/neuron_067_values.mem", neuron_67.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_68 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_68),
		.result(neuron_result[68]),
		.result_valid(neuron_valid[68]),
		.busy(neuron_busy[68])
	);
	// BRAM init: $readmemh("mem/neuron_068_keys.mem", neuron_68.key_mem);
	// BRAM init: $readmemh("mem/neuron_068_values.mem", neuron_68.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(326),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_69 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_69),
		.result(neuron_result[69]),
		.result_valid(neuron_valid[69]),
		.busy(neuron_busy[69])
	);
	// BRAM init: $readmemh("mem/neuron_069_keys.mem", neuron_69.key_mem);
	// BRAM init: $readmemh("mem/neuron_069_values.mem", neuron_69.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(796),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_70 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_70),
		.result(neuron_result[70]),
		.result_valid(neuron_valid[70]),
		.busy(neuron_busy[70])
	);
	// BRAM init: $readmemh("mem/neuron_070_keys.mem", neuron_70.key_mem);
	// BRAM init: $readmemh("mem/neuron_070_values.mem", neuron_70.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_71 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_71),
		.result(neuron_result[71]),
		.result_valid(neuron_valid[71]),
		.busy(neuron_busy[71])
	);
	// BRAM init: $readmemh("mem/neuron_071_keys.mem", neuron_71.key_mem);
	// BRAM init: $readmemh("mem/neuron_071_values.mem", neuron_71.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(772),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_72 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_72),
		.result(neuron_result[72]),
		.result_valid(neuron_valid[72]),
		.busy(neuron_busy[72])
	);
	// BRAM init: $readmemh("mem/neuron_072_keys.mem", neuron_72.key_mem);
	// BRAM init: $readmemh("mem/neuron_072_values.mem", neuron_72.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(718),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_73 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_73),
		.result(neuron_result[73]),
		.result_valid(neuron_valid[73]),
		.busy(neuron_busy[73])
	);
	// BRAM init: $readmemh("mem/neuron_073_keys.mem", neuron_73.key_mem);
	// BRAM init: $readmemh("mem/neuron_073_values.mem", neuron_73.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(386),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_74 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_74),
		.result(neuron_result[74]),
		.result_valid(neuron_valid[74]),
		.busy(neuron_busy[74])
	);
	// BRAM init: $readmemh("mem/neuron_074_keys.mem", neuron_74.key_mem);
	// BRAM init: $readmemh("mem/neuron_074_values.mem", neuron_74.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1881),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_75 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_75),
		.result(neuron_result[75]),
		.result_valid(neuron_valid[75]),
		.busy(neuron_busy[75])
	);
	// BRAM init: $readmemh("mem/neuron_075_keys.mem", neuron_75.key_mem);
	// BRAM init: $readmemh("mem/neuron_075_values.mem", neuron_75.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(927),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_76 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_76),
		.result(neuron_result[76]),
		.result_valid(neuron_valid[76]),
		.busy(neuron_busy[76])
	);
	// BRAM init: $readmemh("mem/neuron_076_keys.mem", neuron_76.key_mem);
	// BRAM init: $readmemh("mem/neuron_076_values.mem", neuron_76.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_77 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_77),
		.result(neuron_result[77]),
		.result_valid(neuron_valid[77]),
		.busy(neuron_busy[77])
	);
	// BRAM init: $readmemh("mem/neuron_077_keys.mem", neuron_77.key_mem);
	// BRAM init: $readmemh("mem/neuron_077_values.mem", neuron_77.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(955),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_78 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_78),
		.result(neuron_result[78]),
		.result_valid(neuron_valid[78]),
		.busy(neuron_busy[78])
	);
	// BRAM init: $readmemh("mem/neuron_078_keys.mem", neuron_78.key_mem);
	// BRAM init: $readmemh("mem/neuron_078_values.mem", neuron_78.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1756),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_79 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_79),
		.result(neuron_result[79]),
		.result_valid(neuron_valid[79]),
		.busy(neuron_busy[79])
	);
	// BRAM init: $readmemh("mem/neuron_079_keys.mem", neuron_79.key_mem);
	// BRAM init: $readmemh("mem/neuron_079_values.mem", neuron_79.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_80 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_80),
		.result(neuron_result[80]),
		.result_valid(neuron_valid[80]),
		.busy(neuron_busy[80])
	);
	// BRAM init: $readmemh("mem/neuron_080_keys.mem", neuron_80.key_mem);
	// BRAM init: $readmemh("mem/neuron_080_values.mem", neuron_80.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(481),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_81 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_81),
		.result(neuron_result[81]),
		.result_valid(neuron_valid[81]),
		.busy(neuron_busy[81])
	);
	// BRAM init: $readmemh("mem/neuron_081_keys.mem", neuron_81.key_mem);
	// BRAM init: $readmemh("mem/neuron_081_values.mem", neuron_81.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(420),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_82 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_82),
		.result(neuron_result[82]),
		.result_valid(neuron_valid[82]),
		.busy(neuron_busy[82])
	);
	// BRAM init: $readmemh("mem/neuron_082_keys.mem", neuron_82.key_mem);
	// BRAM init: $readmemh("mem/neuron_082_values.mem", neuron_82.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(840),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_83 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_83),
		.result(neuron_result[83]),
		.result_valid(neuron_valid[83]),
		.busy(neuron_busy[83])
	);
	// BRAM init: $readmemh("mem/neuron_083_keys.mem", neuron_83.key_mem);
	// BRAM init: $readmemh("mem/neuron_083_values.mem", neuron_83.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1588),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_84 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_84),
		.result(neuron_result[84]),
		.result_valid(neuron_valid[84]),
		.busy(neuron_busy[84])
	);
	// BRAM init: $readmemh("mem/neuron_084_keys.mem", neuron_84.key_mem);
	// BRAM init: $readmemh("mem/neuron_084_values.mem", neuron_84.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_85 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_85),
		.result(neuron_result[85]),
		.result_valid(neuron_valid[85]),
		.busy(neuron_busy[85])
	);
	// BRAM init: $readmemh("mem/neuron_085_keys.mem", neuron_85.key_mem);
	// BRAM init: $readmemh("mem/neuron_085_values.mem", neuron_85.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(814),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_86 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_86),
		.result(neuron_result[86]),
		.result_valid(neuron_valid[86]),
		.busy(neuron_busy[86])
	);
	// BRAM init: $readmemh("mem/neuron_086_keys.mem", neuron_86.key_mem);
	// BRAM init: $readmemh("mem/neuron_086_values.mem", neuron_86.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(927),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_87 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_87),
		.result(neuron_result[87]),
		.result_valid(neuron_valid[87]),
		.busy(neuron_busy[87])
	);
	// BRAM init: $readmemh("mem/neuron_087_keys.mem", neuron_87.key_mem);
	// BRAM init: $readmemh("mem/neuron_087_values.mem", neuron_87.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(910),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_88 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_88),
		.result(neuron_result[88]),
		.result_valid(neuron_valid[88]),
		.busy(neuron_busy[88])
	);
	// BRAM init: $readmemh("mem/neuron_088_keys.mem", neuron_88.key_mem);
	// BRAM init: $readmemh("mem/neuron_088_values.mem", neuron_88.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(927),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_89 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_89),
		.result(neuron_result[89]),
		.result_valid(neuron_valid[89]),
		.busy(neuron_busy[89])
	);
	// BRAM init: $readmemh("mem/neuron_089_keys.mem", neuron_89.key_mem);
	// BRAM init: $readmemh("mem/neuron_089_values.mem", neuron_89.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1291),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_90 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_90),
		.result(neuron_result[90]),
		.result_valid(neuron_valid[90]),
		.busy(neuron_busy[90])
	);
	// BRAM init: $readmemh("mem/neuron_090_keys.mem", neuron_90.key_mem);
	// BRAM init: $readmemh("mem/neuron_090_values.mem", neuron_90.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(763),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_91 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_91),
		.result(neuron_result[91]),
		.result_valid(neuron_valid[91]),
		.busy(neuron_busy[91])
	);
	// BRAM init: $readmemh("mem/neuron_091_keys.mem", neuron_91.key_mem);
	// BRAM init: $readmemh("mem/neuron_091_values.mem", neuron_91.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_92 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_92),
		.result(neuron_result[92]),
		.result_valid(neuron_valid[92]),
		.busy(neuron_busy[92])
	);
	// BRAM init: $readmemh("mem/neuron_092_keys.mem", neuron_92.key_mem);
	// BRAM init: $readmemh("mem/neuron_092_values.mem", neuron_92.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(814),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_93 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_93),
		.result(neuron_result[93]),
		.result_valid(neuron_valid[93]),
		.busy(neuron_busy[93])
	);
	// BRAM init: $readmemh("mem/neuron_093_keys.mem", neuron_93.key_mem);
	// BRAM init: $readmemh("mem/neuron_093_values.mem", neuron_93.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(656),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_94 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_94),
		.result(neuron_result[94]),
		.result_valid(neuron_valid[94]),
		.busy(neuron_busy[94])
	);
	// BRAM init: $readmemh("mem/neuron_094_keys.mem", neuron_94.key_mem);
	// BRAM init: $readmemh("mem/neuron_094_values.mem", neuron_94.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1066),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_95 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_95),
		.result(neuron_result[95]),
		.result_valid(neuron_valid[95]),
		.busy(neuron_busy[95])
	);
	// BRAM init: $readmemh("mem/neuron_095_keys.mem", neuron_95.key_mem);
	// BRAM init: $readmemh("mem/neuron_095_values.mem", neuron_95.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_96 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_96),
		.result(neuron_result[96]),
		.result_valid(neuron_valid[96]),
		.busy(neuron_busy[96])
	);
	// BRAM init: $readmemh("mem/neuron_096_keys.mem", neuron_96.key_mem);
	// BRAM init: $readmemh("mem/neuron_096_values.mem", neuron_96.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_97 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_97),
		.result(neuron_result[97]),
		.result_valid(neuron_valid[97]),
		.busy(neuron_busy[97])
	);
	// BRAM init: $readmemh("mem/neuron_097_keys.mem", neuron_97.key_mem);
	// BRAM init: $readmemh("mem/neuron_097_values.mem", neuron_97.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(897),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_98 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_98),
		.result(neuron_result[98]),
		.result_valid(neuron_valid[98]),
		.busy(neuron_busy[98])
	);
	// BRAM init: $readmemh("mem/neuron_098_keys.mem", neuron_98.key_mem);
	// BRAM init: $readmemh("mem/neuron_098_values.mem", neuron_98.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_99 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_99),
		.result(neuron_result[99]),
		.result_valid(neuron_valid[99]),
		.busy(neuron_busy[99])
	);
	// BRAM init: $readmemh("mem/neuron_099_keys.mem", neuron_99.key_mem);
	// BRAM init: $readmemh("mem/neuron_099_values.mem", neuron_99.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(915),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_100 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_100),
		.result(neuron_result[100]),
		.result_valid(neuron_valid[100]),
		.busy(neuron_busy[100])
	);
	// BRAM init: $readmemh("mem/neuron_100_keys.mem", neuron_100.key_mem);
	// BRAM init: $readmemh("mem/neuron_100_values.mem", neuron_100.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(735),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_101 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_101),
		.result(neuron_result[101]),
		.result_valid(neuron_valid[101]),
		.busy(neuron_busy[101])
	);
	// BRAM init: $readmemh("mem/neuron_101_keys.mem", neuron_101.key_mem);
	// BRAM init: $readmemh("mem/neuron_101_values.mem", neuron_101.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1330),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_102 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_102),
		.result(neuron_result[102]),
		.result_valid(neuron_valid[102]),
		.busy(neuron_busy[102])
	);
	// BRAM init: $readmemh("mem/neuron_102_keys.mem", neuron_102.key_mem);
	// BRAM init: $readmemh("mem/neuron_102_values.mem", neuron_102.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(621),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_103 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_103),
		.result(neuron_result[103]),
		.result_valid(neuron_valid[103]),
		.busy(neuron_busy[103])
	);
	// BRAM init: $readmemh("mem/neuron_103_keys.mem", neuron_103.key_mem);
	// BRAM init: $readmemh("mem/neuron_103_values.mem", neuron_103.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(823),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_104 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_104),
		.result(neuron_result[104]),
		.result_valid(neuron_valid[104]),
		.busy(neuron_busy[104])
	);
	// BRAM init: $readmemh("mem/neuron_104_keys.mem", neuron_104.key_mem);
	// BRAM init: $readmemh("mem/neuron_104_values.mem", neuron_104.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(438),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_105 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_105),
		.result(neuron_result[105]),
		.result_valid(neuron_valid[105]),
		.busy(neuron_busy[105])
	);
	// BRAM init: $readmemh("mem/neuron_105_keys.mem", neuron_105.key_mem);
	// BRAM init: $readmemh("mem/neuron_105_values.mem", neuron_105.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_106 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_106),
		.result(neuron_result[106]),
		.result_valid(neuron_valid[106]),
		.busy(neuron_busy[106])
	);
	// BRAM init: $readmemh("mem/neuron_106_keys.mem", neuron_106.key_mem);
	// BRAM init: $readmemh("mem/neuron_106_values.mem", neuron_106.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1352),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_107 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_107),
		.result(neuron_result[107]),
		.result_valid(neuron_valid[107]),
		.busy(neuron_busy[107])
	);
	// BRAM init: $readmemh("mem/neuron_107_keys.mem", neuron_107.key_mem);
	// BRAM init: $readmemh("mem/neuron_107_values.mem", neuron_107.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_108 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_108),
		.result(neuron_result[108]),
		.result_valid(neuron_valid[108]),
		.busy(neuron_busy[108])
	);
	// BRAM init: $readmemh("mem/neuron_108_keys.mem", neuron_108.key_mem);
	// BRAM init: $readmemh("mem/neuron_108_values.mem", neuron_108.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_109 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_109),
		.result(neuron_result[109]),
		.result_valid(neuron_valid[109]),
		.busy(neuron_busy[109])
	);
	// BRAM init: $readmemh("mem/neuron_109_keys.mem", neuron_109.key_mem);
	// BRAM init: $readmemh("mem/neuron_109_values.mem", neuron_109.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1744),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_110 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_110),
		.result(neuron_result[110]),
		.result_valid(neuron_valid[110]),
		.busy(neuron_busy[110])
	);
	// BRAM init: $readmemh("mem/neuron_110_keys.mem", neuron_110.key_mem);
	// BRAM init: $readmemh("mem/neuron_110_values.mem", neuron_110.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(254),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_111 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_111),
		.result(neuron_result[111]),
		.result_valid(neuron_valid[111]),
		.busy(neuron_busy[111])
	);
	// BRAM init: $readmemh("mem/neuron_111_keys.mem", neuron_111.key_mem);
	// BRAM init: $readmemh("mem/neuron_111_values.mem", neuron_111.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(840),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_112 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_112),
		.result(neuron_result[112]),
		.result_valid(neuron_valid[112]),
		.busy(neuron_busy[112])
	);
	// BRAM init: $readmemh("mem/neuron_112_keys.mem", neuron_112.key_mem);
	// BRAM init: $readmemh("mem/neuron_112_values.mem", neuron_112.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(917),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_113 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_113),
		.result(neuron_result[113]),
		.result_valid(neuron_valid[113]),
		.busy(neuron_busy[113])
	);
	// BRAM init: $readmemh("mem/neuron_113_keys.mem", neuron_113.key_mem);
	// BRAM init: $readmemh("mem/neuron_113_values.mem", neuron_113.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_114 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_114),
		.result(neuron_result[114]),
		.result_valid(neuron_valid[114]),
		.busy(neuron_busy[114])
	);
	// BRAM init: $readmemh("mem/neuron_114_keys.mem", neuron_114.key_mem);
	// BRAM init: $readmemh("mem/neuron_114_values.mem", neuron_114.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(511),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_115 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_115),
		.result(neuron_result[115]),
		.result_valid(neuron_valid[115]),
		.busy(neuron_busy[115])
	);
	// BRAM init: $readmemh("mem/neuron_115_keys.mem", neuron_115.key_mem);
	// BRAM init: $readmemh("mem/neuron_115_values.mem", neuron_115.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_116 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_116),
		.result(neuron_result[116]),
		.result_valid(neuron_valid[116]),
		.busy(neuron_busy[116])
	);
	// BRAM init: $readmemh("mem/neuron_116_keys.mem", neuron_116.key_mem);
	// BRAM init: $readmemh("mem/neuron_116_values.mem", neuron_116.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(232),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_117 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_117),
		.result(neuron_result[117]),
		.result_valid(neuron_valid[117]),
		.busy(neuron_busy[117])
	);
	// BRAM init: $readmemh("mem/neuron_117_keys.mem", neuron_117.key_mem);
	// BRAM init: $readmemh("mem/neuron_117_values.mem", neuron_117.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(326),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_118 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_118),
		.result(neuron_result[118]),
		.result_valid(neuron_valid[118]),
		.busy(neuron_busy[118])
	);
	// BRAM init: $readmemh("mem/neuron_118_keys.mem", neuron_118.key_mem);
	// BRAM init: $readmemh("mem/neuron_118_values.mem", neuron_118.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(284),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_119 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_119),
		.result(neuron_result[119]),
		.result_valid(neuron_valid[119]),
		.busy(neuron_busy[119])
	);
	// BRAM init: $readmemh("mem/neuron_119_keys.mem", neuron_119.key_mem);
	// BRAM init: $readmemh("mem/neuron_119_values.mem", neuron_119.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(460),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_120 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_120),
		.result(neuron_result[120]),
		.result_valid(neuron_valid[120]),
		.busy(neuron_busy[120])
	);
	// BRAM init: $readmemh("mem/neuron_120_keys.mem", neuron_120.key_mem);
	// BRAM init: $readmemh("mem/neuron_120_values.mem", neuron_120.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_121 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_121),
		.result(neuron_result[121]),
		.result_valid(neuron_valid[121]),
		.busy(neuron_busy[121])
	);
	// BRAM init: $readmemh("mem/neuron_121_keys.mem", neuron_121.key_mem);
	// BRAM init: $readmemh("mem/neuron_121_values.mem", neuron_121.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_122 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_122),
		.result(neuron_result[122]),
		.result_valid(neuron_valid[122]),
		.busy(neuron_busy[122])
	);
	// BRAM init: $readmemh("mem/neuron_122_keys.mem", neuron_122.key_mem);
	// BRAM init: $readmemh("mem/neuron_122_values.mem", neuron_122.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_123 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_123),
		.result(neuron_result[123]),
		.result_valid(neuron_valid[123]),
		.busy(neuron_busy[123])
	);
	// BRAM init: $readmemh("mem/neuron_123_keys.mem", neuron_123.key_mem);
	// BRAM init: $readmemh("mem/neuron_123_values.mem", neuron_123.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(740),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_124 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_124),
		.result(neuron_result[124]),
		.result_valid(neuron_valid[124]),
		.busy(neuron_busy[124])
	);
	// BRAM init: $readmemh("mem/neuron_124_keys.mem", neuron_124.key_mem);
	// BRAM init: $readmemh("mem/neuron_124_values.mem", neuron_124.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_125 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_125),
		.result(neuron_result[125]),
		.result_valid(neuron_valid[125]),
		.busy(neuron_busy[125])
	);
	// BRAM init: $readmemh("mem/neuron_125_keys.mem", neuron_125.key_mem);
	// BRAM init: $readmemh("mem/neuron_125_values.mem", neuron_125.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_126 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_126),
		.result(neuron_result[126]),
		.result_valid(neuron_valid[126]),
		.busy(neuron_busy[126])
	);
	// BRAM init: $readmemh("mem/neuron_126_keys.mem", neuron_126.key_mem);
	// BRAM init: $readmemh("mem/neuron_126_values.mem", neuron_126.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_127 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_127),
		.result(neuron_result[127]),
		.result_valid(neuron_valid[127]),
		.busy(neuron_busy[127])
	);
	// BRAM init: $readmemh("mem/neuron_127_keys.mem", neuron_127.key_mem);
	// BRAM init: $readmemh("mem/neuron_127_values.mem", neuron_127.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_128 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_128),
		.result(neuron_result[128]),
		.result_valid(neuron_valid[128]),
		.busy(neuron_busy[128])
	);
	// BRAM init: $readmemh("mem/neuron_128_keys.mem", neuron_128.key_mem);
	// BRAM init: $readmemh("mem/neuron_128_values.mem", neuron_128.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(917),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_129 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_129),
		.result(neuron_result[129]),
		.result_valid(neuron_valid[129]),
		.busy(neuron_busy[129])
	);
	// BRAM init: $readmemh("mem/neuron_129_keys.mem", neuron_129.key_mem);
	// BRAM init: $readmemh("mem/neuron_129_values.mem", neuron_129.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1066),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_130 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_130),
		.result(neuron_result[130]),
		.result_valid(neuron_valid[130]),
		.busy(neuron_busy[130])
	);
	// BRAM init: $readmemh("mem/neuron_130_keys.mem", neuron_130.key_mem);
	// BRAM init: $readmemh("mem/neuron_130_values.mem", neuron_130.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(718),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_131 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_131),
		.result(neuron_result[131]),
		.result_valid(neuron_valid[131]),
		.busy(neuron_busy[131])
	);
	// BRAM init: $readmemh("mem/neuron_131_keys.mem", neuron_131.key_mem);
	// BRAM init: $readmemh("mem/neuron_131_values.mem", neuron_131.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(546),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_132 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_132),
		.result(neuron_result[132]),
		.result_valid(neuron_valid[132]),
		.busy(neuron_busy[132])
	);
	// BRAM init: $readmemh("mem/neuron_132_keys.mem", neuron_132.key_mem);
	// BRAM init: $readmemh("mem/neuron_132_values.mem", neuron_132.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_133 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_133),
		.result(neuron_result[133]),
		.result_valid(neuron_valid[133]),
		.busy(neuron_busy[133])
	);
	// BRAM init: $readmemh("mem/neuron_133_keys.mem", neuron_133.key_mem);
	// BRAM init: $readmemh("mem/neuron_133_values.mem", neuron_133.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(718),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_134 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_134),
		.result(neuron_result[134]),
		.result_valid(neuron_valid[134]),
		.busy(neuron_busy[134])
	);
	// BRAM init: $readmemh("mem/neuron_134_keys.mem", neuron_134.key_mem);
	// BRAM init: $readmemh("mem/neuron_134_values.mem", neuron_134.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(957),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_135 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_135),
		.result(neuron_result[135]),
		.result_valid(neuron_valid[135]),
		.busy(neuron_busy[135])
	);
	// BRAM init: $readmemh("mem/neuron_135_keys.mem", neuron_135.key_mem);
	// BRAM init: $readmemh("mem/neuron_135_values.mem", neuron_135.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_136 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_136),
		.result(neuron_result[136]),
		.result_valid(neuron_valid[136]),
		.busy(neuron_busy[136])
	);
	// BRAM init: $readmemh("mem/neuron_136_keys.mem", neuron_136.key_mem);
	// BRAM init: $readmemh("mem/neuron_136_values.mem", neuron_136.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_137 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_137),
		.result(neuron_result[137]),
		.result_valid(neuron_valid[137]),
		.busy(neuron_busy[137])
	);
	// BRAM init: $readmemh("mem/neuron_137_keys.mem", neuron_137.key_mem);
	// BRAM init: $readmemh("mem/neuron_137_values.mem", neuron_137.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(447),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_138 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_138),
		.result(neuron_result[138]),
		.result_valid(neuron_valid[138]),
		.busy(neuron_busy[138])
	);
	// BRAM init: $readmemh("mem/neuron_138_keys.mem", neuron_138.key_mem);
	// BRAM init: $readmemh("mem/neuron_138_values.mem", neuron_138.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_139 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_139),
		.result(neuron_result[139]),
		.result_valid(neuron_valid[139]),
		.busy(neuron_busy[139])
	);
	// BRAM init: $readmemh("mem/neuron_139_keys.mem", neuron_139.key_mem);
	// BRAM init: $readmemh("mem/neuron_139_values.mem", neuron_139.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(447),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_140 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_140),
		.result(neuron_result[140]),
		.result_valid(neuron_valid[140]),
		.busy(neuron_busy[140])
	);
	// BRAM init: $readmemh("mem/neuron_140_keys.mem", neuron_140.key_mem);
	// BRAM init: $readmemh("mem/neuron_140_values.mem", neuron_140.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(420),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_141 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_141),
		.result(neuron_result[141]),
		.result_valid(neuron_valid[141]),
		.busy(neuron_busy[141])
	);
	// BRAM init: $readmemh("mem/neuron_141_keys.mem", neuron_141.key_mem);
	// BRAM init: $readmemh("mem/neuron_141_values.mem", neuron_141.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_142 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_142),
		.result(neuron_result[142]),
		.result_valid(neuron_valid[142]),
		.busy(neuron_busy[142])
	);
	// BRAM init: $readmemh("mem/neuron_142_keys.mem", neuron_142.key_mem);
	// BRAM init: $readmemh("mem/neuron_142_values.mem", neuron_142.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_143 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_143),
		.result(neuron_result[143]),
		.result_valid(neuron_valid[143]),
		.busy(neuron_busy[143])
	);
	// BRAM init: $readmemh("mem/neuron_143_keys.mem", neuron_143.key_mem);
	// BRAM init: $readmemh("mem/neuron_143_values.mem", neuron_143.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(511),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_144 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_144),
		.result(neuron_result[144]),
		.result_valid(neuron_valid[144]),
		.busy(neuron_busy[144])
	);
	// BRAM init: $readmemh("mem/neuron_144_keys.mem", neuron_144.key_mem);
	// BRAM init: $readmemh("mem/neuron_144_values.mem", neuron_144.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(232),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_145 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_145),
		.result(neuron_result[145]),
		.result_valid(neuron_valid[145]),
		.busy(neuron_busy[145])
	);
	// BRAM init: $readmemh("mem/neuron_145_keys.mem", neuron_145.key_mem);
	// BRAM init: $readmemh("mem/neuron_145_values.mem", neuron_145.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(284),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_146 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_146),
		.result(neuron_result[146]),
		.result_valid(neuron_valid[146]),
		.busy(neuron_busy[146])
	);
	// BRAM init: $readmemh("mem/neuron_146_keys.mem", neuron_146.key_mem);
	// BRAM init: $readmemh("mem/neuron_146_values.mem", neuron_146.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_147 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_147),
		.result(neuron_result[147]),
		.result_valid(neuron_valid[147]),
		.busy(neuron_busy[147])
	);
	// BRAM init: $readmemh("mem/neuron_147_keys.mem", neuron_147.key_mem);
	// BRAM init: $readmemh("mem/neuron_147_values.mem", neuron_147.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_148 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_148),
		.result(neuron_result[148]),
		.result_valid(neuron_valid[148]),
		.busy(neuron_busy[148])
	);
	// BRAM init: $readmemh("mem/neuron_148_keys.mem", neuron_148.key_mem);
	// BRAM init: $readmemh("mem/neuron_148_values.mem", neuron_148.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_149 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_149),
		.result(neuron_result[149]),
		.result_valid(neuron_valid[149]),
		.busy(neuron_busy[149])
	);
	// BRAM init: $readmemh("mem/neuron_149_keys.mem", neuron_149.key_mem);
	// BRAM init: $readmemh("mem/neuron_149_values.mem", neuron_149.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(764),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_150 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_150),
		.result(neuron_result[150]),
		.result_valid(neuron_valid[150]),
		.busy(neuron_busy[150])
	);
	// BRAM init: $readmemh("mem/neuron_150_keys.mem", neuron_150.key_mem);
	// BRAM init: $readmemh("mem/neuron_150_values.mem", neuron_150.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(522),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_151 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_151),
		.result(neuron_result[151]),
		.result_valid(neuron_valid[151]),
		.busy(neuron_busy[151])
	);
	// BRAM init: $readmemh("mem/neuron_151_keys.mem", neuron_151.key_mem);
	// BRAM init: $readmemh("mem/neuron_151_values.mem", neuron_151.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_152 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_152),
		.result(neuron_result[152]),
		.result_valid(neuron_valid[152]),
		.busy(neuron_busy[152])
	);
	// BRAM init: $readmemh("mem/neuron_152_keys.mem", neuron_152.key_mem);
	// BRAM init: $readmemh("mem/neuron_152_values.mem", neuron_152.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(460),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_153 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_153),
		.result(neuron_result[153]),
		.result_valid(neuron_valid[153]),
		.busy(neuron_busy[153])
	);
	// BRAM init: $readmemh("mem/neuron_153_keys.mem", neuron_153.key_mem);
	// BRAM init: $readmemh("mem/neuron_153_values.mem", neuron_153.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(438),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_154 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_154),
		.result(neuron_result[154]),
		.result_valid(neuron_valid[154]),
		.busy(neuron_busy[154])
	);
	// BRAM init: $readmemh("mem/neuron_154_keys.mem", neuron_154.key_mem);
	// BRAM init: $readmemh("mem/neuron_154_values.mem", neuron_154.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_155 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_155),
		.result(neuron_result[155]),
		.result_valid(neuron_valid[155]),
		.busy(neuron_busy[155])
	);
	// BRAM init: $readmemh("mem/neuron_155_keys.mem", neuron_155.key_mem);
	// BRAM init: $readmemh("mem/neuron_155_values.mem", neuron_155.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_156 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_156),
		.result(neuron_result[156]),
		.result_valid(neuron_valid[156]),
		.busy(neuron_busy[156])
	);
	// BRAM init: $readmemh("mem/neuron_156_keys.mem", neuron_156.key_mem);
	// BRAM init: $readmemh("mem/neuron_156_values.mem", neuron_156.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(271),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_157 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_157),
		.result(neuron_result[157]),
		.result_valid(neuron_valid[157]),
		.busy(neuron_busy[157])
	);
	// BRAM init: $readmemh("mem/neuron_157_keys.mem", neuron_157.key_mem);
	// BRAM init: $readmemh("mem/neuron_157_values.mem", neuron_157.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(840),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_158 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_158),
		.result(neuron_result[158]),
		.result_valid(neuron_valid[158]),
		.busy(neuron_busy[158])
	);
	// BRAM init: $readmemh("mem/neuron_158_keys.mem", neuron_158.key_mem);
	// BRAM init: $readmemh("mem/neuron_158_values.mem", neuron_158.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_159 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_159),
		.result(neuron_result[159]),
		.result_valid(neuron_valid[159]),
		.busy(neuron_busy[159])
	);
	// BRAM init: $readmemh("mem/neuron_159_keys.mem", neuron_159.key_mem);
	// BRAM init: $readmemh("mem/neuron_159_values.mem", neuron_159.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(763),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_160 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_160),
		.result(neuron_result[160]),
		.result_valid(neuron_valid[160]),
		.busy(neuron_busy[160])
	);
	// BRAM init: $readmemh("mem/neuron_160_keys.mem", neuron_160.key_mem);
	// BRAM init: $readmemh("mem/neuron_160_values.mem", neuron_160.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(764),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_161 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_161),
		.result(neuron_result[161]),
		.result_valid(neuron_valid[161]),
		.busy(neuron_busy[161])
	);
	// BRAM init: $readmemh("mem/neuron_161_keys.mem", neuron_161.key_mem);
	// BRAM init: $readmemh("mem/neuron_161_values.mem", neuron_161.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(383),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_162 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_162),
		.result(neuron_result[162]),
		.result_valid(neuron_valid[162]),
		.busy(neuron_busy[162])
	);
	// BRAM init: $readmemh("mem/neuron_162_keys.mem", neuron_162.key_mem);
	// BRAM init: $readmemh("mem/neuron_162_values.mem", neuron_162.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(943),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_163 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_163),
		.result(neuron_result[163]),
		.result_valid(neuron_valid[163]),
		.busy(neuron_busy[163])
	);
	// BRAM init: $readmemh("mem/neuron_163_keys.mem", neuron_163.key_mem);
	// BRAM init: $readmemh("mem/neuron_163_values.mem", neuron_163.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_164 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_164),
		.result(neuron_result[164]),
		.result_valid(neuron_valid[164]),
		.busy(neuron_busy[164])
	);
	// BRAM init: $readmemh("mem/neuron_164_keys.mem", neuron_164.key_mem);
	// BRAM init: $readmemh("mem/neuron_164_values.mem", neuron_164.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(840),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_165 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_165),
		.result(neuron_result[165]),
		.result_valid(neuron_valid[165]),
		.busy(neuron_busy[165])
	);
	// BRAM init: $readmemh("mem/neuron_165_keys.mem", neuron_165.key_mem);
	// BRAM init: $readmemh("mem/neuron_165_values.mem", neuron_165.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(546),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_166 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_166),
		.result(neuron_result[166]),
		.result_valid(neuron_valid[166]),
		.busy(neuron_busy[166])
	);
	// BRAM init: $readmemh("mem/neuron_166_keys.mem", neuron_166.key_mem);
	// BRAM init: $readmemh("mem/neuron_166_values.mem", neuron_166.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1102),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_167 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_167),
		.result(neuron_result[167]),
		.result_valid(neuron_valid[167]),
		.busy(neuron_busy[167])
	);
	// BRAM init: $readmemh("mem/neuron_167_keys.mem", neuron_167.key_mem);
	// BRAM init: $readmemh("mem/neuron_167_values.mem", neuron_167.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(254),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_168 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_168),
		.result(neuron_result[168]),
		.result_valid(neuron_valid[168]),
		.busy(neuron_busy[168])
	);
	// BRAM init: $readmemh("mem/neuron_168_keys.mem", neuron_168.key_mem);
	// BRAM init: $readmemh("mem/neuron_168_values.mem", neuron_168.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_169 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_169),
		.result(neuron_result[169]),
		.result_valid(neuron_valid[169]),
		.busy(neuron_busy[169])
	);
	// BRAM init: $readmemh("mem/neuron_169_keys.mem", neuron_169.key_mem);
	// BRAM init: $readmemh("mem/neuron_169_values.mem", neuron_169.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(669),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_170 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_170),
		.result(neuron_result[170]),
		.result_valid(neuron_valid[170]),
		.busy(neuron_busy[170])
	);
	// BRAM init: $readmemh("mem/neuron_170_keys.mem", neuron_170.key_mem);
	// BRAM init: $readmemh("mem/neuron_170_values.mem", neuron_170.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_171 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_171),
		.result(neuron_result[171]),
		.result_valid(neuron_valid[171]),
		.busy(neuron_busy[171])
	);
	// BRAM init: $readmemh("mem/neuron_171_keys.mem", neuron_171.key_mem);
	// BRAM init: $readmemh("mem/neuron_171_values.mem", neuron_171.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_172 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_172),
		.result(neuron_result[172]),
		.result_valid(neuron_valid[172]),
		.busy(neuron_busy[172])
	);
	// BRAM init: $readmemh("mem/neuron_172_keys.mem", neuron_172.key_mem);
	// BRAM init: $readmemh("mem/neuron_172_values.mem", neuron_172.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(621),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_173 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_173),
		.result(neuron_result[173]),
		.result_valid(neuron_valid[173]),
		.busy(neuron_busy[173])
	);
	// BRAM init: $readmemh("mem/neuron_173_keys.mem", neuron_173.key_mem);
	// BRAM init: $readmemh("mem/neuron_173_values.mem", neuron_173.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(897),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_174 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_174),
		.result(neuron_result[174]),
		.result_valid(neuron_valid[174]),
		.busy(neuron_busy[174])
	);
	// BRAM init: $readmemh("mem/neuron_174_keys.mem", neuron_174.key_mem);
	// BRAM init: $readmemh("mem/neuron_174_values.mem", neuron_174.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_175 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_175),
		.result(neuron_result[175]),
		.result_valid(neuron_valid[175]),
		.busy(neuron_busy[175])
	);
	// BRAM init: $readmemh("mem/neuron_175_keys.mem", neuron_175.key_mem);
	// BRAM init: $readmemh("mem/neuron_175_values.mem", neuron_175.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_176 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_176),
		.result(neuron_result[176]),
		.result_valid(neuron_valid[176]),
		.busy(neuron_busy[176])
	);
	// BRAM init: $readmemh("mem/neuron_176_keys.mem", neuron_176.key_mem);
	// BRAM init: $readmemh("mem/neuron_176_values.mem", neuron_176.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(730),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_177 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_177),
		.result(neuron_result[177]),
		.result_valid(neuron_valid[177]),
		.busy(neuron_busy[177])
	);
	// BRAM init: $readmemh("mem/neuron_177_keys.mem", neuron_177.key_mem);
	// BRAM init: $readmemh("mem/neuron_177_values.mem", neuron_177.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_178 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_178),
		.result(neuron_result[178]),
		.result_valid(neuron_valid[178]),
		.busy(neuron_busy[178])
	);
	// BRAM init: $readmemh("mem/neuron_178_keys.mem", neuron_178.key_mem);
	// BRAM init: $readmemh("mem/neuron_178_values.mem", neuron_178.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_179 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_179),
		.result(neuron_result[179]),
		.result_valid(neuron_valid[179]),
		.busy(neuron_busy[179])
	);
	// BRAM init: $readmemh("mem/neuron_179_keys.mem", neuron_179.key_mem);
	// BRAM init: $readmemh("mem/neuron_179_values.mem", neuron_179.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(698),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_180 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_180),
		.result(neuron_result[180]),
		.result_valid(neuron_valid[180]),
		.busy(neuron_busy[180])
	);
	// BRAM init: $readmemh("mem/neuron_180_keys.mem", neuron_180.key_mem);
	// BRAM init: $readmemh("mem/neuron_180_values.mem", neuron_180.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_181 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_181),
		.result(neuron_result[181]),
		.result_valid(neuron_valid[181]),
		.busy(neuron_busy[181])
	);
	// BRAM init: $readmemh("mem/neuron_181_keys.mem", neuron_181.key_mem);
	// BRAM init: $readmemh("mem/neuron_181_values.mem", neuron_181.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(479),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_182 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_182),
		.result(neuron_result[182]),
		.result_valid(neuron_valid[182]),
		.busy(neuron_busy[182])
	);
	// BRAM init: $readmemh("mem/neuron_182_keys.mem", neuron_182.key_mem);
	// BRAM init: $readmemh("mem/neuron_182_values.mem", neuron_182.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1352),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_183 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_183),
		.result(neuron_result[183]),
		.result_valid(neuron_valid[183]),
		.busy(neuron_busy[183])
	);
	// BRAM init: $readmemh("mem/neuron_183_keys.mem", neuron_183.key_mem);
	// BRAM init: $readmemh("mem/neuron_183_values.mem", neuron_183.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(326),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_184 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_184),
		.result(neuron_result[184]),
		.result_valid(neuron_valid[184]),
		.busy(neuron_busy[184])
	);
	// BRAM init: $readmemh("mem/neuron_184_keys.mem", neuron_184.key_mem);
	// BRAM init: $readmemh("mem/neuron_184_values.mem", neuron_184.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_185 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_185),
		.result(neuron_result[185]),
		.result_valid(neuron_valid[185]),
		.busy(neuron_busy[185])
	);
	// BRAM init: $readmemh("mem/neuron_185_keys.mem", neuron_185.key_mem);
	// BRAM init: $readmemh("mem/neuron_185_values.mem", neuron_185.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1291),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_186 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_186),
		.result(neuron_result[186]),
		.result_valid(neuron_valid[186]),
		.busy(neuron_busy[186])
	);
	// BRAM init: $readmemh("mem/neuron_186_keys.mem", neuron_186.key_mem);
	// BRAM init: $readmemh("mem/neuron_186_values.mem", neuron_186.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(730),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_187 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_187),
		.result(neuron_result[187]),
		.result_valid(neuron_valid[187]),
		.busy(neuron_busy[187])
	);
	// BRAM init: $readmemh("mem/neuron_187_keys.mem", neuron_187.key_mem);
	// BRAM init: $readmemh("mem/neuron_187_values.mem", neuron_187.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_188 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_188),
		.result(neuron_result[188]),
		.result_valid(neuron_valid[188]),
		.busy(neuron_busy[188])
	);
	// BRAM init: $readmemh("mem/neuron_188_keys.mem", neuron_188.key_mem);
	// BRAM init: $readmemh("mem/neuron_188_values.mem", neuron_188.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_189 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_189),
		.result(neuron_result[189]),
		.result_valid(neuron_valid[189]),
		.busy(neuron_busy[189])
	);
	// BRAM init: $readmemh("mem/neuron_189_keys.mem", neuron_189.key_mem);
	// BRAM init: $readmemh("mem/neuron_189_values.mem", neuron_189.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_190 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_190),
		.result(neuron_result[190]),
		.result_valid(neuron_valid[190]),
		.busy(neuron_busy[190])
	);
	// BRAM init: $readmemh("mem/neuron_190_keys.mem", neuron_190.key_mem);
	// BRAM init: $readmemh("mem/neuron_190_values.mem", neuron_190.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_191 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_191),
		.result(neuron_result[191]),
		.result_valid(neuron_valid[191]),
		.busy(neuron_busy[191])
	);
	// BRAM init: $readmemh("mem/neuron_191_keys.mem", neuron_191.key_mem);
	// BRAM init: $readmemh("mem/neuron_191_values.mem", neuron_191.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_192 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_192),
		.result(neuron_result[192]),
		.result_valid(neuron_valid[192]),
		.busy(neuron_busy[192])
	);
	// BRAM init: $readmemh("mem/neuron_192_keys.mem", neuron_192.key_mem);
	// BRAM init: $readmemh("mem/neuron_192_values.mem", neuron_192.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(511),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_193 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_193),
		.result(neuron_result[193]),
		.result_valid(neuron_valid[193]),
		.busy(neuron_busy[193])
	);
	// BRAM init: $readmemh("mem/neuron_193_keys.mem", neuron_193.key_mem);
	// BRAM init: $readmemh("mem/neuron_193_values.mem", neuron_193.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1352),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_194 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_194),
		.result(neuron_result[194]),
		.result_valid(neuron_valid[194]),
		.busy(neuron_busy[194])
	);
	// BRAM init: $readmemh("mem/neuron_194_keys.mem", neuron_194.key_mem);
	// BRAM init: $readmemh("mem/neuron_194_values.mem", neuron_194.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(630),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_195 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_195),
		.result(neuron_result[195]),
		.result_valid(neuron_valid[195]),
		.busy(neuron_busy[195])
	);
	// BRAM init: $readmemh("mem/neuron_195_keys.mem", neuron_195.key_mem);
	// BRAM init: $readmemh("mem/neuron_195_values.mem", neuron_195.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_196 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_196),
		.result(neuron_result[196]),
		.result_valid(neuron_valid[196]),
		.busy(neuron_busy[196])
	);
	// BRAM init: $readmemh("mem/neuron_196_keys.mem", neuron_196.key_mem);
	// BRAM init: $readmemh("mem/neuron_196_values.mem", neuron_196.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(420),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_197 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_197),
		.result(neuron_result[197]),
		.result_valid(neuron_valid[197]),
		.busy(neuron_busy[197])
	);
	// BRAM init: $readmemh("mem/neuron_197_keys.mem", neuron_197.key_mem);
	// BRAM init: $readmemh("mem/neuron_197_values.mem", neuron_197.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(740),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_198 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_198),
		.result(neuron_result[198]),
		.result_valid(neuron_valid[198]),
		.busy(neuron_busy[198])
	);
	// BRAM init: $readmemh("mem/neuron_198_keys.mem", neuron_198.key_mem);
	// BRAM init: $readmemh("mem/neuron_198_values.mem", neuron_198.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(415),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_199 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_199),
		.result(neuron_result[199]),
		.result_valid(neuron_valid[199]),
		.busy(neuron_busy[199])
	);
	// BRAM init: $readmemh("mem/neuron_199_keys.mem", neuron_199.key_mem);
	// BRAM init: $readmemh("mem/neuron_199_values.mem", neuron_199.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_200 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_200),
		.result(neuron_result[200]),
		.result_valid(neuron_valid[200]),
		.busy(neuron_busy[200])
	);
	// BRAM init: $readmemh("mem/neuron_200_keys.mem", neuron_200.key_mem);
	// BRAM init: $readmemh("mem/neuron_200_values.mem", neuron_200.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_201 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_201),
		.result(neuron_result[201]),
		.result_valid(neuron_valid[201]),
		.busy(neuron_busy[201])
	);
	// BRAM init: $readmemh("mem/neuron_201_keys.mem", neuron_201.key_mem);
	// BRAM init: $readmemh("mem/neuron_201_values.mem", neuron_201.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(892),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_202 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_202),
		.result(neuron_result[202]),
		.result_valid(neuron_valid[202]),
		.busy(neuron_busy[202])
	);
	// BRAM init: $readmemh("mem/neuron_202_keys.mem", neuron_202.key_mem);
	// BRAM init: $readmemh("mem/neuron_202_values.mem", neuron_202.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(631),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_203 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_203),
		.result(neuron_result[203]),
		.result_valid(neuron_valid[203]),
		.busy(neuron_busy[203])
	);
	// BRAM init: $readmemh("mem/neuron_203_keys.mem", neuron_203.key_mem);
	// BRAM init: $readmemh("mem/neuron_203_values.mem", neuron_203.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_204 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_204),
		.result(neuron_result[204]),
		.result_valid(neuron_valid[204]),
		.busy(neuron_busy[204])
	);
	// BRAM init: $readmemh("mem/neuron_204_keys.mem", neuron_204.key_mem);
	// BRAM init: $readmemh("mem/neuron_204_values.mem", neuron_204.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(407),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_205 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_205),
		.result(neuron_result[205]),
		.result_valid(neuron_valid[205]),
		.busy(neuron_busy[205])
	);
	// BRAM init: $readmemh("mem/neuron_205_keys.mem", neuron_205.key_mem);
	// BRAM init: $readmemh("mem/neuron_205_values.mem", neuron_205.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(553),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_206 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_206),
		.result(neuron_result[206]),
		.result_valid(neuron_valid[206]),
		.busy(neuron_busy[206])
	);
	// BRAM init: $readmemh("mem/neuron_206_keys.mem", neuron_206.key_mem);
	// BRAM init: $readmemh("mem/neuron_206_values.mem", neuron_206.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1756),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_207 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_207),
		.result(neuron_result[207]),
		.result_valid(neuron_valid[207]),
		.busy(neuron_busy[207])
	);
	// BRAM init: $readmemh("mem/neuron_207_keys.mem", neuron_207.key_mem);
	// BRAM init: $readmemh("mem/neuron_207_values.mem", neuron_207.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_208 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_208),
		.result(neuron_result[208]),
		.result_valid(neuron_valid[208]),
		.busy(neuron_busy[208])
	);
	// BRAM init: $readmemh("mem/neuron_208_keys.mem", neuron_208.key_mem);
	// BRAM init: $readmemh("mem/neuron_208_values.mem", neuron_208.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_209 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_209),
		.result(neuron_result[209]),
		.result_valid(neuron_valid[209]),
		.busy(neuron_busy[209])
	);
	// BRAM init: $readmemh("mem/neuron_209_keys.mem", neuron_209.key_mem);
	// BRAM init: $readmemh("mem/neuron_209_values.mem", neuron_209.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(479),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_210 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_210),
		.result(neuron_result[210]),
		.result_valid(neuron_valid[210]),
		.busy(neuron_busy[210])
	);
	// BRAM init: $readmemh("mem/neuron_210_keys.mem", neuron_210.key_mem);
	// BRAM init: $readmemh("mem/neuron_210_values.mem", neuron_210.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(787),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_211 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_211),
		.result(neuron_result[211]),
		.result_valid(neuron_valid[211]),
		.busy(neuron_busy[211])
	);
	// BRAM init: $readmemh("mem/neuron_211_keys.mem", neuron_211.key_mem);
	// BRAM init: $readmemh("mem/neuron_211_values.mem", neuron_211.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(447),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_212 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_212),
		.result(neuron_result[212]),
		.result_valid(neuron_valid[212]),
		.busy(neuron_busy[212])
	);
	// BRAM init: $readmemh("mem/neuron_212_keys.mem", neuron_212.key_mem);
	// BRAM init: $readmemh("mem/neuron_212_values.mem", neuron_212.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(630),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_213 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_213),
		.result(neuron_result[213]),
		.result_valid(neuron_valid[213]),
		.busy(neuron_busy[213])
	);
	// BRAM init: $readmemh("mem/neuron_213_keys.mem", neuron_213.key_mem);
	// BRAM init: $readmemh("mem/neuron_213_values.mem", neuron_213.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(917),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_214 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_214),
		.result(neuron_result[214]),
		.result_valid(neuron_valid[214]),
		.busy(neuron_busy[214])
	);
	// BRAM init: $readmemh("mem/neuron_214_keys.mem", neuron_214.key_mem);
	// BRAM init: $readmemh("mem/neuron_214_values.mem", neuron_214.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1109),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_215 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_215),
		.result(neuron_result[215]),
		.result_valid(neuron_valid[215]),
		.busy(neuron_busy[215])
	);
	// BRAM init: $readmemh("mem/neuron_215_keys.mem", neuron_215.key_mem);
	// BRAM init: $readmemh("mem/neuron_215_values.mem", neuron_215.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(897),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_216 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_216),
		.result(neuron_result[216]),
		.result_valid(neuron_valid[216]),
		.busy(neuron_busy[216])
	);
	// BRAM init: $readmemh("mem/neuron_216_keys.mem", neuron_216.key_mem);
	// BRAM init: $readmemh("mem/neuron_216_values.mem", neuron_216.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(545),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_217 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_217),
		.result(neuron_result[217]),
		.result_valid(neuron_valid[217]),
		.busy(neuron_busy[217])
	);
	// BRAM init: $readmemh("mem/neuron_217_keys.mem", neuron_217.key_mem);
	// BRAM init: $readmemh("mem/neuron_217_values.mem", neuron_217.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_218 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_218),
		.result(neuron_result[218]),
		.result_valid(neuron_valid[218]),
		.busy(neuron_busy[218])
	);
	// BRAM init: $readmemh("mem/neuron_218_keys.mem", neuron_218.key_mem);
	// BRAM init: $readmemh("mem/neuron_218_values.mem", neuron_218.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_219 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_219),
		.result(neuron_result[219]),
		.result_valid(neuron_valid[219]),
		.busy(neuron_busy[219])
	);
	// BRAM init: $readmemh("mem/neuron_219_keys.mem", neuron_219.key_mem);
	// BRAM init: $readmemh("mem/neuron_219_values.mem", neuron_219.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(840),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_220 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_220),
		.result(neuron_result[220]),
		.result_valid(neuron_valid[220]),
		.busy(neuron_busy[220])
	);
	// BRAM init: $readmemh("mem/neuron_220_keys.mem", neuron_220.key_mem);
	// BRAM init: $readmemh("mem/neuron_220_values.mem", neuron_220.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1588),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_221 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_221),
		.result(neuron_result[221]),
		.result_valid(neuron_valid[221]),
		.busy(neuron_busy[221])
	);
	// BRAM init: $readmemh("mem/neuron_221_keys.mem", neuron_221.key_mem);
	// BRAM init: $readmemh("mem/neuron_221_values.mem", neuron_221.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_222 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_222),
		.result(neuron_result[222]),
		.result_valid(neuron_valid[222]),
		.busy(neuron_busy[222])
	);
	// BRAM init: $readmemh("mem/neuron_222_keys.mem", neuron_222.key_mem);
	// BRAM init: $readmemh("mem/neuron_222_values.mem", neuron_222.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_223 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_223),
		.result(neuron_result[223]),
		.result_valid(neuron_valid[223]),
		.busy(neuron_busy[223])
	);
	// BRAM init: $readmemh("mem/neuron_223_keys.mem", neuron_223.key_mem);
	// BRAM init: $readmemh("mem/neuron_223_values.mem", neuron_223.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(955),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_224 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_224),
		.result(neuron_result[224]),
		.result_valid(neuron_valid[224]),
		.busy(neuron_busy[224])
	);
	// BRAM init: $readmemh("mem/neuron_224_keys.mem", neuron_224.key_mem);
	// BRAM init: $readmemh("mem/neuron_224_values.mem", neuron_224.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(840),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_225 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_225),
		.result(neuron_result[225]),
		.result_valid(neuron_valid[225]),
		.busy(neuron_busy[225])
	);
	// BRAM init: $readmemh("mem/neuron_225_keys.mem", neuron_225.key_mem);
	// BRAM init: $readmemh("mem/neuron_225_values.mem", neuron_225.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(383),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_226 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_226),
		.result(neuron_result[226]),
		.result_valid(neuron_valid[226]),
		.busy(neuron_busy[226])
	);
	// BRAM init: $readmemh("mem/neuron_226_keys.mem", neuron_226.key_mem);
	// BRAM init: $readmemh("mem/neuron_226_values.mem", neuron_226.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_227 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_227),
		.result(neuron_result[227]),
		.result_valid(neuron_valid[227]),
		.busy(neuron_busy[227])
	);
	// BRAM init: $readmemh("mem/neuron_227_keys.mem", neuron_227.key_mem);
	// BRAM init: $readmemh("mem/neuron_227_values.mem", neuron_227.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_228 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_228),
		.result(neuron_result[228]),
		.result_valid(neuron_valid[228]),
		.busy(neuron_busy[228])
	);
	// BRAM init: $readmemh("mem/neuron_228_keys.mem", neuron_228.key_mem);
	// BRAM init: $readmemh("mem/neuron_228_values.mem", neuron_228.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(917),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_229 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_229),
		.result(neuron_result[229]),
		.result_valid(neuron_valid[229]),
		.busy(neuron_busy[229])
	);
	// BRAM init: $readmemh("mem/neuron_229_keys.mem", neuron_229.key_mem);
	// BRAM init: $readmemh("mem/neuron_229_values.mem", neuron_229.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1352),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_230 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_230),
		.result(neuron_result[230]),
		.result_valid(neuron_valid[230]),
		.busy(neuron_busy[230])
	);
	// BRAM init: $readmemh("mem/neuron_230_keys.mem", neuron_230.key_mem);
	// BRAM init: $readmemh("mem/neuron_230_values.mem", neuron_230.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(303),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_231 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_231),
		.result(neuron_result[231]),
		.result_valid(neuron_valid[231]),
		.busy(neuron_busy[231])
	);
	// BRAM init: $readmemh("mem/neuron_231_keys.mem", neuron_231.key_mem);
	// BRAM init: $readmemh("mem/neuron_231_values.mem", neuron_231.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_232 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_232),
		.result(neuron_result[232]),
		.result_valid(neuron_valid[232]),
		.busy(neuron_busy[232])
	);
	// BRAM init: $readmemh("mem/neuron_232_keys.mem", neuron_232.key_mem);
	// BRAM init: $readmemh("mem/neuron_232_values.mem", neuron_232.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(915),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_233 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_233),
		.result(neuron_result[233]),
		.result_valid(neuron_valid[233]),
		.busy(neuron_busy[233])
	);
	// BRAM init: $readmemh("mem/neuron_233_keys.mem", neuron_233.key_mem);
	// BRAM init: $readmemh("mem/neuron_233_values.mem", neuron_233.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(892),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_234 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_234),
		.result(neuron_result[234]),
		.result_valid(neuron_valid[234]),
		.busy(neuron_busy[234])
	);
	// BRAM init: $readmemh("mem/neuron_234_keys.mem", neuron_234.key_mem);
	// BRAM init: $readmemh("mem/neuron_234_values.mem", neuron_234.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1265),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_235 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_235),
		.result(neuron_result[235]),
		.result_valid(neuron_valid[235]),
		.busy(neuron_busy[235])
	);
	// BRAM init: $readmemh("mem/neuron_235_keys.mem", neuron_235.key_mem);
	// BRAM init: $readmemh("mem/neuron_235_values.mem", neuron_235.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(532),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_236 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_236),
		.result(neuron_result[236]),
		.result_valid(neuron_valid[236]),
		.busy(neuron_busy[236])
	);
	// BRAM init: $readmemh("mem/neuron_236_keys.mem", neuron_236.key_mem);
	// BRAM init: $readmemh("mem/neuron_236_values.mem", neuron_236.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(927),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_237 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_237),
		.result(neuron_result[237]),
		.result_valid(neuron_valid[237]),
		.busy(neuron_busy[237])
	);
	// BRAM init: $readmemh("mem/neuron_237_keys.mem", neuron_237.key_mem);
	// BRAM init: $readmemh("mem/neuron_237_values.mem", neuron_237.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_238 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_238),
		.result(neuron_result[238]),
		.result_valid(neuron_valid[238]),
		.busy(neuron_busy[238])
	);
	// BRAM init: $readmemh("mem/neuron_238_keys.mem", neuron_238.key_mem);
	// BRAM init: $readmemh("mem/neuron_238_values.mem", neuron_238.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(814),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_239 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_239),
		.result(neuron_result[239]),
		.result_valid(neuron_valid[239]),
		.busy(neuron_busy[239])
	);
	// BRAM init: $readmemh("mem/neuron_239_keys.mem", neuron_239.key_mem);
	// BRAM init: $readmemh("mem/neuron_239_values.mem", neuron_239.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_240 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_240),
		.result(neuron_result[240]),
		.result_valid(neuron_valid[240]),
		.busy(neuron_busy[240])
	);
	// BRAM init: $readmemh("mem/neuron_240_keys.mem", neuron_240.key_mem);
	// BRAM init: $readmemh("mem/neuron_240_values.mem", neuron_240.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(796),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_241 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_241),
		.result(neuron_result[241]),
		.result_valid(neuron_valid[241]),
		.busy(neuron_busy[241])
	);
	// BRAM init: $readmemh("mem/neuron_241_keys.mem", neuron_241.key_mem);
	// BRAM init: $readmemh("mem/neuron_241_values.mem", neuron_241.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_242 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_242),
		.result(neuron_result[242]),
		.result_valid(neuron_valid[242]),
		.busy(neuron_busy[242])
	);
	// BRAM init: $readmemh("mem/neuron_242_keys.mem", neuron_242.key_mem);
	// BRAM init: $readmemh("mem/neuron_242_values.mem", neuron_242.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_243 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_243),
		.result(neuron_result[243]),
		.result_valid(neuron_valid[243]),
		.busy(neuron_busy[243])
	);
	// BRAM init: $readmemh("mem/neuron_243_keys.mem", neuron_243.key_mem);
	// BRAM init: $readmemh("mem/neuron_243_values.mem", neuron_243.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(326),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_244 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_244),
		.result(neuron_result[244]),
		.result_valid(neuron_valid[244]),
		.busy(neuron_busy[244])
	);
	// BRAM init: $readmemh("mem/neuron_244_keys.mem", neuron_244.key_mem);
	// BRAM init: $readmemh("mem/neuron_244_values.mem", neuron_244.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(343),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_245 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_245),
		.result(neuron_result[245]),
		.result_valid(neuron_valid[245]),
		.busy(neuron_busy[245])
	);
	// BRAM init: $readmemh("mem/neuron_245_keys.mem", neuron_245.key_mem);
	// BRAM init: $readmemh("mem/neuron_245_values.mem", neuron_245.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_246 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_246),
		.result(neuron_result[246]),
		.result_valid(neuron_valid[246]),
		.busy(neuron_busy[246])
	);
	// BRAM init: $readmemh("mem/neuron_246_keys.mem", neuron_246.key_mem);
	// BRAM init: $readmemh("mem/neuron_246_values.mem", neuron_246.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(381),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_247 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_247),
		.result(neuron_result[247]),
		.result_valid(neuron_valid[247]),
		.busy(neuron_busy[247])
	);
	// BRAM init: $readmemh("mem/neuron_247_keys.mem", neuron_247.key_mem);
	// BRAM init: $readmemh("mem/neuron_247_values.mem", neuron_247.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(630),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_248 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_248),
		.result(neuron_result[248]),
		.result_valid(neuron_valid[248]),
		.busy(neuron_busy[248])
	);
	// BRAM init: $readmemh("mem/neuron_248_keys.mem", neuron_248.key_mem);
	// BRAM init: $readmemh("mem/neuron_248_values.mem", neuron_248.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_249 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_249),
		.result(neuron_result[249]),
		.result_valid(neuron_valid[249]),
		.busy(neuron_busy[249])
	);
	// BRAM init: $readmemh("mem/neuron_249_keys.mem", neuron_249.key_mem);
	// BRAM init: $readmemh("mem/neuron_249_values.mem", neuron_249.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1265),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_250 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_250),
		.result(neuron_result[250]),
		.result_valid(neuron_valid[250]),
		.busy(neuron_busy[250])
	);
	// BRAM init: $readmemh("mem/neuron_250_keys.mem", neuron_250.key_mem);
	// BRAM init: $readmemh("mem/neuron_250_values.mem", neuron_250.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_251 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_251),
		.result(neuron_result[251]),
		.result_valid(neuron_valid[251]),
		.busy(neuron_busy[251])
	);
	// BRAM init: $readmemh("mem/neuron_251_keys.mem", neuron_251.key_mem);
	// BRAM init: $readmemh("mem/neuron_251_values.mem", neuron_251.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(718),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_252 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_252),
		.result(neuron_result[252]),
		.result_valid(neuron_valid[252]),
		.busy(neuron_busy[252])
	);
	// BRAM init: $readmemh("mem/neuron_252_keys.mem", neuron_252.key_mem);
	// BRAM init: $readmemh("mem/neuron_252_values.mem", neuron_252.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(981),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_253 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_253),
		.result(neuron_result[253]),
		.result_valid(neuron_valid[253]),
		.busy(neuron_busy[253])
	);
	// BRAM init: $readmemh("mem/neuron_253_keys.mem", neuron_253.key_mem);
	// BRAM init: $readmemh("mem/neuron_253_values.mem", neuron_253.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_254 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_254),
		.result(neuron_result[254]),
		.result_valid(neuron_valid[254]),
		.busy(neuron_busy[254])
	);
	// BRAM init: $readmemh("mem/neuron_254_keys.mem", neuron_254.key_mem);
	// BRAM init: $readmemh("mem/neuron_254_values.mem", neuron_254.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(631),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_255 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_255),
		.result(neuron_result[255]),
		.result_valid(neuron_valid[255]),
		.busy(neuron_busy[255])
	);
	// BRAM init: $readmemh("mem/neuron_255_keys.mem", neuron_255.key_mem);
	// BRAM init: $readmemh("mem/neuron_255_values.mem", neuron_255.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_256 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_256),
		.result(neuron_result[256]),
		.result_valid(neuron_valid[256]),
		.busy(neuron_busy[256])
	);
	// BRAM init: $readmemh("mem/neuron_256_keys.mem", neuron_256.key_mem);
	// BRAM init: $readmemh("mem/neuron_256_values.mem", neuron_256.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1066),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_257 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_257),
		.result(neuron_result[257]),
		.result_valid(neuron_valid[257]),
		.busy(neuron_busy[257])
	);
	// BRAM init: $readmemh("mem/neuron_257_keys.mem", neuron_257.key_mem);
	// BRAM init: $readmemh("mem/neuron_257_values.mem", neuron_257.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(522),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_258 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_258),
		.result(neuron_result[258]),
		.result_valid(neuron_valid[258]),
		.busy(neuron_busy[258])
	);
	// BRAM init: $readmemh("mem/neuron_258_keys.mem", neuron_258.key_mem);
	// BRAM init: $readmemh("mem/neuron_258_values.mem", neuron_258.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(730),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_259 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_259),
		.result(neuron_result[259]),
		.result_valid(neuron_valid[259]),
		.busy(neuron_busy[259])
	);
	// BRAM init: $readmemh("mem/neuron_259_keys.mem", neuron_259.key_mem);
	// BRAM init: $readmemh("mem/neuron_259_values.mem", neuron_259.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(284),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_260 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_260),
		.result(neuron_result[260]),
		.result_valid(neuron_valid[260]),
		.busy(neuron_busy[260])
	);
	// BRAM init: $readmemh("mem/neuron_260_keys.mem", neuron_260.key_mem);
	// BRAM init: $readmemh("mem/neuron_260_values.mem", neuron_260.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1352),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_261 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_261),
		.result(neuron_result[261]),
		.result_valid(neuron_valid[261]),
		.busy(neuron_busy[261])
	);
	// BRAM init: $readmemh("mem/neuron_261_keys.mem", neuron_261.key_mem);
	// BRAM init: $readmemh("mem/neuron_261_values.mem", neuron_261.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_262 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_262),
		.result(neuron_result[262]),
		.result_valid(neuron_valid[262]),
		.busy(neuron_busy[262])
	);
	// BRAM init: $readmemh("mem/neuron_262_keys.mem", neuron_262.key_mem);
	// BRAM init: $readmemh("mem/neuron_262_values.mem", neuron_262.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_263 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_263),
		.result(neuron_result[263]),
		.result_valid(neuron_valid[263]),
		.busy(neuron_busy[263])
	);
	// BRAM init: $readmemh("mem/neuron_263_keys.mem", neuron_263.key_mem);
	// BRAM init: $readmemh("mem/neuron_263_values.mem", neuron_263.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(453),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_264 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_264),
		.result(neuron_result[264]),
		.result_valid(neuron_valid[264]),
		.busy(neuron_busy[264])
	);
	// BRAM init: $readmemh("mem/neuron_264_keys.mem", neuron_264.key_mem);
	// BRAM init: $readmemh("mem/neuron_264_values.mem", neuron_264.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_265 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_265),
		.result(neuron_result[265]),
		.result_valid(neuron_valid[265]),
		.busy(neuron_busy[265])
	);
	// BRAM init: $readmemh("mem/neuron_265_keys.mem", neuron_265.key_mem);
	// BRAM init: $readmemh("mem/neuron_265_values.mem", neuron_265.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_266 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_266),
		.result(neuron_result[266]),
		.result_valid(neuron_valid[266]),
		.busy(neuron_busy[266])
	);
	// BRAM init: $readmemh("mem/neuron_266_keys.mem", neuron_266.key_mem);
	// BRAM init: $readmemh("mem/neuron_266_values.mem", neuron_266.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(352),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_267 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_267),
		.result(neuron_result[267]),
		.result_valid(neuron_valid[267]),
		.busy(neuron_busy[267])
	);
	// BRAM init: $readmemh("mem/neuron_267_keys.mem", neuron_267.key_mem);
	// BRAM init: $readmemh("mem/neuron_267_values.mem", neuron_267.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(553),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_268 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_268),
		.result(neuron_result[268]),
		.result_valid(neuron_valid[268]),
		.busy(neuron_busy[268])
	);
	// BRAM init: $readmemh("mem/neuron_268_keys.mem", neuron_268.key_mem);
	// BRAM init: $readmemh("mem/neuron_268_values.mem", neuron_268.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_269 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_269),
		.result(neuron_result[269]),
		.result_valid(neuron_valid[269]),
		.busy(neuron_busy[269])
	);
	// BRAM init: $readmemh("mem/neuron_269_keys.mem", neuron_269.key_mem);
	// BRAM init: $readmemh("mem/neuron_269_values.mem", neuron_269.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(460),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_270 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_270),
		.result(neuron_result[270]),
		.result_valid(neuron_valid[270]),
		.busy(neuron_busy[270])
	);
	// BRAM init: $readmemh("mem/neuron_270_keys.mem", neuron_270.key_mem);
	// BRAM init: $readmemh("mem/neuron_270_values.mem", neuron_270.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(814),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_271 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_271),
		.result(neuron_result[271]),
		.result_valid(neuron_valid[271]),
		.busy(neuron_busy[271])
	);
	// BRAM init: $readmemh("mem/neuron_271_keys.mem", neuron_271.key_mem);
	// BRAM init: $readmemh("mem/neuron_271_values.mem", neuron_271.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1744),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_272 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_272),
		.result(neuron_result[272]),
		.result_valid(neuron_valid[272]),
		.busy(neuron_busy[272])
	);
	// BRAM init: $readmemh("mem/neuron_272_keys.mem", neuron_272.key_mem);
	// BRAM init: $readmemh("mem/neuron_272_values.mem", neuron_272.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(576),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_273 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_273),
		.result(neuron_result[273]),
		.result_valid(neuron_valid[273]),
		.busy(neuron_busy[273])
	);
	// BRAM init: $readmemh("mem/neuron_273_keys.mem", neuron_273.key_mem);
	// BRAM init: $readmemh("mem/neuron_273_values.mem", neuron_273.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(656),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_274 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_274),
		.result(neuron_result[274]),
		.result_valid(neuron_valid[274]),
		.busy(neuron_busy[274])
	);
	// BRAM init: $readmemh("mem/neuron_274_keys.mem", neuron_274.key_mem);
	// BRAM init: $readmemh("mem/neuron_274_values.mem", neuron_274.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(383),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_275 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_275),
		.result(neuron_result[275]),
		.result_valid(neuron_valid[275]),
		.busy(neuron_busy[275])
	);
	// BRAM init: $readmemh("mem/neuron_275_keys.mem", neuron_275.key_mem);
	// BRAM init: $readmemh("mem/neuron_275_values.mem", neuron_275.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(981),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_276 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_276),
		.result(neuron_result[276]),
		.result_valid(neuron_valid[276]),
		.busy(neuron_busy[276])
	);
	// BRAM init: $readmemh("mem/neuron_276_keys.mem", neuron_276.key_mem);
	// BRAM init: $readmemh("mem/neuron_276_values.mem", neuron_276.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(507),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_277 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_277),
		.result(neuron_result[277]),
		.result_valid(neuron_valid[277]),
		.busy(neuron_busy[277])
	);
	// BRAM init: $readmemh("mem/neuron_277_keys.mem", neuron_277.key_mem);
	// BRAM init: $readmemh("mem/neuron_277_values.mem", neuron_277.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(514),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_278 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_278),
		.result(neuron_result[278]),
		.result_valid(neuron_valid[278]),
		.busy(neuron_busy[278])
	);
	// BRAM init: $readmemh("mem/neuron_278_keys.mem", neuron_278.key_mem);
	// BRAM init: $readmemh("mem/neuron_278_values.mem", neuron_278.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_279 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_279),
		.result(neuron_result[279]),
		.result_valid(neuron_valid[279]),
		.busy(neuron_busy[279])
	);
	// BRAM init: $readmemh("mem/neuron_279_keys.mem", neuron_279.key_mem);
	// BRAM init: $readmemh("mem/neuron_279_values.mem", neuron_279.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(656),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_280 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_280),
		.result(neuron_result[280]),
		.result_valid(neuron_valid[280]),
		.busy(neuron_busy[280])
	);
	// BRAM init: $readmemh("mem/neuron_280_keys.mem", neuron_280.key_mem);
	// BRAM init: $readmemh("mem/neuron_280_values.mem", neuron_280.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1756),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_281 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_281),
		.result(neuron_result[281]),
		.result_valid(neuron_valid[281]),
		.busy(neuron_busy[281])
	);
	// BRAM init: $readmemh("mem/neuron_281_keys.mem", neuron_281.key_mem);
	// BRAM init: $readmemh("mem/neuron_281_values.mem", neuron_281.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(553),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_282 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_282),
		.result(neuron_result[282]),
		.result_valid(neuron_valid[282]),
		.busy(neuron_busy[282])
	);
	// BRAM init: $readmemh("mem/neuron_282_keys.mem", neuron_282.key_mem);
	// BRAM init: $readmemh("mem/neuron_282_values.mem", neuron_282.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(927),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_283 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_283),
		.result(neuron_result[283]),
		.result_valid(neuron_valid[283]),
		.busy(neuron_busy[283])
	);
	// BRAM init: $readmemh("mem/neuron_283_keys.mem", neuron_283.key_mem);
	// BRAM init: $readmemh("mem/neuron_283_values.mem", neuron_283.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_284 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_284),
		.result(neuron_result[284]),
		.result_valid(neuron_valid[284]),
		.busy(neuron_busy[284])
	);
	// BRAM init: $readmemh("mem/neuron_284_keys.mem", neuron_284.key_mem);
	// BRAM init: $readmemh("mem/neuron_284_values.mem", neuron_284.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_285 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_285),
		.result(neuron_result[285]),
		.result_valid(neuron_valid[285]),
		.busy(neuron_busy[285])
	);
	// BRAM init: $readmemh("mem/neuron_285_keys.mem", neuron_285.key_mem);
	// BRAM init: $readmemh("mem/neuron_285_values.mem", neuron_285.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(772),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_286 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_286),
		.result(neuron_result[286]),
		.result_valid(neuron_valid[286]),
		.busy(neuron_busy[286])
	);
	// BRAM init: $readmemh("mem/neuron_286_keys.mem", neuron_286.key_mem);
	// BRAM init: $readmemh("mem/neuron_286_values.mem", neuron_286.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_287 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_287),
		.result(neuron_result[287]),
		.result_valid(neuron_valid[287]),
		.busy(neuron_busy[287])
	);
	// BRAM init: $readmemh("mem/neuron_287_keys.mem", neuron_287.key_mem);
	// BRAM init: $readmemh("mem/neuron_287_values.mem", neuron_287.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(718),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_288 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_288),
		.result(neuron_result[288]),
		.result_valid(neuron_valid[288]),
		.busy(neuron_busy[288])
	);
	// BRAM init: $readmemh("mem/neuron_288_keys.mem", neuron_288.key_mem);
	// BRAM init: $readmemh("mem/neuron_288_values.mem", neuron_288.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(917),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_289 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_289),
		.result(neuron_result[289]),
		.result_valid(neuron_valid[289]),
		.busy(neuron_busy[289])
	);
	// BRAM init: $readmemh("mem/neuron_289_keys.mem", neuron_289.key_mem);
	// BRAM init: $readmemh("mem/neuron_289_values.mem", neuron_289.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_290 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_290),
		.result(neuron_result[290]),
		.result_valid(neuron_valid[290]),
		.busy(neuron_busy[290])
	);
	// BRAM init: $readmemh("mem/neuron_290_keys.mem", neuron_290.key_mem);
	// BRAM init: $readmemh("mem/neuron_290_values.mem", neuron_290.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_291 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_291),
		.result(neuron_result[291]),
		.result_valid(neuron_valid[291]),
		.busy(neuron_busy[291])
	);
	// BRAM init: $readmemh("mem/neuron_291_keys.mem", neuron_291.key_mem);
	// BRAM init: $readmemh("mem/neuron_291_values.mem", neuron_291.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_292 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_292),
		.result(neuron_result[292]),
		.result_valid(neuron_valid[292]),
		.busy(neuron_busy[292])
	);
	// BRAM init: $readmemh("mem/neuron_292_keys.mem", neuron_292.key_mem);
	// BRAM init: $readmemh("mem/neuron_292_values.mem", neuron_292.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(248),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_293 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_293),
		.result(neuron_result[293]),
		.result_valid(neuron_valid[293]),
		.busy(neuron_busy[293])
	);
	// BRAM init: $readmemh("mem/neuron_293_keys.mem", neuron_293.key_mem);
	// BRAM init: $readmemh("mem/neuron_293_values.mem", neuron_293.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_294 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_294),
		.result(neuron_result[294]),
		.result_valid(neuron_valid[294]),
		.busy(neuron_busy[294])
	);
	// BRAM init: $readmemh("mem/neuron_294_keys.mem", neuron_294.key_mem);
	// BRAM init: $readmemh("mem/neuron_294_values.mem", neuron_294.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(729),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_295 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_295),
		.result(neuron_result[295]),
		.result_valid(neuron_valid[295]),
		.busy(neuron_busy[295])
	);
	// BRAM init: $readmemh("mem/neuron_295_keys.mem", neuron_295.key_mem);
	// BRAM init: $readmemh("mem/neuron_295_values.mem", neuron_295.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_296 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_296),
		.result(neuron_result[296]),
		.result_valid(neuron_valid[296]),
		.busy(neuron_busy[296])
	);
	// BRAM init: $readmemh("mem/neuron_296_keys.mem", neuron_296.key_mem);
	// BRAM init: $readmemh("mem/neuron_296_values.mem", neuron_296.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(897),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_297 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_297),
		.result(neuron_result[297]),
		.result_valid(neuron_valid[297]),
		.busy(neuron_busy[297])
	);
	// BRAM init: $readmemh("mem/neuron_297_keys.mem", neuron_297.key_mem);
	// BRAM init: $readmemh("mem/neuron_297_values.mem", neuron_297.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_298 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_298),
		.result(neuron_result[298]),
		.result_valid(neuron_valid[298]),
		.busy(neuron_busy[298])
	);
	// BRAM init: $readmemh("mem/neuron_298_keys.mem", neuron_298.key_mem);
	// BRAM init: $readmemh("mem/neuron_298_values.mem", neuron_298.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(669),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_299 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_299),
		.result(neuron_result[299]),
		.result_valid(neuron_valid[299]),
		.busy(neuron_busy[299])
	);
	// BRAM init: $readmemh("mem/neuron_299_keys.mem", neuron_299.key_mem);
	// BRAM init: $readmemh("mem/neuron_299_values.mem", neuron_299.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(981),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_300 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_300),
		.result(neuron_result[300]),
		.result_valid(neuron_valid[300]),
		.busy(neuron_busy[300])
	);
	// BRAM init: $readmemh("mem/neuron_300_keys.mem", neuron_300.key_mem);
	// BRAM init: $readmemh("mem/neuron_300_values.mem", neuron_300.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_301 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_301),
		.result(neuron_result[301]),
		.result_valid(neuron_valid[301]),
		.busy(neuron_busy[301])
	);
	// BRAM init: $readmemh("mem/neuron_301_keys.mem", neuron_301.key_mem);
	// BRAM init: $readmemh("mem/neuron_301_values.mem", neuron_301.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(981),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_302 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_302),
		.result(neuron_result[302]),
		.result_valid(neuron_valid[302]),
		.busy(neuron_busy[302])
	);
	// BRAM init: $readmemh("mem/neuron_302_keys.mem", neuron_302.key_mem);
	// BRAM init: $readmemh("mem/neuron_302_values.mem", neuron_302.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_303 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_303),
		.result(neuron_result[303]),
		.result_valid(neuron_valid[303]),
		.busy(neuron_busy[303])
	);
	// BRAM init: $readmemh("mem/neuron_303_keys.mem", neuron_303.key_mem);
	// BRAM init: $readmemh("mem/neuron_303_values.mem", neuron_303.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(630),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_304 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_304),
		.result(neuron_result[304]),
		.result_valid(neuron_valid[304]),
		.busy(neuron_busy[304])
	);
	// BRAM init: $readmemh("mem/neuron_304_keys.mem", neuron_304.key_mem);
	// BRAM init: $readmemh("mem/neuron_304_values.mem", neuron_304.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(499),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_305 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_305),
		.result(neuron_result[305]),
		.result_valid(neuron_valid[305]),
		.busy(neuron_busy[305])
	);
	// BRAM init: $readmemh("mem/neuron_305_keys.mem", neuron_305.key_mem);
	// BRAM init: $readmemh("mem/neuron_305_values.mem", neuron_305.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_306 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_306),
		.result(neuron_result[306]),
		.result_valid(neuron_valid[306]),
		.busy(neuron_busy[306])
	);
	// BRAM init: $readmemh("mem/neuron_306_keys.mem", neuron_306.key_mem);
	// BRAM init: $readmemh("mem/neuron_306_values.mem", neuron_306.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(740),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_307 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_307),
		.result(neuron_result[307]),
		.result_valid(neuron_valid[307]),
		.busy(neuron_busy[307])
	);
	// BRAM init: $readmemh("mem/neuron_307_keys.mem", neuron_307.key_mem);
	// BRAM init: $readmemh("mem/neuron_307_values.mem", neuron_307.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_308 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_308),
		.result(neuron_result[308]),
		.result_valid(neuron_valid[308]),
		.busy(neuron_busy[308])
	);
	// BRAM init: $readmemh("mem/neuron_308_keys.mem", neuron_308.key_mem);
	// BRAM init: $readmemh("mem/neuron_308_values.mem", neuron_308.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1744),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_309 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_309),
		.result(neuron_result[309]),
		.result_valid(neuron_valid[309]),
		.busy(neuron_busy[309])
	);
	// BRAM init: $readmemh("mem/neuron_309_keys.mem", neuron_309.key_mem);
	// BRAM init: $readmemh("mem/neuron_309_values.mem", neuron_309.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_310 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_310),
		.result(neuron_result[310]),
		.result_valid(neuron_valid[310]),
		.busy(neuron_busy[310])
	);
	// BRAM init: $readmemh("mem/neuron_310_keys.mem", neuron_310.key_mem);
	// BRAM init: $readmemh("mem/neuron_310_values.mem", neuron_310.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(815),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_311 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_311),
		.result(neuron_result[311]),
		.result_valid(neuron_valid[311]),
		.busy(neuron_busy[311])
	);
	// BRAM init: $readmemh("mem/neuron_311_keys.mem", neuron_311.key_mem);
	// BRAM init: $readmemh("mem/neuron_311_values.mem", neuron_311.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_312 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_312),
		.result(neuron_result[312]),
		.result_valid(neuron_valid[312]),
		.busy(neuron_busy[312])
	);
	// BRAM init: $readmemh("mem/neuron_312_keys.mem", neuron_312.key_mem);
	// BRAM init: $readmemh("mem/neuron_312_values.mem", neuron_312.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(718),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_313 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_313),
		.result(neuron_result[313]),
		.result_valid(neuron_valid[313]),
		.busy(neuron_busy[313])
	);
	// BRAM init: $readmemh("mem/neuron_313_keys.mem", neuron_313.key_mem);
	// BRAM init: $readmemh("mem/neuron_313_values.mem", neuron_313.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1615),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_314 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_314),
		.result(neuron_result[314]),
		.result_valid(neuron_valid[314]),
		.busy(neuron_busy[314])
	);
	// BRAM init: $readmemh("mem/neuron_314_keys.mem", neuron_314.key_mem);
	// BRAM init: $readmemh("mem/neuron_314_values.mem", neuron_314.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_315 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_315),
		.result(neuron_result[315]),
		.result_valid(neuron_valid[315]),
		.busy(neuron_busy[315])
	);
	// BRAM init: $readmemh("mem/neuron_315_keys.mem", neuron_315.key_mem);
	// BRAM init: $readmemh("mem/neuron_315_values.mem", neuron_315.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(764),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_316 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_316),
		.result(neuron_result[316]),
		.result_valid(neuron_valid[316]),
		.busy(neuron_busy[316])
	);
	// BRAM init: $readmemh("mem/neuron_316_keys.mem", neuron_316.key_mem);
	// BRAM init: $readmemh("mem/neuron_316_values.mem", neuron_316.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(901),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_317 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_317),
		.result(neuron_result[317]),
		.result_valid(neuron_valid[317]),
		.busy(neuron_busy[317])
	);
	// BRAM init: $readmemh("mem/neuron_317_keys.mem", neuron_317.key_mem);
	// BRAM init: $readmemh("mem/neuron_317_values.mem", neuron_317.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1020),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_318 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_318),
		.result(neuron_result[318]),
		.result_valid(neuron_valid[318]),
		.busy(neuron_busy[318])
	);
	// BRAM init: $readmemh("mem/neuron_318_keys.mem", neuron_318.key_mem);
	// BRAM init: $readmemh("mem/neuron_318_values.mem", neuron_318.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_319 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_319),
		.result(neuron_result[319]),
		.result_valid(neuron_valid[319]),
		.busy(neuron_busy[319])
	);
	// BRAM init: $readmemh("mem/neuron_319_keys.mem", neuron_319.key_mem);
	// BRAM init: $readmemh("mem/neuron_319_values.mem", neuron_319.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_320 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_320),
		.result(neuron_result[320]),
		.result_valid(neuron_valid[320]),
		.busy(neuron_busy[320])
	);
	// BRAM init: $readmemh("mem/neuron_320_keys.mem", neuron_320.key_mem);
	// BRAM init: $readmemh("mem/neuron_320_values.mem", neuron_320.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(927),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_321 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_321),
		.result(neuron_result[321]),
		.result_valid(neuron_valid[321]),
		.busy(neuron_busy[321])
	);
	// BRAM init: $readmemh("mem/neuron_321_keys.mem", neuron_321.key_mem);
	// BRAM init: $readmemh("mem/neuron_321_values.mem", neuron_321.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_322 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_322),
		.result(neuron_result[322]),
		.result_valid(neuron_valid[322]),
		.busy(neuron_busy[322])
	);
	// BRAM init: $readmemh("mem/neuron_322_keys.mem", neuron_322.key_mem);
	// BRAM init: $readmemh("mem/neuron_322_values.mem", neuron_322.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_323 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_323),
		.result(neuron_result[323]),
		.result_valid(neuron_valid[323]),
		.busy(neuron_busy[323])
	);
	// BRAM init: $readmemh("mem/neuron_323_keys.mem", neuron_323.key_mem);
	// BRAM init: $readmemh("mem/neuron_323_values.mem", neuron_323.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1881),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_324 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_324),
		.result(neuron_result[324]),
		.result_valid(neuron_valid[324]),
		.busy(neuron_busy[324])
	);
	// BRAM init: $readmemh("mem/neuron_324_keys.mem", neuron_324.key_mem);
	// BRAM init: $readmemh("mem/neuron_324_values.mem", neuron_324.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(248),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(8)
	) neuron_325 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_325),
		.result(neuron_result[325]),
		.result_valid(neuron_valid[325]),
		.busy(neuron_busy[325])
	);
	// BRAM init: $readmemh("mem/neuron_325_keys.mem", neuron_325.key_mem);
	// BRAM init: $readmemh("mem/neuron_325_values.mem", neuron_325.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(271),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_326 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_326),
		.result(neuron_result[326]),
		.result_valid(neuron_valid[326]),
		.busy(neuron_busy[326])
	);
	// BRAM init: $readmemh("mem/neuron_326_keys.mem", neuron_326.key_mem);
	// BRAM init: $readmemh("mem/neuron_326_values.mem", neuron_326.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1881),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_327 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_327),
		.result(neuron_result[327]),
		.result_valid(neuron_valid[327]),
		.busy(neuron_busy[327])
	);
	// BRAM init: $readmemh("mem/neuron_327_keys.mem", neuron_327.key_mem);
	// BRAM init: $readmemh("mem/neuron_327_values.mem", neuron_327.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_328 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_328),
		.result(neuron_result[328]),
		.result_valid(neuron_valid[328]),
		.busy(neuron_busy[328])
	);
	// BRAM init: $readmemh("mem/neuron_328_keys.mem", neuron_328.key_mem);
	// BRAM init: $readmemh("mem/neuron_328_values.mem", neuron_328.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(981),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_329 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_329),
		.result(neuron_result[329]),
		.result_valid(neuron_valid[329]),
		.busy(neuron_busy[329])
	);
	// BRAM init: $readmemh("mem/neuron_329_keys.mem", neuron_329.key_mem);
	// BRAM init: $readmemh("mem/neuron_329_values.mem", neuron_329.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(943),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_330 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_330),
		.result(neuron_result[330]),
		.result_valid(neuron_valid[330]),
		.busy(neuron_busy[330])
	);
	// BRAM init: $readmemh("mem/neuron_330_keys.mem", neuron_330.key_mem);
	// BRAM init: $readmemh("mem/neuron_330_values.mem", neuron_330.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(840),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_331 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_331),
		.result(neuron_result[331]),
		.result_valid(neuron_valid[331]),
		.busy(neuron_busy[331])
	);
	// BRAM init: $readmemh("mem/neuron_331_keys.mem", neuron_331.key_mem);
	// BRAM init: $readmemh("mem/neuron_331_values.mem", neuron_331.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(403),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_332 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_332),
		.result(neuron_result[332]),
		.result_valid(neuron_valid[332]),
		.busy(neuron_busy[332])
	);
	// BRAM init: $readmemh("mem/neuron_332_keys.mem", neuron_332.key_mem);
	// BRAM init: $readmemh("mem/neuron_332_values.mem", neuron_332.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_333 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_333),
		.result(neuron_result[333]),
		.result_valid(neuron_valid[333]),
		.busy(neuron_busy[333])
	);
	// BRAM init: $readmemh("mem/neuron_333_keys.mem", neuron_333.key_mem);
	// BRAM init: $readmemh("mem/neuron_333_values.mem", neuron_333.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(978),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_334 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_334),
		.result(neuron_result[334]),
		.result_valid(neuron_valid[334]),
		.busy(neuron_busy[334])
	);
	// BRAM init: $readmemh("mem/neuron_334_keys.mem", neuron_334.key_mem);
	// BRAM init: $readmemh("mem/neuron_334_values.mem", neuron_334.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(368),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_335 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_335),
		.result(neuron_result[335]),
		.result_valid(neuron_valid[335]),
		.busy(neuron_busy[335])
	);
	// BRAM init: $readmemh("mem/neuron_335_keys.mem", neuron_335.key_mem);
	// BRAM init: $readmemh("mem/neuron_335_values.mem", neuron_335.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(284),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_336 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_336),
		.result(neuron_result[336]),
		.result_valid(neuron_valid[336]),
		.busy(neuron_busy[336])
	);
	// BRAM init: $readmemh("mem/neuron_336_keys.mem", neuron_336.key_mem);
	// BRAM init: $readmemh("mem/neuron_336_values.mem", neuron_336.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1066),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_337 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_337),
		.result(neuron_result[337]),
		.result_valid(neuron_valid[337]),
		.busy(neuron_busy[337])
	);
	// BRAM init: $readmemh("mem/neuron_337_keys.mem", neuron_337.key_mem);
	// BRAM init: $readmemh("mem/neuron_337_values.mem", neuron_337.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(563),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_338 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_338),
		.result(neuron_result[338]),
		.result_valid(neuron_valid[338]),
		.busy(neuron_busy[338])
	);
	// BRAM init: $readmemh("mem/neuron_338_keys.mem", neuron_338.key_mem);
	// BRAM init: $readmemh("mem/neuron_338_values.mem", neuron_338.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(787),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_339 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_339),
		.result(neuron_result[339]),
		.result_valid(neuron_valid[339]),
		.busy(neuron_busy[339])
	);
	// BRAM init: $readmemh("mem/neuron_339_keys.mem", neuron_339.key_mem);
	// BRAM init: $readmemh("mem/neuron_339_values.mem", neuron_339.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_340 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_340),
		.result(neuron_result[340]),
		.result_valid(neuron_valid[340]),
		.busy(neuron_busy[340])
	);
	// BRAM init: $readmemh("mem/neuron_340_keys.mem", neuron_340.key_mem);
	// BRAM init: $readmemh("mem/neuron_340_values.mem", neuron_340.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(718),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_341 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_341),
		.result(neuron_result[341]),
		.result_valid(neuron_valid[341]),
		.busy(neuron_busy[341])
	);
	// BRAM init: $readmemh("mem/neuron_341_keys.mem", neuron_341.key_mem);
	// BRAM init: $readmemh("mem/neuron_341_values.mem", neuron_341.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_342 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_342),
		.result(neuron_result[342]),
		.result_valid(neuron_valid[342]),
		.busy(neuron_busy[342])
	);
	// BRAM init: $readmemh("mem/neuron_342_keys.mem", neuron_342.key_mem);
	// BRAM init: $readmemh("mem/neuron_342_values.mem", neuron_342.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(553),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_343 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_343),
		.result(neuron_result[343]),
		.result_valid(neuron_valid[343]),
		.busy(neuron_busy[343])
	);
	// BRAM init: $readmemh("mem/neuron_343_keys.mem", neuron_343.key_mem);
	// BRAM init: $readmemh("mem/neuron_343_values.mem", neuron_343.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(656),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_344 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_344),
		.result(neuron_result[344]),
		.result_valid(neuron_valid[344]),
		.busy(neuron_busy[344])
	);
	// BRAM init: $readmemh("mem/neuron_344_keys.mem", neuron_344.key_mem);
	// BRAM init: $readmemh("mem/neuron_344_values.mem", neuron_344.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(630),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_345 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_345),
		.result(neuron_result[345]),
		.result_valid(neuron_valid[345]),
		.busy(neuron_busy[345])
	);
	// BRAM init: $readmemh("mem/neuron_345_keys.mem", neuron_345.key_mem);
	// BRAM init: $readmemh("mem/neuron_345_values.mem", neuron_345.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(522),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_346 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_346),
		.result(neuron_result[346]),
		.result_valid(neuron_valid[346]),
		.busy(neuron_busy[346])
	);
	// BRAM init: $readmemh("mem/neuron_346_keys.mem", neuron_346.key_mem);
	// BRAM init: $readmemh("mem/neuron_346_values.mem", neuron_346.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(952),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_347 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_347),
		.result(neuron_result[347]),
		.result_valid(neuron_valid[347]),
		.busy(neuron_busy[347])
	);
	// BRAM init: $readmemh("mem/neuron_347_keys.mem", neuron_347.key_mem);
	// BRAM init: $readmemh("mem/neuron_347_values.mem", neuron_347.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(915),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_348 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_348),
		.result(neuron_result[348]),
		.result_valid(neuron_valid[348]),
		.busy(neuron_busy[348])
	);
	// BRAM init: $readmemh("mem/neuron_348_keys.mem", neuron_348.key_mem);
	// BRAM init: $readmemh("mem/neuron_348_values.mem", neuron_348.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(271),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_349 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_349),
		.result(neuron_result[349]),
		.result_valid(neuron_valid[349]),
		.busy(neuron_busy[349])
	);
	// BRAM init: $readmemh("mem/neuron_349_keys.mem", neuron_349.key_mem);
	// BRAM init: $readmemh("mem/neuron_349_values.mem", neuron_349.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(981),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_350 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_350),
		.result(neuron_result[350]),
		.result_valid(neuron_valid[350]),
		.busy(neuron_busy[350])
	);
	// BRAM init: $readmemh("mem/neuron_350_keys.mem", neuron_350.key_mem);
	// BRAM init: $readmemh("mem/neuron_350_values.mem", neuron_350.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(955),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_351 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_351),
		.result(neuron_result[351]),
		.result_valid(neuron_valid[351]),
		.busy(neuron_busy[351])
	);
	// BRAM init: $readmemh("mem/neuron_351_keys.mem", neuron_351.key_mem);
	// BRAM init: $readmemh("mem/neuron_351_values.mem", neuron_351.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1352),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_352 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_352),
		.result(neuron_result[352]),
		.result_valid(neuron_valid[352]),
		.busy(neuron_busy[352])
	);
	// BRAM init: $readmemh("mem/neuron_352_keys.mem", neuron_352.key_mem);
	// BRAM init: $readmemh("mem/neuron_352_values.mem", neuron_352.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_353 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_353),
		.result(neuron_result[353]),
		.result_valid(neuron_valid[353]),
		.busy(neuron_busy[353])
	);
	// BRAM init: $readmemh("mem/neuron_353_keys.mem", neuron_353.key_mem);
	// BRAM init: $readmemh("mem/neuron_353_values.mem", neuron_353.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(481),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_354 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_354),
		.result(neuron_result[354]),
		.result_valid(neuron_valid[354]),
		.busy(neuron_busy[354])
	);
	// BRAM init: $readmemh("mem/neuron_354_keys.mem", neuron_354.key_mem);
	// BRAM init: $readmemh("mem/neuron_354_values.mem", neuron_354.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1099),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_355 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_355),
		.result(neuron_result[355]),
		.result_valid(neuron_valid[355]),
		.busy(neuron_busy[355])
	);
	// BRAM init: $readmemh("mem/neuron_355_keys.mem", neuron_355.key_mem);
	// BRAM init: $readmemh("mem/neuron_355_values.mem", neuron_355.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(784),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_356 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_356),
		.result(neuron_result[356]),
		.result_valid(neuron_valid[356]),
		.busy(neuron_busy[356])
	);
	// BRAM init: $readmemh("mem/neuron_356_keys.mem", neuron_356.key_mem);
	// BRAM init: $readmemh("mem/neuron_356_values.mem", neuron_356.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1756),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_357 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_357),
		.result(neuron_result[357]),
		.result_valid(neuron_valid[357]),
		.busy(neuron_busy[357])
	);
	// BRAM init: $readmemh("mem/neuron_357_keys.mem", neuron_357.key_mem);
	// BRAM init: $readmemh("mem/neuron_357_values.mem", neuron_357.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_358 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_358),
		.result(neuron_result[358]),
		.result_valid(neuron_valid[358]),
		.busy(neuron_busy[358])
	);
	// BRAM init: $readmemh("mem/neuron_358_keys.mem", neuron_358.key_mem);
	// BRAM init: $readmemh("mem/neuron_358_values.mem", neuron_358.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(833),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_359 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_359),
		.result(neuron_result[359]),
		.result_valid(neuron_valid[359]),
		.busy(neuron_busy[359])
	);
	// BRAM init: $readmemh("mem/neuron_359_keys.mem", neuron_359.key_mem);
	// BRAM init: $readmemh("mem/neuron_359_values.mem", neuron_359.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(369),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(9)
	) neuron_360 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_360),
		.result(neuron_result[360]),
		.result_valid(neuron_valid[360]),
		.busy(neuron_busy[360])
	);
	// BRAM init: $readmemh("mem/neuron_360_keys.mem", neuron_360.key_mem);
	// BRAM init: $readmemh("mem/neuron_360_values.mem", neuron_360.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1201),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_361 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_361),
		.result(neuron_result[361]),
		.result_valid(neuron_valid[361]),
		.busy(neuron_busy[361])
	);
	// BRAM init: $readmemh("mem/neuron_361_keys.mem", neuron_361.key_mem);
	// BRAM init: $readmemh("mem/neuron_361_values.mem", neuron_361.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(514),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_362 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_362),
		.result(neuron_result[362]),
		.result_valid(neuron_valid[362]),
		.busy(neuron_busy[362])
	);
	// BRAM init: $readmemh("mem/neuron_362_keys.mem", neuron_362.key_mem);
	// BRAM init: $readmemh("mem/neuron_362_values.mem", neuron_362.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(754),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_363 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_363),
		.result(neuron_result[363]),
		.result_valid(neuron_valid[363]),
		.busy(neuron_busy[363])
	);
	// BRAM init: $readmemh("mem/neuron_363_keys.mem", neuron_363.key_mem);
	// BRAM init: $readmemh("mem/neuron_363_values.mem", neuron_363.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1265),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_364 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_364),
		.result(neuron_result[364]),
		.result_valid(neuron_valid[364]),
		.busy(neuron_busy[364])
	);
	// BRAM init: $readmemh("mem/neuron_364_keys.mem", neuron_364.key_mem);
	// BRAM init: $readmemh("mem/neuron_364_values.mem", neuron_364.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1109),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_365 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_365),
		.result(neuron_result[365]),
		.result_valid(neuron_valid[365]),
		.busy(neuron_busy[365])
	);
	// BRAM init: $readmemh("mem/neuron_365_keys.mem", neuron_365.key_mem);
	// BRAM init: $readmemh("mem/neuron_365_values.mem", neuron_365.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(1291),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(11)
	) neuron_366 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_366),
		.result(neuron_result[366]),
		.result_valid(neuron_valid[366]),
		.busy(neuron_busy[366])
	);
	// BRAM init: $readmemh("mem/neuron_366_keys.mem", neuron_366.key_mem);
	// BRAM init: $readmemh("mem/neuron_366_values.mem", neuron_366.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(924),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_367 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_367),
		.result(neuron_result[367]),
		.result_valid(neuron_valid[367]),
		.busy(neuron_busy[367])
	);
	// BRAM init: $readmemh("mem/neuron_367_keys.mem", neuron_367.key_mem);
	// BRAM init: $readmemh("mem/neuron_367_values.mem", neuron_367.value_mem);

	wnn_neuron #(
		.NUM_ENTRIES(698),
		.ADDR_BITS(32),
		.INPUT_BITS(152),
		.SEARCH_DEPTH(10)
	) neuron_368 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_368),
		.result(neuron_result[368]),
		.result_valid(neuron_valid[368]),
		.busy(neuron_busy[368])
	);
	// BRAM init: $readmemh("mem/neuron_368_keys.mem", neuron_368.key_mem);
	// BRAM init: $readmemh("mem/neuron_368_values.mem", neuron_368.value_mem);

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
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[11]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[12]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[13]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[14]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[15]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[16]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[17]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[18]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[19]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[20]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[21]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[22]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[23]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[24]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[25]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[26]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[27]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[28]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[29]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[30]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[31]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[32]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[33]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[34]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[35]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[36]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[37]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[38]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[39]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[40]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[41]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[42]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[43]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[44]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[45]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[46]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[47]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[48]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[49]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[50]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[51]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[52]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[53]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[54]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[55]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[56]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[57]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[58]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[59]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[60]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[61]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[62]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[63]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[64]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[65]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[66]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[67]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[68]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[69]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[70]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[71]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[72]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[73]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[74]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[75]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[76]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[77]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[78]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[79]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[80]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[81]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[82]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[83]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[84]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[85]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[86]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[87]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[88]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[89]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[90]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[91]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[92]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[93]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[94]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[95]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[96]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[97]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[98]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[99]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[100]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[101]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[102]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[103]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[104]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[105]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[106]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[107]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[108]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[109]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[110]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[111]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[112]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[113]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[114]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[115]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[116]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[117]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[118]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[119]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[120]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[121]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[122]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[123]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[124]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[125]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[126]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[127]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[128]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[129]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[130]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[131]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[132]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[133]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[134]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[135]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[136]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[137]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[138]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[139]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[140]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[141]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[142]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[143]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[144]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[145]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[146]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[147]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[148]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[149]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[150]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[151]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[152]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[153]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[154]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[155]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[156]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[157]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[158]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[159]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[160]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[161]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[162]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[163]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[164]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[165]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[166]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[167]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[168]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[169]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[170]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[171]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[172]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[173]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[174]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[175]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[176]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[177]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[178]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[179]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[180]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[181]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[182]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[183]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[184]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[185]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[186]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[187]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[188]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[189]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[190]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[191]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[192]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[193]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[194]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[195]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[196]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[197]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[198]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[199]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[200]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[201]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[202]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[203]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[204]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[205]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[206]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[207]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[208]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[209]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[210]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[211]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[212]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[213]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[214]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[215]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[216]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[217]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[218]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[219]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[220]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[221]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[222]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[223]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[224]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[225]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[226]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[227]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[228]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[229]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[230]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[231]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[232]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[233]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[234]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[235]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[236]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[237]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[238]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[239]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[240]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[241]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[242]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[243]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[244]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[245]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[246]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[247]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[248]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[249]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[250]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[251]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[252]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[253]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[254]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[255]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[256]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[257]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[258]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[259]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[260]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[261]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[262]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[263]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[264]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[265]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[266]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[267]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[268]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[269]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[270]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[271]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[272]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[273]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[274]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[275]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[276]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[277]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[278]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[279]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[280]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[281]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[282]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[283]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[284]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[285]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[286]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[287]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[288]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[289]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[290]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[291]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[292]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[293]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[294]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[295]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[296]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[297]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[298]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[299]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[300]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[301]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[302]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[303]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[304]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[305]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[306]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[307]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[308]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[309]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[310]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[311]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[312]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[313]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[314]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[315]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[316]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[317]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[318]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[319]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[320]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[321]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[322]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[323]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[324]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[325]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[326]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[327]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[328]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[329]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[330]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[331]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[332]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[333]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[334]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[335]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[336]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[337]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[338]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[339]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[340]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[341]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[342]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[343]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[344]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[345]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[346]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[347]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[348]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[349]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[350]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[351]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[352]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[353]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[354]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[355]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[356]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[357]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[358]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[359]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[360]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[361]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[362]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[363]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[364]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[365]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[366]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[367]);
		weighted_sum = weighted_sum + ACC_BITS'(neuron_result[368]);
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
