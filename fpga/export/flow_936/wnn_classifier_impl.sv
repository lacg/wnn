// Auto-generated WNN classifier implementation
// Genome: 68ffd76829c46e0a from flow 936
// 91 neurons × 32-bit addresses
// 232,466 total sparse entries
// Dataset: cicids2017 (random split)
// Thermometer: 16-bit (320 input bits)

module wnn_classifier_impl #(
	parameter int THRESHOLD = 0
) (
	input  logic                     clk,
	input  logic                     rst_n,
	input  logic                     input_valid,
	input  logic [319:0]  input_vec,

	output logic                     class_out,
	output logic [9-1:0] score_out,
	output logic                     output_valid,
	output logic                     busy
);

	localparam int NUM_NEURONS = 91;
	localparam int ADDR_BITS   = 32;
	localparam int INPUT_BITS  = 320;
	localparam int ACC_BITS    = 9;

	// Per-neuron signals
	logic [7:0]  neuron_result [91];
	logic [91-1:0] neuron_valid;
	logic [91-1:0] neuron_busy;
	logic neuron_start;

	assign neuron_start = input_valid & ~(|neuron_busy);
	assign busy = |neuron_busy;

	// --- Per-neuron address formation (evolved connections) ---
	// Neuron 0: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_0;
	assign addr_0 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 1: 1952 entries, bits from features [0, 1, 2, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_1;
	assign addr_1 = {input_vec[45], input_vec[273], input_vec[19], input_vec[139], input_vec[291], input_vec[299], input_vec[150], input_vec[21], input_vec[136], input_vec[193], input_vec[23], input_vec[116], input_vec[18], input_vec[0], input_vec[272], input_vec[22], input_vec[210], input_vec[4], input_vec[15], input_vec[233], input_vec[157], input_vec[196], input_vec[86], input_vec[275], input_vec[270], input_vec[184], input_vec[176], input_vec[23], input_vec[128], input_vec[313], input_vec[294], input_vec[236]};

	// Neuron 2: 1871 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]
	logic [31:0] addr_2;
	assign addr_2 = {input_vec[14], input_vec[48], input_vec[291], input_vec[51], input_vec[167], input_vec[279], input_vec[132], input_vec[78], input_vec[257], input_vec[295], input_vec[115], input_vec[265], input_vec[145], input_vec[212], input_vec[83], input_vec[315], input_vec[259], input_vec[140], input_vec[126], input_vec[149], input_vec[49], input_vec[108], input_vec[160], input_vec[209], input_vec[167], input_vec[203], input_vec[279], input_vec[59], input_vec[218], input_vec[13], input_vec[22], input_vec[179]};

	// Neuron 3: 2955 entries, bits from features [0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_3;
	assign addr_3 = {input_vec[293], input_vec[234], input_vec[155], input_vec[200], input_vec[293], input_vec[12], input_vec[303], input_vec[280], input_vec[292], input_vec[314], input_vec[129], input_vec[187], input_vec[262], input_vec[314], input_vec[127], input_vec[274], input_vec[254], input_vec[284], input_vec[76], input_vec[142], input_vec[208], input_vec[138], input_vec[27], input_vec[23], input_vec[170], input_vec[145], input_vec[195], input_vec[192], input_vec[125], input_vec[172], input_vec[192], input_vec[29]};

	// Neuron 4: 2087 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_4;
	assign addr_4 = {input_vec[258], input_vec[180], input_vec[183], input_vec[40], input_vec[226], input_vec[26], input_vec[91], input_vec[159], input_vec[297], input_vec[26], input_vec[66], input_vec[163], input_vec[230], input_vec[247], input_vec[152], input_vec[67], input_vec[255], input_vec[32], input_vec[280], input_vec[224], input_vec[283], input_vec[306], input_vec[214], input_vec[158], input_vec[195], input_vec[117], input_vec[301], input_vec[113], input_vec[56], input_vec[0], input_vec[117], input_vec[112]};

	// Neuron 5: 4296 entries, bits from features [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19]
	logic [31:0] addr_5;
	assign addr_5 = {input_vec[153], input_vec[230], input_vec[25], input_vec[306], input_vec[190], input_vec[104], input_vec[300], input_vec[311], input_vec[293], input_vec[305], input_vec[50], input_vec[313], input_vec[190], input_vec[266], input_vec[316], input_vec[127], input_vec[130], input_vec[84], input_vec[312], input_vec[182], input_vec[139], input_vec[18], input_vec[179], input_vec[301], input_vec[94], input_vec[114], input_vec[46], input_vec[155], input_vec[271], input_vec[215], input_vec[162], input_vec[56]};

	// Neuron 6: 3420 entries, bits from features [0, 1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_6;
	assign addr_6 = {input_vec[106], input_vec[35], input_vec[312], input_vec[301], input_vec[280], input_vec[256], input_vec[242], input_vec[152], input_vec[6], input_vec[215], input_vec[226], input_vec[44], input_vec[99], input_vec[280], input_vec[70], input_vec[189], input_vec[85], input_vec[277], input_vec[221], input_vec[289], input_vec[7], input_vec[189], input_vec[85], input_vec[191], input_vec[298], input_vec[314], input_vec[23], input_vec[175], input_vec[39], input_vec[176], input_vec[252], input_vec[5]};

	// Neuron 7: 3924 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_7;
	assign addr_7 = {input_vec[218], input_vec[224], input_vec[76], input_vec[219], input_vec[230], input_vec[269], input_vec[312], input_vec[90], input_vec[136], input_vec[123], input_vec[142], input_vec[291], input_vec[51], input_vec[233], input_vec[286], input_vec[161], input_vec[263], input_vec[264], input_vec[39], input_vec[163], input_vec[221], input_vec[227], input_vec[10], input_vec[104], input_vec[152], input_vec[26], input_vec[107], input_vec[282], input_vec[269], input_vec[29], input_vec[230], input_vec[298]};

	// Neuron 8: 1903 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_8;
	assign addr_8 = {input_vec[314], input_vec[188], input_vec[285], input_vec[118], input_vec[252], input_vec[68], input_vec[11], input_vec[263], input_vec[260], input_vec[5], input_vec[48], input_vec[115], input_vec[82], input_vec[9], input_vec[267], input_vec[96], input_vec[33], input_vec[5], input_vec[159], input_vec[284], input_vec[270], input_vec[33], input_vec[231], input_vec[74], input_vec[180], input_vec[1], input_vec[60], input_vec[302], input_vec[217], input_vec[89], input_vec[19], input_vec[144]};

	// Neuron 9: 2860 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_9;
	assign addr_9 = {input_vec[222], input_vec[51], input_vec[107], input_vec[317], input_vec[50], input_vec[2], input_vec[111], input_vec[72], input_vec[236], input_vec[281], input_vec[294], input_vec[135], input_vec[145], input_vec[99], input_vec[203], input_vec[139], input_vec[156], input_vec[29], input_vec[74], input_vec[265], input_vec[132], input_vec[115], input_vec[166], input_vec[128], input_vec[152], input_vec[288], input_vec[208], input_vec[253], input_vec[35], input_vec[249], input_vec[91], input_vec[123]};

	// Neuron 10: 1702 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_10;
	assign addr_10 = {input_vec[148], input_vec[146], input_vec[98], input_vec[161], input_vec[46], input_vec[103], input_vec[112], input_vec[280], input_vec[156], input_vec[144], input_vec[295], input_vec[107], input_vec[106], input_vec[79], input_vec[145], input_vec[267], input_vec[240], input_vec[217], input_vec[100], input_vec[76], input_vec[292], input_vec[237], input_vec[35], input_vec[24], input_vec[27], input_vec[82], input_vec[268], input_vec[285], input_vec[13], input_vec[273], input_vec[145], input_vec[155]};

	// Neuron 11: 1785 entries, bits from features [0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 19]
	logic [31:0] addr_11;
	assign addr_11 = {input_vec[277], input_vec[115], input_vec[22], input_vec[240], input_vec[215], input_vec[203], input_vec[9], input_vec[219], input_vec[217], input_vec[254], input_vec[111], input_vec[139], input_vec[2], input_vec[64], input_vec[125], input_vec[107], input_vec[71], input_vec[85], input_vec[85], input_vec[110], input_vec[101], input_vec[114], input_vec[147], input_vec[315], input_vec[155], input_vec[312], input_vec[17], input_vec[256], input_vec[282], input_vec[102], input_vec[104], input_vec[67]};

	// Neuron 12: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_12;
	assign addr_12 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 13: 3268 entries, bits from features [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_13;
	assign addr_13 = {input_vec[69], input_vec[311], input_vec[278], input_vec[194], input_vec[310], input_vec[64], input_vec[208], input_vec[130], input_vec[149], input_vec[178], input_vec[107], input_vec[108], input_vec[116], input_vec[230], input_vec[227], input_vec[127], input_vec[141], input_vec[148], input_vec[182], input_vec[307], input_vec[154], input_vec[239], input_vec[267], input_vec[185], input_vec[207], input_vec[265], input_vec[132], input_vec[2], input_vec[90], input_vec[270], input_vec[293], input_vec[30]};

	// Neuron 14: 1697 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_14;
	assign addr_14 = {input_vec[82], input_vec[254], input_vec[143], input_vec[157], input_vec[0], input_vec[23], input_vec[184], input_vec[75], input_vec[127], input_vec[41], input_vec[63], input_vec[277], input_vec[290], input_vec[71], input_vec[97], input_vec[227], input_vec[268], input_vec[72], input_vec[213], input_vec[103], input_vec[67], input_vec[116], input_vec[232], input_vec[175], input_vec[8], input_vec[180], input_vec[177], input_vec[99], input_vec[195], input_vec[68], input_vec[249], input_vec[71]};

	// Neuron 15: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_15;
	assign addr_15 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 16: 1697 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_16;
	assign addr_16 = {input_vec[82], input_vec[254], input_vec[143], input_vec[157], input_vec[0], input_vec[23], input_vec[184], input_vec[75], input_vec[127], input_vec[41], input_vec[63], input_vec[277], input_vec[290], input_vec[71], input_vec[97], input_vec[227], input_vec[268], input_vec[72], input_vec[213], input_vec[103], input_vec[67], input_vec[116], input_vec[232], input_vec[175], input_vec[8], input_vec[180], input_vec[177], input_vec[99], input_vec[195], input_vec[68], input_vec[249], input_vec[71]};

	// Neuron 17: 3268 entries, bits from features [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_17;
	assign addr_17 = {input_vec[69], input_vec[311], input_vec[278], input_vec[194], input_vec[310], input_vec[64], input_vec[208], input_vec[130], input_vec[149], input_vec[178], input_vec[107], input_vec[108], input_vec[116], input_vec[230], input_vec[227], input_vec[127], input_vec[141], input_vec[148], input_vec[182], input_vec[307], input_vec[154], input_vec[239], input_vec[267], input_vec[185], input_vec[207], input_vec[265], input_vec[132], input_vec[2], input_vec[90], input_vec[270], input_vec[293], input_vec[30]};

	// Neuron 18: 1952 entries, bits from features [0, 1, 2, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_18;
	assign addr_18 = {input_vec[45], input_vec[273], input_vec[19], input_vec[139], input_vec[291], input_vec[299], input_vec[150], input_vec[21], input_vec[136], input_vec[193], input_vec[23], input_vec[116], input_vec[18], input_vec[0], input_vec[272], input_vec[22], input_vec[210], input_vec[4], input_vec[15], input_vec[233], input_vec[157], input_vec[196], input_vec[86], input_vec[275], input_vec[270], input_vec[184], input_vec[176], input_vec[23], input_vec[128], input_vec[313], input_vec[294], input_vec[236]};

	// Neuron 19: 3420 entries, bits from features [0, 1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_19;
	assign addr_19 = {input_vec[106], input_vec[35], input_vec[312], input_vec[301], input_vec[280], input_vec[256], input_vec[242], input_vec[152], input_vec[6], input_vec[215], input_vec[226], input_vec[44], input_vec[99], input_vec[280], input_vec[70], input_vec[189], input_vec[85], input_vec[277], input_vec[221], input_vec[289], input_vec[7], input_vec[189], input_vec[85], input_vec[191], input_vec[298], input_vec[314], input_vec[23], input_vec[175], input_vec[39], input_vec[176], input_vec[252], input_vec[5]};

	// Neuron 20: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_20;
	assign addr_20 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 21: 2581 entries, bits from features [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_21;
	assign addr_21 = {input_vec[87], input_vec[164], input_vec[99], input_vec[234], input_vec[186], input_vec[274], input_vec[180], input_vec[73], input_vec[190], input_vec[279], input_vec[84], input_vec[111], input_vec[100], input_vec[256], input_vec[150], input_vec[161], input_vec[137], input_vec[194], input_vec[319], input_vec[27], input_vec[86], input_vec[224], input_vec[196], input_vec[3], input_vec[67], input_vec[114], input_vec[219], input_vec[95], input_vec[120], input_vec[189], input_vec[239], input_vec[301]};

	// Neuron 22: 3924 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_22;
	assign addr_22 = {input_vec[218], input_vec[224], input_vec[76], input_vec[219], input_vec[230], input_vec[269], input_vec[312], input_vec[90], input_vec[136], input_vec[123], input_vec[142], input_vec[291], input_vec[51], input_vec[233], input_vec[286], input_vec[161], input_vec[263], input_vec[264], input_vec[39], input_vec[163], input_vec[221], input_vec[227], input_vec[10], input_vec[104], input_vec[152], input_vec[26], input_vec[107], input_vec[282], input_vec[269], input_vec[29], input_vec[230], input_vec[298]};

	// Neuron 23: 2181 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_23;
	assign addr_23 = {input_vec[318], input_vec[170], input_vec[41], input_vec[114], input_vec[268], input_vec[80], input_vec[259], input_vec[95], input_vec[108], input_vec[90], input_vec[14], input_vec[52], input_vec[246], input_vec[166], input_vec[83], input_vec[208], input_vec[204], input_vec[20], input_vec[218], input_vec[306], input_vec[74], input_vec[26], input_vec[236], input_vec[204], input_vec[285], input_vec[27], input_vec[289], input_vec[115], input_vec[272], input_vec[309], input_vec[79], input_vec[260]};

	// Neuron 24: 2640 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
	logic [31:0] addr_24;
	assign addr_24 = {input_vec[82], input_vec[169], input_vec[244], input_vec[195], input_vec[167], input_vec[12], input_vec[301], input_vec[71], input_vec[190], input_vec[260], input_vec[203], input_vec[49], input_vec[82], input_vec[22], input_vec[102], input_vec[151], input_vec[32], input_vec[91], input_vec[129], input_vec[280], input_vec[248], input_vec[189], input_vec[83], input_vec[307], input_vec[194], input_vec[174], input_vec[17], input_vec[146], input_vec[114], input_vec[129], input_vec[210], input_vec[173]};

	// Neuron 25: 1644 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19]
	logic [31:0] addr_25;
	assign addr_25 = {input_vec[221], input_vec[69], input_vec[134], input_vec[171], input_vec[150], input_vec[292], input_vec[208], input_vec[162], input_vec[74], input_vec[237], input_vec[246], input_vec[50], input_vec[195], input_vec[13], input_vec[318], input_vec[200], input_vec[10], input_vec[143], input_vec[155], input_vec[98], input_vec[14], input_vec[163], input_vec[9], input_vec[32], input_vec[114], input_vec[103], input_vec[310], input_vec[27], input_vec[178], input_vec[310], input_vec[87], input_vec[310]};

	// Neuron 26: 2860 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_26;
	assign addr_26 = {input_vec[222], input_vec[51], input_vec[107], input_vec[317], input_vec[50], input_vec[2], input_vec[111], input_vec[72], input_vec[236], input_vec[281], input_vec[294], input_vec[135], input_vec[145], input_vec[99], input_vec[203], input_vec[139], input_vec[156], input_vec[29], input_vec[74], input_vec[265], input_vec[132], input_vec[115], input_vec[166], input_vec[128], input_vec[152], input_vec[288], input_vec[208], input_vec[253], input_vec[35], input_vec[249], input_vec[91], input_vec[123]};

	// Neuron 27: 2181 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_27;
	assign addr_27 = {input_vec[318], input_vec[170], input_vec[41], input_vec[114], input_vec[268], input_vec[80], input_vec[259], input_vec[95], input_vec[108], input_vec[90], input_vec[14], input_vec[52], input_vec[246], input_vec[166], input_vec[83], input_vec[208], input_vec[204], input_vec[20], input_vec[218], input_vec[306], input_vec[74], input_vec[26], input_vec[236], input_vec[204], input_vec[285], input_vec[27], input_vec[289], input_vec[115], input_vec[272], input_vec[309], input_vec[79], input_vec[260]};

	// Neuron 28: 2348 entries, bits from features [3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_28;
	assign addr_28 = {input_vec[299], input_vec[105], input_vec[155], input_vec[272], input_vec[90], input_vec[97], input_vec[240], input_vec[253], input_vec[313], input_vec[214], input_vec[94], input_vec[281], input_vec[163], input_vec[118], input_vec[208], input_vec[79], input_vec[225], input_vec[82], input_vec[273], input_vec[110], input_vec[174], input_vec[60], input_vec[300], input_vec[135], input_vec[87], input_vec[265], input_vec[292], input_vec[225], input_vec[83], input_vec[318], input_vec[173], input_vec[70]};

	// Neuron 29: 2348 entries, bits from features [3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_29;
	assign addr_29 = {input_vec[299], input_vec[105], input_vec[155], input_vec[272], input_vec[90], input_vec[97], input_vec[240], input_vec[253], input_vec[313], input_vec[214], input_vec[94], input_vec[281], input_vec[163], input_vec[118], input_vec[208], input_vec[79], input_vec[225], input_vec[82], input_vec[273], input_vec[110], input_vec[174], input_vec[60], input_vec[300], input_vec[135], input_vec[87], input_vec[265], input_vec[292], input_vec[225], input_vec[83], input_vec[318], input_vec[173], input_vec[70]};

	// Neuron 30: 1871 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]
	logic [31:0] addr_30;
	assign addr_30 = {input_vec[14], input_vec[48], input_vec[291], input_vec[51], input_vec[167], input_vec[279], input_vec[132], input_vec[78], input_vec[257], input_vec[295], input_vec[115], input_vec[265], input_vec[145], input_vec[212], input_vec[83], input_vec[315], input_vec[259], input_vec[140], input_vec[126], input_vec[149], input_vec[49], input_vec[108], input_vec[160], input_vec[209], input_vec[167], input_vec[203], input_vec[279], input_vec[59], input_vec[218], input_vec[13], input_vec[22], input_vec[179]};

	// Neuron 31: 1903 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_31;
	assign addr_31 = {input_vec[314], input_vec[188], input_vec[285], input_vec[118], input_vec[252], input_vec[68], input_vec[11], input_vec[263], input_vec[260], input_vec[5], input_vec[48], input_vec[115], input_vec[82], input_vec[9], input_vec[267], input_vec[96], input_vec[33], input_vec[5], input_vec[159], input_vec[284], input_vec[270], input_vec[33], input_vec[231], input_vec[74], input_vec[180], input_vec[1], input_vec[60], input_vec[302], input_vec[217], input_vec[89], input_vec[19], input_vec[144]};

	// Neuron 32: 4296 entries, bits from features [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19]
	logic [31:0] addr_32;
	assign addr_32 = {input_vec[153], input_vec[230], input_vec[25], input_vec[306], input_vec[190], input_vec[104], input_vec[300], input_vec[311], input_vec[293], input_vec[305], input_vec[50], input_vec[313], input_vec[190], input_vec[266], input_vec[316], input_vec[127], input_vec[130], input_vec[84], input_vec[312], input_vec[182], input_vec[139], input_vec[18], input_vec[179], input_vec[301], input_vec[94], input_vec[114], input_vec[46], input_vec[155], input_vec[271], input_vec[215], input_vec[162], input_vec[56]};

	// Neuron 33: 1697 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_33;
	assign addr_33 = {input_vec[82], input_vec[254], input_vec[143], input_vec[157], input_vec[0], input_vec[23], input_vec[184], input_vec[75], input_vec[127], input_vec[41], input_vec[63], input_vec[277], input_vec[290], input_vec[71], input_vec[97], input_vec[227], input_vec[268], input_vec[72], input_vec[213], input_vec[103], input_vec[67], input_vec[116], input_vec[232], input_vec[175], input_vec[8], input_vec[180], input_vec[177], input_vec[99], input_vec[195], input_vec[68], input_vec[249], input_vec[71]};

	// Neuron 34: 2206 entries, bits from features [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]
	logic [31:0] addr_34;
	assign addr_34 = {input_vec[63], input_vec[227], input_vec[113], input_vec[199], input_vec[148], input_vec[5], input_vec[132], input_vec[115], input_vec[187], input_vec[19], input_vec[317], input_vec[196], input_vec[307], input_vec[61], input_vec[17], input_vec[38], input_vec[70], input_vec[272], input_vec[205], input_vec[204], input_vec[209], input_vec[189], input_vec[246], input_vec[175], input_vec[112], input_vec[172], input_vec[300], input_vec[236], input_vec[115], input_vec[27], input_vec[11], input_vec[58]};

	// Neuron 35: 2331 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 19]
	logic [31:0] addr_35;
	assign addr_35 = {input_vec[192], input_vec[88], input_vec[64], input_vec[220], input_vec[91], input_vec[251], input_vec[183], input_vec[136], input_vec[14], input_vec[194], input_vec[315], input_vec[41], input_vec[36], input_vec[318], input_vec[61], input_vec[139], input_vec[44], input_vec[104], input_vec[39], input_vec[308], input_vec[247], input_vec[77], input_vec[282], input_vec[188], input_vec[202], input_vec[161], input_vec[103], input_vec[55], input_vec[23], input_vec[196], input_vec[243], input_vec[267]};

	// Neuron 36: 2640 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
	logic [31:0] addr_36;
	assign addr_36 = {input_vec[82], input_vec[169], input_vec[244], input_vec[195], input_vec[167], input_vec[12], input_vec[301], input_vec[71], input_vec[190], input_vec[260], input_vec[203], input_vec[49], input_vec[82], input_vec[22], input_vec[102], input_vec[151], input_vec[32], input_vec[91], input_vec[129], input_vec[280], input_vec[248], input_vec[189], input_vec[83], input_vec[307], input_vec[194], input_vec[174], input_vec[17], input_vec[146], input_vec[114], input_vec[129], input_vec[210], input_vec[173]};

	// Neuron 37: 2342 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_37;
	assign addr_37 = {input_vec[282], input_vec[60], input_vec[288], input_vec[63], input_vec[78], input_vec[104], input_vec[296], input_vec[9], input_vec[288], input_vec[54], input_vec[49], input_vec[47], input_vec[241], input_vec[215], input_vec[158], input_vec[165], input_vec[93], input_vec[138], input_vec[131], input_vec[217], input_vec[194], input_vec[264], input_vec[194], input_vec[309], input_vec[10], input_vec[247], input_vec[23], input_vec[314], input_vec[177], input_vec[62], input_vec[229], input_vec[168]};

	// Neuron 38: 2728 entries, bits from features [0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_38;
	assign addr_38 = {input_vec[141], input_vec[93], input_vec[310], input_vec[2], input_vec[262], input_vec[30], input_vec[5], input_vec[105], input_vec[294], input_vec[25], input_vec[204], input_vec[253], input_vec[306], input_vec[135], input_vec[193], input_vec[242], input_vec[193], input_vec[300], input_vec[160], input_vec[183], input_vec[1], input_vec[305], input_vec[233], input_vec[174], input_vec[7], input_vec[286], input_vec[270], input_vec[264], input_vec[73], input_vec[173], input_vec[194], input_vec[150]};

	// Neuron 39: 1952 entries, bits from features [0, 1, 2, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_39;
	assign addr_39 = {input_vec[45], input_vec[273], input_vec[19], input_vec[139], input_vec[291], input_vec[299], input_vec[150], input_vec[21], input_vec[136], input_vec[193], input_vec[23], input_vec[116], input_vec[18], input_vec[0], input_vec[272], input_vec[22], input_vec[210], input_vec[4], input_vec[15], input_vec[233], input_vec[157], input_vec[196], input_vec[86], input_vec[275], input_vec[270], input_vec[184], input_vec[176], input_vec[23], input_vec[128], input_vec[313], input_vec[294], input_vec[236]};

	// Neuron 40: 2479 entries, bits from features [0, 1, 2, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_40;
	assign addr_40 = {input_vec[15], input_vec[15], input_vec[265], input_vec[16], input_vec[258], input_vec[103], input_vec[148], input_vec[314], input_vec[166], input_vec[81], input_vec[47], input_vec[12], input_vec[85], input_vec[122], input_vec[234], input_vec[204], input_vec[206], input_vec[249], input_vec[250], input_vec[157], input_vec[164], input_vec[174], input_vec[123], input_vec[220], input_vec[22], input_vec[301], input_vec[282], input_vec[269], input_vec[149], input_vec[22], input_vec[254], input_vec[47]};

	// Neuron 41: 2443 entries, bits from features [0, 1, 2, 4, 7, 8, 9, 10, 11, 13, 14, 15, 18, 19]
	logic [31:0] addr_41;
	assign addr_41 = {input_vec[65], input_vec[65], input_vec[75], input_vec[210], input_vec[126], input_vec[64], input_vec[25], input_vec[8], input_vec[176], input_vec[185], input_vec[65], input_vec[295], input_vec[23], input_vec[34], input_vec[20], input_vec[231], input_vec[184], input_vec[34], input_vec[42], input_vec[74], input_vec[168], input_vec[112], input_vec[154], input_vec[243], input_vec[309], input_vec[8], input_vec[75], input_vec[133], input_vec[184], input_vec[191], input_vec[227], input_vec[145]};

	// Neuron 42: 3268 entries, bits from features [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_42;
	assign addr_42 = {input_vec[69], input_vec[311], input_vec[278], input_vec[194], input_vec[310], input_vec[64], input_vec[208], input_vec[130], input_vec[149], input_vec[178], input_vec[107], input_vec[108], input_vec[116], input_vec[230], input_vec[227], input_vec[127], input_vec[141], input_vec[148], input_vec[182], input_vec[307], input_vec[154], input_vec[239], input_vec[267], input_vec[185], input_vec[207], input_vec[265], input_vec[132], input_vec[2], input_vec[90], input_vec[270], input_vec[293], input_vec[30]};

	// Neuron 43: 2108 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_43;
	assign addr_43 = {input_vec[50], input_vec[44], input_vec[13], input_vec[22], input_vec[8], input_vec[162], input_vec[165], input_vec[265], input_vec[90], input_vec[115], input_vec[17], input_vec[257], input_vec[124], input_vec[51], input_vec[134], input_vec[116], input_vec[218], input_vec[204], input_vec[79], input_vec[77], input_vec[17], input_vec[266], input_vec[103], input_vec[83], input_vec[57], input_vec[156], input_vec[286], input_vec[267], input_vec[247], input_vec[296], input_vec[121], input_vec[276]};

	// Neuron 44: 3268 entries, bits from features [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_44;
	assign addr_44 = {input_vec[69], input_vec[311], input_vec[278], input_vec[194], input_vec[310], input_vec[64], input_vec[208], input_vec[130], input_vec[149], input_vec[178], input_vec[107], input_vec[108], input_vec[116], input_vec[230], input_vec[227], input_vec[127], input_vec[141], input_vec[148], input_vec[182], input_vec[307], input_vec[154], input_vec[239], input_vec[267], input_vec[185], input_vec[207], input_vec[265], input_vec[132], input_vec[2], input_vec[90], input_vec[270], input_vec[293], input_vec[30]};

	// Neuron 45: 1799 entries, bits from features [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_45;
	assign addr_45 = {input_vec[50], input_vec[155], input_vec[107], input_vec[284], input_vec[11], input_vec[210], input_vec[33], input_vec[303], input_vec[281], input_vec[60], input_vec[306], input_vec[207], input_vec[287], input_vec[215], input_vec[311], input_vec[14], input_vec[71], input_vec[93], input_vec[161], input_vec[102], input_vec[69], input_vec[107], input_vec[270], input_vec[35], input_vec[155], input_vec[235], input_vec[149], input_vec[233], input_vec[240], input_vec[302], input_vec[182], input_vec[86]};

	// Neuron 46: 2108 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18]
	logic [31:0] addr_46;
	assign addr_46 = {input_vec[50], input_vec[44], input_vec[13], input_vec[22], input_vec[8], input_vec[162], input_vec[165], input_vec[265], input_vec[90], input_vec[115], input_vec[17], input_vec[257], input_vec[124], input_vec[51], input_vec[134], input_vec[116], input_vec[218], input_vec[204], input_vec[79], input_vec[77], input_vec[17], input_vec[266], input_vec[103], input_vec[83], input_vec[57], input_vec[156], input_vec[286], input_vec[267], input_vec[247], input_vec[296], input_vec[121], input_vec[276]};

	// Neuron 47: 2640 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
	logic [31:0] addr_47;
	assign addr_47 = {input_vec[82], input_vec[169], input_vec[244], input_vec[195], input_vec[167], input_vec[12], input_vec[301], input_vec[71], input_vec[190], input_vec[260], input_vec[203], input_vec[49], input_vec[82], input_vec[22], input_vec[102], input_vec[151], input_vec[32], input_vec[91], input_vec[129], input_vec[280], input_vec[248], input_vec[189], input_vec[83], input_vec[307], input_vec[194], input_vec[174], input_vec[17], input_vec[146], input_vec[114], input_vec[129], input_vec[210], input_vec[173]};

	// Neuron 48: 2439 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 18, 19]
	logic [31:0] addr_48;
	assign addr_48 = {input_vec[177], input_vec[211], input_vec[29], input_vec[209], input_vec[121], input_vec[191], input_vec[56], input_vec[51], input_vec[92], input_vec[29], input_vec[275], input_vec[155], input_vec[61], input_vec[155], input_vec[86], input_vec[71], input_vec[315], input_vec[160], input_vec[110], input_vec[17], input_vec[85], input_vec[134], input_vec[160], input_vec[173], input_vec[98], input_vec[23], input_vec[182], input_vec[114], input_vec[13], input_vec[49], input_vec[267], input_vec[290]};

	// Neuron 49: 2139 entries, bits from features [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
	logic [31:0] addr_49;
	assign addr_49 = {input_vec[39], input_vec[216], input_vec[15], input_vec[318], input_vec[270], input_vec[308], input_vec[185], input_vec[152], input_vec[17], input_vec[24], input_vec[213], input_vec[54], input_vec[289], input_vec[167], input_vec[238], input_vec[150], input_vec[225], input_vec[17], input_vec[209], input_vec[79], input_vec[244], input_vec[296], input_vec[265], input_vec[158], input_vec[181], input_vec[231], input_vec[16], input_vec[175], input_vec[307], input_vec[166], input_vec[206], input_vec[219]};

	// Neuron 50: 3039 entries, bits from features [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_50;
	assign addr_50 = {input_vec[97], input_vec[266], input_vec[307], input_vec[135], input_vec[274], input_vec[224], input_vec[106], input_vec[300], input_vec[72], input_vec[211], input_vec[7], input_vec[179], input_vec[185], input_vec[79], input_vec[313], input_vec[15], input_vec[146], input_vec[226], input_vec[53], input_vec[70], input_vec[204], input_vec[248], input_vec[155], input_vec[167], input_vec[74], input_vec[104], input_vec[284], input_vec[61], input_vec[50], input_vec[36], input_vec[162], input_vec[248]};

	// Neuron 51: 4296 entries, bits from features [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19]
	logic [31:0] addr_51;
	assign addr_51 = {input_vec[153], input_vec[230], input_vec[25], input_vec[306], input_vec[190], input_vec[104], input_vec[300], input_vec[311], input_vec[293], input_vec[305], input_vec[50], input_vec[313], input_vec[190], input_vec[266], input_vec[316], input_vec[127], input_vec[130], input_vec[84], input_vec[312], input_vec[182], input_vec[139], input_vec[18], input_vec[179], input_vec[301], input_vec[94], input_vec[114], input_vec[46], input_vec[155], input_vec[271], input_vec[215], input_vec[162], input_vec[56]};

	// Neuron 52: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_52;
	assign addr_52 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 53: 3670 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_53;
	assign addr_53 = {input_vec[151], input_vec[164], input_vec[71], input_vec[232], input_vec[1], input_vec[171], input_vec[303], input_vec[97], input_vec[275], input_vec[40], input_vec[224], input_vec[34], input_vec[10], input_vec[296], input_vec[270], input_vec[207], input_vec[186], input_vec[309], input_vec[86], input_vec[140], input_vec[29], input_vec[57], input_vec[83], input_vec[128], input_vec[60], input_vec[302], input_vec[243], input_vec[168], input_vec[310], input_vec[6], input_vec[103], input_vec[185]};

	// Neuron 54: 1868 entries, bits from features [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_54;
	assign addr_54 = {input_vec[78], input_vec[149], input_vec[130], input_vec[25], input_vec[145], input_vec[271], input_vec[114], input_vec[222], input_vec[66], input_vec[266], input_vec[287], input_vec[187], input_vec[238], input_vec[220], input_vec[204], input_vec[193], input_vec[247], input_vec[263], input_vec[223], input_vec[300], input_vec[88], input_vec[198], input_vec[114], input_vec[17], input_vec[248], input_vec[145], input_vec[137], input_vec[238], input_vec[281], input_vec[166], input_vec[45], input_vec[25]};

	// Neuron 55: 2342 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_55;
	assign addr_55 = {input_vec[282], input_vec[60], input_vec[288], input_vec[63], input_vec[78], input_vec[104], input_vec[296], input_vec[9], input_vec[288], input_vec[54], input_vec[49], input_vec[47], input_vec[241], input_vec[215], input_vec[158], input_vec[165], input_vec[93], input_vec[138], input_vec[131], input_vec[217], input_vec[194], input_vec[264], input_vec[194], input_vec[309], input_vec[10], input_vec[247], input_vec[23], input_vec[314], input_vec[177], input_vec[62], input_vec[229], input_vec[168]};

	// Neuron 56: 2263 entries, bits from features [0, 1, 2, 3, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18]
	logic [31:0] addr_56;
	assign addr_56 = {input_vec[158], input_vec[301], input_vec[26], input_vec[127], input_vec[296], input_vec[230], input_vec[136], input_vec[193], input_vec[200], input_vec[82], input_vec[195], input_vec[27], input_vec[253], input_vec[156], input_vec[271], input_vec[33], input_vec[92], input_vec[225], input_vec[143], input_vec[171], input_vec[199], input_vec[16], input_vec[248], input_vec[127], input_vec[252], input_vec[148], input_vec[3], input_vec[52], input_vec[161], input_vec[21], input_vec[284], input_vec[289]};

	// Neuron 57: 3924 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_57;
	assign addr_57 = {input_vec[218], input_vec[224], input_vec[76], input_vec[219], input_vec[230], input_vec[269], input_vec[312], input_vec[90], input_vec[136], input_vec[123], input_vec[142], input_vec[291], input_vec[51], input_vec[233], input_vec[286], input_vec[161], input_vec[263], input_vec[264], input_vec[39], input_vec[163], input_vec[221], input_vec[227], input_vec[10], input_vec[104], input_vec[152], input_vec[26], input_vec[107], input_vec[282], input_vec[269], input_vec[29], input_vec[230], input_vec[298]};

	// Neuron 58: 3715 entries, bits from features [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_58;
	assign addr_58 = {input_vec[282], input_vec[149], input_vec[181], input_vec[146], input_vec[292], input_vec[155], input_vec[307], input_vec[261], input_vec[116], input_vec[232], input_vec[214], input_vec[144], input_vec[229], input_vec[72], input_vec[258], input_vec[127], input_vec[208], input_vec[111], input_vec[234], input_vec[174], input_vec[30], input_vec[122], input_vec[4], input_vec[294], input_vec[60], input_vec[275], input_vec[296], input_vec[23], input_vec[45], input_vec[126], input_vec[107], input_vec[34]};

	// Neuron 59: 1644 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19]
	logic [31:0] addr_59;
	assign addr_59 = {input_vec[221], input_vec[69], input_vec[134], input_vec[171], input_vec[150], input_vec[292], input_vec[208], input_vec[162], input_vec[74], input_vec[237], input_vec[246], input_vec[50], input_vec[195], input_vec[13], input_vec[318], input_vec[200], input_vec[10], input_vec[143], input_vec[155], input_vec[98], input_vec[14], input_vec[163], input_vec[9], input_vec[32], input_vec[114], input_vec[103], input_vec[310], input_vec[27], input_vec[178], input_vec[310], input_vec[87], input_vec[310]};

	// Neuron 60: 2206 entries, bits from features [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]
	logic [31:0] addr_60;
	assign addr_60 = {input_vec[63], input_vec[227], input_vec[113], input_vec[199], input_vec[148], input_vec[5], input_vec[132], input_vec[115], input_vec[187], input_vec[19], input_vec[317], input_vec[196], input_vec[307], input_vec[61], input_vec[17], input_vec[38], input_vec[70], input_vec[272], input_vec[205], input_vec[204], input_vec[209], input_vec[189], input_vec[246], input_vec[175], input_vec[112], input_vec[172], input_vec[300], input_vec[236], input_vec[115], input_vec[27], input_vec[11], input_vec[58]};

	// Neuron 61: 3924 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19]
	logic [31:0] addr_61;
	assign addr_61 = {input_vec[218], input_vec[224], input_vec[76], input_vec[219], input_vec[230], input_vec[269], input_vec[312], input_vec[90], input_vec[136], input_vec[123], input_vec[142], input_vec[291], input_vec[51], input_vec[233], input_vec[286], input_vec[161], input_vec[263], input_vec[264], input_vec[39], input_vec[163], input_vec[221], input_vec[227], input_vec[10], input_vec[104], input_vec[152], input_vec[26], input_vec[107], input_vec[282], input_vec[269], input_vec[29], input_vec[230], input_vec[298]};

	// Neuron 62: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_62;
	assign addr_62 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 63: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_63;
	assign addr_63 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 64: 2444 entries, bits from features [0, 1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
	logic [31:0] addr_64;
	assign addr_64 = {input_vec[195], input_vec[155], input_vec[36], input_vec[143], input_vec[37], input_vec[268], input_vec[39], input_vec[227], input_vec[26], input_vec[244], input_vec[94], input_vec[29], input_vec[312], input_vec[154], input_vec[28], input_vec[284], input_vec[213], input_vec[19], input_vec[319], input_vec[133], input_vec[196], input_vec[66], input_vec[26], input_vec[72], input_vec[195], input_vec[6], input_vec[178], input_vec[149], input_vec[153], input_vec[166], input_vec[89], input_vec[162]};

	// Neuron 65: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_65;
	assign addr_65 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 66: 1756 entries, bits from features [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 17, 18]
	logic [31:0] addr_66;
	assign addr_66 = {input_vec[84], input_vec[152], input_vec[80], input_vec[79], input_vec[155], input_vec[62], input_vec[172], input_vec[192], input_vec[115], input_vec[134], input_vec[297], input_vec[1], input_vec[166], input_vec[202], input_vec[121], input_vec[219], input_vec[124], input_vec[68], input_vec[299], input_vec[286], input_vec[36], input_vec[190], input_vec[58], input_vec[32], input_vec[43], input_vec[274], input_vec[58], input_vec[60], input_vec[197], input_vec[156], input_vec[298], input_vec[131]};

	// Neuron 67: 2860 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_67;
	assign addr_67 = {input_vec[222], input_vec[51], input_vec[107], input_vec[317], input_vec[50], input_vec[2], input_vec[111], input_vec[72], input_vec[236], input_vec[281], input_vec[294], input_vec[135], input_vec[145], input_vec[99], input_vec[203], input_vec[139], input_vec[156], input_vec[29], input_vec[74], input_vec[265], input_vec[132], input_vec[115], input_vec[166], input_vec[128], input_vec[152], input_vec[288], input_vec[208], input_vec[253], input_vec[35], input_vec[249], input_vec[91], input_vec[123]};

	// Neuron 68: 2087 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_68;
	assign addr_68 = {input_vec[258], input_vec[180], input_vec[183], input_vec[40], input_vec[226], input_vec[26], input_vec[91], input_vec[159], input_vec[297], input_vec[26], input_vec[66], input_vec[163], input_vec[230], input_vec[247], input_vec[152], input_vec[67], input_vec[255], input_vec[32], input_vec[280], input_vec[224], input_vec[283], input_vec[306], input_vec[214], input_vec[158], input_vec[195], input_vec[117], input_vec[301], input_vec[113], input_vec[56], input_vec[0], input_vec[117], input_vec[112]};

	// Neuron 69: 2706 entries, bits from features [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_69;
	assign addr_69 = {input_vec[102], input_vec[25], input_vec[292], input_vec[224], input_vec[309], input_vec[91], input_vec[143], input_vec[59], input_vec[66], input_vec[143], input_vec[221], input_vec[197], input_vec[245], input_vec[79], input_vec[262], input_vec[183], input_vec[103], input_vec[164], input_vec[55], input_vec[62], input_vec[84], input_vec[113], input_vec[122], input_vec[191], input_vec[178], input_vec[283], input_vec[202], input_vec[149], input_vec[291], input_vec[289], input_vec[148], input_vec[227]};

	// Neuron 70: 2139 entries, bits from features [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
	logic [31:0] addr_70;
	assign addr_70 = {input_vec[39], input_vec[216], input_vec[15], input_vec[318], input_vec[270], input_vec[308], input_vec[185], input_vec[152], input_vec[17], input_vec[24], input_vec[213], input_vec[54], input_vec[289], input_vec[167], input_vec[238], input_vec[150], input_vec[225], input_vec[17], input_vec[209], input_vec[79], input_vec[244], input_vec[296], input_vec[265], input_vec[158], input_vec[181], input_vec[231], input_vec[16], input_vec[175], input_vec[307], input_vec[166], input_vec[206], input_vec[219]};

	// Neuron 71: 2121 entries, bits from features [0, 1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_71;
	assign addr_71 = {input_vec[165], input_vec[269], input_vec[123], input_vec[251], input_vec[39], input_vec[249], input_vec[138], input_vec[205], input_vec[149], input_vec[19], input_vec[94], input_vec[114], input_vec[229], input_vec[6], input_vec[232], input_vec[19], input_vec[113], input_vec[216], input_vec[247], input_vec[281], input_vec[23], input_vec[174], input_vec[80], input_vec[291], input_vec[209], input_vec[192], input_vec[190], input_vec[225], input_vec[31], input_vec[6], input_vec[233], input_vec[256]};

	// Neuron 72: 2139 entries, bits from features [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
	logic [31:0] addr_72;
	assign addr_72 = {input_vec[39], input_vec[216], input_vec[15], input_vec[318], input_vec[270], input_vec[308], input_vec[185], input_vec[152], input_vec[17], input_vec[24], input_vec[213], input_vec[54], input_vec[289], input_vec[167], input_vec[238], input_vec[150], input_vec[225], input_vec[17], input_vec[209], input_vec[79], input_vec[244], input_vec[296], input_vec[265], input_vec[158], input_vec[181], input_vec[231], input_vec[16], input_vec[175], input_vec[307], input_vec[166], input_vec[206], input_vec[219]};

	// Neuron 73: 2955 entries, bits from features [0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_73;
	assign addr_73 = {input_vec[293], input_vec[234], input_vec[155], input_vec[200], input_vec[293], input_vec[12], input_vec[303], input_vec[280], input_vec[292], input_vec[314], input_vec[129], input_vec[187], input_vec[262], input_vec[314], input_vec[127], input_vec[274], input_vec[254], input_vec[284], input_vec[76], input_vec[142], input_vec[208], input_vec[138], input_vec[27], input_vec[23], input_vec[170], input_vec[145], input_vec[195], input_vec[192], input_vec[125], input_vec[172], input_vec[192], input_vec[29]};

	// Neuron 74: 2640 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
	logic [31:0] addr_74;
	assign addr_74 = {input_vec[82], input_vec[169], input_vec[244], input_vec[195], input_vec[167], input_vec[12], input_vec[301], input_vec[71], input_vec[190], input_vec[260], input_vec[203], input_vec[49], input_vec[82], input_vec[22], input_vec[102], input_vec[151], input_vec[32], input_vec[91], input_vec[129], input_vec[280], input_vec[248], input_vec[189], input_vec[83], input_vec[307], input_vec[194], input_vec[174], input_vec[17], input_vec[146], input_vec[114], input_vec[129], input_vec[210], input_vec[173]};

	// Neuron 75: 2087 entries, bits from features [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_75;
	assign addr_75 = {input_vec[258], input_vec[180], input_vec[183], input_vec[40], input_vec[226], input_vec[26], input_vec[91], input_vec[159], input_vec[297], input_vec[26], input_vec[66], input_vec[163], input_vec[230], input_vec[247], input_vec[152], input_vec[67], input_vec[255], input_vec[32], input_vec[280], input_vec[224], input_vec[283], input_vec[306], input_vec[214], input_vec[158], input_vec[195], input_vec[117], input_vec[301], input_vec[113], input_vec[56], input_vec[0], input_vec[117], input_vec[112]};

	// Neuron 76: 2149 entries, bits from features [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18]
	logic [31:0] addr_76;
	assign addr_76 = {input_vec[14], input_vec[202], input_vec[234], input_vec[107], input_vec[40], input_vec[166], input_vec[92], input_vec[21], input_vec[172], input_vec[89], input_vec[86], input_vec[7], input_vec[33], input_vec[271], input_vec[46], input_vec[135], input_vec[22], input_vec[15], input_vec[294], input_vec[101], input_vec[165], input_vec[160], input_vec[195], input_vec[159], input_vec[43], input_vec[190], input_vec[14], input_vec[74], input_vec[0], input_vec[98], input_vec[254], input_vec[126]};

	// Neuron 77: 1551 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19]
	logic [31:0] addr_77;
	assign addr_77 = {input_vec[22], input_vec[97], input_vec[248], input_vec[46], input_vec[209], input_vec[272], input_vec[288], input_vec[89], input_vec[203], input_vec[163], input_vec[15], input_vec[134], input_vec[216], input_vec[305], input_vec[317], input_vec[99], input_vec[177], input_vec[66], input_vec[65], input_vec[82], input_vec[165], input_vec[152], input_vec[89], input_vec[310], input_vec[204], input_vec[57], input_vec[175], input_vec[291], input_vec[272], input_vec[313], input_vec[301], input_vec[162]};

	// Neuron 78: 1799 entries, bits from features [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_78;
	assign addr_78 = {input_vec[50], input_vec[155], input_vec[107], input_vec[284], input_vec[11], input_vec[210], input_vec[33], input_vec[303], input_vec[281], input_vec[60], input_vec[306], input_vec[207], input_vec[287], input_vec[215], input_vec[311], input_vec[14], input_vec[71], input_vec[93], input_vec[161], input_vec[102], input_vec[69], input_vec[107], input_vec[270], input_vec[35], input_vec[155], input_vec[235], input_vec[149], input_vec[233], input_vec[240], input_vec[302], input_vec[182], input_vec[86]};

	// Neuron 79: 2342 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_79;
	assign addr_79 = {input_vec[282], input_vec[60], input_vec[288], input_vec[63], input_vec[78], input_vec[104], input_vec[296], input_vec[9], input_vec[288], input_vec[54], input_vec[49], input_vec[47], input_vec[241], input_vec[215], input_vec[158], input_vec[165], input_vec[93], input_vec[138], input_vec[131], input_vec[217], input_vec[194], input_vec[264], input_vec[194], input_vec[309], input_vec[10], input_vec[247], input_vec[23], input_vec[314], input_vec[177], input_vec[62], input_vec[229], input_vec[168]};

	// Neuron 80: 2686 entries, bits from features [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 16, 17, 18, 19]
	logic [31:0] addr_80;
	assign addr_80 = {input_vec[274], input_vec[133], input_vec[207], input_vec[109], input_vec[151], input_vec[191], input_vec[62], input_vec[132], input_vec[271], input_vec[93], input_vec[147], input_vec[31], input_vec[16], input_vec[301], input_vec[260], input_vec[198], input_vec[55], input_vec[234], input_vec[64], input_vec[25], input_vec[196], input_vec[176], input_vec[285], input_vec[300], input_vec[88], input_vec[17], input_vec[135], input_vec[203], input_vec[318], input_vec[314], input_vec[143], input_vec[41]};

	// Neuron 81: 4584 entries, bits from features [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 16, 17, 18, 19]
	logic [31:0] addr_81;
	assign addr_81 = {input_vec[286], input_vec[153], input_vec[256], input_vec[31], input_vec[280], input_vec[189], input_vec[33], input_vec[294], input_vec[129], input_vec[109], input_vec[23], input_vec[104], input_vec[110], input_vec[57], input_vec[264], input_vec[82], input_vec[122], input_vec[37], input_vec[316], input_vec[265], input_vec[311], input_vec[22], input_vec[317], input_vec[96], input_vec[140], input_vec[225], input_vec[20], input_vec[120], input_vec[188], input_vec[68], input_vec[230], input_vec[54]};

	// Neuron 82: 1871 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]
	logic [31:0] addr_82;
	assign addr_82 = {input_vec[14], input_vec[48], input_vec[291], input_vec[51], input_vec[167], input_vec[279], input_vec[132], input_vec[78], input_vec[257], input_vec[295], input_vec[115], input_vec[265], input_vec[145], input_vec[212], input_vec[83], input_vec[315], input_vec[259], input_vec[140], input_vec[126], input_vec[149], input_vec[49], input_vec[108], input_vec[160], input_vec[209], input_vec[167], input_vec[203], input_vec[279], input_vec[59], input_vec[218], input_vec[13], input_vec[22], input_vec[179]};

	// Neuron 83: 2860 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_83;
	assign addr_83 = {input_vec[222], input_vec[51], input_vec[107], input_vec[317], input_vec[50], input_vec[2], input_vec[111], input_vec[72], input_vec[236], input_vec[281], input_vec[294], input_vec[135], input_vec[145], input_vec[99], input_vec[203], input_vec[139], input_vec[156], input_vec[29], input_vec[74], input_vec[265], input_vec[132], input_vec[115], input_vec[166], input_vec[128], input_vec[152], input_vec[288], input_vec[208], input_vec[253], input_vec[35], input_vec[249], input_vec[91], input_vec[123]};

	// Neuron 84: 2121 entries, bits from features [0, 1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
	logic [31:0] addr_84;
	assign addr_84 = {input_vec[165], input_vec[269], input_vec[123], input_vec[251], input_vec[39], input_vec[249], input_vec[138], input_vec[205], input_vec[149], input_vec[19], input_vec[94], input_vec[114], input_vec[229], input_vec[6], input_vec[232], input_vec[19], input_vec[113], input_vec[216], input_vec[247], input_vec[281], input_vec[23], input_vec[174], input_vec[80], input_vec[291], input_vec[209], input_vec[192], input_vec[190], input_vec[225], input_vec[31], input_vec[6], input_vec[233], input_vec[256]};

	// Neuron 85: 2640 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
	logic [31:0] addr_85;
	assign addr_85 = {input_vec[82], input_vec[169], input_vec[244], input_vec[195], input_vec[167], input_vec[12], input_vec[301], input_vec[71], input_vec[190], input_vec[260], input_vec[203], input_vec[49], input_vec[82], input_vec[22], input_vec[102], input_vec[151], input_vec[32], input_vec[91], input_vec[129], input_vec[280], input_vec[248], input_vec[189], input_vec[83], input_vec[307], input_vec[194], input_vec[174], input_vec[17], input_vec[146], input_vec[114], input_vec[129], input_vec[210], input_vec[173]};

	// Neuron 86: 2139 entries, bits from features [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
	logic [31:0] addr_86;
	assign addr_86 = {input_vec[39], input_vec[216], input_vec[15], input_vec[318], input_vec[270], input_vec[308], input_vec[185], input_vec[152], input_vec[17], input_vec[24], input_vec[213], input_vec[54], input_vec[289], input_vec[167], input_vec[238], input_vec[150], input_vec[225], input_vec[17], input_vec[209], input_vec[79], input_vec[244], input_vec[296], input_vec[265], input_vec[158], input_vec[181], input_vec[231], input_vec[16], input_vec[175], input_vec[307], input_vec[166], input_vec[206], input_vec[219]};

	// Neuron 87: 2439 entries, bits from features [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 18, 19]
	logic [31:0] addr_87;
	assign addr_87 = {input_vec[177], input_vec[211], input_vec[29], input_vec[209], input_vec[121], input_vec[191], input_vec[56], input_vec[51], input_vec[92], input_vec[29], input_vec[275], input_vec[155], input_vec[61], input_vec[155], input_vec[86], input_vec[71], input_vec[315], input_vec[160], input_vec[110], input_vec[17], input_vec[85], input_vec[134], input_vec[160], input_vec[173], input_vec[98], input_vec[23], input_vec[182], input_vec[114], input_vec[13], input_vec[49], input_vec[267], input_vec[290]};

	// Neuron 88: 1799 entries, bits from features [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_88;
	assign addr_88 = {input_vec[50], input_vec[155], input_vec[107], input_vec[284], input_vec[11], input_vec[210], input_vec[33], input_vec[303], input_vec[281], input_vec[60], input_vec[306], input_vec[207], input_vec[287], input_vec[215], input_vec[311], input_vec[14], input_vec[71], input_vec[93], input_vec[161], input_vec[102], input_vec[69], input_vec[107], input_vec[270], input_vec[35], input_vec[155], input_vec[235], input_vec[149], input_vec[233], input_vec[240], input_vec[302], input_vec[182], input_vec[86]};

	// Neuron 89: 2732 entries, bits from features [0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_89;
	assign addr_89 = {input_vec[121], input_vec[246], input_vec[23], input_vec[166], input_vec[277], input_vec[312], input_vec[168], input_vec[255], input_vec[90], input_vec[130], input_vec[304], input_vec[226], input_vec[75], input_vec[212], input_vec[65], input_vec[21], input_vec[91], input_vec[304], input_vec[153], input_vec[162], input_vec[295], input_vec[15], input_vec[103], input_vec[92], input_vec[194], input_vec[291], input_vec[145], input_vec[259], input_vec[306], input_vec[298], input_vec[104], input_vec[278]};

	// Neuron 90: 2860 entries, bits from features [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
	logic [31:0] addr_90;
	assign addr_90 = {input_vec[222], input_vec[51], input_vec[107], input_vec[317], input_vec[50], input_vec[2], input_vec[111], input_vec[72], input_vec[236], input_vec[281], input_vec[294], input_vec[135], input_vec[145], input_vec[99], input_vec[203], input_vec[139], input_vec[156], input_vec[29], input_vec[74], input_vec[265], input_vec[132], input_vec[115], input_vec[166], input_vec[128], input_vec[152], input_vec[288], input_vec[208], input_vec[253], input_vec[35], input_vec[249], input_vec[91], input_vec[123]};

	// --- Neuron instances ---
	wnn_neuron #(
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(1952),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(1871),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(2955),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2087),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(4296),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(3420),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(3924),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(1903),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2860),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(1702),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(1785),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3268),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1697),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1697),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(3268),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1952),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(3420),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2581),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3924),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2181),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2640),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1644),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2860),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2181),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2348),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2348),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1871),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(1903),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(4296),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(13)
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
		.NUM_ENTRIES(1697),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2206),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2331),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2640),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2342),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2728),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1952),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
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
		.NUM_ENTRIES(2479),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2443),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3268),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2108),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3268),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1799),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2108),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2640),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2439),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2139),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3039),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(4296),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(13)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3670),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1868),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2342),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2263),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3924),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3715),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1644),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2206),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(3924),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2444),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1756),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2860),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2087),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2706),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2139),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2121),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2139),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2955),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2640),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2087),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2149),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1551),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(1799),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2342),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2686),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(4584),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(13)
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
		.NUM_ENTRIES(1871),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2860),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2121),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2640),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2139),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2439),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(1799),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(11)
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
		.NUM_ENTRIES(2732),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
		.NUM_ENTRIES(2860),
		.ADDR_BITS(32),
		.INPUT_BITS(320),
		.SEARCH_DEPTH(12)
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
