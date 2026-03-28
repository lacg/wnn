// Auto-generated dense WNN classifier (O(1) lookup)
// Genome: 98ff5233e9aa9ffe from flow 1024
// 83 neurons × 12-bit addresses (dense)
// Dataset: unsw-nb15 (random split)
// Thermometer: 16-bit (176 input bits)

module wnn_classifier_dense #(
	parameter int THRESHOLD = 0
) (
	input  logic                     clk,
	input  logic                     rst_n,
	input  logic                     input_valid,
	input  logic [175:0]  input_vec,

	output logic                     class_out,
	output logic [8-1:0]    score_out,
	output logic                     output_valid,
	output logic                     busy
);

	localparam int NUM_NEURONS = 83;
	localparam int ADDR_BITS   = 12;
	localparam int INPUT_BITS  = 176;
	localparam int ACC_BITS    = 8;

	logic [7:0]  neuron_result [83];
	logic [83-1:0] neuron_valid;
	logic neuron_start;
	assign neuron_start = input_valid;
	assign busy = 1'b0;  // Dense: always ready

	// --- Per-neuron address formation (evolved connections) ---
	logic [11:0] addr_0;
	assign addr_0 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_1;
	assign addr_1 = {input_vec[89], input_vec[166], input_vec[106], input_vec[108], input_vec[65], input_vec[112], input_vec[143], input_vec[10], input_vec[15], input_vec[19], input_vec[126], input_vec[76]};
	logic [11:0] addr_2;
	assign addr_2 = {input_vec[93], input_vec[21], input_vec[120], input_vec[2], input_vec[41], input_vec[168], input_vec[73], input_vec[12], input_vec[102], input_vec[135], input_vec[171], input_vec[34]};
	logic [11:0] addr_3;
	assign addr_3 = {input_vec[24], input_vec[100], input_vec[106], input_vec[60], input_vec[145], input_vec[169], input_vec[120], input_vec[101], input_vec[30], input_vec[60], input_vec[104], input_vec[45]};
	logic [11:0] addr_4;
	assign addr_4 = {input_vec[108], input_vec[174], input_vec[23], input_vec[74], input_vec[13], input_vec[156], input_vec[47], input_vec[7], input_vec[92], input_vec[86], input_vec[113], input_vec[46]};
	logic [11:0] addr_5;
	assign addr_5 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_6;
	assign addr_6 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_7;
	assign addr_7 = {input_vec[172], input_vec[58], input_vec[143], input_vec[131], input_vec[112], input_vec[28], input_vec[1], input_vec[102], input_vec[22], input_vec[107], input_vec[110], input_vec[50]};
	logic [11:0] addr_8;
	assign addr_8 = {input_vec[132], input_vec[60], input_vec[79], input_vec[11], input_vec[22], input_vec[69], input_vec[1], input_vec[141], input_vec[47], input_vec[15], input_vec[97], input_vec[109]};
	logic [11:0] addr_9;
	assign addr_9 = {input_vec[18], input_vec[169], input_vec[110], input_vec[41], input_vec[142], input_vec[119], input_vec[93], input_vec[100], input_vec[85], input_vec[124], input_vec[1], input_vec[53]};
	logic [11:0] addr_10;
	assign addr_10 = {input_vec[1], input_vec[19], input_vec[120], input_vec[90], input_vec[34], input_vec[141], input_vec[161], input_vec[10], input_vec[158], input_vec[64], input_vec[127], input_vec[11]};
	logic [11:0] addr_11;
	assign addr_11 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_12;
	assign addr_12 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_13;
	assign addr_13 = {input_vec[1], input_vec[19], input_vec[120], input_vec[90], input_vec[34], input_vec[141], input_vec[161], input_vec[10], input_vec[158], input_vec[64], input_vec[127], input_vec[11]};
	logic [11:0] addr_14;
	assign addr_14 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_15;
	assign addr_15 = {input_vec[22], input_vec[67], input_vec[0], input_vec[84], input_vec[50], input_vec[82], input_vec[6], input_vec[146], input_vec[163], input_vec[163], input_vec[115], input_vec[129]};
	logic [11:0] addr_16;
	assign addr_16 = {input_vec[21], input_vec[144], input_vec[137], input_vec[140], input_vec[137], input_vec[34], input_vec[175], input_vec[130], input_vec[51], input_vec[119], input_vec[41], input_vec[36]};
	logic [11:0] addr_17;
	assign addr_17 = {input_vec[132], input_vec[60], input_vec[79], input_vec[11], input_vec[22], input_vec[69], input_vec[1], input_vec[141], input_vec[47], input_vec[15], input_vec[97], input_vec[109]};
	logic [11:0] addr_18;
	assign addr_18 = {input_vec[86], input_vec[70], input_vec[112], input_vec[15], input_vec[72], input_vec[67], input_vec[2], input_vec[32], input_vec[95], input_vec[51], input_vec[164], input_vec[136]};
	logic [11:0] addr_19;
	assign addr_19 = {input_vec[169], input_vec[4], input_vec[73], input_vec[77], input_vec[100], input_vec[166], input_vec[41], input_vec[47], input_vec[56], input_vec[87], input_vec[129], input_vec[67]};
	logic [11:0] addr_20;
	assign addr_20 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_21;
	assign addr_21 = {input_vec[22], input_vec[67], input_vec[0], input_vec[84], input_vec[50], input_vec[82], input_vec[6], input_vec[146], input_vec[163], input_vec[163], input_vec[115], input_vec[129]};
	logic [11:0] addr_22;
	assign addr_22 = {input_vec[46], input_vec[2], input_vec[34], input_vec[59], input_vec[85], input_vec[153], input_vec[18], input_vec[59], input_vec[173], input_vec[74], input_vec[115], input_vec[113]};
	logic [11:0] addr_23;
	assign addr_23 = {input_vec[21], input_vec[144], input_vec[137], input_vec[140], input_vec[137], input_vec[34], input_vec[175], input_vec[130], input_vec[51], input_vec[119], input_vec[41], input_vec[36]};
	logic [11:0] addr_24;
	assign addr_24 = {input_vec[108], input_vec[174], input_vec[23], input_vec[74], input_vec[13], input_vec[156], input_vec[47], input_vec[7], input_vec[92], input_vec[86], input_vec[113], input_vec[46]};
	logic [11:0] addr_25;
	assign addr_25 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_26;
	assign addr_26 = {input_vec[12], input_vec[57], input_vec[151], input_vec[66], input_vec[18], input_vec[158], input_vec[79], input_vec[132], input_vec[34], input_vec[15], input_vec[36], input_vec[96]};
	logic [11:0] addr_27;
	assign addr_27 = {input_vec[122], input_vec[110], input_vec[41], input_vec[69], input_vec[137], input_vec[99], input_vec[143], input_vec[152], input_vec[140], input_vec[149], input_vec[144], input_vec[46]};
	logic [11:0] addr_28;
	assign addr_28 = {input_vec[94], input_vec[0], input_vec[3], input_vec[149], input_vec[85], input_vec[57], input_vec[37], input_vec[34], input_vec[46], input_vec[170], input_vec[121], input_vec[131]};
	logic [11:0] addr_29;
	assign addr_29 = {input_vec[44], input_vec[97], input_vec[10], input_vec[30], input_vec[95], input_vec[66], input_vec[121], input_vec[34], input_vec[26], input_vec[88], input_vec[58], input_vec[168]};
	logic [11:0] addr_30;
	assign addr_30 = {input_vec[24], input_vec[100], input_vec[106], input_vec[60], input_vec[145], input_vec[169], input_vec[120], input_vec[101], input_vec[30], input_vec[60], input_vec[104], input_vec[45]};
	logic [11:0] addr_31;
	assign addr_31 = {input_vec[174], input_vec[127], input_vec[47], input_vec[122], input_vec[129], input_vec[57], input_vec[47], input_vec[42], input_vec[54], input_vec[70], input_vec[166], input_vec[155]};
	logic [11:0] addr_32;
	assign addr_32 = {input_vec[108], input_vec[174], input_vec[23], input_vec[74], input_vec[13], input_vec[156], input_vec[47], input_vec[7], input_vec[92], input_vec[86], input_vec[113], input_vec[46]};
	logic [11:0] addr_33;
	assign addr_33 = {input_vec[108], input_vec[174], input_vec[23], input_vec[74], input_vec[13], input_vec[156], input_vec[47], input_vec[7], input_vec[92], input_vec[86], input_vec[113], input_vec[46]};
	logic [11:0] addr_34;
	assign addr_34 = {input_vec[108], input_vec[26], input_vec[124], input_vec[113], input_vec[136], input_vec[125], input_vec[143], input_vec[36], input_vec[1], input_vec[82], input_vec[127], input_vec[93]};
	logic [11:0] addr_35;
	assign addr_35 = {input_vec[18], input_vec[169], input_vec[110], input_vec[41], input_vec[142], input_vec[119], input_vec[93], input_vec[100], input_vec[85], input_vec[124], input_vec[1], input_vec[53]};
	logic [11:0] addr_36;
	assign addr_36 = {input_vec[64], input_vec[151], input_vec[82], input_vec[114], input_vec[157], input_vec[117], input_vec[45], input_vec[118], input_vec[41], input_vec[38], input_vec[2], input_vec[49]};
	logic [11:0] addr_37;
	assign addr_37 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_38;
	assign addr_38 = {input_vec[18], input_vec[169], input_vec[110], input_vec[41], input_vec[142], input_vec[119], input_vec[93], input_vec[100], input_vec[85], input_vec[124], input_vec[1], input_vec[53]};
	logic [11:0] addr_39;
	assign addr_39 = {input_vec[161], input_vec[131], input_vec[121], input_vec[3], input_vec[144], input_vec[101], input_vec[171], input_vec[153], input_vec[23], input_vec[160], input_vec[25], input_vec[79]};
	logic [11:0] addr_40;
	assign addr_40 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_41;
	assign addr_41 = {input_vec[12], input_vec[57], input_vec[151], input_vec[66], input_vec[18], input_vec[158], input_vec[79], input_vec[132], input_vec[34], input_vec[15], input_vec[36], input_vec[96]};
	logic [11:0] addr_42;
	assign addr_42 = {input_vec[31], input_vec[154], input_vec[168], input_vec[146], input_vec[152], input_vec[102], input_vec[23], input_vec[21], input_vec[67], input_vec[87], input_vec[21], input_vec[49]};
	logic [11:0] addr_43;
	assign addr_43 = {input_vec[23], input_vec[28], input_vec[32], input_vec[53], input_vec[67], input_vec[64], input_vec[101], input_vec[84], input_vec[138], input_vec[131], input_vec[117], input_vec[26]};
	logic [11:0] addr_44;
	assign addr_44 = {input_vec[127], input_vec[143], input_vec[11], input_vec[54], input_vec[10], input_vec[113], input_vec[155], input_vec[126], input_vec[22], input_vec[53], input_vec[116], input_vec[18]};
	logic [11:0] addr_45;
	assign addr_45 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_46;
	assign addr_46 = {input_vec[93], input_vec[21], input_vec[120], input_vec[2], input_vec[41], input_vec[168], input_vec[73], input_vec[12], input_vec[102], input_vec[135], input_vec[171], input_vec[34]};
	logic [11:0] addr_47;
	assign addr_47 = {input_vec[108], input_vec[26], input_vec[124], input_vec[113], input_vec[136], input_vec[125], input_vec[143], input_vec[36], input_vec[1], input_vec[82], input_vec[127], input_vec[93]};
	logic [11:0] addr_48;
	assign addr_48 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_49;
	assign addr_49 = {input_vec[24], input_vec[100], input_vec[106], input_vec[60], input_vec[145], input_vec[169], input_vec[120], input_vec[101], input_vec[30], input_vec[60], input_vec[104], input_vec[45]};
	logic [11:0] addr_50;
	assign addr_50 = {input_vec[108], input_vec[26], input_vec[124], input_vec[113], input_vec[136], input_vec[125], input_vec[143], input_vec[36], input_vec[1], input_vec[82], input_vec[127], input_vec[93]};
	logic [11:0] addr_51;
	assign addr_51 = {input_vec[122], input_vec[133], input_vec[156], input_vec[145], input_vec[130], input_vec[169], input_vec[14], input_vec[118], input_vec[168], input_vec[119], input_vec[18], input_vec[15]};
	logic [11:0] addr_52;
	assign addr_52 = {input_vec[18], input_vec[169], input_vec[110], input_vec[41], input_vec[142], input_vec[119], input_vec[93], input_vec[100], input_vec[85], input_vec[124], input_vec[1], input_vec[53]};
	logic [11:0] addr_53;
	assign addr_53 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_54;
	assign addr_54 = {input_vec[118], input_vec[80], input_vec[137], input_vec[59], input_vec[43], input_vec[7], input_vec[70], input_vec[21], input_vec[18], input_vec[163], input_vec[107], input_vec[5]};
	logic [11:0] addr_55;
	assign addr_55 = {input_vec[46], input_vec[2], input_vec[34], input_vec[59], input_vec[85], input_vec[153], input_vec[18], input_vec[59], input_vec[173], input_vec[74], input_vec[115], input_vec[113]};
	logic [11:0] addr_56;
	assign addr_56 = {input_vec[112], input_vec[131], input_vec[3], input_vec[100], input_vec[152], input_vec[54], input_vec[114], input_vec[137], input_vec[110], input_vec[165], input_vec[113], input_vec[162]};
	logic [11:0] addr_57;
	assign addr_57 = {input_vec[122], input_vec[110], input_vec[41], input_vec[69], input_vec[137], input_vec[99], input_vec[143], input_vec[152], input_vec[140], input_vec[149], input_vec[144], input_vec[46]};
	logic [11:0] addr_58;
	assign addr_58 = {input_vec[108], input_vec[26], input_vec[124], input_vec[113], input_vec[136], input_vec[125], input_vec[143], input_vec[36], input_vec[1], input_vec[82], input_vec[127], input_vec[93]};
	logic [11:0] addr_59;
	assign addr_59 = {input_vec[169], input_vec[4], input_vec[73], input_vec[77], input_vec[100], input_vec[166], input_vec[41], input_vec[47], input_vec[56], input_vec[87], input_vec[129], input_vec[67]};
	logic [11:0] addr_60;
	assign addr_60 = {input_vec[122], input_vec[110], input_vec[41], input_vec[69], input_vec[137], input_vec[99], input_vec[143], input_vec[152], input_vec[140], input_vec[149], input_vec[144], input_vec[46]};
	logic [11:0] addr_61;
	assign addr_61 = {input_vec[44], input_vec[119], input_vec[80], input_vec[18], input_vec[147], input_vec[57], input_vec[20], input_vec[164], input_vec[1], input_vec[74], input_vec[142], input_vec[86]};
	logic [11:0] addr_62;
	assign addr_62 = {input_vec[94], input_vec[13], input_vec[137], input_vec[1], input_vec[102], input_vec[28], input_vec[148], input_vec[23], input_vec[32], input_vec[113], input_vec[121], input_vec[0]};
	logic [11:0] addr_63;
	assign addr_63 = {input_vec[122], input_vec[133], input_vec[156], input_vec[145], input_vec[130], input_vec[169], input_vec[14], input_vec[118], input_vec[168], input_vec[119], input_vec[18], input_vec[15]};
	logic [11:0] addr_64;
	assign addr_64 = {input_vec[23], input_vec[28], input_vec[32], input_vec[53], input_vec[67], input_vec[64], input_vec[101], input_vec[84], input_vec[138], input_vec[131], input_vec[117], input_vec[26]};
	logic [11:0] addr_65;
	assign addr_65 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_66;
	assign addr_66 = {input_vec[108], input_vec[26], input_vec[124], input_vec[113], input_vec[136], input_vec[125], input_vec[143], input_vec[36], input_vec[1], input_vec[82], input_vec[127], input_vec[93]};
	logic [11:0] addr_67;
	assign addr_67 = {input_vec[47], input_vec[131], input_vec[151], input_vec[126], input_vec[111], input_vec[30], input_vec[98], input_vec[173], input_vec[73], input_vec[99], input_vec[84], input_vec[3]};
	logic [11:0] addr_68;
	assign addr_68 = {input_vec[166], input_vec[47], input_vec[136], input_vec[58], input_vec[72], input_vec[128], input_vec[68], input_vec[13], input_vec[57], input_vec[77], input_vec[172], input_vec[146]};
	logic [11:0] addr_69;
	assign addr_69 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_70;
	assign addr_70 = {input_vec[172], input_vec[58], input_vec[143], input_vec[131], input_vec[112], input_vec[28], input_vec[1], input_vec[102], input_vec[22], input_vec[107], input_vec[110], input_vec[50]};
	logic [11:0] addr_71;
	assign addr_71 = {input_vec[153], input_vec[147], input_vec[166], input_vec[52], input_vec[121], input_vec[22], input_vec[109], input_vec[158], input_vec[41], input_vec[141], input_vec[96], input_vec[77]};
	logic [11:0] addr_72;
	assign addr_72 = {input_vec[166], input_vec[47], input_vec[136], input_vec[58], input_vec[72], input_vec[128], input_vec[68], input_vec[13], input_vec[57], input_vec[77], input_vec[172], input_vec[146]};
	logic [11:0] addr_73;
	assign addr_73 = {input_vec[132], input_vec[60], input_vec[79], input_vec[11], input_vec[22], input_vec[69], input_vec[1], input_vec[141], input_vec[47], input_vec[15], input_vec[97], input_vec[109]};
	logic [11:0] addr_74;
	assign addr_74 = {input_vec[93], input_vec[21], input_vec[120], input_vec[2], input_vec[41], input_vec[168], input_vec[73], input_vec[12], input_vec[102], input_vec[135], input_vec[171], input_vec[34]};
	logic [11:0] addr_75;
	assign addr_75 = {input_vec[132], input_vec[60], input_vec[79], input_vec[11], input_vec[22], input_vec[69], input_vec[1], input_vec[141], input_vec[47], input_vec[15], input_vec[97], input_vec[109]};
	logic [11:0] addr_76;
	assign addr_76 = {input_vec[18], input_vec[169], input_vec[110], input_vec[41], input_vec[142], input_vec[119], input_vec[93], input_vec[100], input_vec[85], input_vec[124], input_vec[1], input_vec[53]};
	logic [11:0] addr_77;
	assign addr_77 = {input_vec[118], input_vec[80], input_vec[137], input_vec[59], input_vec[43], input_vec[7], input_vec[70], input_vec[21], input_vec[18], input_vec[163], input_vec[107], input_vec[5]};
	logic [11:0] addr_78;
	assign addr_78 = {input_vec[22], input_vec[67], input_vec[0], input_vec[84], input_vec[50], input_vec[82], input_vec[6], input_vec[146], input_vec[163], input_vec[163], input_vec[115], input_vec[129]};
	logic [11:0] addr_79;
	assign addr_79 = {input_vec[18], input_vec[169], input_vec[110], input_vec[41], input_vec[142], input_vec[119], input_vec[93], input_vec[100], input_vec[85], input_vec[124], input_vec[1], input_vec[53]};
	logic [11:0] addr_80;
	assign addr_80 = {input_vec[18], input_vec[169], input_vec[110], input_vec[41], input_vec[142], input_vec[119], input_vec[93], input_vec[100], input_vec[85], input_vec[124], input_vec[1], input_vec[53]};
	logic [11:0] addr_81;
	assign addr_81 = {input_vec[98], input_vec[40], input_vec[146], input_vec[48], input_vec[121], input_vec[18], input_vec[33], input_vec[116], input_vec[157], input_vec[6], input_vec[154], input_vec[22]};
	logic [11:0] addr_82;
	assign addr_82 = {input_vec[24], input_vec[133], input_vec[134], input_vec[93], input_vec[103], input_vec[40], input_vec[163], input_vec[139], input_vec[173], input_vec[100], input_vec[45], input_vec[89]};

	// --- Dense neuron instances (O(1) lookup) ---
	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_0 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_0),
		.result(neuron_result[0]),
		.result_valid(neuron_valid[0]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_000.mem", neuron_0.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_1 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_1),
		.result(neuron_result[1]),
		.result_valid(neuron_valid[1]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_001.mem", neuron_1.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_2 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_2),
		.result(neuron_result[2]),
		.result_valid(neuron_valid[2]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_002.mem", neuron_2.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_3 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_3),
		.result(neuron_result[3]),
		.result_valid(neuron_valid[3]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_003.mem", neuron_3.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_4 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_4),
		.result(neuron_result[4]),
		.result_valid(neuron_valid[4]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_004.mem", neuron_4.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_5 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_5),
		.result(neuron_result[5]),
		.result_valid(neuron_valid[5]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_005.mem", neuron_5.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_6 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_6),
		.result(neuron_result[6]),
		.result_valid(neuron_valid[6]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_006.mem", neuron_6.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_7 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_7),
		.result(neuron_result[7]),
		.result_valid(neuron_valid[7]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_007.mem", neuron_7.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_8 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_8),
		.result(neuron_result[8]),
		.result_valid(neuron_valid[8]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_008.mem", neuron_8.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_9 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_9),
		.result(neuron_result[9]),
		.result_valid(neuron_valid[9]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_009.mem", neuron_9.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_10 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_10),
		.result(neuron_result[10]),
		.result_valid(neuron_valid[10]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_010.mem", neuron_10.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_11 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_11),
		.result(neuron_result[11]),
		.result_valid(neuron_valid[11]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_011.mem", neuron_11.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_12 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_12),
		.result(neuron_result[12]),
		.result_valid(neuron_valid[12]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_012.mem", neuron_12.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_13 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_13),
		.result(neuron_result[13]),
		.result_valid(neuron_valid[13]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_013.mem", neuron_13.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_14 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_14),
		.result(neuron_result[14]),
		.result_valid(neuron_valid[14]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_014.mem", neuron_14.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_15 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_15),
		.result(neuron_result[15]),
		.result_valid(neuron_valid[15]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_015.mem", neuron_15.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_16 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_16),
		.result(neuron_result[16]),
		.result_valid(neuron_valid[16]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_016.mem", neuron_16.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_17 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_17),
		.result(neuron_result[17]),
		.result_valid(neuron_valid[17]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_017.mem", neuron_17.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_18 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_18),
		.result(neuron_result[18]),
		.result_valid(neuron_valid[18]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_018.mem", neuron_18.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_19 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_19),
		.result(neuron_result[19]),
		.result_valid(neuron_valid[19]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_019.mem", neuron_19.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_20 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_20),
		.result(neuron_result[20]),
		.result_valid(neuron_valid[20]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_020.mem", neuron_20.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_21 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_21),
		.result(neuron_result[21]),
		.result_valid(neuron_valid[21]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_021.mem", neuron_21.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_22 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_22),
		.result(neuron_result[22]),
		.result_valid(neuron_valid[22]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_022.mem", neuron_22.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_23 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_23),
		.result(neuron_result[23]),
		.result_valid(neuron_valid[23]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_023.mem", neuron_23.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_24 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_24),
		.result(neuron_result[24]),
		.result_valid(neuron_valid[24]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_024.mem", neuron_24.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_25 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_25),
		.result(neuron_result[25]),
		.result_valid(neuron_valid[25]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_025.mem", neuron_25.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_26 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_26),
		.result(neuron_result[26]),
		.result_valid(neuron_valid[26]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_026.mem", neuron_26.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_27 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_27),
		.result(neuron_result[27]),
		.result_valid(neuron_valid[27]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_027.mem", neuron_27.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_28 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_28),
		.result(neuron_result[28]),
		.result_valid(neuron_valid[28]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_028.mem", neuron_28.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_29 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_29),
		.result(neuron_result[29]),
		.result_valid(neuron_valid[29]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_029.mem", neuron_29.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_30 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_30),
		.result(neuron_result[30]),
		.result_valid(neuron_valid[30]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_030.mem", neuron_30.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_31 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_31),
		.result(neuron_result[31]),
		.result_valid(neuron_valid[31]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_031.mem", neuron_31.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_32 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_32),
		.result(neuron_result[32]),
		.result_valid(neuron_valid[32]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_032.mem", neuron_32.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_33 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_33),
		.result(neuron_result[33]),
		.result_valid(neuron_valid[33]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_033.mem", neuron_33.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_34 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_34),
		.result(neuron_result[34]),
		.result_valid(neuron_valid[34]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_034.mem", neuron_34.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_35 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_35),
		.result(neuron_result[35]),
		.result_valid(neuron_valid[35]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_035.mem", neuron_35.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_36 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_36),
		.result(neuron_result[36]),
		.result_valid(neuron_valid[36]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_036.mem", neuron_36.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_37 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_37),
		.result(neuron_result[37]),
		.result_valid(neuron_valid[37]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_037.mem", neuron_37.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_38 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_38),
		.result(neuron_result[38]),
		.result_valid(neuron_valid[38]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_038.mem", neuron_38.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_39 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_39),
		.result(neuron_result[39]),
		.result_valid(neuron_valid[39]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_039.mem", neuron_39.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_40 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_40),
		.result(neuron_result[40]),
		.result_valid(neuron_valid[40]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_040.mem", neuron_40.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_41 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_41),
		.result(neuron_result[41]),
		.result_valid(neuron_valid[41]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_041.mem", neuron_41.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_42 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_42),
		.result(neuron_result[42]),
		.result_valid(neuron_valid[42]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_042.mem", neuron_42.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_43 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_43),
		.result(neuron_result[43]),
		.result_valid(neuron_valid[43]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_043.mem", neuron_43.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_44 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_44),
		.result(neuron_result[44]),
		.result_valid(neuron_valid[44]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_044.mem", neuron_44.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_45 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_45),
		.result(neuron_result[45]),
		.result_valid(neuron_valid[45]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_045.mem", neuron_45.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_46 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_46),
		.result(neuron_result[46]),
		.result_valid(neuron_valid[46]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_046.mem", neuron_46.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_47 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_47),
		.result(neuron_result[47]),
		.result_valid(neuron_valid[47]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_047.mem", neuron_47.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_48 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_48),
		.result(neuron_result[48]),
		.result_valid(neuron_valid[48]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_048.mem", neuron_48.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_49 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_49),
		.result(neuron_result[49]),
		.result_valid(neuron_valid[49]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_049.mem", neuron_49.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_50 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_50),
		.result(neuron_result[50]),
		.result_valid(neuron_valid[50]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_050.mem", neuron_50.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_51 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_51),
		.result(neuron_result[51]),
		.result_valid(neuron_valid[51]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_051.mem", neuron_51.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_52 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_52),
		.result(neuron_result[52]),
		.result_valid(neuron_valid[52]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_052.mem", neuron_52.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_53 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_53),
		.result(neuron_result[53]),
		.result_valid(neuron_valid[53]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_053.mem", neuron_53.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_54 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_54),
		.result(neuron_result[54]),
		.result_valid(neuron_valid[54]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_054.mem", neuron_54.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_55 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_55),
		.result(neuron_result[55]),
		.result_valid(neuron_valid[55]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_055.mem", neuron_55.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_56 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_56),
		.result(neuron_result[56]),
		.result_valid(neuron_valid[56]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_056.mem", neuron_56.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_57 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_57),
		.result(neuron_result[57]),
		.result_valid(neuron_valid[57]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_057.mem", neuron_57.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_58 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_58),
		.result(neuron_result[58]),
		.result_valid(neuron_valid[58]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_058.mem", neuron_58.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_59 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_59),
		.result(neuron_result[59]),
		.result_valid(neuron_valid[59]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_059.mem", neuron_59.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_60 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_60),
		.result(neuron_result[60]),
		.result_valid(neuron_valid[60]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_060.mem", neuron_60.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_61 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_61),
		.result(neuron_result[61]),
		.result_valid(neuron_valid[61]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_061.mem", neuron_61.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_62 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_62),
		.result(neuron_result[62]),
		.result_valid(neuron_valid[62]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_062.mem", neuron_62.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_63 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_63),
		.result(neuron_result[63]),
		.result_valid(neuron_valid[63]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_063.mem", neuron_63.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_64 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_64),
		.result(neuron_result[64]),
		.result_valid(neuron_valid[64]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_064.mem", neuron_64.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_65 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_65),
		.result(neuron_result[65]),
		.result_valid(neuron_valid[65]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_065.mem", neuron_65.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_66 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_66),
		.result(neuron_result[66]),
		.result_valid(neuron_valid[66]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_066.mem", neuron_66.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_67 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_67),
		.result(neuron_result[67]),
		.result_valid(neuron_valid[67]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_067.mem", neuron_67.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_68 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_68),
		.result(neuron_result[68]),
		.result_valid(neuron_valid[68]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_068.mem", neuron_68.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_69 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_69),
		.result(neuron_result[69]),
		.result_valid(neuron_valid[69]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_069.mem", neuron_69.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_70 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_70),
		.result(neuron_result[70]),
		.result_valid(neuron_valid[70]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_070.mem", neuron_70.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_71 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_71),
		.result(neuron_result[71]),
		.result_valid(neuron_valid[71]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_071.mem", neuron_71.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_72 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_72),
		.result(neuron_result[72]),
		.result_valid(neuron_valid[72]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_072.mem", neuron_72.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_73 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_73),
		.result(neuron_result[73]),
		.result_valid(neuron_valid[73]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_073.mem", neuron_73.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_74 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_74),
		.result(neuron_result[74]),
		.result_valid(neuron_valid[74]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_074.mem", neuron_74.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_75 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_75),
		.result(neuron_result[75]),
		.result_valid(neuron_valid[75]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_075.mem", neuron_75.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_76 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_76),
		.result(neuron_result[76]),
		.result_valid(neuron_valid[76]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_076.mem", neuron_76.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_77 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_77),
		.result(neuron_result[77]),
		.result_valid(neuron_valid[77]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_077.mem", neuron_77.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_78 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_78),
		.result(neuron_result[78]),
		.result_valid(neuron_valid[78]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_078.mem", neuron_78.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_79 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_79),
		.result(neuron_result[79]),
		.result_valid(neuron_valid[79]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_079.mem", neuron_79.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_80 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_80),
		.result(neuron_result[80]),
		.result_valid(neuron_valid[80]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_080.mem", neuron_80.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_81 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_81),
		.result(neuron_result[81]),
		.result_valid(neuron_valid[81]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_081.mem", neuron_81.mem);

	wnn_neuron_dense #(
		.ADDR_BITS(12),
		.INPUT_BITS(176)
	) neuron_82 (
		.clk(clk),
		.rst_n(rst_n),
		.start(neuron_start),
		.input_vec(input_vec),
		.address(addr_82),
		.result(neuron_result[82]),
		.result_valid(neuron_valid[82]),
		.busy()
	);
	// Init: $readmemh("mem/neuron_082.mem", neuron_82.mem);

	// --- Weighted accumulation (all neurons valid same cycle) ---
	logic all_valid;
	assign all_valid = neuron_valid[0];  // All fire on same cycle

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
