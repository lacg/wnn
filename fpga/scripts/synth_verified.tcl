# WNN synthesis that PROVES the model is in the design.
#
# WHY (31/08/2026): every prior run in fpga/results/ reports 0 BRAM and 0
# LUT-as-Memory, and the LUT counts cannot possibly hold the keys — the memory
# was being optimised away (conditional $readmemh + ram_style on a ROM, both
# fixed in rtl/wnn_neuron.sv). Synthesis SUCCEEDED and produced a number that
# did not contain the model. A green run is therefore NOT evidence; the only
# evidence is a nonzero memory count, so this script FAILS THE RUN when the
# memory is absent rather than writing a report nobody will re-check.
#
# Usage:
#   vivado -mode batch -source fpga/scripts/synth_verified.tcl \
#          -tclargs <export_dir> [part]
#
# Run fpga/scripts/fpga_fit_check.py FIRST — this script will not tell you that a
# design is 31x too large for the part, only that its memory reached the netlist.

if {$argc < 1} { puts "ERROR: usage: ... -tclargs <export_dir> \[part\]"; exit 1 }
set export_dir [lindex $argv 0]
set part       [expr {$argc > 1 ? [lindex $argv 1] : "xc7z020clg400-1"}]
set rtl_dir    [file normalize [file join [file dirname [info script]] .. rtl]]
set name       [file tail $export_dir]

puts "=== synth_verified: $name on $part ==="

create_project -in_memory -part $part

read_verilog -sv [file join $rtl_dir wnn_neuron.sv]
read_verilog -sv [file join $export_dir wnn_classifier_impl.sv]

# $readmemh resolves relative to the search path, NOT to the source file. Without
# this the reads silently find nothing and the memory disappears again — the very
# failure this script exists to catch.
set_property include_dirs [list [file join $export_dir mem] $export_dir] [current_fileset]
foreach f [glob -nocomplain [file join $export_dir mem *.mem]] { add_files $f }

synth_design -top wnn_classifier_impl -part $part -mode out_of_context

set rpt [file join $export_dir synth_verified]
file mkdir $rpt
report_utilization -file [file join $rpt utilization.rpt]
report_timing_summary -file [file join $rpt timing.rpt]

# ---- THE ASSERTION -------------------------------------------------------
# Count real memory primitives in the netlist. RAMB* = block RAM; RAMD/RAMS/SRL
# = LUT distributed memory. A design whose keys are present must show some.
set bram [llength [get_cells -hier -quiet -filter {PRIMITIVE_TYPE =~ BMEM.*.*}]]
set lutram [llength [get_cells -hier -quiet -filter {PRIMITIVE_TYPE =~ *.SRL.* || PRIMITIVE_TYPE =~ *.LUTRAM.*}]]
set luts [llength [get_cells -hier -quiet -filter {PRIMITIVE_TYPE =~ LUT.*}]]
puts "=== netlist: BRAM=$bram  LUTRAM/SRL=$lutram  LUT=$luts ==="

if {$bram == 0 && $lutram == 0} {
	puts "############################################################"
	puts "FAIL: the netlist contains NO memory primitives."
	puts "The keys were optimised away — this LUT count ($luts) does NOT"
	puts "contain the model and MUST NOT be reported as a footprint."
	puts "Check: \$readmemh paths resolve, rom_style attribute present,"
	puts "and that mem/*.mem were added to the fileset."
	puts "############################################################"
	exit 1
}
puts "PASS: model is in the netlist (BRAM=$bram, LUTRAM/SRL=$lutram)."
write_checkpoint -force [file join $rpt post_synth.dcp]
exit 0
