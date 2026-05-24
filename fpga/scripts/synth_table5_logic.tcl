# Logic-only synthesis for Table 5 genomes: isolates the on-chip binary-search
# FSMs (64-bit datapath) with memories capped+distributed so they fit Z-7020.
# Reports LUT/FF/LUTRAM/BRAM/Fmax/Power per genome. Run:
#   vivado -mode batch -source synth_table5_logic.tcl
set base_dir /home/ubuntu/wnn_fpga

set genomes {
    {flow_2693_logic "211n x 64b best_acc (logic-only)"}
    {flow_2470_logic "247n x 64b best_fpr (logic-only)"}
    {flow_2452_logic "245n x 67b best_ce  (logic-only)"}
}

foreach genome $genomes {
    set d [lindex $genome 0]
    set desc [lindex $genome 1]
    set ed ${base_dir}/export/${d}
    if {![file exists ${ed}/wnn_classifier_impl.sv]} {
        puts "SKIP ${d}: not found"; continue
    }
    puts "===== Synthesizing ${d} (${desc}) ====="
    create_project -force s_${d} ${ed}/proj -part xc7z020clg400-1
    add_files ${ed}/wnn_neuron.sv
    add_files ${ed}/wnn_classifier_impl.sv
    set_property file_type {Memory File} [add_files [glob ${ed}/mem/*.mem]]
    set_property top wnn_classifier_impl [current_fileset]
    synth_design -top wnn_classifier_impl -part xc7z020clg400-1 -mode out_of_context
    create_clock -period 5.0 -name clk [get_ports clk]

    report_utilization -file ${ed}/utilization.rpt
    report_timing_summary -file ${ed}/timing.rpt
    report_power -file ${ed}/power.rpt

    set luts   [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
    set ffs    [llength [get_cells -hier -filter {REF_NAME =~ FD*}]]
    set bram36 [llength [get_cells -hier -filter {REF_NAME =~ RAMB36*}]]
    set bram18 [llength [get_cells -hier -filter {REF_NAME =~ RAMB18*}]]

    set fp [open ${ed}/summary.txt w]
    puts $fp "Genome: ${d}"
    puts $fp "Desc: ${desc}"
    puts $fp "LUTs: ${luts}"
    puts $fp "FFs: ${ffs}"
    puts $fp "RAMB36: ${bram36}"
    puts $fp "RAMB18: ${bram18}"
    set tp [get_timing_paths -max_paths 1 -quiet]
    if {[llength $tp] > 0} {
        set wns [get_property SLACK [lindex $tp 0]]
        puts $fp "WNS_ns: ${wns}"
        puts $fp "Fmax_MHz: [expr {1000.0/(5.0-$wns)}]"
    }
    close $fp
    puts "DONE ${d}: LUTs=${luts} FFs=${ffs} RAMB18=${bram18} RAMB36=${bram36}"
    close_project
}
puts "===== TABLE 5 LOGIC-ONLY SYNTH COMPLETE ====="
