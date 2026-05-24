# Vivado batch synthesis for Table 5 (CIC-IoT-2023 1.14M cohort) genomes.
# The 3 distinct GA-converged Pareto genomes behind the 6 Table-5 rows:
#   flow_2693  211n x 64b  (best_acc) -> Best F1 / Best Acc rows
#   flow_2470  247n x 64b  (best_fpr) -> Best F1 (FPR<6%) row
#   flow_2452  245n x 67b  (best_ce)  -> Best F1 (FPR<5%/<4%) rows
# OOC synth on Z-7020; reports utilization/timing/power. No auto-shutdown
# (we pull results first). Run: vivado -mode batch -source synth_table5.tcl

set base_dir /home/ubuntu/wnn_fpga
set rtl_dir ${base_dir}/rtl

set genomes {
    {flow_2693 "211n x 64b best_acc"}
    {flow_2470 "247n x 64b best_fpr"}
    {flow_2452 "245n x 67b best_ce"}
}

foreach genome $genomes {
    set flow_dir [lindex $genome 0]
    set desc [lindex $genome 1]
    set export_dir ${base_dir}/export/${flow_dir}

    if {![file exists ${export_dir}/wnn_classifier_impl.sv]} {
        puts "SKIPPING ${flow_dir}: classifier RTL not found at ${export_dir}"
        continue
    }

    puts "============================================="
    puts "Synthesizing: ${flow_dir} (${desc})"
    puts "  started [clock format [clock seconds]]"
    puts "============================================="

    create_project -force synth_${flow_dir} ${export_dir}/vivado_project -part xc7z020clg400-1

    add_files ${rtl_dir}/wnn_neuron.sv
    add_files ${export_dir}/wnn_classifier_impl.sv
    if {[file exists ${export_dir}/mem]} {
        set_property file_type {Memory File} [add_files [glob ${export_dir}/mem/*.mem]]
    }

    set_property top wnn_classifier_impl [current_fileset]
    synth_design -top wnn_classifier_impl -part xc7z020clg400-1 -mode out_of_context
    create_clock -period 5.0 -name clk [get_ports clk]

    report_utilization -file ${export_dir}/utilization.rpt
    report_timing_summary -file ${export_dir}/timing.rpt
    report_power -file ${export_dir}/power.rpt

    # Pull the key numbers straight into a summary via design queries so we
    # don't have to parse the .rpt files later.
    set fp [open ${export_dir}/summary.txt w]
    puts $fp "Genome dir: ${flow_dir}"
    puts $fp "Description: ${desc}"

    set luts  [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
    set ffs   [llength [get_cells -hier -filter {REF_NAME =~ FD*}]]
    set bram36 [llength [get_cells -hier -filter {REF_NAME =~ RAMB36*}]]
    set bram18 [llength [get_cells -hier -filter {REF_NAME =~ RAMB18*}]]
    set lutram [llength [get_cells -hier -filter {REF_NAME =~ RAM*  && PRIMITIVE_GROUP == LUTRAM}]]
    puts $fp "LUTs: ${luts}"
    puts $fp "FFs: ${ffs}"
    puts $fp "RAMB36: ${bram36}"
    puts $fp "RAMB18: ${bram18}"

    set timing_paths [get_timing_paths -max_paths 1 -quiet]
    if {[llength $timing_paths] > 0} {
        set wns [get_property SLACK [lindex $timing_paths 0]]
        set fmax_mhz [expr {1000.0 / (5.0 - $wns)}]
        puts $fp "WNS: ${wns} ns"
        puts $fp "Fmax_MHz: ${fmax_mhz}"
    } else {
        puts $fp "WNS: N/A"
        puts $fp "Fmax_MHz: N/A"
    }
    close $fp

    puts "DONE ${flow_dir}: LUTs=${luts} RAMB36=${bram36} RAMB18=${bram18}"
    puts "  finished [clock format [clock seconds]]"
    close_project
}

puts "===== TABLE 5 SYNTHESIS COMPLETE ====="
