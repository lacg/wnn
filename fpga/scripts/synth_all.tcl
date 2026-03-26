# Vivado batch synthesis for all 6 WNN genomes
# Target: xc7z020clg400-1 (Zynq Z-7020)
# Run: vivado -mode batch -source synth_all.tcl

set base_dir /home/ubuntu/wnn_fpga
set rtl_dir ${base_dir}/rtl
set results_dir ${base_dir}/results

file mkdir ${results_dir}

# List of genomes to synthesize: {flow_id description}
set genomes {
    {973 "CICIDS_11n_best"}
    {936 "CICIDS_91n_mid"}
    {706 "CICIDS_500n_large"}
    {532 "UNSW_92n_small"}
    {601 "UNSW_369n_best"}
    {517 "UNSW_500n_large"}
}

foreach genome $genomes {
    set flow_id [lindex $genome 0]
    set desc [lindex $genome 1]
    set export_dir ${base_dir}/export/flow_${flow_id}
    set out_dir ${results_dir}/${desc}

    file mkdir ${out_dir}

    puts "============================================="
    puts "Synthesizing: ${desc} (flow ${flow_id})"
    puts "============================================="

    # Create project
    create_project -force synth_${flow_id} ${out_dir}/project -part xc7z020clg400-1

    # Add sources
    add_files ${rtl_dir}/wnn_neuron.sv
    add_files ${export_dir}/wnn_classifier_impl.sv

    # Set top
    set_property top wnn_classifier_impl [current_fileset]

    # Synthesize (out-of-context: no I/O placement constraints)
    synth_design -top wnn_classifier_impl -part xc7z020clg400-1 -mode out_of_context

    # Add clock constraint for timing and power analysis (200 MHz target)
    create_clock -period 5.0 -name clk [get_ports clk]

    # Reports
    report_utilization -file ${out_dir}/utilization.rpt
    report_timing_summary -file ${out_dir}/timing.rpt
    report_power -file ${out_dir}/power.rpt

    # Extract key metrics to a summary file
    set fp [open ${out_dir}/summary.txt w]
    puts $fp "Flow: ${flow_id}"
    puts $fp "Description: ${desc}"

    # Get utilization from report (post-synthesis)
    puts $fp "--- Post-Synthesis Utilization ---"

    # Get timing - WNS from timing report
    set timing_paths [get_timing_paths -max_paths 1 -quiet]
    if {[llength $timing_paths] > 0} {
        set wns [get_property SLACK [lindex $timing_paths 0]]
        set period 5.0
        set actual_period [expr {$period - $wns}]
        set fmax_mhz [expr {1000.0 / $actual_period}]
        puts $fp "WNS: ${wns} ns"
        puts $fp "Fmax: ${fmax_mhz} MHz"
    } else {
        puts $fp "WNS: N/A (no timing paths)"
        puts $fp "Fmax: N/A"
    }

    close $fp

    puts "Results written to ${out_dir}/"
    puts ""

    # Close project before next
    close_project
}

puts "===== ALL SYNTHESES COMPLETE ====="
