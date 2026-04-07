# Vivado batch synthesis for WNN genomes
# Target: xc7z020clg400-1 (Zynq Z-7020)
# Run: vivado -mode batch -source synth_all.tcl

set base_dir /home/ubuntu/wnn_fpga
set rtl_dir ${base_dir}/rtl
set results_dir ${base_dir}/results

file mkdir ${results_dir}

# List of genomes to synthesize: {flow_dir description mode top_module}
# mode: "sparse" (binary search, wnn_neuron.sv + wnn_classifier_impl.sv)
#       "dense"  (direct lookup, wnn_neuron_dense.sv + wnn_classifier_dense.sv)
set genomes {
    {flow_973           "CICIDS_11n_best"          sparse wnn_classifier_impl}
    {flow_936           "CICIDS_91n_mid"           sparse wnn_classifier_impl}
    {flow_706           "CICIDS_500n_large"        sparse wnn_classifier_impl}
    {flow_532           "UNSW_92n_small"           sparse wnn_classifier_impl}
    {flow_601           "UNSW_369n_best"           sparse wnn_classifier_impl}
    {flow_517           "UNSW_500n_large"          sparse wnn_classifier_impl}
    {flow_798_ciciot_200n4b "CICIOT_200n_4b_dense" dense  wnn_classifier_dense}
}

foreach genome $genomes {
    set flow_dir [lindex $genome 0]
    set desc [lindex $genome 1]
    set mode [lindex $genome 2]
    set top_module [lindex $genome 3]
    set export_dir ${base_dir}/export/${flow_dir}
    set out_dir ${results_dir}/${desc}

    file mkdir ${out_dir}

    puts "============================================="
    puts "Synthesizing: ${desc} (${flow_dir}, ${mode})"
    puts "============================================="

    # Create project
    create_project -force synth_${desc} ${out_dir}/project -part xc7z020clg400-1

    # Add sources based on mode
    if {${mode} eq "sparse"} {
        add_files ${rtl_dir}/wnn_neuron.sv
        add_files ${export_dir}/wnn_classifier_impl.sv
    } else {
        add_files ${rtl_dir}/wnn_neuron_dense.sv
        add_files ${export_dir}/wnn_classifier_dense.sv
        # Dense .mem files need to be on the search path for $readmemh
        set_property file_type {Memory File} [add_files [glob ${export_dir}/mem/*.mem]]
    }

    # Set top
    set_property top ${top_module} [current_fileset]

    # Synthesize (out-of-context: no I/O placement constraints)
    synth_design -top ${top_module} -part xc7z020clg400-1 -mode out_of_context

    # Add clock constraint for timing and power analysis (200 MHz target)
    create_clock -period 5.0 -name clk [get_ports clk]

    # Reports
    report_utilization -file ${out_dir}/utilization.rpt
    report_timing_summary -file ${out_dir}/timing.rpt
    report_power -file ${out_dir}/power.rpt

    # Extract key metrics to a summary file
    set fp [open ${out_dir}/summary.txt w]
    puts $fp "Flow: ${flow_dir}"
    puts $fp "Description: ${desc}"
    puts $fp "Mode: ${mode}"

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
