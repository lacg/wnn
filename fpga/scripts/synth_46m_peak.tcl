# Vivado batch synthesis for 46M Peak tier genomes
# Target: xc7z020clg400-1 (Zynq Z-7020)
# Run: vivado -mode batch -source synth_46m_peak.tcl

set base_dir /home/ubuntu/wnn_fpga
set rtl_dir ${base_dir}/rtl
set results_dir ${base_dir}/results

file mkdir ${results_dir}

set genomes {
    {ciciot46m_96n32b    "CICIOT46M_96n_32b"    sparse wnn_classifier_impl}
    {ciciot46m_198n32b   "CICIOT46M_198n_32b"   sparse wnn_classifier_impl}
    {ciciot46m_245n32b   "CICIOT46M_245n_32b"   sparse wnn_classifier_impl}
    {ciciot46m_500n34b   "CICIOT46M_500n_34b"   sparse wnn_classifier_impl}
}

foreach genome $genomes {
    set flow_dir [lindex $genome 0]
    set desc [lindex $genome 1]
    set mode [lindex $genome 2]
    set top_module [lindex $genome 3]
    set export_dir ${base_dir}/export/${flow_dir}
    set out_dir ${results_dir}/${desc}

    # Skip if export directory doesn't exist yet (500n may arrive later)
    if {![file exists ${export_dir}/wnn_classifier_impl.sv]} {
        puts "SKIPPING ${desc}: export not found at ${export_dir}"
        continue
    }

    file mkdir ${out_dir}

    puts "============================================="
    puts "Synthesizing: ${desc} (${flow_dir}, ${mode})"
    puts "============================================="

    create_project -force synth_${desc} ${out_dir}/project -part xc7z020clg400-1

    add_files ${rtl_dir}/wnn_neuron.sv
    add_files ${export_dir}/wnn_classifier_impl.sv
    if {[file exists ${export_dir}/mem]} {
        set_property file_type {Memory File} [add_files [glob ${export_dir}/mem/*.mem]]
    }

    set_property top ${top_module} [current_fileset]
    synth_design -top ${top_module} -part xc7z020clg400-1 -mode out_of_context
    create_clock -period 5.0 -name clk [get_ports clk]

    report_utilization -file ${out_dir}/utilization.rpt
    report_timing_summary -file ${out_dir}/timing.rpt
    report_power -file ${out_dir}/power.rpt

    set fp [open ${out_dir}/summary.txt w]
    puts $fp "Genome: ${flow_dir}"
    puts $fp "Description: ${desc}"
    puts $fp "Mode: ${mode}"

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
    close_project
}

puts "===== 46M PEAK SYNTHESIS COMPLETE ====="
