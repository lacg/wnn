# Vivado synthesis for 46M 500n×34b sparse genome
set base_dir /home/ubuntu/wnn_fpga
set rtl_dir ${base_dir}/rtl
set results_dir ${base_dir}/results
set export_dir ${base_dir}/export/ciciot46m_500n34b
set desc "CICIOT46M_500n_34b"
set out_dir ${results_dir}/${desc}

file mkdir ${out_dir}

puts "Synthesizing: ${desc}"
create_project -force synth_${desc} ${out_dir}/project -part xc7z020clg400-1
add_files ${rtl_dir}/wnn_neuron.sv
add_files ${export_dir}/wnn_classifier_impl.sv
if {[file exists ${export_dir}/mem]} {
    set_property file_type {Memory File} [add_files [glob ${export_dir}/mem/*.mem]]
}
set_property top wnn_classifier_impl [current_fileset]
synth_design -top wnn_classifier_impl -part xc7z020clg400-1 -mode out_of_context
create_clock -period 5.0 -name clk [get_ports clk]
report_utilization -file ${out_dir}/utilization.rpt
report_timing_summary -file ${out_dir}/timing.rpt
report_power -file ${out_dir}/power.rpt

set fp [open ${out_dir}/summary.txt w]
puts $fp "Genome: ciciot46m_500n34b"
puts $fp "Description: ${desc}"
puts $fp "Mode: sparse"
set timing_paths [get_timing_paths -max_paths 1 -quiet]
if {[llength $timing_paths] > 0} {
    set wns [get_property SLACK [lindex $timing_paths 0]]
    set fmax_mhz [expr {1000.0 / (5.0 - $wns)}]
    puts $fp "WNS: ${wns} ns"
    puts $fp "Fmax: ${fmax_mhz} MHz"
}
close $fp
close_project
puts "===== 500n SYNTHESIS COMPLETE ====="
