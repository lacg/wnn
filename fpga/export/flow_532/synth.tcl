# Vivado synthesis script for WNN classifier
# Genome: 3c5bcd8c1e82cf0e (92n × 32b)
# Target: xc7z020clg400-1

create_project wnn_synth fpga/export/flow_532/vivado_project -part xc7z020clg400-1 -force

# Add RTL sources
add_files fpga/export/rtl/wnn_neuron.sv
add_files fpga/export/flow_532/wnn_classifier_impl.sv

# Add BRAM initialization files
add_files fpga/export/flow_532/mem/

# Set top module
set_property top wnn_classifier_impl [current_fileset]

# Synthesis
synth_design -top wnn_classifier_impl -part xc7z020clg400-1

# Reports
report_utilization -file fpga/export/flow_532/utilization.rpt
report_timing_summary -file fpga/export/flow_532/timing.rpt
report_power -file fpga/export/flow_532/power.rpt

# Write checkpoint
write_checkpoint -force fpga/export/flow_532/post_synth.dcp

puts "Synthesis complete!"
puts "Utilization: fpga/export/flow_532/utilization.rpt"
puts "Timing:      fpga/export/flow_532/timing.rpt"
puts "Power:       fpga/export/flow_532/power.rpt"
