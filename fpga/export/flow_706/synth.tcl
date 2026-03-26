# Vivado synthesis script for WNN classifier
# Genome: 0baca0c0f8b4d091 (500n × 32b)
# Target: xc7z020clg400-1

create_project wnn_synth fpga/export/flow_706/vivado_project -part xc7z020clg400-1 -force

# Add RTL sources
add_files fpga/export/rtl/wnn_neuron.sv
add_files fpga/export/flow_706/wnn_classifier_impl.sv

# Add BRAM initialization files
add_files fpga/export/flow_706/mem/

# Set top module
set_property top wnn_classifier_impl [current_fileset]

# Synthesis
synth_design -top wnn_classifier_impl -part xc7z020clg400-1

# Reports
report_utilization -file fpga/export/flow_706/utilization.rpt
report_timing_summary -file fpga/export/flow_706/timing.rpt
report_power -file fpga/export/flow_706/power.rpt

# Write checkpoint
write_checkpoint -force fpga/export/flow_706/post_synth.dcp

puts "Synthesis complete!"
puts "Utilization: fpga/export/flow_706/utilization.rpt"
puts "Timing:      fpga/export/flow_706/timing.rpt"
puts "Power:       fpga/export/flow_706/power.rpt"
