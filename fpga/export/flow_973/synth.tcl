# Vivado synthesis script for WNN classifier
# Genome: 46cbd295dd2796f5 (11n × 34b)
# Target: xc7z020clg400-1

create_project wnn_synth fpga/export/flow_973/vivado_project -part xc7z020clg400-1 -force

# Add RTL sources
add_files fpga/export/rtl/wnn_neuron.sv
add_files fpga/export/flow_973/wnn_classifier_impl.sv

# Add BRAM initialization files
add_files fpga/export/flow_973/mem/

# Set top module
set_property top wnn_classifier_impl [current_fileset]

# Synthesis
synth_design -top wnn_classifier_impl -part xc7z020clg400-1

# Reports
report_utilization -file fpga/export/flow_973/utilization.rpt
report_timing_summary -file fpga/export/flow_973/timing.rpt
report_power -file fpga/export/flow_973/power.rpt

# Write checkpoint
write_checkpoint -force fpga/export/flow_973/post_synth.dcp

puts "Synthesis complete!"
puts "Utilization: fpga/export/flow_973/utilization.rpt"
puts "Timing:      fpga/export/flow_973/timing.rpt"
puts "Power:       fpga/export/flow_973/power.rpt"
