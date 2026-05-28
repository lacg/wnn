# Vivado batch synthesis for the 2747+2748 OI cohort exports.
#
# Targets: 1 standard sparse synth (5n × 40b small detector, FITS Z-7020 on-chip
# BRAM) + 3 logic-only variants (250n × {60-64}b, NUM_ENTRIES capped to 32 with
# distributed RAM — the 64-bit comparator critical path + per-neuron FSM still
# represent the real on-chip search logic; full sparse tables live in external
# DRAM at deploy time per fpga/scripts/make_logic_only.py).
#
# Run: vivado -mode batch -source synth_2747_2748_cohort.tcl
# Expects: /home/ubuntu/wnn_fpga/export/<genome_dir>/ with wnn_classifier_impl.sv
#          + wnn_neuron.sv (for logic-only) + mem/*.mem

set base_dir /home/ubuntu/wnn_fpga

# (dir_name, description, neuron_rtl_override)
# neuron_rtl_override == "" → use the standard fpga/rtl/wnn_neuron.sv from the
# repo copy at base_dir/rtl/wnn_neuron.sv (sparse Z-7020 BRAM variant).
# Otherwise use the per-export logic-only neuron sv that ships in the bundle.
set genomes {
    {flow_2747_best_fpr            "2747 best_fpr 5n x 40b (standard sparse, Z-7020 BRAM)"             ""}
    {flow_2747_best_f1_logic       "2747 best_f1  250n x 60-64b (logic-only, cap=32)"                  "wnn_neuron.sv"}
    {flow_2747_best_fitness_logic  "2747 best_fitness 250n x 60-64b (logic-only, cap=32)"              "wnn_neuron.sv"}
    {flow_2748_best_fitness_logic  "2748 best_fitness 250n x 60-64b (logic-only, cap=32)"              "wnn_neuron.sv"}
}

foreach g $genomes {
    set d        [lindex $g 0]
    set desc     [lindex $g 1]
    set neuron_local [lindex $g 2]
    set ed ${base_dir}/export/${d}

    if {![file exists ${ed}/wnn_classifier_impl.sv]} {
        puts "===== SKIP ${d}: classifier sv missing ====="
        continue
    }

    puts "===== Synthesizing ${d} (${desc}) ====="
    create_project -force s_${d} ${ed}/proj -part xc7z020clg400-1

    # Pick the neuron RTL: bundled (logic-only) or shared (sparse).
    if {$neuron_local ne ""} {
        add_files ${ed}/${neuron_local}
    } else {
        add_files ${base_dir}/rtl/wnn_neuron.sv
    }
    add_files ${ed}/wnn_classifier_impl.sv
    set_property file_type {Memory File} [add_files [glob ${ed}/mem/*.mem]]
    set_property top wnn_classifier_impl [current_fileset]

    synth_design -top wnn_classifier_impl -part xc7z020clg400-1 -mode out_of_context

    # Clock for timing/power reports (5 ns target = 200 MHz).
    create_clock -period 5.0 -name clk [get_ports clk]

    report_utilization      -file ${ed}/utilization.rpt
    report_timing_summary   -file ${ed}/timing.rpt
    report_power            -file ${ed}/power.rpt

    # Capture WNS + Fmax for the summary file.
    set timing_paths [get_timing_paths -max_paths 1 -quiet]
    set fp [open ${ed}/summary.txt w]
    puts $fp "Genome: ${d}"
    puts $fp "Description: ${desc}"
    puts $fp "Clock target: 5.0 ns (200 MHz)"
    if {[llength $timing_paths] > 0} {
        set wns [get_property SLACK [lindex $timing_paths 0]]
        set fmax_ns [expr 5.0 - $wns]
        set fmax_mhz [expr 1000.0 / $fmax_ns]
        puts $fp "WNS: ${wns} ns"
        puts $fp "Fmax: [format "%.1f" $fmax_mhz] MHz"
    } else {
        puts $fp "WNS: N/A (no timing paths)"
    }
    close $fp

    close_project
}

puts "===== ALL 4 GENOMES SYNTHESIZED ====="
exit
