#!/bin/bash
# Run functional simulation of WNN classifier on Vivado xsim
# Usage: bash run_sim.sh
# Must be run from ~/wnn_fpga on EC2 with Vivado sourced

set -e

FLOW_ID=973
EXPORT_DIR=export/flow_${FLOW_ID}
TB_DIR=${EXPORT_DIR}/testbench
SIM_DIR=${EXPORT_DIR}/sim
RTL_DIR=rtl

echo "========================================="
echo "  WNN FPGA Functional Simulation"
echo "  Flow: ${FLOW_ID}"
echo "========================================="

# Create sim directory
rm -rf ${SIM_DIR}
mkdir -p ${SIM_DIR}
cd ${SIM_DIR}

# Copy test data files to sim working directory
cp ../../${TB_DIR}/test_inputs.mem .
cp ../../${TB_DIR}/test_expected.mem .

# Create symlink to mem files
ln -sf ../../${EXPORT_DIR}/mem mem

echo ""
echo "1. Compiling RTL..."
xvlog --sv \
    ../../${RTL_DIR}/wnn_neuron.sv \
    ../../${EXPORT_DIR}/wnn_classifier_impl.sv \
    ../../tb/wnn_tb.sv \
    2>&1 | tail -5

echo ""
echo "2. Elaborating..."
xelab wnn_tb -s sim -timescale 1ns/1ps 2>&1 | tail -5

echo ""
echo "3. Running simulation..."
xsim sim -runall -log sim.log 2>&1

echo ""
echo "Simulation log: ${SIM_DIR}/sim.log"
echo "Done!"
