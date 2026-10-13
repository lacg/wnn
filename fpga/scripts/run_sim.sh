#!/bin/bash
# HW-vs-Rust parity simulation (Vivado xsim) of a unit-emitted export.
# Usage: bash run_sim.sh <export_dir> [sparse|dense]
# Run from the fpga/ directory (rtl/ beside export dirs) with Vivado sourced.
# PASS = every test vector's score_out equals the Rust score x 4 x logical_neurons.

set -e

EXPORT_DIR=${1:?usage: run_sim.sh <export_dir> [sparse|dense]}
KIND=${2:-sparse}
RTL_DIR=$(cd "$(dirname "$0")/../rtl" && pwd)
EXPORT_DIR=$(cd "${EXPORT_DIR}" && pwd)
SIM_DIR=${EXPORT_DIR}/sim

if [ "${KIND}" = "dense" ]; then
	SOURCES="${RTL_DIR}/wnn_neuron_dense.sv ${EXPORT_DIR}/wnn_classifier_dense.sv"
else
	SOURCES="${RTL_DIR}/wnn_neuron.sv ${EXPORT_DIR}/wnn_classifier_impl.sv"
fi

rm -rf "${SIM_DIR}" && mkdir -p "${SIM_DIR}" && cd "${SIM_DIR}"
cp "${EXPORT_DIR}"/testbench/test_inputs.mem "${EXPORT_DIR}"/testbench/test_expected.mem .
cp "${EXPORT_DIR}"/mem/*.mem .   # leaf neurons $readmemh their unit files by name

echo "1. Compiling RTL (${KIND})..."
xvlog --sv ${SOURCES} "${EXPORT_DIR}/testbench/wnn_tb_generated.sv" 2>&1 | tail -5
echo "2. Elaborating..."
xelab wnn_tb -s sim -timescale 1ns/1ps 2>&1 | tail -5
echo "3. Running simulation..."
xsim sim -runall -log sim.log 2>&1 | grep -E "HW-vs-Rust|RESULT|MISMATCH"
