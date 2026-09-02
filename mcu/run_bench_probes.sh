#!/usr/bin/env bash
# Probe accounting for the incremental control step (02/09/2026).
#
# WHAT THIS ANSWERS. Instructions are nearly free at 480 MHz (1 ms = 480,000
# cycles), so the cost of moving a too-large model into external memory-mapped
# QSPI NOR is not instructions — it is RANDOM READS, each paying a
# command+address+dummy phase before data moves. This script measures how many
# key-array reads a control step actually performs, so that cost can be priced.
#
# It uses -DINC_LOOKUP_BINSEARCH: the open-addressed hash is not a candidate at
# this size (8 bytes/slot at >=2x the keys is ~16 MB for n=256, larger than the
# H743's SRAM and larger than this board's whole RAM), and binary search reads
# the key array in place, which is exactly what an XIP flash mapping gives.
#
# -DPROBE_STATS is NEVER combined with an instr/step measurement: counting costs
# instructions. The two builds answer different questions. run_bench.sh remains
# the timing harness and is untouched.
set -eu
S="$(cd "$(dirname "$0")" && pwd)"
N="${N:-500}"
MODEL="${MODEL:-wnn_model.h}"   # -DWNN_MODEL_HEADER; see bench_inc.c
JITTERS="${JITTERS:-1 2 4 8}"
CC=arm-none-eabi-gcc
CFLAGS="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 \
 -nostdlib -nostartfiles -ffreestanding -T $S/link.ld"
QEMU="qemu-system-arm -M mps2-an386 -cpu cortex-m4 -nographic \
 -semihosting-config enable=on,target=native"

echo "model: $MODEL   steps: $N"
printf '%-8s %10s %10s %12s %12s %14s\n' jitter dirty/step dirty_max reads/step reads_max reads/dirty
for j in $JITTERS; do
	elf="$S/p_$$.elf"
	$CC $CFLAGS -DBENCH_INC -DPROBE_STATS -DINC_LOOKUP_BINSEARCH \
	    "-DWNN_MODEL_HEADER=\"$MODEL\"" \
	    "-DITERS=$N" "-DSTEP_JITTER=$j" -I"$S" -o "$elf" "$S/bench_inc.c"
	out=$(timeout 900 $QEMU -kernel "$elf" 2>&1 || true)
	rm -f "$elf"
	echo "$out" | awk -F'\t' -v j="$j" '
		/^CHECK/ { if ($3 != 0 || $5 != 0) printf "  !! EQUIVALENCE FAILED addr=%s lookup=%s\n", $3, $5 }
		/^PROBES/ {
			# field 1 is the row tag; the key/value pairs start at 2.
			for (i = 2; i < NF; i += 2) v[$i] = $(i+1)
			s = v["steps"] + 0; d = v["dirty_tot"] + 0; m = v["memreads_tot"] + 0
			printf "%-8s %10.2f %10s %12.2f %12s %14.2f\n", j,
				(s ? d/s : 0), v["dirty_max"],
				(s ? m/s : 0), v["memreads_max"],
				(d ? m/d : 0)
		}'
done
echo
echo "NOTE: counts, not cycles. Multiply reads/step by the external-memory random"
echo "      read latency to price an off-chip model; compare against 1 ms."
