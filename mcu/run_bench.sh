#!/usr/bin/env bash
# Build each variant at ITERS=0 and ITERS=N, count retired instructions under
# QEMU (mps2-an386 = Cortex-M4), and report per-control-step cost.
#
# Per-step = (I_N - I_0) / N, then minus the NONE row to remove make_input().
# The double subtraction is what makes this trustworthy: startup, semihosting
# and input generation all cancel exactly, so what is left is the control step.
set -eu

S="$(cd "$(dirname "$0")" && pwd)"
N="${N:-200}"
CC=arm-none-eabi-gcc
# Hardware FPU: the STM32F405 is a Cortex-M4F, so PID/MLP float work belongs on
# the VFP, not in soft-float helpers. -nostdlib also drops libgcc, so without
# this the float variants fail to link on __aeabi_fadd rather than run slowly.
CFLAGS="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 \
 -nostdlib -nostartfiles -ffreestanding -T $S/link.ld"
QEMU="qemu-system-arm -M mps2-an386 -cpu cortex-m4 -nographic \
 -semihosting-config enable=on,target=native -plugin $S/insncount.dylib -d plugin"

count() {  # $1 = -DBENCH_X, $2 = iters -> instructions retired
	local elf="$S/b_$$.elf"
	$CC $CFLAGS "-D$1" "-DITERS=$2" -I"$S" -o "$elf" "$S/bench.c"
	# .text size is reported separately; here we only need the instruction count
	local out
	out=$(timeout 600 $QEMU -kernel "$elf" 2>&1 || true)
	rm -f "$elf"
	echo "$out" | awk -F'\t' '/^INSNS/{print $2}'
}

textsize() {  # $1 = -DBENCH_X -> .text bytes (the flash footprint)
	local elf="$S/t_$$.elf"
	$CC $CFLAGS "-D$1" "-DITERS=1" -I"$S" -o "$elf" "$S/bench.c"
	arm-none-eabi-size -A "$elf" | awk '/^\.text/{print $2}'
	rm -f "$elf"
}

printf '%-18s %12s %12s %12s %10s\n' variant "I(0)" "I($N)" "per-step" ".text B"
for v in BENCH_NONE BENCH_WNN_ADDR BENCH_WNN BENCH_WNN_FAST BENCH_PID BENCH_MLP; do
	i0=$(count "$v" 0)
	iN=$(count "$v" "$N")
	ps=$(( (iN - i0) / N ))
	eval "PS_$v=$ps"
	printf '%-18s %12s %12s %12s %10s\n' "$v" "$i0" "$iN" "$ps" "$(textsize "$v")"
done

echo
echo "per control step, input generation subtracted out:"
eval "base=\$PS_BENCH_NONE"
for v in BENCH_WNN_ADDR BENCH_WNN BENCH_WNN_FAST BENCH_PID BENCH_MLP; do
	eval "cur=\$PS_$v"
	printf '  %-14s %8d instructions\n' "${v#BENCH_}" "$(( cur - base ))"
done
echo
echo "NOTE: retired INSTRUCTIONS under emulation, not silicon cycles."
echo "      QEMU's mps2-an386 does not implement DWT_CYCCNT. Real Cortex-M4"
echo "      cycles run above this (loads ~2, taken branches ~2-3, flash waits)."
