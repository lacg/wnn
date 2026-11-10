#!/usr/bin/env bash
# CTRL-51: WNN cascade vs MPCOF cascade teacher on the same Cortex-M4 bench.
#
# Unlike run_bench.sh (no libc, synthetic inputs, mean only), this LINKS newlib's
# libm + libgcc — the MPCOF needs atan2/asin/sqrt in double, the WNN atan2f/sqrtf/
# powf and the Mahony estimator's double sqrt; on the M4F every double op is a
# libgcc soft-float call, which is exactly the cost being measured. The homebrew
# arm-none-eabi-gcc has no newlib, so ARM_TC must point at a full Arm GNU
# Toolchain bin/ (e.g. arm-gnu-toolchain-15.3.rel2 unpacked under /tmp).
#
# Inputs (all from the Rust reference, never synthetic):
#   $REF/traj_{nominal,stress,upset}.bin   mcu/ctrl51/export_mpcof.py
#   $REF/wnn_a49_flights.bin               mcu/ctrl51/record_wnn_flights.py
#   $WNN_MODEL                             scripts/export_controller_c.py (a49 dres)
# Output: $OUT/<variant>.regions (one instruction count per call) + .log, then
# mcu/ctrl51/summarize_bench.py prints the table.
set -eu
S="$(cd "$(dirname "$0")" && pwd)"
ARM_TC="${ARM_TC:-/tmp/ctrl51/tc/arm-gnu-toolchain-15.3.rel2-darwin-arm64-arm-none-eabi/bin}"
REF="${REF:-/tmp/ctrl51/ref}"
OUT="${OUT:-/tmp/ctrl51/bench}"
WNN_MODEL="${WNN_MODEL:-/tmp/ctrl51/wnn_model_a49.h}"
PLUGIN="${PLUGIN:-/tmp/ctrl51/insnmark.dylib}"
CC="$ARM_TC/arm-none-eabi-gcc"
SIZE="$ARM_TC/arm-none-eabi-size"
mkdir -p "$OUT"
# -ffp-contract=off: the build the host parity proves bit-exact (README "Build flags").
CFLAGS="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 -std=c11 \
 -ffp-contract=off -ffreestanding -nostartfiles --specs=nano.specs --specs=nosys.specs \
 -ffunction-sections -fdata-sections -Wl,--gc-sections -T $S/link.ld -I$S"
LIBS="-lm -lc -lgcc"
QEMU="qemu-system-arm -M mps2-an386 -cpu cortex-m4 -nographic -semihosting-config enable=on,target=native"

MPCOF_SRC="$S/mpcof_cascade.c $S/cascade_dhat.c $S/wnn_ratefw.c"
WNN_SRC="$S/cascade_dhat.c $S/wnn_ratefw.c"   # wnn_cascade.c is #included by the bench

build() {  # $1 name, $2 defines, $3 sources (named objects: the map attributes bytes per file)
	local objs="" src o
	mkdir -p "$OUT/obj_$1"
	for src in "$S/bench_ctrl51.c" $3; do
		o="$OUT/obj_$1/$(basename "${src%.c}").o"
		$CC $CFLAGS $2 -c -o "$o" "$src"
		objs="$objs $o"
	done
	$CC $CFLAGS -o "$OUT/$1.elf" $objs $LIBS -Wl,-Map,"$OUT/$1.map"
	$SIZE -A "$OUT/$1.elf" | awk '/^\.text|^\.data|^\.bss/{printf "%s=%s ", $1, $2}' > "$OUT/$1.size"
}

# Entry addresses of the soft-double primitives and libm functions in an ELF, as
# insnmark.c plugin args (Thumb bit cleared, aliases deduplicated). The cd*cmp
# helpers are left out: dcmplt & co. call them, which would count a compare twice.
SOFT_RE=' __aeabi_(dadd|dsub|drsub|dmul|ddiv|drdiv|dcmp(eq|lt|le|ge|gt|un)|f2d|d2f|i2d|ui2d|l2d|ul2d|d2iz|d2uiz|d2lz|d2ulz)$'
LIBM_RE=' (atan2|asin|sqrt|tan|cos|atan2f|asinf|sqrtf|powf|fmaxf)$'
fn_args() {  # $1 elf, $2 regex, $3 arg name
	local seen="" addr _t _n a
	$ARM_TC/arm-none-eabi-nm "$1" | grep -E "$2" | while read -r addr _t _n; do
		a=$(printf '%x' $(( 16#$addr & ~1 )))
		case " $seen " in *" $a "*) continue ;; esac
		seen="$seen $a"
		printf ',%s=0x%s' "$3" "$a"
	done
}

run() {  # $1 name, $2 trajectory file
	local fns
	fns="$(fn_args "$OUT/$1.elf" "$SOFT_RE" soft)$(fn_args "$OUT/$1.elf" "$LIBM_RE" lm)"
	timeout 3600 $QEMU -kernel "$OUT/$1.elf" -plugin "$PLUGIN,out=$OUT/$1.regions$fns" -d plugin \
		> "$OUT/$1.log" 2>&1 || true
	echo "== $1 ($(basename "$2")): $(tr '\n' ' ' < "$OUT/$1.log")"
}

variant() {  # $1 name, $2 defines, $3 sources, $4 trajectory
	build "$1" "$2 -DTRAJ_PATH=\"$4\"" "$3"
	run "$1" "$4"
}

ONLY="${ONLY:-all}"   # all | mpcof | wnn
[ "$ONLY" = wnn ] || for r in nominal stress upset; do
	variant "mpcof_f64_$r" "-DBENCH_MPCOF" "$MPCOF_SRC" "$REF/traj_$r.bin"
	variant "mpcof_f32_$r" "-DBENCH_MPCOF -DMPCOF_F32" "$MPCOF_SRC" "$REF/traj_$r.bin"
done
[ "$ONLY" = mpcof ] && exit 0
WDEF="-DBENCH_WNN -DWNN_MODEL_HEADER=\"$WNN_MODEL\""
variant "wnn_a49_f64obs" "$WDEF" "$WNN_SRC" "$REF/wnn_a49_flights.bin"
variant "wnn_a49_f32obs" "$WDEF -DWNN_DHAT_F32" "$WNN_SRC" "$REF/wnn_a49_flights.bin"
