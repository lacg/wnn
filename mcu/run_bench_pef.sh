#!/usr/bin/env bash
# pef64 vs uint32 key store, SAME winner: retired instructions under QEMU.
# CPU=m4 (default): mps2-an386 Cortex-M4, the harness every banked mcu/ number uses.
# CPU=m7: mps2-an500 Cortex-M7 (the STM32H743's core; same link.ld memory map).
#
#   U32=<uint32 header> PEF=<pef64 header> TRACE=<ctrl_trace_*.npz> \
#     [CPU=m4|m7] [N=200] [JITTER=2] [TICKS=200] [PY=python3] bash mcu/run_bench_pef.sh
#
# Headers: scripts/export_controller_c.py --tables-only [--key-format pef64].
# 1. bench_inc.c, synthetic thermometer walk (the run_bench.sh method: ITERS=0 vs
#    ITERS=N, NONE subtracted): GATHER, BASE (every neuron looked up), INC
#    (incremental addressing, lookups only where dirty). One lookup = (BASE-GATHER)/neurons.
# 2. bench_pef_replay.c, the winner's REAL VAL addresses (TRACE), insnmark.c regions:
#    per lookup, per decision tick with every neuron, per decision tick incremental.
#    mean / p50 / p99 / max, marker overhead subtracted. Both formats must report
#    the same hit count (an on-target equivalence check).
# INSTRUCTIONS, not cycles: see mcu/README.md.
set -eu
S="$(cd "$(dirname "$0")" && pwd)"
N="${N:-200}"; JITTER="${JITTER:-2}"; TICKS="${TICKS:-200}"; PY="${PY:-python3}"
: "${U32:?uint32 header}" "${PEF:?pef64 header}" "${TRACE:?ctrl_trace npz}"
COUNT_PLUGIN="${COUNT_PLUGIN:-$S/insncount.dylib}"
MARK_PLUGIN="${MARK_PLUGIN:-$S/insnmark.dylib}"
case "${CPU:-m4}" in
	m4) ARCH=(-mcpu=cortex-m4 -mfpu=fpv4-sp-d16); BOARD=(-M mps2-an386 -cpu cortex-m4) ;;
	m7) ARCH=(-mcpu=cortex-m7 -mfpu=fpv5-d16); BOARD=(-M mps2-an500 -cpu cortex-m7) ;;
	*) echo "CPU must be m4 or m7" >&2; exit 2 ;;
esac
CFLAGS=("${ARCH[@]}" -mthumb -mfloat-abi=hard -O2
        -nostdlib -nostartfiles -ffreestanding -T "$S/link.ld")
QEMU=(qemu-system-arm "${BOARD[@]}" -nographic
      -semihosting-config enable=on,target=native)
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT

build() {  # $1 out.elf  $2 source  $3 model header  $4.. extra -D flags
	local out="$1" src="$2" hdr="$3"; shift 3
	arm-none-eabi-gcc "${CFLAGS[@]}" "-DWNN_MODEL_HEADER=\"$hdr\"" "$@" -I"$S" -o "$out" "$S/$src"
}

insns() {  # $1 header  $2 variant  $3 iters -> retired instructions (bench_inc.c)
	build "$W/i.elf" bench_inc.c "$1" "-D$2" "-DITERS=$3" "-DSTEP_JITTER=$JITTER" -DINC_LOOKUP_BINSEARCH
	timeout 900 "${QEMU[@]}" -plugin "$COUNT_PLUGIN" -d plugin -kernel "$W/i.elf" 2>&1 \
		| tee "$W/last.out" | awk -F'\t' '/^INSNS/{print $2}'
	if grep -q '^CHECK' "$W/last.out"; then grep '^CHECK' "$W/last.out" | sed 's/^/    /' >&2; fi
}

per_step() {  # $1 header  $2 variant -> instructions per step
	local i0 iN
	i0=$(insns "$1" "$2" 0); iN=$(insns "$1" "$2" "$N")
	echo $(( (iN - i0) / N ))
}

synthetic() {  # $1 label  $2 header
	local nrn none g b i
	nrn=$(awk '/#define WNN_NEURONS/{print $3}' "$2")
	none=$(per_step "$2" BENCH_NONE); g=$(per_step "$2" BENCH_GATHER)
	b=$(per_step "$2" BENCH_BASE); i=$(per_step "$2" BENCH_INC)
	build "$W/t.elf" bench_inc.c "$2" -DBENCH_BASE -DITERS=1 -DINC_LOOKUP_BINSEARCH
	printf '%-7s %7d %9d %9d %9d %11d %12s\n' "$1" "$nrn" $((g - none)) $((b - none)) $((i - none)) \
		$(( (b - g) / nrn )) "$(arm-none-eabi-size -A "$W/t.elf" | awk '/^\.text/{print $2}')"
}

stats() {  # region file -> "mean p50 p99 max regions" (first region = marker overhead)
	awk 'NR==1{o=$1; next} {print $1 - o}' "$1" | sort -n | awk '
		{v[NR] = $1; s += $1}
		END {printf "%9.1f %7d %7d %7d %8d", s / NR, v[int(NR * 0.50) + 1], v[int(NR * 0.99) + 1], v[NR], NR}'
}

replay() {  # $1 label  $2 header  $3 mode
	build "$W/r.elf" bench_pef_replay.c "$2" "-D$3" "-DQ_HEADER=\"$W/q.h\""
	timeout 900 "${QEMU[@]}" -plugin "$MARK_PLUGIN,out=$W/regions.txt" -d plugin -kernel "$W/r.elf" > "$W/r.out" 2>&1 || true
	printf '%-7s %-14s %s   %s\n' "$1" "${3#REPLAY_}" "$(stats "$W/regions.txt")" \
		"$(awk -F'\t' '/^REPLAY/{print "hits " $8}' "$W/r.out")"
}

echo "CPU ${CPU:-m4}"
echo "== 1. synthetic walk (bench_inc.c), jitter $JITTER, N=$N, NONE subtracted =="
printf '%-7s %7s %9s %9s %9s %11s %12s\n' format neurons GATHER BASE INC lookup/BASE ".text B"
synthetic uint32 "$U32"
synthetic pef64 "$PEF"

echo
echo "== 2. real VAL addresses (bench_pef_replay.c), $TICKS decision ticks of $(basename "$TRACE") =="
"$PY" "$S/pef_replay_queries.py" --trace "$TRACE" --out "$W/q.h" --ticks "$TICKS"
printf '%-7s %-14s %9s %7s %7s %7s %8s\n' format region mean p50 p99 max regions
for mode in REPLAY_LOOKUP REPLAY_FULL REPLAY_INC; do
	replay uint32 "$U32" "$mode"
	replay pef64 "$PEF" "$mode"
done
echo
echo "NOTE: retired INSTRUCTIONS under emulation, not silicon cycles;"
echo "      flash wait states / cache misses on the H743 are NOT in these numbers."
