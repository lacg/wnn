#!/usr/bin/env bash
# Per-DECISION instruction cost of an IDS classification across real genome shapes.
# Same double-subtraction method as run_bench.sh: (I_N - I_0)/N, minus the NONE row.
#
# Compiler stderr is NOT suppressed: an earlier version hid it and a shape that
# failed to build looked like a shape that produced no row.
set -u
S="$(cd "$(dirname "$0")" && pwd)"
N="${N:-100}"
CC=arm-none-eabi-gcc
CFLAGS="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 \
 -nostdlib -nostartfiles -ffreestanding -T $S/link.ld"
QEMU="qemu-system-arm -M mps2-an386 -cpu cortex-m4 -nographic \
 -semihosting-config enable=on,target=native -plugin $S/insncount.dylib -d plugin"

count() {
	local elf="$S/i_$$.elf" out
	if ! $CC $CFLAGS "-D$1" "-DITERS=$2" -I"$S" -o "$elf" "$S/bench_ids.c"; then
		echo "BUILD-FAILED" >&2; echo 0; return
	fi
	out=$(timeout 900 $QEMU -kernel "$elf" 2>&1 || true)
	rm -f "$elf"
	echo "$out" | awk -F'\t' '/^INSNS/{print $2}'
}

SHAPES=(
	"64:30:1797:CONTROLLER-ANCHOR (classify only; bench.c BASE incl. decode = 20231)"
	"400:34:256:live genome, flows DB"
	"400:34:1024:live genome, deeper memory"
	"500:34:256:the shape Vivado synthesised"
	"250:100:256:production winner (>64b -> splitmix64)"
	"53:48:1024:what the GA picks"
)

printf '%-20s %9s %9s %9s %9s   %s\n' shape total gather search keys_KB provenance
for row in "${SHAPES[@]}"; do
	IFS=: read -r n b k prov <<< "$row"
	kb=$(python3 "$S/gen_ids_model.py" --neurons "$n" --bits "$b" \
		--keys-per-neuron "$k" --out "$S/ids_model.h" | sed -E 's/.*-> [0-9]+ keys, ([0-9]+) KB.*/\1/')
	i0=$(count BENCH_NONE 0);     iN=$(count BENCH_NONE "$N");     none=$(( (iN - i0) / N ))
	i0=$(count BENCH_IDS_ADDR 0); iN=$(count BENCH_IDS_ADDR "$N"); addr=$(( (iN - i0) / N - none ))
	i0=$(count BENCH_IDS 0);      iN=$(count BENCH_IDS "$N");      full=$(( (iN - i0) / N - none ))
	printf '%-20s %9d %9d %9d %9s   %s\n' "${n}n x ${b}b k=${k}" "$full" "$addr" "$((full-addr))" "$kb" "$prov"
done
echo
echo "Retired INSTRUCTIONS under QEMU mps2-an386, not silicon cycles. Input synthesis"
echo "subtracted. NO temporal coherence between decisions -- that is the workload, not"
echo "a harness limitation (see bench_ids.c header). keys_KB is the KEY ARRAY ONLY."
