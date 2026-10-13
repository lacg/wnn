"""wnn_neuron.sv binary search finds EVERY stored key (RTL simulation via Yosys CXXRTL).

Regression: the search bounds were `IDX_BITS'(NUM_ENTRIES) >> 1`, a size cast of
a SIGNED int parameter. For a power-of-two NUM_ENTRIES the cast's MSB is set and
Yosys sign-extends before the shift: the search started at mid = 1.5 N and every
key in the upper half read the miss cell (flow 6947 best_f1: units with 512 and
1024 entries, 269/400 HW-vs-Rust vectors wrong). Skipped without yosys/clang++.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

RTL = Path(__file__).resolve().parents[1] / "fpga" / "rtl" / "wnn_neuron.sv"
MISS = 1
pytestmark = pytest.mark.skipif(not (shutil.which("yosys") and shutil.which("yosys-config") and shutil.which("clang++")),
	reason="needs yosys (CXXRTL) and clang++")

TOP = """module top(input logic clk, input logic rst_n, input logic start, input logic [33:0] address,
	output logic [7:0] result, output logic result_valid, output logic busy);
	wnn_neuron #(.NUM_ENTRIES({n}), .ADDR_BITS(34), .INPUT_BITS(8), .DEFAULT_VALUE(8'd{miss}),
		.KEY_MEM_FILE("keys.mem"), .VALUE_MEM_FILE("values.mem"))
		u (.clk(clk), .rst_n(rst_n), .start(start), .input_vec(8'd0), .address(address),
		.result(result), .result_valid(result_valid), .busy(busy));
endmodule
"""

HARNESS = r"""#include <cstdio>
#include "d.cc"
int main() {
	FILE *f = fopen("probe.txt", "r"); unsigned long k; int expect, bad = 0, n = 0;
	cxxrtl_design::p_top t;
	auto tick = [&]() { t.p_clk.set<bool>(false); t.step(); t.p_clk.set<bool>(true); t.step(); };
	t.p_rst__n.set<bool>(false); tick(); t.p_rst__n.set<bool>(true); tick();
	while (fscanf(f, "%lx %d", &k, &expect) == 2) {
		t.p_address.set<uint64_t>(k); t.p_start.set<bool>(true); tick(); t.p_start.set<bool>(false);
		for (int c = 0; c < 1000 && !t.p_result__valid.get<bool>(); c++) tick();
		bad += (int)t.p_result.get<uint32_t>() != expect; n++; tick();
	}
	printf("%d %d\n", bad, n);
}
"""


def _write_case(d: Path, n: int) -> None:
	rng = np.random.RandomState(n)
	keys = np.unique(rng.randint(0, 1 << 34, size=3 * n, dtype=np.int64))[:n]
	values = rng.choice([0, 2, 3], size=n)  # never the miss cell, so a miss is visible
	(d / "keys.mem").write_text("".join(f"{int(k):016x}\n" for k in keys))
	(d / "values.mem").write_text("".join(f"{int(v):02x}\n" for v in values))
	absent = [int(k) + 1 for k in keys[:20] if int(k) + 1 not in set(keys.tolist())]
	rows = [(int(k), int(v)) for k, v in zip(keys, values)] + [(a, MISS) for a in absent]
	(d / "probe.txt").write_text("".join(f"{k:x} {v}\n" for k, v in rows))
	(d / "top.sv").write_text(TOP.format(n=n, miss=MISS))
	# Yosys's front end has no `parameter string`; the untyped form is the same parameter.
	(d / "wnn_neuron.sv").write_text(RTL.read_text().replace("parameter string ", "parameter "))
	(d / "tb.cc").write_text(HARNESS)


@pytest.mark.parametrize("n", [512, 586, 1024])
def test_every_stored_key_is_found(tmp_path, n):
	_write_case(tmp_path, n)
	run = lambda cmd: subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True)
	run(["yosys", "-q", "-p", "read_verilog -sv wnn_neuron.sv; read_verilog -sv top.sv; hierarchy -top top; "
		"proc; flatten; opt_clean; write_cxxrtl d.cc"])
	inc = subprocess.run(["yosys-config", "--datdir"], check=True, capture_output=True, text=True).stdout.strip()
	run(["clang++", "-O1", "-w", "-DNDEBUG", "-std=c++17", f"-I{inc}/include/backends/cxxrtl/runtime", "-I.", "-o", "tb", "tb.cc"])
	bad, total = map(int, run(["./tb"]).stdout.split())
	assert total >= n and bad == 0, f"{bad}/{total} lookups wrong at NUM_ENTRIES={n}"
