"""make_logic_only must actually steer the neuron ROMs to distributed.

Regression: it replaced `ram_style = "block"`, which wnn_neuron.sv stopped
carrying on 31/08/2026 (ROMs need rom_style). The replace was a silent no-op,
so the "logic-only" variant still requested block ROM for every neuron.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "fpga" / "scripts"))

from make_logic_only import distributed_neuron  # noqa: E402


def test_shipped_neuron_rtl_is_steered_to_distributed():
	out = distributed_neuron((REPO / "fpga" / "rtl" / "wnn_neuron.sv").read_text())
	assert 'rom_style = "block"' not in out
	assert out.count('rom_style = "distributed"') == 2  # key_mem + value_mem


def test_rtl_without_the_attribute_is_refused():
	with pytest.raises(ValueError):
		distributed_neuron('(* ram_style = "block" *) logic [63:0] key_mem [0:31];')
