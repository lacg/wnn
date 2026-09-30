"""Tests for load_population_member — pulling ONE ranked genome out of a
schema-2 checkpoint without materializing the rest of final_population
(30/09/2026: counting stage-select HEADLINE genomes, `STAGE#i`, from multi-GB
stage checkpoints).

Run: PYTHONPATH=src/wnn python tests/test_population_member.py
"""
import sys
import tempfile
from pathlib import Path

from wnn.ram.strategies.phased import PhaseCheckpoint, save_checkpoint, load_population_member
import wnn.ram.strategies.phased.checkpoint as ckmod

PASS = "[PASS]"


class _DictCodec:
	"""Toy codec: genomes are nested flow mappings/sequences, like the real one."""
	name = "dict_codec"

	def encode(self, genome):
		return genome

	def decode(self, data):
		return data


def _genome(i: int) -> dict:
	return {"id": i, "rows": [[i, j, [j, {"k": j}]] for j in range(40)], "tag": f"g{i}"}


def _write(tmp: Path, n: int) -> Path:
	pop = [_genome(i) for i in range(n)]
	ck = PhaseCheckpoint(phase_key="1", phase_name="neurons", strategy_type="GA",
	                     best_genome=_genome(99), final_population=pop,
	                     extra={"spec": {"num_motors": 4}})
	return save_checkpoint(tmp / "stage1_neurons.yaml.gz", ck, _DictCodec())


def test_every_member_matches():
	with tempfile.TemporaryDirectory() as d:
		p = _write(Path(d), 25)
		for i in range(25):
			g, extra = load_population_member(p, _DictCodec(), i)
			assert g == _genome(i), f"member {i} mismatch"
			assert extra["spec"] == {"num_motors": 4}
	print(f"  {PASS} all 25 members round-trip; extra carries the spec")


def test_members_spanning_chunk_boundaries():
	# A tiny scan chunk forces members (and the population marker) to straddle reads.
	saved = ckmod._POP_CHUNK
	ckmod._POP_CHUNK = 97
	try:
		with tempfile.TemporaryDirectory() as d:
			p = _write(Path(d), 12)
			for i in (0, 5, 11):
				g, _ = load_population_member(p, _DictCodec(), i)
				assert g == _genome(i), f"member {i} mismatch across chunks"
	finally:
		ckmod._POP_CHUNK = saved
	print(f"  {PASS} members straddling 97-byte scan chunks")


def test_out_of_range_raises():
	with tempfile.TemporaryDirectory() as d:
		p = _write(Path(d), 7)
		try:
			load_population_member(p, _DictCodec(), 7)
		except IndexError as e:
			assert "has 7 members" in str(e), str(e)
		else:
			raise AssertionError("index past the end must raise")
	print(f"  {PASS} out-of-range index raises IndexError with the true count")


if __name__ == "__main__":
	test_every_member_matches()
	test_members_spanning_chunk_boundaries()
	test_out_of_range_raises()
	sys.exit(0)
