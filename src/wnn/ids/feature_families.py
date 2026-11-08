"""
Feature families for IDS receptive fields (flow param `ids_receptive_fields`,
worker ABI 27; docs/ids_receptive_fields_design.md).

A family is a declared group of features with one mechanism (UNSW ttl_state,
handshake, volume, ...). The maps are versioned package data next to the
ranking lists — `feature_families/<FAMILY_MAP_VERSION>/<dataset>.json` — for
the same reason as feature_rankings/: the worker reads the tree, and a change
goes to a new directory + version so the validation cache re-keys instead of
a map silently changing in place.

Connections index ENCODED bits, so the map that reaches Rust is per bit:
`bit_families(encoder, total_bits, dataset)` walks the encoder's actual layout
(`feature_bit_ranges`, after ids_encoding_dedupe dropped bits / features, with
the is_invalid flag bit belonging to its feature). It refuses a layout it
cannot map exactly: a feature with no family, a feature in two families, or an
encoder whose ranges do not tile the dataset's bits (top20_split joins two
encoders; only the first is on the dataset).
"""

import json
from pathlib import Path

FAMILY_MAP_VERSION = "families_v1"
_DIR = Path(__file__).resolve().parent / "feature_families" / "v1"


def family_file(dataset_name: str) -> Path:
	"""The map for a flow's ids_dataset (both CIC-IoT sizes share one)."""
	key = "ciciot2023" if dataset_name.startswith("ciciot2023") else dataset_name
	path = _DIR / f"{key}.json"
	if not path.exists():
		raise FileNotFoundError(f"ids_receptive_fields: no family map for {dataset_name!r} ({path})")
	return path


def load_family_map(dataset_name: str) -> tuple[dict[str, str], list[str]]:
	"""(feature -> family, families in file order). Refuses a feature listed twice."""
	families = json.loads(family_file(dataset_name).read_text())["families"]
	owner: dict[str, str] = {}
	for family, spec in families.items():
		for feature in spec["features"]:
			if feature in owner:
				raise ValueError(f"family map {dataset_name}: {feature!r} in both {owner[feature]!r} and {family!r}")
			owner[feature] = family
	return owner, list(families)


def bit_families(encoder, total_bits: int, dataset_name: str) -> tuple[list[int], list[str]]:
	"""(family index per encoded bit, family names indexed by it). Families with
	no encoded bit are dropped, so every index has at least one bit."""
	owner, order = load_family_map(dataset_name)
	ranges = _tiling_ranges(encoder, total_bits)
	missing = [f for f in ranges if f not in owner]
	if missing:
		raise ValueError(f"ids_receptive_fields: features without a family in {family_file(dataset_name).name}: {missing}")
	present = [fam for fam in order if any(owner[f] == fam for f in ranges)]
	index = {fam: i for i, fam in enumerate(present)}
	bit_family = [0] * total_bits
	for feature, (start, end) in ranges.items():
		bit_family[start:end] = [index[owner[feature]]] * (end - start)
	return bit_family, present


def _tiling_ranges(encoder, total_bits: int) -> dict[str, tuple[int, int]]:
	"""The encoder's feature -> [start, end) ranges, verified to tile [0, total_bits)."""
	ranges = encoder.feature_bit_ranges()
	cursor = 0
	for feature, (start, end) in sorted(ranges.items(), key=lambda kv: kv[1]):
		if start != cursor or end < start:
			raise ValueError(f"ids_receptive_fields: encoder layout has a gap/overlap at {feature!r} ({start}, {end})")
		cursor = end
	if cursor != total_bits:
		raise ValueError(
			f"ids_receptive_fields: the encoder maps {cursor} bits but the dataset has {total_bits} "
			"(top20_split joins two encoders — not supported)")
	return ranges
