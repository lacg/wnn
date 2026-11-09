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

MERGE (Luiz, IDS-55 follow-up): a family may declare `"merge_into": "<family>"`.
When its encoded-bit count (after dedupe) is below `host_bits` — the bits a
family needs to host the run's WIDEST neuron, max_bits - floor(rf_cross x
max_bits), the Rust eligibility rule — its features join the declared partner
(chained through the partner's own merge_into if that is still too small).
A family that can host keeps exactly its own map. v2 (08/11) added UNSW
handshake->timing and protocol->context; protocol (~12 bits on the legacy list)
merges with dedupe OFF too, hence the version bump (v1 = the IDS-55 A/B's map).
"""

import json
import math
from pathlib import Path

FAMILY_MAP_VERSION = "families_v2"
_DIR = Path(__file__).resolve().parent / "feature_families" / "v2"


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


def merge_targets(dataset_name: str) -> dict[str, str]:
	"""family -> declared merge_into partner. Refuses an unknown partner, a
	self-merge and a cycle (a chain must end at a family that never merges)."""
	families = json.loads(family_file(dataset_name).read_text())["families"]
	targets = {f: spec["merge_into"] for f, spec in families.items() if "merge_into" in spec}
	for family, partner in targets.items():
		if partner not in families or partner == family:
			raise ValueError(f"family map {dataset_name}: {family!r} merge_into {partner!r} is not another family")
		seen = {family}
		while partner in targets:
			if partner in seen:
				raise ValueError(f"family map {dataset_name}: merge_into cycle through {family!r}")
			seen.add(partner)
			partner = targets[partner]
	return targets


def bit_families(encoder, total_bits: int, dataset_name: str, host_bits: int = 0) -> tuple[list[int], list[str], list[str]]:
	"""(family index per encoded bit, family names indexed by it, merges applied
	as "family(Nb)->partner"). Families with no encoded bit are dropped, so every
	index has at least one bit. host_bits = 0 never merges."""
	owner, order = load_family_map(dataset_name)
	ranges = _tiling_ranges(encoder, total_bits)
	missing = [f for f in ranges if f not in owner]
	if missing:
		raise ValueError(f"ids_receptive_fields: features without a family in {family_file(dataset_name).name}: {missing}")
	owner, merges = _merge_small_families(owner, ranges, merge_targets(dataset_name), host_bits)
	present = [fam for fam in order if any(owner[f] == fam for f in ranges)]
	index = {fam: i for i, fam in enumerate(present)}
	bit_family = [0] * total_bits
	for feature, (start, end) in ranges.items():
		bit_family[start:end] = [index[owner[feature]]] * (end - start)
	return bit_family, present, merges


def _merge_small_families(owner: dict, ranges: dict, targets: dict, host_bits: int) -> tuple[dict, list[str]]:
	"""Move every declared-merge family below host_bits into its partner,
	repeating so a partner that is still too small follows its own merge_into."""
	owner, merges = dict(owner), []
	while True:
		sizes = _family_sizes(owner, ranges)
		small = [f for f in targets if 0 < sizes.get(f, 0) < host_bits]
		if not small:
			return owner, merges
		family = small[0]
		owner = {feat: (targets[family] if fam == family else fam) for feat, fam in owner.items()}
		merges.append(f"{family}({sizes[family]}b)->{targets[family]}")


def _family_sizes(owner: dict, ranges: dict) -> dict[str, int]:
	sizes: dict[str, int] = {}
	for feature, (start, end) in ranges.items():
		sizes[owner[feature]] = sizes.get(owner[feature], 0) + end - start
	return sizes


def host_bits(max_bits: int, cross: float) -> int:
	"""Encoded bits a family needs to host the run's widest neuron (the Rust
	receptive_fields eligibility rule at width max_bits)."""
	return max_bits - math.floor(cross * max_bits + 1e-9)


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
