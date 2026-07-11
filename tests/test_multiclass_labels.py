"""Multiclass label-mapping unit tests (docs/MULTICLASS_DESIGN.md §1).

Asserts the per-dataset class structure the multiclass extension relies on —
class counts (UNSW 10 / CICIDS 15 / CIC-IoT 8), benign at index 0, and label
normalization — WITHOUT downloading any dataset (mapping dicts + a tiny
synthetic DataFrame through encode_and_build_dataset).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))

from wnn.ids.dataset import (
	ATTACK_CATEGORIES as UNSW_CATEGORIES,
	_CATEGORY_ALIASES as UNSW_ALIASES,
	encode_and_build_dataset,
	map_multiclass_labels,
)
from wnn.ids.cicids2017 import (
	ATTACK_CATEGORIES as CICIDS_CATEGORIES,
	_LABEL_ALIASES as CICIDS_ALIASES,
	normalize_labels as cicids_normalize_labels,
)
from wnn.ids.ciciot2023 import ATTACK_CLASSES as CICIOT_CLASSES


# ── UNSW-NB15: 10 classes = 9 attack cats + Normal, benign (Normal) at 0 ──

def test_unsw_class_structure():
	assert len(UNSW_CATEGORIES) == 10
	assert UNSW_CATEGORIES[0] == "Normal"  # benign index 0 (benign-margin decode relies on this)
	assert len(set(UNSW_CATEGORIES)) == 10  # no duplicates


def test_unsw_aliases_resolve_to_canonical():
	"""Every alias target must be a canonical category (raw HF data has both
	'Backdoor' and 'Backdoors' — verified against random_3way on 11/07/2026)."""
	canonical = {c.lower(): c for c in UNSW_CATEGORIES}
	for raw, target in UNSW_ALIASES.items():
		assert target in UNSW_CATEGORIES, f"alias {raw!r} → {target!r} not canonical"
	assert UNSW_ALIASES["backdoors"] == "Backdoor"
	assert canonical["normal"] == "Normal"


# ── CICIDS2017: 15 classes = 14 attack labels + BENIGN, benign at 0 ──

def test_cicids_class_structure():
	assert len(CICIDS_CATEGORIES) == 15
	assert CICIDS_CATEGORIES[0] == "BENIGN"  # benign index 0
	assert len(set(CICIDS_CATEGORIES)) == 15


def test_cicids_mojibake_labels_normalize_to_canonical():
	"""The HF copy's actual Web Attack labels carry U+FFFD and 'Sql' casing
	(verified against lacg030175/CICIDS2017 random_3way on 11/07/2026).
	Un-normalized they'd silently map to index 0 = BENIGN."""
	raw_hf_values = pd.Series([
		"Web Attack � Brute Force",
		"Web Attack � Sql Injection",
		"Web Attack � XSS",
	])
	normalized = cicids_normalize_labels(raw_hf_values)
	for value in normalized:
		assert value in CICIDS_CATEGORIES, f"{value!r} not in ATTACK_CATEGORIES"
	assert list(normalized) == [
		"Web Attack - Brute Force",
		"Web Attack - SQL Injection",
		"Web Attack - XSS",
	]


def test_cicids_alias_targets_are_canonical():
	for raw, target in CICIDS_ALIASES.items():
		assert target in CICIDS_CATEGORIES, f"alias {raw!r} → {target!r} not canonical"


def test_cicids_normalize_passes_canonical_through():
	series = pd.Series(CICIDS_CATEGORIES)
	assert list(cicids_normalize_labels(series)) == CICIDS_CATEGORIES


# ── CIC-IoT-2023: 8 classes = Neto's 7 attack groups + Benign, benign at 0 ──

def test_ciciot_class_structure():
	assert len(CICIOT_CLASSES) == 8
	assert CICIOT_CLASSES[0] == "Benign"  # benign index 0
	# Neto et al.'s own 33-attack → 7-category grouping (applied at HF dataset
	# build time via NETO_LABEL_MAP; attack_class column carries these values —
	# verified against neto-subsample random_3way on 11/07/2026).
	assert set(CICIOT_CLASSES) == {
		"Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based",
	}


# ── map_multiclass_labels: fallback + warning behavior ──

def test_map_multiclass_labels_maps_and_falls_back(capsys):
	class_to_idx = {"Benign": 0, "AttackA": 1, "AttackB": 2}
	series = pd.Series(["Benign", "AttackB", "Zzz-unknown", "AttackA"])
	mapped = map_multiclass_labels(series, class_to_idx, "train")
	assert mapped.dtype == np.int64
	assert list(mapped) == [0, 2, 0, 1]  # unknown → index 0 (benign)
	out = capsys.readouterr().out
	assert "[MULTICLASS]" in out and "Zzz-unknown" in out


def test_map_multiclass_labels_silent_when_all_mapped(capsys):
	class_to_idx = {"Benign": 0, "AttackA": 1}
	mapped = map_multiclass_labels(pd.Series(["AttackA", "Benign"]), class_to_idx, "test")
	assert list(mapped) == [1, 0]
	assert "[MULTICLASS]" not in capsys.readouterr().out


# ── End-to-end: encode_and_build_dataset multiclass label extraction ──

def test_encode_and_build_dataset_multiclass_labels():
	rng = np.random.default_rng(7)
	n = 60
	categories = ["Benign", "AttackA", "AttackB"]
	multi = np.array(categories)[rng.integers(0, 3, size=n)]
	df = pd.DataFrame({
		"feat_a": rng.normal(0, 1, size=n),
		"feat_b": rng.uniform(0, 10, size=n),
		"label": (multi != "Benign").astype(int),
		"attack_class": multi,
	})
	ds = encode_and_build_dataset(
		df.iloc[:40].copy(), df.iloc[40:].copy(), None,
		common_features=["feat_a", "feat_b"],
		top_features=[],
		category_names=categories,
		label_binary_col="label",
		label_multi_col="attack_class",
		n_bits=4,
	)
	assert ds.category_names == categories
	expected_train = [categories.index(c) for c in multi[:40]]
	expected_test = [categories.index(c) for c in multi[40:]]
	assert list(ds.y_train_multi) == expected_train
	assert list(ds.y_test_multi) == expected_test
	# binary/multi consistency: multi==0 ⇔ binary==0 (benign at index 0)
	assert ((ds.y_train_multi == 0) == (ds.y_train_binary == 0)).all()
