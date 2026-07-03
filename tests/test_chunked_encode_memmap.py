"""Parity tests for encode_df_to_memmap (chunked encode → memmap).

03/07/2026: the one-shot ThermometerEncoder.transform() on 46M-row frames
materializes ~n_rows × total_bits bool intermediates (~150 GB at 96b × 20f)
and gets SIGKILLed. encode_df_to_memmap streams encoder.iter_chunks() slabs
into a pre-allocated memmap instead. These tests pin the byte-exact parity
between the two paths, including uneven chunk boundaries and invalid values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))

from wnn.ids.encoded_array import MemmapEncoded, encode_df_to_memmap
from wnn.representations.thermometer import ThermometerEncoder, ThermometerType


def _make_df(n_rows: int, seed: int = 7) -> pd.DataFrame:
	"""Mixed-type frame: numeric (with NaN/Inf), binary, categorical (with NaN)."""
	rng = np.random.default_rng(seed)
	num_a = rng.normal(0.0, 5.0, n_rows)
	num_a[rng.random(n_rows) < 0.03] = np.nan
	num_a[rng.random(n_rows) < 0.02] = np.inf
	num_b = rng.exponential(2.0, n_rows)
	binary = rng.integers(0, 2, n_rows)
	cats = rng.choice(["tcp", "udp", "icmp", "other"], n_rows).astype(object)
	cats[rng.random(n_rows) < 0.05] = None
	return pd.DataFrame({"num_a": num_a, "num_b": num_b, "flag": binary, "proto": cats})


def _fit_encoder(df: pd.DataFrame, n_bits: int = 12) -> ThermometerEncoder:
	enc = ThermometerEncoder(n_bits=n_bits, method=ThermometerType.DISTRIBUTIVE)
	enc.fit(df)
	return enc


def test_chunked_memmap_matches_oneshot_transform(tmp_path):
	"""Byte-exact parity: chunked memmap == one-shot transform (uneven chunks)."""
	df = _make_df(1000)
	enc = _fit_encoder(df)
	packed_oneshot, total_bits = enc.transform(df)

	# chunk_size=333 forces uneven boundaries (333+333+333+1)
	mm = encode_df_to_memmap(enc, df, storage_dir=tmp_path, chunk_size=333)
	assert isinstance(mm, MemmapEncoded)
	assert mm.n_rows == len(df)
	assert mm.total_bits == total_bits
	assert np.array_equal(np.asarray(mm.as_packed_uint8()), packed_oneshot), \
		"chunked memmap bytes must equal one-shot transform bytes"


def test_chunked_memmap_single_oversized_chunk(tmp_path):
	"""chunk_size > n_rows degenerates to one chunk — still exact."""
	df = _make_df(57, seed=11)
	enc = _fit_encoder(df, n_bits=8)
	packed_oneshot, total_bits = enc.transform(df)
	mm = encode_df_to_memmap(enc, df, storage_dir=tmp_path, chunk_size=10_000)
	assert np.array_equal(np.asarray(mm.as_packed_uint8()), packed_oneshot)
	assert mm.total_bits == total_bits


def test_chunked_memmap_row_access_and_bool_roundtrip(tmp_path):
	"""Row reads through the memmap decode to the same bools as the packed truth."""
	df = _make_df(200, seed=23)
	enc = _fit_encoder(df)
	packed_oneshot, total_bits = enc.transform(df)
	truth_bool = np.unpackbits(packed_oneshot, axis=1, bitorder='little')[:, :total_bits].astype(bool)

	mm = encode_df_to_memmap(enc, df, storage_dir=tmp_path, chunk_size=64)
	got_bool = mm.to_numpy_bool()
	assert np.array_equal(got_bool, truth_bool)

	idx = np.array([0, 3, 199, 42])
	assert np.array_equal(np.asarray(mm[idx]), packed_oneshot[idx])


def test_tmp_suffix_cleanup(tmp_path):
	""".tmp memmap files from the chunked path are removed on __del__."""
	df = _make_df(50, seed=3)
	enc = _fit_encoder(df, n_bits=6)
	mm = encode_df_to_memmap(enc, df, storage_dir=tmp_path, chunk_size=16)
	path = mm.path
	assert path.exists() and path.suffix == ".tmp"
	del mm
	assert not path.exists(), ".tmp file must be cleaned up on __del__"


if __name__ == "__main__":
	import tempfile
	for fn in (
		test_chunked_memmap_matches_oneshot_transform,
		test_chunked_memmap_single_oversized_chunk,
		test_chunked_memmap_row_access_and_bool_roundtrip,
		test_tmp_suffix_cleanup,
	):
		with tempfile.TemporaryDirectory() as d:
			fn(Path(d))
		print(f"PASS {fn.__name__}")
