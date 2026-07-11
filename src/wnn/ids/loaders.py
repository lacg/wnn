"""Unified IDS dataset loading via the template-method pattern.

One pipeline (fetch → encode → build `IDSDataset`), three datasets. `BaseIDSLoader`
owns the shared pipeline (`load()`); each subclass declares only its dataset-specific
bits — HF fetch, feature list, label columns, categories, raw support. Adding a new
dataset is a small subclass + one registry entry; changing one dataset is a localized
edit to its subclass. Entry point: `load_ids_dataset(name, **kwargs)`.

Import discipline (avoids circular imports): this module top-imports only `dataset`
(the base module, which never imports this one). The CICIDS/CIC-IoT helpers are
lazy-imported inside the subclass methods, and the public `load_*` compat wrappers in
the three dataset modules lazy-import from here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .dataset import (
	IDSDataset,
	ThermometerType,
	encode_and_build_dataset,
	# UNSW lives in dataset.py, so its helpers/constants are safe to import at top.
	TOP20_RF_FEATURES as _UNSW_TOP20,
	ATTACK_CATEGORIES as _UNSW_CATS,
	_CATEGORY_ALIASES as _UNSW_ALIASES,
	_load_from_huggingface as _unsw_hf,
)


@dataclass
class LoadSpec:
	"""The uniform load interface — identical for every dataset."""
	split: str = "random"
	n_bits: object = 8                 # int OR "auto"
	method: ThermometerType = ThermometerType.DISTRIBUTIVE
	feature_selection: str = "all"
	rest_bits: Optional[int] = None
	auto_max_bits: int = 32
	invalid_encoding: Optional[str] = None
	encoded_storage: str = "memory"
	storage_dir: object = None
	raw: bool = False


class BaseIDSLoader(ABC):
	"""Template method: subclasses declare specifics; `load()` runs the pipeline."""

	name: str = ""
	valid_splits: tuple = ("random",)
	attack_categories: list = []   # overridden per subclass (plain class attr)
	binary_col: str = "label"
	multiclass_col: Optional[str] = "attack_class"
	supports_raw: bool = True
	intrinsically_raw: bool = False    # dataset preserves NaN/inf by construction

	# Datasets with a STATIC feature list set this; others override `_feature_list`.
	_FEATURES: list = []

	@abstractmethod
	def _fetch(self, spec: "LoadSpec"):
		"""Dataset-specific. Return (df_train, df_test, common_features, df_val)."""

	def _feature_list(self, spec: "LoadSpec") -> list:
		return self._FEATURES

	def _resolve_invalid(self, spec: "LoadSpec") -> str:
		if spec.invalid_encoding is not None:
			return spec.invalid_encoding
		return "single_bit" if (spec.raw or self.intrinsically_raw) else "none"

	def load(self, spec: "LoadSpec") -> IDSDataset:
		if spec.split not in self.valid_splits:
			raise ValueError(f"{self.name}: split must be one of {self.valid_splits}, got {spec.split!r}")
		if spec.raw and not self.supports_raw:
			raise NotImplementedError(f"{self.name} has no raw variant (its HuggingFace copy is pre-cleaned)")
		feats = self._feature_list(spec)
		inv = self._resolve_invalid(spec)

		# Chained build: the DataFrames live ONLY as encode_and_build_dataset params,
		# so its inline `del` actually frees them (refcount → 0). Preserves the
		# ~10 GB memory drop on 46M datasets.
		def _build(df_train, df_test, common_features, df_val):
			return encode_and_build_dataset(
				df_train, df_test, df_val,
				common_features=common_features,
				top_features=feats or [],
				category_names=self.attack_categories,
				label_binary_col=self.binary_col,
				label_multi_col=self.multiclass_col,
				n_bits=spec.n_bits, method=spec.method,
				feature_selection=spec.feature_selection,
				rest_bits=spec.rest_bits, invalid_encoding=inv,
				auto_max_bits=spec.auto_max_bits,
				encoded_storage=spec.encoded_storage, storage_dir=spec.storage_dir,
			)

		return _build(*self._fetch(spec))


class UNSWLoader(BaseIDSLoader):
	name = "unsw-nb15"
	valid_splits = ("temporal", "random", "temporal_3way", "random_3way")
	attack_categories = _UNSW_CATS
	binary_col = "label"
	multiclass_col = "attack_cat"
	supports_raw = False            # #1: HF copy is pre-cleaned; no raw variant
	_FEATURES = _UNSW_TOP20

	def _fetch(self, spec):
		# #3: HF-only (the data_dir / local-CSV fallback was removed).
		df_train, df_test, common, df_val = _unsw_hf(spec.split)
		# Normalize attack_cat to canonical names so the simple cat→idx map in
		# encode_and_build_dataset matches (UNSW's alias quirk, isolated here).
		def _norm(s):
			v = s.fillna("Normal").astype(str).str.strip().replace("", "Normal")
			return v.map(lambda x: _UNSW_ALIASES.get(x.strip().lower(), x.strip()))
		for d in (df_train, df_test, df_val):
			if d is not None and "attack_cat" in d.columns:
				d["attack_cat"] = _norm(d["attack_cat"])
		return df_train, df_test, common, df_val


class CICIDSLoader(BaseIDSLoader):
	name = "cicids2017"
	valid_splits = ("temporal", "random", "temporal_3way", "random_3way")
	binary_col = "label"
	multiclass_col = "Label"

	@property
	def attack_categories(self):
		from .cicids2017 import ATTACK_CATEGORIES
		return ATTACK_CATEGORIES

	def _feature_list(self, spec):
		from .cicids2017 import TOP20_RF_FEATURES
		return TOP20_RF_FEATURES

	def _fetch(self, spec):
		from .cicids2017 import _load_from_huggingface, normalize_labels
		df_train, df_test, common, df_val = _load_from_huggingface(spec.split, raw=spec.raw)
		# Normalize Label mojibake ("Web Attack � ..." → "Web Attack - ...")
		# so the cat→idx map in encode_and_build_dataset matches — otherwise
		# those rows silently become BENIGN in the multiclass path (CICIDS
		# quirk, isolated here; mirrors the UNSW alias normalization above).
		for d in (df_train, df_test, df_val):
			if d is not None and "Label" in d.columns:
				d["Label"] = normalize_labels(d["Label"])
		return df_train, df_test, common, df_val


class CICIoTLoader(BaseIDSLoader):
	name = "ciciot2023"
	valid_splits = ("random", "random_3way")
	binary_col = "label"
	multiclass_col = "attack_class"
	intrinsically_raw = True        # neto variants preserve NaN/inf by construction

	# #4: only the two canonical Neto variants survive; the bencorn-derived
	# subsample/full/canonical are deprecated.
	_VALID_SIZES = ("neto_full", "neto_subsample")

	def __init__(self, dataset_size: str = "neto_subsample"):
		if dataset_size not in self._VALID_SIZES:
			raise ValueError(
				f"dataset_size {dataset_size!r} is deprecated/removed; use one of "
				f"{self._VALID_SIZES} (the bencorn subsample/full/canonical were dropped)."
			)
		self.dataset_size = dataset_size

	@property
	def attack_categories(self):
		from .ciciot2023 import ATTACK_CLASSES
		return ATTACK_CLASSES

	def _feature_list(self, spec):
		from .ciciot2023 import FEATURE_LIST_BY_NAME, _load_ranked_features
		if spec.feature_selection in FEATURE_LIST_BY_NAME:
			return FEATURE_LIST_BY_NAME[spec.feature_selection]
		feats = _load_ranked_features()
		if not feats and spec.feature_selection in ("top15", "top25", "top20_split"):
			raise ValueError("Could not load top-N features from HuggingFace")
		return feats

	def _fetch(self, spec):
		from .ciciot2023 import _load_from_huggingface
		return _load_from_huggingface(spec.split, self.dataset_size, raw=spec.raw)

	def load(self, spec):
		# CIC-IoT-only: the streaming path uses a different builder (never
		# materializes full DataFrames). Delegate; everything else uses the base.
		if getattr(spec, "streaming", False):
			return self._load_streaming(spec)
		return super().load(spec)

	def _load_streaming(self, spec):
		from .ciciot2023 import _load_from_huggingface_streaming
		from .dataset import encode_and_build_dataset_streaming
		if spec.encoded_storage != "memory":
			raise NotImplementedError(
				f"streaming=True is incompatible with encoded_storage={spec.encoded_storage!r}."
			)
		feats = self._feature_list(spec)
		inv = self._resolve_invalid(spec)
		(train_f, test_f, val_f, n_tr, n_te, n_va, common) = _load_from_huggingface_streaming(
			spec.split, self.dataset_size, raw=spec.raw,
			chunk_size=getattr(spec, "streaming_chunk_size", 1_000_000),
		)
		return encode_and_build_dataset_streaming(
			train_f, test_f, val_f, n_tr, n_te, n_va,
			common_features=common, top_features=feats or [],
			category_names=self.attack_categories,
			label_binary_col=self.binary_col, label_multi_col=self.multiclass_col,
			n_bits=spec.n_bits, method=spec.method, feature_selection=spec.feature_selection,
			rest_bits=spec.rest_bits, invalid_encoding=inv, auto_max_bits=spec.auto_max_bits,
		)


# Registry: dataset name → factory. Adding a dataset = one line here + a subclass.
_REGISTRY = {
	"unsw-nb15": UNSWLoader,
	"cicids2017": CICIDSLoader,
	"ciciot2023_neto_full": lambda: CICIoTLoader("neto_full"),
	"ciciot2023_neto_subsample": lambda: CICIoTLoader("neto_subsample"),
}


def get_loader(dataset: str) -> BaseIDSLoader:
	"""Instantiate the loader for a dataset name."""
	if dataset not in _REGISTRY:
		raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(_REGISTRY)}")
	return _REGISTRY[dataset]()


def load_ids_dataset(dataset: str, **kwargs) -> IDSDataset:
	"""Unified entry point. `dataset` is a registry key; `kwargs` map to LoadSpec
	(split, n_bits, method, feature_selection, rest_bits, auto_max_bits,
	invalid_encoding, encoded_storage, storage_dir, raw). Streaming kwargs
	(streaming, streaming_chunk_size) are accepted and forwarded to CIC-IoT."""
	# streaming is CIC-IoT-only and not a LoadSpec field; stash it on the spec.
	streaming = kwargs.pop("streaming", False)
	streaming_chunk_size = kwargs.pop("streaming_chunk_size", 1_000_000)
	spec = LoadSpec(**kwargs)
	spec.streaming = streaming                      # type: ignore[attr-defined]
	spec.streaming_chunk_size = streaming_chunk_size  # type: ignore[attr-defined]
	return get_loader(dataset).load(spec)


__all__ = ["LoadSpec", "BaseIDSLoader", "UNSWLoader", "CICIDSLoader",
           "CICIoTLoader", "get_loader", "load_ids_dataset"]
