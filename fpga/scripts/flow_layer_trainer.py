"""FlowLayerTrainer: train a flow's genome EXACTLY as the worker's final validation did.

Shared by every FPGA export script. Evaluators come from the worker's own
IdsEvaluatorFactory (the flow's real dataset / split / encoding / feature
selection / neuron_sample_rate / memory_mode / OI / class weights / run seed);
training is the Rust IDSCacheWrapper.export_trained (the same training call as
evaluate_at_thresholds — the multiplicity layer at rate >= 1.0); the result
must pass the FidelityGate against the recorded DB row before anything is
written. No Python re-implementation of training or scoring anywhere.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fidelity_gate import FidelityGate  # noqa: E402
from flow_genome_source import FlowGenomeSource  # noqa: E402


@dataclass(frozen=True)
class TrainedFlowGenome:
	source: FlowGenomeSource
	settings: object        # IdsEvaluatorSettings
	layer: object           # TrainedLayerExport
	dataset: object         # the flow's IDSDataset (test rows for test vectors)
	labels: list[int]       # held-out labels the scores refer to
	modes_verified: int


class FlowLayerTrainer:

	def __init__(self, log=print):
		self._log = log

	def train(self, source: FlowGenomeSource) -> TrainedFlowGenome:
		from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
		from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
		settings = IdsEvaluatorSettings.from_params(source.params)
		self._require_exportable(settings)
		factory = IdsEvaluatorFactory(self._log)
		dataset = factory.load(settings)
		_, test_eval = factory.flat(dataset, settings)
		layer = self._export_with_flow_oi(test_eval, source, settings.order_independent_train)
		gate = FidelityGate(list(test_eval._y_test), test_eval._normal_class)
		modes = gate.require(layer.eval_scores, source.threshold_metadata)
		self._log(f"[FIDELITY] {modes} recorded threshold modes reproduced exactly")
		return TrainedFlowGenome(source, settings, layer, dataset, list(test_eval._y_test), modes)

	@staticmethod
	def _export_with_flow_oi(test_eval, source: FlowGenomeSource, oi: bool):
		"""Train with the env var the Rust trainer reads set EXACTLY to the
		flow's OI setting (on -> "1", off -> unset, whatever the caller's shell
		has), restoring the caller's environment afterwards."""
		import os
		from wnn.ram.experiments.IdsEvaluatorSettings import OI_ENV
		previous = os.environ.pop(OI_ENV, None)
		if oi:
			os.environ[OI_ENV] = "1"
		try:
			return test_eval.export_trained_layer(source.genome())
		finally:
			os.environ.pop(OI_ENV, None)
			if previous is not None:
				os.environ[OI_ENV] = previous

	@staticmethod
	def _require_exportable(settings) -> None:
		"""The deployed IDS designs are single-cluster binary discriminators."""
		if settings.classification != "binary" or not settings.single_cluster:
			raise NotImplementedError(
				f"FPGA export supports single-cluster binary flows only "
				f"(classification={settings.classification}, single_cluster={settings.single_cluster})")
