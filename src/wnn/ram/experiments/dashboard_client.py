"""
HTTP client for communicating with the WNN Dashboard API.

Provides methods for:
- Creating and managing flows
- Registering and querying checkpoints
- Sending lifecycle events (flow started, completed, etc.)
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urljoin

try:
	import requests
	HAS_REQUESTS = True
except ImportError:
	HAS_REQUESTS = False


@dataclass
class DashboardClientConfig:
	"""Configuration for dashboard client."""
	base_url: str = "http://localhost:3000"
	timeout: float = 30.0
	retry_count: int = 3
	retry_delay: float = 1.0


@dataclass
class FlowConfig:
	"""Configuration for a flow (sequence of experiments)."""
	name: str
	experiments: list[dict[str, Any]]
	template: Optional[str] = None
	description: Optional[str] = None
	params: Optional[dict[str, Any]] = None

	@classmethod
	def standard_6_phase(
		cls,
		name: str,
		ga_generations: int = 250,
		ts_iterations: int = 250,
		population_size: int = 50,
		neighbors_per_iter: int = 50,
		patience: int = 10,
		phase_order: str = "neurons_first",
		**kwargs,
	) -> "FlowConfig":
		"""
		Create a standard 6-phase flow configuration.

		Phase order depends on phase_order parameter:
		- "neurons_first": neurons -> bits -> connections
		- "bits_first": bits -> neurons -> connections
		"""
		if phase_order == "bits_first":
			phases = [
				("Phase 1a: GA Bits", "ga", True, False, False),
				("Phase 1b: TS Bits", "ts", True, False, False),
				("Phase 2a: GA Neurons", "ga", False, True, False),
				("Phase 2b: TS Neurons", "ts", False, True, False),
				("Phase 3a: GA Connections", "ga", False, False, True),
				("Phase 3b: TS Connections", "ts", False, False, True),
			]
		else:
			phases = [
				("Phase 1a: GA Neurons", "ga", False, True, False),
				("Phase 1b: TS Neurons", "ts", False, True, False),
				("Phase 2a: GA Bits", "ga", True, False, False),
				("Phase 2b: TS Bits", "ts", True, False, False),
				("Phase 3a: GA Connections", "ga", False, False, True),
				("Phase 3b: TS Connections", "ts", False, False, True),
			]

		experiments = []
		for phase_name, exp_type, opt_bits, opt_neurons, opt_conns in phases:
			exp = {
				"name": phase_name,
				"experiment_type": exp_type,
				"optimize_bits": opt_bits,
				"optimize_neurons": opt_neurons,
				"optimize_connections": opt_conns,
				"params": {
					"patience": patience,
				},
			}
			if exp_type == "ga":
				exp["params"]["generations"] = ga_generations
				exp["params"]["population_size"] = population_size
			else:
				exp["params"]["iterations"] = ts_iterations
				exp["params"]["neighbors_per_iter"] = neighbors_per_iter
			experiments.append(exp)

		return cls(
			name=name,
			experiments=experiments,
			template="standard-6-phase",
			params={
				"phase_order": phase_order,
				**kwargs,
			},
		)


class DashboardClient:
	"""
	HTTP client for the WNN Dashboard API.

	Example usage:
		client = DashboardClient()

		# Create a flow
		flow_config = FlowConfig.standard_6_phase("Pass 1", patience=10)
		flow_id = client.create_flow(flow_config)

		# Notify flow started
		client.flow_started(flow_id)

		# Register checkpoints as they're created
		ckpt_id = client.checkpoint_created(
			experiment_id=1,
			file_path="/path/to/checkpoint.json.gz",
			name="Phase 1a Final",
			final_fitness=10.5,
			final_accuracy=0.05,
		)

		# Complete flow
		client.flow_completed(flow_id)
	"""

	def __init__(
		self,
		config: Optional[DashboardClientConfig] = None,
		logger: Optional[Callable[[str], None]] = None,
	):
		if not HAS_REQUESTS:
			raise ImportError("requests library required for DashboardClient. Install with: pip install requests")

		self._config = config or DashboardClientConfig()
		self._logger = logger or (lambda x: None)
		self._session = requests.Session()
		self._session.headers.update({
			"Content-Type": "application/json",
			"Accept": "application/json",
		})

	def _url(self, path: str) -> str:
		"""Build full URL from path."""
		return urljoin(self._config.base_url, path)

	def _request(
		self,
		method: str,
		path: str,
		json_data: Optional[dict] = None,
		params: Optional[dict] = None,
	) -> Optional[dict]:
		"""Make HTTP request with retries."""
		import time

		url = self._url(path)
		last_error = None

		for attempt in range(self._config.retry_count):
			try:
				response = self._session.request(
					method=method,
					url=url,
					json=json_data,
					params=params,
					timeout=self._config.timeout,
				)

				if response.status_code == 204:
					return None

				response.raise_for_status()
				return response.json()

			except requests.RequestException as e:
				last_error = e
				self._logger(f"Request failed (attempt {attempt + 1}): {e}")
				if attempt < self._config.retry_count - 1:
					time.sleep(self._config.retry_delay)

		raise ConnectionError(f"Failed after {self._config.retry_count} attempts: {last_error}")

	# =========================================================================
	# Flow methods
	# =========================================================================

	def create_flow(
		self,
		config: FlowConfig,
		seed_checkpoint_id: Optional[int] = None,
	) -> int:
		"""
		Create a new flow.

		Args:
			config: Flow configuration
			seed_checkpoint_id: Optional checkpoint to seed from

		Returns:
			Flow ID
		"""
		data = {
			"name": config.name,
			"description": config.description,
			"config": {
				"experiments": config.experiments,
				"template": config.template,
				"params": config.params or {},
			},
			"seed_checkpoint_id": seed_checkpoint_id,
		}

		result = self._request("POST", "/api/flows", json_data=data)
		flow_id = result["id"]
		self._logger(f"Created flow {flow_id}: {config.name}")
		return flow_id

	def list_flows(
		self,
		status: Optional[str] = None,
		limit: int = 20,
		offset: int = 0,
	) -> list[dict]:
		"""List flows with optional status filter."""
		params = {"limit": limit, "offset": offset}
		if status:
			params["status"] = status
		return self._request("GET", "/api/flows", params=params)

	def get_flow(self, flow_id: int) -> dict:
		"""Get flow by ID."""
		return self._request("GET", f"/api/flows/{flow_id}")

	def update_flow(
		self,
		flow_id: int,
		name: Optional[str] = None,
		description: Optional[str] = None,
		status: Optional[str] = None,
	) -> dict:
		"""Update flow fields."""
		data = {}
		if name is not None:
			data["name"] = name
		if description is not None:
			data["description"] = description
		if status is not None:
			data["status"] = status
		return self._request("PATCH", f"/api/flows/{flow_id}", json_data=data)

	def delete_flow(self, flow_id: int) -> None:
		"""Delete a flow."""
		self._request("DELETE", f"/api/flows/{flow_id}")
		self._logger(f"Deleted flow {flow_id}")

	def flow_started(self, flow_id: int) -> dict:
		"""Mark flow as started."""
		return self.update_flow(flow_id, status="running")

	def flow_completed(self, flow_id: int) -> dict:
		"""Mark flow as completed."""
		return self.update_flow(flow_id, status="completed")

	def flow_failed(self, flow_id: int, error: Optional[str] = None) -> dict:
		"""Mark flow as failed."""
		result = self.update_flow(flow_id, status="failed")
		if error:
			self._logger(f"Flow {flow_id} failed: {error}")
		return result

	# =========================================================================
	# Checkpoint methods
	# =========================================================================

	def checkpoint_created(
		self,
		experiment_id: int,
		file_path: str,
		name: str,
		final_fitness: Optional[float] = None,
		final_accuracy: Optional[float] = None,
		iterations_run: Optional[int] = None,
		genome_stats: Optional[dict] = None,
		is_final: bool = False,
	) -> int:
		"""
		Register a checkpoint with the dashboard.

		Args:
			experiment_id: ID of the parent experiment
			file_path: Path to the checkpoint file
			name: Human-readable name
			final_fitness: CE loss value
			final_accuracy: Accuracy value
			iterations_run: Number of iterations completed
			genome_stats: Optional genome statistics
			is_final: Whether this is the final checkpoint for the experiment

		Returns:
			Checkpoint ID
		"""
		# Get file size
		file_size = None
		if os.path.exists(file_path):
			file_size = os.path.getsize(file_path)

		data = {
			"experiment_id": experiment_id,
			"name": name,
			"file_path": file_path,
			"file_size_bytes": file_size,
			"final_fitness": final_fitness,
			"final_accuracy": final_accuracy,
			"iterations_run": iterations_run,
			"genome_stats": genome_stats,
			"is_final": is_final,
		}

		result = self._request("POST", "/api/checkpoints", json_data=data)
		ckpt_id = result["id"]
		self._logger(f"Registered checkpoint {ckpt_id}: {name}")
		return ckpt_id

	def list_checkpoints(
		self,
		experiment_id: Optional[int] = None,
		is_final: Optional[bool] = None,
		limit: int = 50,
		offset: int = 0,
	) -> list[dict]:
		"""List checkpoints with optional filters."""
		params = {"limit": limit, "offset": offset}
		if experiment_id is not None:
			params["experiment_id"] = experiment_id
		if is_final is not None:
			params["is_final"] = is_final
		return self._request("GET", "/api/checkpoints", params=params)

	def get_checkpoint(self, checkpoint_id: int) -> dict:
		"""Get checkpoint by ID."""
		return self._request("GET", f"/api/checkpoints/{checkpoint_id}")

	def delete_checkpoint(self, checkpoint_id: int, force: bool = False) -> None:
		"""
		Delete a checkpoint.

		Args:
			checkpoint_id: ID of checkpoint to delete
			force: If True, delete even if referenced by other experiments
		"""
		params = {"force": force} if force else {}
		self._request("DELETE", f"/api/checkpoints/{checkpoint_id}", params=params)
		self._logger(f"Deleted checkpoint {checkpoint_id}")

	# =========================================================================
	# Health check
	# =========================================================================

	def ping(self) -> bool:
		"""Check if dashboard is reachable."""
		try:
			self._request("GET", "/api/flows", params={"limit": 1})
			return True
		except Exception:
			return False
