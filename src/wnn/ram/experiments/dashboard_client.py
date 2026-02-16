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
	import urllib3
	urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
	HAS_REQUESTS = True
except ImportError:
	HAS_REQUESTS = False


@dataclass
class DashboardClientConfig:
	"""Configuration for dashboard client.

	Args:
		base_url: Dashboard API URL (http:// or https://)
		timeout: Request timeout in seconds
		retry_count: Number of retry attempts on failure
		retry_delay: Delay between retries in seconds
		verify_ssl: SSL verification setting:
			- True: Verify SSL certificates (default, production)
			- False: Disable SSL verification (development with self-signed certs)
			- str: Path to CA certificate file for custom CA
	"""
	base_url: str = "https://localhost:3000"
	timeout: float = 30.0
	retry_count: int = 3
	retry_delay: float = 1.0
	verify_ssl: bool | str = False  # Default: skip verify (self-signed cert)


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
		# Configure SSL verification
		self._session.verify = self._config.verify_ssl

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
		"""Make HTTP request with retries.

		Returns None for 204 No Content or 404 Not Found.
		Raises ConnectionError for other failures.
		"""
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

				# 404 means resource doesn't exist - return None (not an error)
				if response.status_code == 404:
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
		seed_checkpoint_id: Optional[int] = None,
	) -> dict:
		"""Update flow fields."""
		data = {}
		if name is not None:
			data["name"] = name
		if description is not None:
			data["description"] = description
		if status is not None:
			data["status"] = status
		if seed_checkpoint_id is not None:
			data["seed_checkpoint_id"] = seed_checkpoint_id
		return self._request("PATCH", f"/api/flows/{flow_id}", json_data=data)

	def set_flow_checkpoint(self, flow_id: int, checkpoint_id: int) -> dict:
		"""Set the seed checkpoint for a flow (for resumption after stop)."""
		return self._request(
			"PATCH",
			f"/api/flows/{flow_id}",
			json_data={"seed_checkpoint_id": checkpoint_id}
		)

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

	def requeue_flow(self, flow_id: int) -> dict:
		"""Re-queue a flow for resumption after graceful shutdown."""
		return self.update_flow(flow_id, status="queued")

	def register_flow_pid(self, flow_id: int, pid: int) -> dict:
		"""Register the worker process PID for a flow.

		This enables stop/restart functionality from the dashboard.

		Args:
			flow_id: Flow ID
			pid: Process ID of the worker

		Returns:
			Response from server
		"""
		return self._request("PATCH", f"/api/flows/{flow_id}/pid", json_data={"pid": pid})

	def send_heartbeat(self, flow_id: int) -> bool:
		"""Send a heartbeat for a running flow.

		Called periodically by the worker to indicate the flow is still being processed.
		If the worker crashes or is killed, the missing heartbeat allows the dashboard
		to detect the stale flow and re-queue it.

		Args:
			flow_id: Flow ID

		Returns:
			True if heartbeat was recorded, False if flow not found
		"""
		result = self._request("POST", f"/api/flows/{flow_id}/heartbeat")
		return result is not None and result.get("success", False)

	def stop_flow(self, flow_id: int) -> dict:
		"""Stop a running flow.

		Sends SIGTERM to the worker process and sets status to cancelled.

		Args:
			flow_id: Flow ID to stop

		Returns:
			Updated flow data
		"""
		return self._request("POST", f"/api/flows/{flow_id}/stop")

	def restart_flow(self, flow_id: int, from_beginning: bool = False) -> dict:
		"""Restart a flow.

		Re-queues the flow for execution.

		Args:
			flow_id: Flow ID to restart
			from_beginning: If True, start fresh without checkpoint

		Returns:
			Updated flow data
		"""
		return self._request("POST", f"/api/flows/{flow_id}/restart", json_data={"from_beginning": from_beginning})

	def watch_log(self, log_path: str) -> dict:
		"""Tell the dashboard to watch a log file.

		Args:
			log_path: Absolute path to the log file

		Returns:
			Response from the server
		"""
		try:
			result = self._request("POST", "/api/watch", json_data={"log_path": log_path})
			self._logger(f"Dashboard now watching: {log_path}")
			return result
		except Exception as e:
			self._logger(f"Warning: Failed to set log watch path: {e}")
			return {}

	def list_flow_experiments(self, flow_id: int) -> list[dict]:
		"""List experiments associated with a flow."""
		return self._request("GET", f"/api/flows/{flow_id}/experiments")

	# =========================================================================
	# Experiment methods
	# =========================================================================

	def create_experiment(
		self,
		name: str,
		log_path: str = "",
		config: Optional[dict] = None,
	) -> int:
		"""
		Create a new experiment.

		Args:
			name: Experiment name
			log_path: Path to log file
			config: Optional experiment configuration

		Returns:
			Experiment ID
		"""
		data = {
			"name": name,
			"log_path": log_path,
			"config": config or {},
		}

		result = self._request("POST", "/api/experiments", json_data=data)
		exp_id = result["id"]
		self._logger(f"Created experiment {exp_id}: {name}")
		return exp_id

	def get_experiment(self, experiment_id: int) -> dict:
		"""Get experiment by ID."""
		return self._request("GET", f"/api/experiments/{experiment_id}")

	def update_experiment(
		self,
		experiment_id: int,
		name: Optional[str] = None,
		status: Optional[str] = None,
		best_ce: Optional[float] = None,
		best_accuracy: Optional[float] = None,
		current_iteration: Optional[int] = None,
		cluster_type: Optional[str] = None,
	) -> dict:
		"""
		Update an experiment.

		Args:
			experiment_id: Experiment ID
			name: New name
			status: New status ('pending', 'running', 'completed', 'failed', 'cancelled')
			best_ce: Best CE achieved
			best_accuracy: Best accuracy achieved
			current_iteration: Current iteration number
			cluster_type: Architecture type ('tiered' or 'bitwise')

		Returns:
			Updated experiment data
		"""
		data = {}
		if name is not None:
			data["name"] = name
		if status is not None:
			data["status"] = status
		if best_ce is not None:
			data["best_ce"] = best_ce
		if best_accuracy is not None:
			data["best_accuracy"] = best_accuracy
		if current_iteration is not None:
			data["current_iteration"] = current_iteration
		if cluster_type is not None:
			data["cluster_type"] = cluster_type
		return self._request("PATCH", f"/api/experiments/{experiment_id}", json_data=data)

	def experiment_started(self, experiment_id: int, cluster_type: Optional[str] = None) -> dict:
		"""Mark experiment as started/running."""
		return self.update_experiment(experiment_id, status="running", cluster_type=cluster_type)

	def experiment_completed(
		self,
		experiment_id: int,
		best_ce: Optional[float] = None,
		best_accuracy: Optional[float] = None,
	) -> dict:
		"""Mark experiment as completed with final metrics."""
		return self.update_experiment(
			experiment_id,
			status="completed",
			best_ce=best_ce,
			best_accuracy=best_accuracy,
		)

	def experiment_failed(self, experiment_id: int) -> dict:
		"""Mark experiment as failed."""
		return self.update_experiment(experiment_id, status="failed")

	def list_experiments(self, limit: int = 50, offset: int = 0) -> list[dict]:
		"""List experiments."""
		params = {"limit": limit, "offset": offset}
		return self._request("GET", "/api/experiments", params=params)

	def link_experiment_to_flow(
		self,
		flow_id: int,
		experiment_id: int,
		sequence_order: int = 0,
	) -> None:
		"""
		Link an experiment to a flow.

		Args:
			flow_id: Flow ID
			experiment_id: Experiment ID to link
			sequence_order: Order of experiment in the flow (default 0)
		"""
		data = {
			"experiment_id": experiment_id,
			"sequence_order": sequence_order,
		}
		self._request("POST", f"/api/flows/{flow_id}/experiments", json_data=data)
		self._logger(f"Linked experiment {experiment_id} to flow {flow_id}")

	# =========================================================================
	# Validation Summary methods
	# =========================================================================

	def check_cached_validation(self, genome_hash: str) -> Optional[tuple[float, float]]:
		"""
		Check if a genome has already been validated.

		Args:
			genome_hash: The genome's config hash

		Returns:
			Tuple of (ce, accuracy) if found, None if not validated yet
		"""
		try:
			result = self._request("GET", "/api/validations/check", params={"genome_hash": genome_hash})
			if result.get("found"):
				return (result["ce"], result["accuracy"])
			return None
		except Exception:
			return None

	def create_validation_summary(
		self,
		experiment_id: int,
		validation_point: str,  # 'init' or 'final'
		genome_type: str,       # 'best_ce', 'best_acc', 'best_fitness'
		genome_hash: str,
		ce: float,
		accuracy: float,
		flow_id: Optional[int] = None,
	) -> dict:
		"""
		Create a validation summary record for a genome at a checkpoint.

		Args:
			experiment_id: Experiment ID
			validation_point: 'init' (start of experiment) or 'final' (end of experiment)
			genome_type: 'best_ce', 'best_acc', or 'best_fitness'
			genome_hash: The genome's config hash (for deduplication)
			ce: Cross-entropy value
			accuracy: Accuracy value
			flow_id: Optional flow ID

		Returns:
			Dict with 'id' of the created/updated summary
		"""
		data = {
			"flow_id": flow_id,
			"validation_point": validation_point,
			"genome_type": genome_type,
			"genome_hash": genome_hash,
			"ce": ce,
			"accuracy": accuracy,
		}
		result = self._request("POST", f"/api/experiments/{experiment_id}/summaries", json_data=data)
		self._logger(f"Created {validation_point}/{genome_type} validation for experiment {experiment_id}")
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
		iteration_id: Optional[int] = None,
		checkpoint_type: str = "auto",
	) -> int:
		"""
		Register a checkpoint with the dashboard.

		Args:
			experiment_id: ID of the parent experiment
			file_path: Path to the checkpoint file
			name: Human-readable name
			final_fitness: CE loss value (best_ce)
			final_accuracy: Accuracy value (best_accuracy)
			iterations_run: Number of iterations completed (unused by API)
			genome_stats: Genome statistics including tier_stats for per-tier averages
			is_final: Whether this is the final checkpoint for the experiment
			iteration_id: Optional iteration ID
			checkpoint_type: Type of checkpoint ('auto', 'user', 'phase_end', 'experiment_end')

		Returns:
			Checkpoint ID
		"""
		# Get file size
		file_size = None
		if os.path.exists(file_path):
			file_size = os.path.getsize(file_path)

		# Build request data matching Rust API's CreateCheckpointRequest
		data = {
			"experiment_id": experiment_id,
			"name": name,
			"file_path": file_path,
			"file_size_bytes": file_size,
			"best_ce": final_fitness,        # API expects best_ce, not final_fitness
			"best_accuracy": final_accuracy,  # API expects best_accuracy, not final_accuracy
			"checkpoint_type": checkpoint_type,
			"iteration_id": iteration_id,
			"genome_stats": genome_stats,    # Per-tier stats and genome info
		}

		result = self._request("POST", "/api/checkpoints", json_data=data)
		ckpt_id = result["id"]
		self._logger(f"Registered checkpoint {ckpt_id}: {name} (type={checkpoint_type})")
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

	def find_checkpoint_by_path(self, file_path: str) -> Optional[int]:
		"""
		Find checkpoint ID by file path.

		Args:
			file_path: Path to checkpoint file

		Returns:
			Checkpoint ID if found, None otherwise
		"""
		# List checkpoints and find matching path
		checkpoints = self.list_checkpoints(limit=100)
		for ckpt in checkpoints:
			if ckpt.get("file_path") == file_path:
				return ckpt.get("id")
		return None

	# =========================================================================
	# Gating Run methods (resource-based API)
	# =========================================================================

	def get_pending_gating_runs(self) -> list[dict]:
		"""
		Get all pending gating runs across all experiments.

		Returns:
			List of GatingRun objects with status='pending'
		"""
		result = self._request("GET", "/api/gating/pending")
		return result if result else []

	def list_gating_runs(self, experiment_id: int) -> list[dict]:
		"""
		List all gating runs for an experiment.

		Args:
			experiment_id: Experiment ID

		Returns:
			List of GatingRun objects
		"""
		result = self._request("GET", f"/api/experiments/{experiment_id}/gating")
		return result if result else []

	def get_gating_run(self, experiment_id: int, gating_run_id: int) -> Optional[dict]:
		"""
		Get a specific gating run.

		Args:
			experiment_id: Experiment ID
			gating_run_id: Gating run ID

		Returns:
			GatingRun object or None if not found
		"""
		return self._request("GET", f"/api/experiments/{experiment_id}/gating/{gating_run_id}")

	def create_gating_run(self, experiment_id: int) -> dict:
		"""
		Create a new gating run for an experiment.

		Args:
			experiment_id: Experiment ID (must be completed)

		Returns:
			Created GatingRun object with id
		"""
		result = self._request("POST", f"/api/experiments/{experiment_id}/gating")
		if result:
			self._logger(f"Created gating run {result.get('id')} for experiment {experiment_id}")
		return result or {}

	def update_gating_run_status(
		self,
		experiment_id: int,
		gating_run_id: int,
		status: str,
	) -> dict:
		"""
		Update the status of a gating run.

		Args:
			experiment_id: Experiment ID
			gating_run_id: Gating run ID
			status: 'pending', 'running', 'completed', 'failed'

		Returns:
			Updated GatingRun object
		"""
		result = self._request(
			"PATCH",
			f"/api/experiments/{experiment_id}/gating/{gating_run_id}",
			json_data={"status": status}
		)
		self._logger(f"Updated gating run {gating_run_id} status to {status}")
		return result or {}

	def update_gating_run_results(
		self,
		experiment_id: int,
		gating_run_id: int,
		results: list[dict],
		genomes_tested: Optional[int] = None,
		error: Optional[str] = None,
	) -> dict:
		"""
		Update a gating run with results (completes the run).

		Args:
			experiment_id: Experiment ID
			gating_run_id: Gating run ID
			results: List of per-genome result dicts with keys:
				genome_type, ce, acc, gated_ce, gated_acc, gating_config
			genomes_tested: Number of genomes tested (defaults to len(results))
			error: Optional error message if failed

		Returns:
			Updated GatingRun object
		"""
		data = {
			"results": results,
			"genomes_tested": genomes_tested or len(results),
		}
		if error:
			data["error"] = error

		result = self._request(
			"PATCH",
			f"/api/experiments/{experiment_id}/gating/{gating_run_id}",
			json_data=data
		)
		self._logger(f"Updated gating run {gating_run_id} with {len(results)} genome results")
		return result or {}

	# =========================================================================
	# HuggingFace export
	# =========================================================================

	def export_genome_hf(
		self,
		checkpoint_id: int,
		output_dir: str = "exports",
	) -> dict:
		"""
		Export a checkpoint's genome as a HuggingFace-compatible model.

		This triggers the export process:
		1. Fetches checkpoint + experiment metadata from the API
		2. Loads the genome from the checkpoint file
		3. Trains memory tables on the full dataset
		4. Saves as HF-compatible directory (config.json + model.safetensors)

		Args:
			checkpoint_id: ID of the checkpoint to export
			output_dir: Directory to save the exported model

		Returns:
			Dict with export metadata including 'output_dir' path
		"""
		# Step 1: Get export metadata from API
		result = self._request(
			"POST",
			f"/api/checkpoints/{checkpoint_id}/export-hf",
			json_data={"output_dir": output_dir}
		)
		if result is None:
			raise ValueError(f"Checkpoint {checkpoint_id} not found")

		self._logger(f"Export metadata received for checkpoint {checkpoint_id}")

		# Step 2: Load genome from checkpoint file
		checkpoint_path = result["checkpoint_path"]
		architecture_type = result.get("architecture_type", "tiered")
		context_size = result.get("context_size", 4)

		import gzip
		genome_data = None
		try:
			if checkpoint_path.endswith(".gz"):
				with gzip.open(checkpoint_path, "rt") as f:
					genome_data = json.load(f)
			else:
				with open(checkpoint_path, "r") as f:
					genome_data = json.load(f)
		except (FileNotFoundError, OSError) as e:
			raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}") from e

		# Step 3: Create HF config and model
		try:
			from wnn.hf import WNNConfig, WNNForCausalLM
		except ImportError:
			raise ImportError(
				"HuggingFace dependencies required for export. "
				"Install with: pip install wnn[hf]"
			)

		# Build config from genome data
		config = WNNConfig.from_genome(genome_data, context_size=context_size)

		# Create model and load genome weights
		model = WNNForCausalLM(config)
		model.load_from_genome(genome_data)

		# Step 4: Save to HF-compatible directory
		export_path = Path(output_dir) / f"wnn-{architecture_type}-ckpt{checkpoint_id}"
		export_path.mkdir(parents=True, exist_ok=True)

		model.save_pretrained(str(export_path))

		# Save tokenizer config
		from wnn.hf.tokenization_wnn import WNNTokenizerConfig
		tokenizer_config = WNNTokenizerConfig()
		tokenizer_config.save_pretrained(str(export_path))

		self._logger(f"Exported HF model to {export_path}")

		return {
			"checkpoint_id": checkpoint_id,
			"output_dir": str(export_path),
			"architecture_type": architecture_type,
			"best_ce": result.get("best_ce"),
			"best_accuracy": result.get("best_accuracy"),
		}

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
