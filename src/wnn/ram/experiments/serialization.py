"""
Serialization protocol and classes for experiment persistence.

Provides a common interface for serializing/deserializing experiment
artifacts like genomes, populations, and checkpoints.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

# Type variable for generic serializable types
T = TypeVar('T', bound='Serializable')


class Serializable(ABC):
	"""
	Protocol for objects that can be serialized to/from dictionaries and files.

	Implementing classes must define serialize() and deserialize() methods.
	The save() and load() methods provide default JSON file I/O.
	"""

	@abstractmethod
	def serialize(self) -> dict[str, Any]:
		"""
		Serialize the object to a dictionary.

		Returns:
			Dictionary representation suitable for JSON serialization.
		"""
		pass

	@classmethod
	@abstractmethod
	def deserialize(cls: type[T], data: dict[str, Any]) -> T:
		"""
		Deserialize an object from a dictionary.

		Args:
			data: Dictionary representation from serialize()

		Returns:
			Reconstructed object instance
		"""
		pass

	def save(self, filepath: str, **metadata: Any) -> None:
		"""
		Save the object to a JSON file.

		Args:
			filepath: Output file path
			**metadata: Additional metadata to include in the file
		"""
		data = self.serialize()
		if metadata:
			data["_metadata"] = metadata

		path = Path(filepath)
		path.parent.mkdir(parents=True, exist_ok=True)

		with open(path, 'w') as f:
			json.dump(data, f, indent=2, default=str)

	@classmethod
	def load(cls: type[T], filepath: str) -> tuple[T, Optional[dict[str, Any]]]:
		"""
		Load an object from a JSON file.

		Args:
			filepath: Input file path

		Returns:
			Tuple of (deserialized object, metadata dict or None)
		"""
		with open(filepath, 'r') as f:
			data = json.load(f)

		metadata = data.pop("_metadata", None)
		return cls.deserialize(data), metadata


# Forward reference for type hints (ClusterGenome is defined elsewhere)
# We use string literals to avoid circular imports
GenomeType = TypeVar('GenomeType')


@dataclass
class Population(Generic[GenomeType], Serializable):
	"""
	A population of genomes with their fitness scores.

	Generic over the genome type to support different genome implementations.
	For use with ClusterGenome, import and use Population[ClusterGenome].
	"""

	items: list[tuple[GenomeType, float]] = field(default_factory=list)

	def __len__(self) -> int:
		return len(self.items)

	def __iter__(self):
		return iter(self.items)

	def __getitem__(self, idx: int) -> tuple[GenomeType, float]:
		return self.items[idx]

	@property
	def best(self) -> Optional[tuple[GenomeType, float]]:
		"""Get the genome with lowest fitness (best CE)."""
		if not self.items:
			return None
		return min(self.items, key=lambda x: x[1])

	@property
	def genomes(self) -> list[GenomeType]:
		"""Get just the genomes without fitness scores."""
		return [g for g, _ in self.items]

	@property
	def fitnesses(self) -> list[float]:
		"""Get just the fitness scores."""
		return [f for _, f in self.items]

	def add(self, genome: GenomeType, fitness: float) -> None:
		"""Add a genome with its fitness to the population."""
		self.items.append((genome, fitness))

	def serialize(self) -> dict[str, Any]:
		"""Serialize the population to a dictionary."""
		return {
			"population": [
				{
					"genome": g.serialize() if hasattr(g, 'serialize') else g,
					"fitness": f,
				}
				for g, f in self.items
			],
			"count": len(self.items),
		}

	@classmethod
	def deserialize(
		cls,
		data: dict[str, Any],
		genome_class: type = None,
	) -> 'Population':
		"""
		Deserialize a population from a dictionary.

		Args:
			data: Dictionary from serialize()
			genome_class: Class to use for deserializing genomes.
				If None, genomes are stored as raw dicts.

		Returns:
			Population instance
		"""
		items = []
		for item in data["population"]:
			genome_data = item["genome"]
			if genome_class is not None and hasattr(genome_class, 'deserialize'):
				genome = genome_class.deserialize(genome_data)
			else:
				genome = genome_data
			items.append((genome, item["fitness"]))

		return cls(items=items)

	@classmethod
	def load(
		cls,
		filepath: str,
		genome_class: type = None,
	) -> tuple['Population', Optional[dict[str, Any]]]:
		"""
		Load a population from a JSON file.

		Args:
			filepath: Input file path
			genome_class: Class to use for deserializing genomes

		Returns:
			Tuple of (population, metadata dict or None)
		"""
		with open(filepath, 'r') as f:
			data = json.load(f)

		metadata = data.pop("_metadata", None)
		return cls.deserialize(data, genome_class=genome_class), metadata


@dataclass
class Checkpoint(Generic[GenomeType], Serializable):
	"""
	A full checkpoint for resuming optimization.

	Contains the best genome, current population, and optimization state.
	"""

	best_genome: GenomeType
	best_fitness: float
	population: Population[GenomeType]
	generation: int
	phase: str
	extra: dict[str, Any] = field(default_factory=dict)

	def serialize(self) -> dict[str, Any]:
		"""Serialize the checkpoint to a dictionary."""
		data = {
			"best": {
				"genome": (
					self.best_genome.serialize()
					if hasattr(self.best_genome, 'serialize')
					else self.best_genome
				),
				"fitness": self.best_fitness,
			},
			"population": self.population.serialize()["population"],
			"population_size": len(self.population),
			"generation": self.generation,
			"phase": self.phase,
		}
		# Add any extra fields
		data.update(self.extra)
		return data

	@classmethod
	def deserialize(
		cls,
		data: dict[str, Any],
		genome_class: type = None,
	) -> 'Checkpoint':
		"""
		Deserialize a checkpoint from a dictionary.

		Args:
			data: Dictionary from serialize()
			genome_class: Class to use for deserializing genomes

		Returns:
			Checkpoint instance
		"""
		# Deserialize best genome
		best_data = data["best"]["genome"]
		if genome_class is not None and hasattr(genome_class, 'deserialize'):
			best_genome = genome_class.deserialize(best_data)
		else:
			best_genome = best_data

		# Deserialize population
		pop_data = {"population": data["population"], "count": len(data["population"])}
		population = Population.deserialize(pop_data, genome_class=genome_class)

		# Extract known fields, rest goes to extra
		known_keys = {"best", "population", "population_size", "generation", "phase"}
		extra = {k: v for k, v in data.items() if k not in known_keys}

		return cls(
			best_genome=best_genome,
			best_fitness=data["best"]["fitness"],
			population=population,
			generation=data["generation"],
			phase=data["phase"],
			extra=extra,
		)

	@classmethod
	def load(
		cls,
		filepath: str,
		genome_class: type = None,
	) -> tuple['Checkpoint', Optional[dict[str, Any]]]:
		"""
		Load a checkpoint from a JSON file.

		Args:
			filepath: Input file path
			genome_class: Class to use for deserializing genomes

		Returns:
			Tuple of (checkpoint, metadata dict or None)
		"""
		with open(filepath, 'r') as f:
			data = json.load(f)

		metadata = data.pop("_metadata", None)
		return cls.deserialize(data, genome_class=genome_class), metadata
