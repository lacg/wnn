"""
Model Serialization for RAM Networks

Provides save/load functionality for RAM models with:
- Configuration preservation (architecture parameters)
- State preservation (memory contents, connections)
- Version compatibility checking

Usage:
    # Save a model
    model.save('model.pt')

    # Load a model
    model = RAMLayer.load('model.pt')

    # Or use the generic loader
    model = load_model('model.pt')
"""

import torch
from torch import Tensor
from pathlib import Path
from typing import Any, TypeVar, Type
from dataclasses import dataclass, asdict, field
import json

# Version for compatibility checking
SERIALIZATION_VERSION = "1.0.0"

T = TypeVar('T')


@dataclass
class ModelConfig:
    """Base configuration for serializable models."""
    model_class: str
    version: str = SERIALIZATION_VERSION
    extra: dict = field(default_factory=dict)


def save_model(
    model: torch.nn.Module,
    path: str | Path,
    config: dict | None = None,
) -> None:
    """
    Save a PyTorch model with its configuration.

    Args:
        model: The model to save
        path: Path to save to (typically .pt extension)
        config: Optional configuration dict (if model doesn't have get_config())
    """
    path = Path(path)

    # Get configuration
    if hasattr(model, 'get_config'):
        config = model.get_config()
    elif config is None:
        config = {}

    # Build save dict
    save_dict = {
        'version': SERIALIZATION_VERSION,
        'model_class': model.__class__.__name__,
        'config': config,
        'state_dict': model.state_dict(),
    }

    torch.save(save_dict, path)


def load_model(
    path: str | Path,
    model_class: Type[T] | None = None,
    device: str | torch.device = 'cpu',
) -> T:
    """
    Load a model from a file.

    Args:
        path: Path to the saved model
        model_class: Optional model class (if not, uses class name from file)
        device: Device to load the model to

    Returns:
        The loaded model
    """
    path = Path(path)
    save_dict = torch.load(path, map_location=device, weights_only=False)

    # Version check
    version = save_dict.get('version', '0.0.0')
    if version != SERIALIZATION_VERSION:
        print(f"Warning: Model saved with version {version}, current is {SERIALIZATION_VERSION}")

    # Get config and state
    config = save_dict.get('config', {})
    state_dict = save_dict.get('state_dict', {})
    saved_class_name = save_dict.get('model_class', '')

    # Determine model class
    if model_class is None:
        model_class = _get_model_class(saved_class_name)

    # Create model from config
    if hasattr(model_class, 'from_config'):
        model = model_class.from_config(config)
    else:
        model = model_class(**config)

    # Load state
    model.load_state_dict(state_dict)

    return model


def _get_model_class(class_name: str) -> type:
    """Get model class from name."""
    # Import here to avoid circular imports
    from wnn.ram.core import (
        RAMLayer,
        Memory,
        RAMRecurrentNetwork,
        RAMSequence,
        RAMMultiHeadSequence,
        RAMMultiHeadKV,
        RAMMultiHeadShared,
        RAMKVMemory,
        RAMAutomaton,
    )
    from wnn.ram.core.models import (
        RAMAttention,
        RAMTransformer,
        RAMSeq2Seq,
        RAMEncoderDecoder,
        PositionOnlyAttention,
    )

    class_map = {
        'RAMLayer': RAMLayer,
        'Memory': Memory,
        'RAMRecurrentNetwork': RAMRecurrentNetwork,
        'RAMSequence': RAMSequence,
        'RAMMultiHeadSequence': RAMMultiHeadSequence,
        'RAMMultiHeadKV': RAMMultiHeadKV,
        'RAMMultiHeadShared': RAMMultiHeadShared,
        'RAMKVMemory': RAMKVMemory,
        'RAMAutomaton': RAMAutomaton,
        'RAMAttention': RAMAttention,
        'RAMTransformer': RAMTransformer,
        'RAMSeq2Seq': RAMSeq2Seq,
        'RAMEncoderDecoder': RAMEncoderDecoder,
        'PositionOnlyAttention': PositionOnlyAttention,
    }

    if class_name not in class_map:
        raise ValueError(f"Unknown model class: {class_name}. "
                        f"Available: {list(class_map.keys())}")

    return class_map[class_name]


class SerializableMixin:
    """
    Mixin class that adds save/load functionality to RAM models.

    Models should implement get_config() and from_config() class method.
    """

    def save(self, path: str | Path) -> None:
        """Save the model to a file."""
        save_model(self, path)

    @classmethod
    def load(cls: Type[T], path: str | Path, device: str | torch.device = 'cpu') -> T:
        """Load a model from a file."""
        return load_model(path, model_class=cls, device=device)

    def get_config(self) -> dict:
        """
        Get configuration dict for recreation.
        Override in subclasses to provide specific config.
        """
        raise NotImplementedError("Subclasses should implement get_config()")

    @classmethod
    def from_config(cls: Type[T], config: dict) -> T:
        """
        Create model from configuration dict.
        Override in subclasses if needed.
        """
        return cls(**config)


# Convenience function to add serialization to existing classes
def add_serialization_to_class(cls: type, config_attrs: list[str]) -> None:
    """
    Add serialization methods to an existing class.

    Args:
        cls: The class to modify
        config_attrs: List of attribute names that form the config
    """
    def get_config(self) -> dict:
        config = {}
        for attr in config_attrs:
            if hasattr(self, attr):
                val = getattr(self, attr)
                # Handle special types
                if isinstance(val, torch.Tensor):
                    val = val.tolist()
                elif hasattr(val, 'value'):  # Enum
                    val = val.value
                config[attr] = val
        return config

    def save(self, path: str | Path) -> None:
        save_model(self, path, self.get_config())

    @classmethod
    def load(cls_inner: Type[T], path: str | Path, device: str = 'cpu') -> T:
        return load_model(path, model_class=cls_inner, device=device)

    @classmethod
    def from_config(cls_inner: Type[T], config: dict) -> T:
        return cls_inner(**config)

    cls.get_config = get_config
    cls.save = save
    cls.load = classmethod(load.__func__)
    cls.from_config = classmethod(from_config.__func__)
