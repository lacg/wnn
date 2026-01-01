"""
Recurrent network enumerations.
"""

from enum import IntEnum


class StateMode(IntEnum):
	"""
	State transition modes for recurrent networks.

	Controls how the state layer computes new_state from (input, prev_state).
	"""
	LEARNED = 0      # State transition learned via EDRA (default)
	XOR = 1          # State = prev_state XOR input (for parity)
	IDENTITY = 2     # State = input (no memory)
	OR = 3           # State = prev_state OR input (for detection)
