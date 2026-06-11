"""Optimization logging: leveled logger with optional file sink + TRACE level."""

import logging
from typing import Callable, Optional

# Custom TRACE level (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class OptimizationLogger:
	"""
	Logger wrapper with TRACE, DEBUG, INFO, ERROR levels.

	TRACE: Filtered candidates, very verbose per-candidate info (stdout only)
	DEBUG: Individual genome info (elites, init genomes)
	INFO: Progress summaries, phase transitions
	ERROR: Errors and warnings

	Usage:
		logger = OptimizationLogger("ArchitectureGA", level=logging.DEBUG)
		logger.debug("Elite details...")
		logger.trace("Filtered candidate...")
		logger.info("Generation complete")

	With file logging:
		file_log = lambda msg: print(msg, file=open("log.txt", "a"))
		logger = OptimizationLogger("GA", file_logger=file_log)
	"""

	def __init__(
		self,
		name: str,
		level: int = logging.DEBUG,
		file_logger: Optional[Callable[[str], None]] = None,
	):
		self._logger = logging.getLogger(f"wnn.optimizer.{name}")
		# Only add StreamHandler if no file_logger (file_logger handles stdout+file)
		if not file_logger and not self._logger.handlers:
			handler = logging.StreamHandler()
			handler.setFormatter(logging.Formatter("%(message)s"))
			self._logger.addHandler(handler)
		self._logger.setLevel(level)
		self._name = name
		self._file_logger = file_logger  # Handles stdout + file when provided

	def trace(self, msg: str) -> None:
		"""Log at TRACE level (filtered candidates, stdout only)."""
		if self._logger.isEnabledFor(TRACE):
			if self._file_logger:
				# file_logger handles stdout+file, but TRACE goes to stdout only
				print(msg)
			else:
				self._logger.log(TRACE, msg)

	def _flush(self) -> None:
		"""Flush all handlers to ensure output is visible immediately."""
		for handler in self._logger.handlers:
			handler.flush()

	def debug(self, msg: str) -> None:
		"""Log at DEBUG level (individual genome info)."""
		if self._logger.isEnabledFor(logging.DEBUG):
			if self._file_logger:
				self._file_logger(msg)  # file_logger handles stdout + file
			else:
				self._logger.debug(msg)
				self._flush()

	def info(self, msg: str) -> None:
		"""Log at INFO level (progress summaries)."""
		if self._logger.isEnabledFor(logging.INFO):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.info(msg)
				self._flush()

	def warning(self, msg: str) -> None:
		"""Log at WARNING level."""
		if self._logger.isEnabledFor(logging.WARNING):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.warning(msg)
				self._flush()

	def error(self, msg: str) -> None:
		"""Log at ERROR level."""
		if self._logger.isEnabledFor(logging.ERROR):
			if self._file_logger:
				self._file_logger(msg)
			else:
				self._logger.error(msg)
				self._flush()

	def __call__(self, msg: str) -> None:
		"""Default: INFO level (backward compatible with print-style logging)."""
		self.info(msg)

	def set_level(self, level: int) -> None:
		"""Change log level dynamically."""
		self._logger.setLevel(level)
