"""
Reusable logging utilities for WNN experiments.

This module provides a Logger class that can be instantiated, configured,
and passed to components. It supports:
- File and console output with timestamps
- Date-based log directory structure (logs/YYYY/MM/DD/)
- Callable interface for easy integration
- Separator and header formatting utilities
"""

import os
import logging
from datetime import datetime
from typing import Optional, Callable


class Logger:
	"""
	Reusable logger that writes to both file and console with timestamps.

	Can be passed to components that accept a logging function (Callable[[str], None]).

	Usage:
		# Create logger
		logger = Logger("my_experiment")

		# Use directly
		logger("Starting experiment...")
		logger.separator("=")
		logger.header("Results")

		# Pass to components that accept a logger function
		optimizer = SomeOptimizer(logger=logger)

		# Or get the callable
		run_search(..., logger=logger.log)

	Attributes:
		name: Logger name (used for log filename)
		log_file: Path to the log file
	"""

	def __init__(
		self,
		name: str = "experiment",
		log_dir: Optional[str] = None,
		project_root: Optional[str] = None,
		console: bool = True,
		timestamp_format: str = '%H:%M:%S',
	):
		"""
		Initialize logger.

		Args:
			name: Base name for the log file (e.g., "adaptive_search")
			log_dir: Override log directory (default: project_root/logs/YYYY/MM/DD/)
			project_root: Project root directory (default: auto-detect from wnn package)
			console: Whether to also log to console
			timestamp_format: strftime format for log timestamps
		"""
		self.name = name
		self._console = console
		self._timestamp_format = timestamp_format

		# Auto-detect project root if not provided
		if project_root is None:
			# Go up from src/wnn/logging.py to project root
			this_dir = os.path.dirname(os.path.abspath(__file__))
			project_root = os.path.dirname(os.path.dirname(this_dir))

		# Create log directory with date structure
		if log_dir is None:
			now = datetime.now()
			log_dir = os.path.join(
				project_root, "logs",
				now.strftime("%Y"),
				now.strftime("%m"),
				now.strftime("%d")
			)
		os.makedirs(log_dir, exist_ok=True)

		# Create log file with timestamp
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

		# Setup Python logger
		self._logger = logging.getLogger(f'wnn.{name}.{timestamp}')
		self._logger.setLevel(logging.INFO)
		self._logger.handlers.clear()

		# Formatter with timestamps
		formatter = logging.Formatter(
			f'%(asctime)s | %(message)s',
			datefmt=timestamp_format
		)

		# File handler
		file_handler = logging.FileHandler(self.log_file)
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		self._logger.addHandler(file_handler)

		# Console handler (optional)
		if console:
			console_handler = logging.StreamHandler()
			console_handler.setLevel(logging.INFO)
			console_handler.setFormatter(formatter)
			self._logger.addHandler(console_handler)

	def __call__(self, message: str = "", flush: bool = True) -> None:
		"""
		Log a message. Makes Logger callable for easy integration.

		Args:
			message: Message to log
			flush: Whether to flush handlers immediately
		"""
		self.log(message, flush=flush)

	def log(self, message: str = "", flush: bool = True) -> None:
		"""
		Log a message to file and console.

		Args:
			message: Message to log
			flush: Whether to flush handlers immediately
		"""
		self._logger.info(message)
		if flush:
			for handler in self._logger.handlers:
				handler.flush()

	def separator(self, char: str = "=", width: int = 70) -> None:
		"""Log a separator line."""
		self.log(char * width)

	def header(self, title: str, char: str = "=", width: int = 70) -> None:
		"""Log a formatted header."""
		self.log()
		self.separator(char, width)
		self.log(f"  {title}")
		self.separator(char, width)

	def section(self, title: str, char: str = "-", width: int = 50) -> None:
		"""Log a section divider."""
		self.log()
		self.separator(char, width)
		self.log(f"  {title}")
		self.separator(char, width)

	def __repr__(self) -> str:
		return f"Logger(name='{self.name}', log_file='{self.log_file}')"


def create_logger(
	name: str = "experiment",
	log_dir: Optional[str] = None,
	console: bool = True,
) -> Logger:
	"""
	Factory function to create a Logger instance.

	Args:
		name: Base name for the log file
		log_dir: Override log directory
		console: Whether to also log to console

	Returns:
		Configured Logger instance
	"""
	return Logger(name=name, log_dir=log_dir, console=console)
