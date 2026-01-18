"""
Reusable genome logging utilities.

Provides consistent formatting for genome logs across GA/TS optimization,
with dynamic padding based on totals.
"""

from enum import IntEnum
from typing import Optional, Union


class GenomeLogType(IntEnum):
    """Type of genome being logged."""
    INITIAL = 0      # Initial random population
    ELITE_CE = 1     # Elite selected by cross-entropy
    ELITE_ACC = 2    # Elite selected by accuracy
    OFFSPRING = 3    # GA offspring
    NEIGHBOR = 4     # TS neighbor
    FINAL = 5        # Final evaluation


# Type labels and their display names (all padded to same width)
# Format: (label, type_indicator) where label is padded to 6 chars (longest is "Genome")
_TYPE_FORMATS = {
    GenomeLogType.INITIAL:   ("Genome", "Init"),
    GenomeLogType.ELITE_CE:  ("Elite ", "CE  "),  # 4 chars like others
    GenomeLogType.ELITE_ACC: ("Elite ", "Acc "),
    GenomeLogType.OFFSPRING: ("Genome", "New "),
    GenomeLogType.NEIGHBOR:  ("Genome", "Nbr "),
    GenomeLogType.FINAL:     ("Genome", "Fin "),
}


def format_genome_log(
    generation: int,
    total_generations: int,
    log_type: GenomeLogType,
    position: int,
    total: int,
    ce: float,
    acc: float,
) -> str:
    """
    Format a genome log line with consistent padding and alignment.

    All log types are aligned with:
    - 6-char label (Elite  or Genome)
    - Position/total with dynamic padding
    - 5-char type indicator in parentheses (CE, Acc, New, Init, Nbr)

    Args:
        generation: Current generation (1-indexed)
        total_generations: Total number of generations
        log_type: Type of genome (Elite, Offspring, etc.)
        position: Position in the batch (1-indexed)
        total: Total items in this batch
        ce: Cross-entropy value
        acc: Accuracy value (0.0 to 1.0)

    Returns:
        Formatted log string like:
        "[Gen 001/100] Elite  01/10 (CE  ): CE=10.3417, Acc=0.0180%"
        "[Gen 001/100] Genome 01/40 (New ): CE=10.3559, Acc=0.0300%"
    """
    # Calculate padding widths based on totals
    gen_width = len(str(total_generations))
    pos_width = len(str(total))

    # Get label and type indicator for this log type
    label, type_ind = _TYPE_FORMATS.get(log_type, ("Genome", "???"))

    # Generation prefix
    if log_type == GenomeLogType.FINAL:
        gen_prefix = "[Final]"
    else:
        gen_prefix = f"[Gen {generation:0{gen_width}d}/{total_generations:0{gen_width}d}]"

    return f"{gen_prefix} {label} {position:0{pos_width}d}/{total} ({type_ind}): CE={ce:.4f}, Acc={acc:.4%}"


def format_gen_prefix(generation: int, total_generations: int) -> str:
    """
    Format just the generation prefix with dynamic padding.

    Args:
        generation: Current generation (1-indexed)
        total_generations: Total number of generations

    Returns:
        Formatted prefix like "[Gen 001/100]"
    """
    gen_width = len(str(total_generations))
    return f"[Gen {generation:0{gen_width}d}/{total_generations:0{gen_width}d}]"


def format_completion_log(
    generation: int,
    total_generations: int,
    elapsed_seconds: float,
    evaluated: int,
    shown: int,
) -> str:
    """
    Format a completion summary log line.

    Args:
        generation: Current generation (1-indexed)
        total_generations: Total number of generations
        elapsed_seconds: Time taken
        evaluated: Number of genomes evaluated
        shown: Number of genomes shown (passing threshold)

    Returns:
        Formatted string like:
        "[Gen 001/100] Completed in 237.5s (50 evaluated, 50 shown)"
    """
    gen_prefix = format_gen_prefix(generation, total_generations)
    return f"{gen_prefix} Completed in {elapsed_seconds:.1f}s ({evaluated} evaluated, {shown} shown)"


def format_offspring_summary(
    generation: int,
    total_generations: int,
    passed: int,
    target: int,
    evaluated: int,
    threshold: float,
) -> str:
    """
    Format offspring search summary.

    Args:
        generation: Current generation (1-indexed)
        total_generations: Total number of generations
        passed: Number that passed threshold
        target: Target count
        evaluated: Total evaluated
        threshold: Accuracy threshold (0.0 to 1.0)

    Returns:
        Formatted string like:
        "[Gen 002/100] Offspring search: 40/40 viable (evaluated 80, threshold 0.0100%)"
    """
    gen_prefix = format_gen_prefix(generation, total_generations)
    return f"{gen_prefix} Offspring search: {passed}/{target} viable (evaluated {evaluated}, threshold {threshold:.4%})"
