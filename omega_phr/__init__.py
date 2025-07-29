"""
Omega-Paradox Hive Recursion (Ω-PHR) Framework

A revolutionary AI security testing framework that integrates:
- Temporal paradox testing
- Synthetic adversarial hive attacks
- Recursive memory inversion
- Generative infinite loop fuzzing

This package provides the core abstractions and utilities for the Ω-PHR framework.
"""

__version__ = "1.0.0"
__author__ = "Venkata Sai Chandu Chitikam"
__email__ = "chandu@portalvii.com"

from .exceptions import (
    HiveCoordinationError,
    MemoryInversionError,
    OmegaPHRError,
    OmegaStateError,
    RecursiveLoopError,
    TemporalParadoxError,
)
from .hive import HiveOrchestrator
from .loops import RecursiveLoopSynthesizer
from .memory import MemoryInverter
from .models import Event, HiveResult, MemoryState, OmegaState, ParadoxResult
from .omega_register import OmegaStateRegister
from .timeline import TimelineLattice

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Models
    "Event",
    "OmegaState",
    "ParadoxResult",
    "HiveResult",
    "MemoryState",
    # Core Components
    "TimelineLattice",
    "HiveOrchestrator",
    "MemoryInverter",
    "RecursiveLoopSynthesizer",
    "OmegaStateRegister",
    # Exceptions
    "OmegaPHRError",
    "TemporalParadoxError",
    "HiveCoordinationError",
    "MemoryInversionError",
    "RecursiveLoopError",
    "OmegaStateError",
]
