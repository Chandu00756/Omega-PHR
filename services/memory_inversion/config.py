"""
Configuration for Memory Inversion Service.
Research-grade stability settings.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class MemoryInversionConfig:
    """Configuration for Memory Inversion Service."""

    # Service settings
    host: str = "localhost"
    port: int = 50053
    log_level: str = "INFO"

    # Analysis settings
    confidence_threshold: float = 0.7
    pattern_detection_window: int = 30  # seconds
    min_traces_for_analysis: int = 3

    # Memory settings
    max_traces_per_source: int = 10000
    trace_retention_days: int = 30

    # Background processing
    analysis_interval: int = 30  # seconds
    cleanup_interval: int = 3600  # seconds (1 hour)

    @classmethod
    def from_env(cls) -> "MemoryInversionConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("MEMORY_HOST", "localhost"),
            port=int(os.getenv("MEMORY_PORT", "50053")),
            log_level=os.getenv("MEMORY_LOG_LEVEL", "INFO"),
            confidence_threshold=float(os.getenv("MEMORY_CONFIDENCE_THRESHOLD", "0.7")),
            pattern_detection_window=int(os.getenv("MEMORY_PATTERN_WINDOW", "30")),
            min_traces_for_analysis=int(os.getenv("MEMORY_MIN_TRACES", "3")),
            max_traces_per_source=int(os.getenv("MEMORY_MAX_TRACES", "10000")),
            trace_retention_days=int(os.getenv("MEMORY_RETENTION_DAYS", "30")),
            analysis_interval=int(os.getenv("MEMORY_ANALYSIS_INTERVAL", "30")),
            cleanup_interval=int(os.getenv("MEMORY_CLEANUP_INTERVAL", "3600"))
        )
