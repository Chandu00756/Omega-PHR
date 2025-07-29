"""
Configuration for Recursive Loop Synthesis Service.
Research-grade stability settings.
"""

import os
from dataclasses import dataclass


@dataclass
class RecursiveLoopSynthConfig:
    """Configuration for Recursive Loop Synthesis Service."""

    # Service settings
    host: str = "localhost"
    port: int = 50055
    log_level: str = "INFO"

    # Analysis settings
    min_cycle_length: int = 2
    max_cycle_length: int = 100
    stability_threshold: float = 0.8
    entropy_threshold: float = 3.0

    # Synthesis settings
    max_synthesis_iterations: int = 1000
    convergence_threshold: float = 0.01
    max_input_patterns: int = 10
    synthesis_timeout: int = 300  # seconds

    # Memory management
    max_patterns: int = 10000
    max_nodes_per_pattern: int = 1000
    cleanup_interval: int = 3600  # seconds
    pattern_retention_hours: int = 24

    # Background processing
    synthesis_interval: int = 60  # seconds
    analysis_batch_size: int = 100

    @classmethod
    def from_env(cls) -> "RecursiveLoopSynthConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("LOOP_HOST", "localhost"),
            port=int(os.getenv("LOOP_PORT", "50055")),
            log_level=os.getenv("LOOP_LOG_LEVEL", "INFO"),
            min_cycle_length=int(os.getenv("LOOP_MIN_CYCLE", "2")),
            max_cycle_length=int(os.getenv("LOOP_MAX_CYCLE", "100")),
            stability_threshold=float(os.getenv("LOOP_STABILITY_THRESHOLD", "0.8")),
            entropy_threshold=float(os.getenv("LOOP_ENTROPY_THRESHOLD", "3.0")),
            max_synthesis_iterations=int(os.getenv("LOOP_MAX_ITERATIONS", "1000")),
            convergence_threshold=float(
                os.getenv("LOOP_CONVERGENCE_THRESHOLD", "0.01")
            ),
            max_input_patterns=int(os.getenv("LOOP_MAX_INPUT_PATTERNS", "10")),
            synthesis_timeout=int(os.getenv("LOOP_SYNTHESIS_TIMEOUT", "300")),
            max_patterns=int(os.getenv("LOOP_MAX_PATTERNS", "10000")),
            max_nodes_per_pattern=int(os.getenv("LOOP_MAX_NODES", "1000")),
            cleanup_interval=int(os.getenv("LOOP_CLEANUP_INTERVAL", "3600")),
            pattern_retention_hours=int(os.getenv("LOOP_RETENTION_HOURS", "24")),
            synthesis_interval=int(os.getenv("LOOP_SYNTHESIS_INTERVAL", "60")),
            analysis_batch_size=int(os.getenv("LOOP_BATCH_SIZE", "100")),
        )
