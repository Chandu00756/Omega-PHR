"""
Configuration for Omega Register Service.
Research-grade stability settings.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OmegaRegisterConfig:
    """Configuration for Omega Register Service."""

    # Service settings
    host: str = "localhost"
    port: int = 50054
    log_level: str = "INFO"

    # Registry settings
    heartbeat_timeout: int = 300  # seconds
    max_agents: int = 1000
    max_tasks_per_agent: int = 100

    # Task scheduling
    task_timeout: int = 3600  # seconds (1 hour)
    retry_attempts: int = 3
    scheduler_interval: int = 10  # seconds

    # Cleanup settings
    cleanup_interval: int = 600  # seconds (10 minutes)
    agent_retention_days: int = 7
    task_retention_days: int = 30

    @classmethod
    def from_env(cls) -> "OmegaRegisterConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("OMEGA_HOST", "localhost"),
            port=int(os.getenv("OMEGA_PORT", "50054")),
            log_level=os.getenv("OMEGA_LOG_LEVEL", "INFO"),
            heartbeat_timeout=int(os.getenv("OMEGA_HEARTBEAT_TIMEOUT", "300")),
            max_agents=int(os.getenv("OMEGA_MAX_AGENTS", "1000")),
            max_tasks_per_agent=int(os.getenv("OMEGA_MAX_TASKS_PER_AGENT", "100")),
            task_timeout=int(os.getenv("OMEGA_TASK_TIMEOUT", "3600")),
            retry_attempts=int(os.getenv("OMEGA_RETRY_ATTEMPTS", "3")),
            scheduler_interval=int(os.getenv("OMEGA_SCHEDULER_INTERVAL", "10")),
            cleanup_interval=int(os.getenv("OMEGA_CLEANUP_INTERVAL", "600")),
            agent_retention_days=int(os.getenv("OMEGA_AGENT_RETENTION_DAYS", "7")),
            task_retention_days=int(os.getenv("OMEGA_TASK_RETENTION_DAYS", "30")),
        )
