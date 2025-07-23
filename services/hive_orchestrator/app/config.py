"""
Hive Orchestrator Configuration

Advanced configuration management for coordinated adversarial testing
with swarm intelligence and adaptive attack strategies.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class HiveConfig:
    """Advanced configuration for Hive Orchestrator service."""

    # Server Configuration
    host: str = field(default_factory=lambda: os.getenv("HIVE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("HIVE_PORT", "50052")))
    max_workers: int = field(default_factory=lambda: int(os.getenv("HIVE_MAX_WORKERS", "10")))

    # Agent Configuration
    max_agents: int = field(default_factory=lambda: int(os.getenv("HIVE_MAX_AGENTS", "100")))
    agent_spawn_rate: float = field(default_factory=lambda: float(os.getenv("HIVE_SPAWN_RATE", "2.0")))
    agent_diversity_factor: float = field(default_factory=lambda: float(os.getenv("HIVE_DIVERSITY", "0.8")))

    # Attack Strategy Configuration
    attack_intensity: float = field(default_factory=lambda: float(os.getenv("HIVE_INTENSITY", "0.7")))
    coordination_protocol: str = field(default_factory=lambda: os.getenv("HIVE_PROTOCOL", "swarm"))
    adaptation_rate: float = field(default_factory=lambda: float(os.getenv("HIVE_ADAPTATION", "0.3")))

    # Swarm Intelligence Parameters
    pheromone_decay: float = field(default_factory=lambda: float(os.getenv("HIVE_PHEROMONE_DECAY", "0.1")))
    exploration_factor: float = field(default_factory=lambda: float(os.getenv("HIVE_EXPLORATION", "0.4")))
    collective_memory_size: int = field(default_factory=lambda: int(os.getenv("HIVE_MEMORY_SIZE", "1000")))

    # Target Configuration
    target_models: List[str] = field(default_factory=lambda: os.getenv("HIVE_TARGETS", "gpt-4,claude-3").split(","))
    target_endpoints: Dict[str, str] = field(default_factory=dict)

    # Safety and Containment
    max_attack_duration: int = field(default_factory=lambda: int(os.getenv("HIVE_MAX_DURATION", "3600")))
    emergency_shutdown_threshold: float = field(default_factory=lambda: float(os.getenv("HIVE_SHUTDOWN_THRESHOLD", "0.95")))
    attack_success_limit: int = field(default_factory=lambda: int(os.getenv("HIVE_SUCCESS_LIMIT", "50")))

    # Logging and Monitoring
    log_level: str = field(default_factory=lambda: os.getenv("HIVE_LOG_LEVEL", "INFO"))
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("HIVE_METRICS", "true").lower() == "true")
    telemetry_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("HIVE_TELEMETRY_ENDPOINT"))

    # Data Storage
    attack_logs_path: str = field(default_factory=lambda: os.getenv("HIVE_LOGS_PATH", "./data/hive_logs"))
    strategy_bank_path: str = field(default_factory=lambda: os.getenv("HIVE_STRATEGY_BANK", "./data/strategies"))
    checkpoint_interval: int = field(default_factory=lambda: int(os.getenv("HIVE_CHECKPOINT_INTERVAL", "300")))

    # Ray/Distributed Configuration
    ray_head_host: Optional[str] = field(default_factory=lambda: os.getenv("RAY_HEAD_HOST"))
    ray_head_port: int = field(default_factory=lambda: int(os.getenv("RAY_HEAD_PORT", "10001")))
    ray_memory_limit: int = field(default_factory=lambda: int(os.getenv("RAY_MEMORY_LIMIT", "4096")))

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure data directories exist
        Path(self.attack_logs_path).mkdir(parents=True, exist_ok=True)
        Path(self.strategy_bank_path).mkdir(parents=True, exist_ok=True)

        # Validate configuration values
        if self.max_agents <= 0:
            raise ValueError("max_agents must be positive")
        if not 0.0 <= self.attack_intensity <= 1.0:
            raise ValueError("attack_intensity must be between 0.0 and 1.0")
        if not 0.0 <= self.adaptation_rate <= 1.0:
            raise ValueError("adaptation_rate must be between 0.0 and 1.0")

        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "max_workers": self.max_workers,
            "max_agents": self.max_agents,
            "agent_spawn_rate": self.agent_spawn_rate,
            "agent_diversity_factor": self.agent_diversity_factor,
            "attack_intensity": self.attack_intensity,
            "coordination_protocol": self.coordination_protocol,
            "adaptation_rate": self.adaptation_rate,
            "pheromone_decay": self.pheromone_decay,
            "exploration_factor": self.exploration_factor,
            "collective_memory_size": self.collective_memory_size,
            "target_models": self.target_models,
            "max_attack_duration": self.max_attack_duration,
            "emergency_shutdown_threshold": self.emergency_shutdown_threshold,
            "attack_success_limit": self.attack_success_limit,
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
            "ray_memory_limit": self.ray_memory_limit
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HiveConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            assert self.max_agents > 0, "max_agents must be positive"
            assert 0.0 <= self.attack_intensity <= 1.0, "attack_intensity must be in [0,1]"
            assert 0.0 <= self.adaptation_rate <= 1.0, "adaptation_rate must be in [0,1]"
            assert 0.0 <= self.pheromone_decay <= 1.0, "pheromone_decay must be in [0,1]"
            assert 0.0 <= self.exploration_factor <= 1.0, "exploration_factor must be in [0,1]"
            assert self.collective_memory_size > 0, "collective_memory_size must be positive"
            assert self.max_attack_duration > 0, "max_attack_duration must be positive"
            assert 0.0 <= self.emergency_shutdown_threshold <= 1.0, "shutdown_threshold must be in [0,1]"
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
