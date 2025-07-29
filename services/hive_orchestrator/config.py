"""
Configuration module for Hive Orchestrator service.

This module provides environment-based configuration management
for the distributed adversarial attack coordination service.
"""

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HiveServiceConfig:
    """Configuration for the Hive Orchestrator service."""

    # Server Configuration
    host: str = field(default_factory=lambda: os.getenv("HIVE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("HIVE_PORT", "50052")))
    use_ssl: bool = field(
        default_factory=lambda: os.getenv("HIVE_USE_SSL", "false").lower() == "true"
    )
    ssl_cert_file: str | None = field(
        default_factory=lambda: os.getenv("HIVE_SSL_CERT")
    )
    ssl_key_file: str | None = field(default_factory=lambda: os.getenv("HIVE_SSL_KEY"))

    # Ray Configuration
    ray_address: str | None = field(default_factory=lambda: os.getenv("RAY_ADDRESS"))
    ray_namespace: str = field(
        default_factory=lambda: os.getenv("RAY_NAMESPACE", "omega-phr-hive")
    )
    max_agents: int = field(
        default_factory=lambda: int(os.getenv("HIVE_MAX_AGENTS", "1000"))
    )
    agent_pool_size: int = field(
        default_factory=lambda: int(os.getenv("HIVE_AGENT_POOL_SIZE", "100"))
    )

    # Attack Configuration
    max_concurrent_attacks: int = field(
        default_factory=lambda: int(os.getenv("HIVE_MAX_CONCURRENT_ATTACKS", "10"))
    )
    default_attack_timeout: int = field(
        default_factory=lambda: int(os.getenv("HIVE_DEFAULT_ATTACK_TIMEOUT", "3600"))
    )
    attack_coordination_interval: float = field(
        default_factory=lambda: float(os.getenv("HIVE_COORDINATION_INTERVAL", "5.0"))
    )

    # Intelligence Configuration
    intelligence_retention_days: int = field(
        default_factory=lambda: int(os.getenv("HIVE_INTELLIGENCE_RETENTION_DAYS", "30"))
    )
    vulnerability_scan_interval: int = field(
        default_factory=lambda: int(os.getenv("HIVE_VULN_SCAN_INTERVAL", "300"))
    )
    payload_generation_timeout: int = field(
        default_factory=lambda: int(os.getenv("HIVE_PAYLOAD_GEN_TIMEOUT", "60"))
    )

    # Security Configuration
    enable_agent_authentication: bool = field(
        default_factory=lambda: os.getenv("HIVE_ENABLE_AGENT_AUTH", "true").lower()
        == "true"
    )
    agent_token_expiry: int = field(
        default_factory=lambda: int(os.getenv("HIVE_AGENT_TOKEN_EXPIRY", "3600"))
    )
    max_failed_auth_attempts: int = field(
        default_factory=lambda: int(os.getenv("HIVE_MAX_FAILED_AUTH", "3"))
    )

    # Storage Configuration
    data_storage_path: str = field(
        default_factory=lambda: os.getenv("HIVE_DATA_PATH", "/tmp/hive-data")
    )
    backup_interval: int = field(
        default_factory=lambda: int(os.getenv("HIVE_BACKUP_INTERVAL", "300"))
    )
    enable_persistence: bool = field(
        default_factory=lambda: os.getenv("HIVE_ENABLE_PERSISTENCE", "true").lower()
        == "true"
    )

    # Monitoring Configuration
    enable_metrics: bool = field(
        default_factory=lambda: os.getenv("HIVE_ENABLE_METRICS", "true").lower()
        == "true"
    )
    metrics_port: int = field(
        default_factory=lambda: int(os.getenv("HIVE_METRICS_PORT", "8082"))
    )
    health_check_interval: float = field(
        default_factory=lambda: float(os.getenv("HIVE_HEALTH_CHECK_INTERVAL", "30.0"))
    )

    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv("HIVE_LOG_LEVEL", "INFO"))
    log_format: str = field(
        default_factory=lambda: os.getenv("HIVE_LOG_FORMAT", "json")
    )
    enable_debug_logs: bool = field(
        default_factory=lambda: os.getenv("HIVE_DEBUG_LOGS", "false").lower() == "true"
    )

    # Advanced Configuration
    enable_agent_clustering: bool = field(
        default_factory=lambda: os.getenv("HIVE_ENABLE_CLUSTERING", "true").lower()
        == "true"
    )
    cluster_rebalance_interval: int = field(
        default_factory=lambda: int(os.getenv("HIVE_CLUSTER_REBALANCE", "60"))
    )
    enable_auto_scaling: bool = field(
        default_factory=lambda: os.getenv("HIVE_ENABLE_AUTO_SCALING", "true").lower()
        == "true"
    )
    scaling_threshold_cpu: float = field(
        default_factory=lambda: float(os.getenv("HIVE_SCALING_CPU_THRESHOLD", "0.8"))
    )
    scaling_threshold_memory: float = field(
        default_factory=lambda: float(os.getenv("HIVE_SCALING_MEMORY_THRESHOLD", "0.8"))
    )

    # AI Model Configuration
    ai_model_endpoint: str | None = field(
        default_factory=lambda: os.getenv("HIVE_AI_MODEL_ENDPOINT")
    )
    ai_model_timeout: int = field(
        default_factory=lambda: int(os.getenv("HIVE_AI_MODEL_TIMEOUT", "30"))
    )
    enable_model_adaptation: bool = field(
        default_factory=lambda: os.getenv(
            "HIVE_ENABLE_MODEL_ADAPTATION", "true"
        ).lower()
        == "true"
    )

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate port ranges
        if not (1 <= self.port <= 65535):
            errors.append(f"Invalid port: {self.port}")

        if not (1 <= self.metrics_port <= 65535):
            errors.append(f"Invalid metrics port: {self.metrics_port}")

        # Validate positive integers
        if self.max_agents <= 0:
            errors.append(f"max_agents must be positive: {self.max_agents}")

        if self.agent_pool_size <= 0:
            errors.append(f"agent_pool_size must be positive: {self.agent_pool_size}")

        if self.max_concurrent_attacks <= 0:
            errors.append(
                f"max_concurrent_attacks must be positive: {self.max_concurrent_attacks}"
            )

        # Validate timeouts
        if self.default_attack_timeout <= 0:
            errors.append(
                f"default_attack_timeout must be positive: {self.default_attack_timeout}"
            )

        if self.agent_token_expiry <= 0:
            errors.append(
                f"agent_token_expiry must be positive: {self.agent_token_expiry}"
            )

        # Validate intervals
        if self.attack_coordination_interval <= 0:
            errors.append(
                f"attack_coordination_interval must be positive: {self.attack_coordination_interval}"
            )

        if self.health_check_interval <= 0:
            errors.append(
                f"health_check_interval must be positive: {self.health_check_interval}"
            )

        # Validate percentages
        if not (0.0 <= self.scaling_threshold_cpu <= 1.0):
            errors.append(
                f"scaling_threshold_cpu must be between 0.0 and 1.0: {self.scaling_threshold_cpu}"
            )

        if not (0.0 <= self.scaling_threshold_memory <= 1.0):
            errors.append(
                f"scaling_threshold_memory must be between 0.0 and 1.0: {self.scaling_threshold_memory}"
            )

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.log_level}")

        # Validate SSL configuration
        if self.use_ssl:
            if not self.ssl_cert_file:
                errors.append("SSL certificate file required when SSL is enabled")
            if not self.ssl_key_file:
                errors.append("SSL key file required when SSL is enabled")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_env(cls) -> "HiveServiceConfig":
        """Create configuration from environment variables."""
        return cls()

    def get_ray_config(self) -> dict[str, Any]:
        """Get Ray-specific configuration."""
        config = {
            "namespace": self.ray_namespace,
            "ignore_reinit_error": True,
        }

        if self.ray_address:
            config["address"] = self.ray_address

        return config

    def get_server_options(self) -> list[tuple]:
        """Get gRPC server options."""
        return [
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ("grpc.max_connection_idle_ms", 10000),
            ("grpc.max_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_workers", self.agent_pool_size),
        ]
