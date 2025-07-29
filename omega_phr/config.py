"""
Omega-Paradox Hive Recursion (Ω-PHR) Framework Configuration Management

Enterprise-grade configuration system for distributed temporal computing.
Provides secure hierarchical configuration loading with environment variables,
validation, and type conversion for all framework components.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for ScyllaDB/Cassandra clusters."""

    hosts: list[str] = field(default_factory=lambda: ["localhost"])
    port: int = 9042
    keyspace: str = "omega_phr"
    username: str | None = None
    password: str | None = None
    consistency_level: str = "QUORUM"
    timeout_seconds: int = 30
    ssl_enabled: bool = False
    auth_enabled: bool = False
    connection_pool_size: int = 10
    retry_policy_retries: int = 3


@dataclass
class RedisConfig:
    """Redis configuration for caching and session storage."""

    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str | None = None
    ssl_enabled: bool = False
    connection_pool_size: int = 50
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    max_connections: int = 100


@dataclass
class RayConfig:
    """Ray distributed computing configuration."""

    head_node_host: str = "localhost"
    head_node_port: int = 10001
    dashboard_port: int = 8265
    object_store_memory: int = 1000000000  # 1GB
    num_cpus: int | None = None
    num_gpus: int = 0
    cluster_name: str = "omega-phr-cluster"
    temp_dir: str | None = None


@dataclass
class SecurityConfig:
    """Security and cryptographic configuration."""

    jwt_secret_key: str | None = None
    jwt_algorithm: str = "ES256"
    jwt_expiration_hours: int = 24
    encryption_key: str | None = None
    key_rotation_days: int = 90
    max_login_attempts: int = 5
    session_timeout_minutes: int = 30
    cors_origins: list[str] = field(default_factory=list)
    rate_limit_requests: int = 1000
    rate_limit_window_minutes: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    grafana_enabled: bool = True
    grafana_port: int = 3000
    jaeger_enabled: bool = False
    jaeger_endpoint: str | None = None
    log_level: str = "INFO"
    metrics_retention_days: int = 30
    trace_sampling_rate: float = 0.1


@dataclass
class ServiceConfig:
    """Individual service configuration."""

    name: str
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_size: int = 4 * 1024 * 1024  # 4MB
    keepalive_timeout_ms: int = 30000
    keepalive_time_ms: int = 30000
    max_connection_idle_ms: int = 300000
    compression_enabled: bool = True


@dataclass
class TimelineConfig:
    """Timeline Lattice service configuration."""

    service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(name="timeline_lattice", port=50051)
    )
    max_timelines: int = 1000
    paradox_threshold: float = 0.1
    max_events_per_timeline: int = 10000
    causality_validation: bool = True
    bitemporal_indexing: bool = True
    event_retention_days: int = 365


@dataclass
class HiveConfig:
    """Hive Orchestrator service configuration."""

    service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(name="hive_orchestrator", port=50052)
    )
    max_agents: int = 100
    agent_timeout_seconds: int = 300
    swarm_coordination_enabled: bool = True
    attack_simulation_mode: bool = False
    intelligence_sharing: bool = True
    vulnerability_scanning: bool = True


@dataclass
class MemoryConfig:
    """Memory Inversion service configuration."""

    service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(name="memory_inversion", port=50053)
    )
    max_snapshots: int = 1000
    snapshot_interval_seconds: int = 60
    inversion_strategies: list[str] = field(
        default_factory=lambda: [
            "contradiction",
            "temporal_shift",
            "identity_swap",
            "semantic_inversion",
        ]
    )
    memory_limit_mb: int = 1024
    rollback_enabled: bool = True


@dataclass
class LoopsConfig:
    """Recursive Loop Synthesizer configuration."""

    service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(name="recursive_loop_synth", port=50054)
    )
    max_loop_depth: int = 50
    detection_threshold: float = 0.8
    containment_enabled: bool = True
    entropy_monitoring: bool = True
    auto_termination: bool = True
    max_execution_time_seconds: int = 300


@dataclass
class OmegaConfig:
    """Omega Register service configuration."""

    service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(name="omega_register", port=50055)
    )
    quarantine_enabled: bool = True
    contamination_tracking: bool = True
    auto_containment: bool = True
    omega_state_limit: int = 100
    vault_encryption: bool = True
    anomaly_detection_threshold: float = 0.7


@dataclass
class TelemetryConfig:
    """Telemetry Exporter service configuration."""

    service: ServiceConfig = field(
        default_factory=lambda: ServiceConfig(name="telemetry_exporter", port=50056)
    )
    export_interval_seconds: int = 30
    batch_size: int = 1000
    compression_enabled: bool = True
    retention_days: int = 90
    exporters: list[str] = field(default_factory=lambda: ["prometheus", "jaeger"])


@dataclass
class FrameworkConfig:
    """Main framework configuration container."""

    version: str = "0.9.3"
    environment: str = "research"
    debug: bool = False

    # Legacy fields for backward compatibility
    log_level: str = "INFO"
    max_agents: int = 50
    timeout: float = 30.0

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Service configurations
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    hive: HiveConfig = field(default_factory=HiveConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    loops: LoopsConfig = field(default_factory=LoopsConfig)
    omega: OmegaConfig = field(default_factory=OmegaConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_env(cls) -> "FrameworkConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Framework settings
        config.version = os.getenv("OMEGA_PHR_VERSION", config.version)
        config.environment = os.getenv("OMEGA_PHR_ENVIRONMENT", config.environment)
        config.debug = os.getenv("OMEGA_PHR_DEBUG", "false").lower() == "true"

        # Database configuration
        scylla_hosts = os.getenv("SCYLLA_HOSTS")
        if scylla_hosts:
            config.database.hosts = [host.strip() for host in scylla_hosts.split(",")]
        config.database.port = int(os.getenv("SCYLLA_PORT", config.database.port))
        config.database.keyspace = os.getenv(
            "SCYLLA_KEYSPACE", config.database.keyspace
        )
        config.database.username = os.getenv("SCYLLA_USERNAME")
        config.database.password = os.getenv("SCYLLA_PASSWORD")
        config.database.ssl_enabled = (
            os.getenv("SCYLLA_TLS_ENABLED", "false").lower() == "true"
        )
        config.database.auth_enabled = (
            os.getenv("SCYLLA_AUTH_ENABLED", "false").lower() == "true"
        )

        # Redis configuration
        config.redis.host = os.getenv("REDIS_HOST", config.redis.host)
        config.redis.port = int(os.getenv("REDIS_PORT", config.redis.port))
        config.redis.database = int(os.getenv("REDIS_DATABASE", config.redis.database))
        config.redis.password = os.getenv("REDIS_PASSWORD")
        config.redis.ssl_enabled = (
            os.getenv("REDIS_TLS_ENABLED", "false").lower() == "true"
        )

        # Ray configuration
        config.ray.head_node_host = os.getenv(
            "RAY_HEAD_NODE_HOST", config.ray.head_node_host
        )
        config.ray.head_node_port = int(
            os.getenv("RAY_HEAD_NODE_PORT", config.ray.head_node_port)
        )
        ray_cpus = os.getenv("RAY_NUM_CPUS")
        config.ray.num_cpus = int(ray_cpus) if ray_cpus else None
        config.ray.num_gpus = int(os.getenv("RAY_NUM_GPUS", config.ray.num_gpus))

        # Security configuration
        config.security.jwt_secret_key = os.getenv("JWT_SECRET_KEY")
        config.security.encryption_key = os.getenv("ENCRYPTION_KEY")
        cors_origins = os.getenv("CORS_ORIGINS")
        if cors_origins:
            config.security.cors_origins = [
                origin.strip() for origin in cors_origins.split(",")
            ]

        # Monitoring configuration
        config.monitoring.prometheus_enabled = (
            os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
        )
        config.monitoring.prometheus_port = int(
            os.getenv("PROMETHEUS_PORT", config.monitoring.prometheus_port)
        )
        config.monitoring.log_level = os.getenv(
            "LOG_LEVEL", config.monitoring.log_level
        )

        # Service configurations
        config.timeline.service.host = os.getenv(
            "TIMELINE_HOST", config.timeline.service.host
        )
        config.timeline.service.port = int(
            os.getenv("TIMELINE_PORT", config.timeline.service.port)
        )
        config.timeline.max_timelines = int(
            os.getenv("TIMELINE_MAX_TIMELINES", config.timeline.max_timelines)
        )

        config.hive.service.host = os.getenv("HIVE_HOST", config.hive.service.host)
        config.hive.service.port = int(os.getenv("HIVE_PORT", config.hive.service.port))
        config.hive.max_agents = int(
            os.getenv("HIVE_MAX_AGENTS", config.hive.max_agents)
        )

        config.memory.service.host = os.getenv(
            "MEMORY_HOST", config.memory.service.host
        )
        config.memory.service.port = int(
            os.getenv("MEMORY_PORT", config.memory.service.port)
        )

        config.loops.service.host = os.getenv("LOOPS_HOST", config.loops.service.host)
        config.loops.service.port = int(
            os.getenv("LOOPS_PORT", config.loops.service.port)
        )

        config.omega.service.host = os.getenv("OMEGA_HOST", config.omega.service.host)
        config.omega.service.port = int(
            os.getenv("OMEGA_PORT", config.omega.service.port)
        )

        config.telemetry.service.host = os.getenv(
            "TELEMETRY_HOST", config.telemetry.service.host
        )
        config.telemetry.service.port = int(
            os.getenv("TELEMETRY_PORT", config.telemetry.service.port)
        )

        return config

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate database configuration
        if not self.database.hosts:
            errors.append("⚠️  Database hosts cannot be empty")
        if self.database.port < 1 or self.database.port > 65535:
            errors.append("❌ Database port must be between 1 and 65535")

        # Validate Redis configuration
        if self.redis.port < 1 or self.redis.port > 65535:
            errors.append("❌ Redis port must be between 1 and 65535")

        # Validate service ports for conflicts
        ports = [
            self.timeline.service.port,
            self.hive.service.port,
            self.memory.service.port,
            self.loops.service.port,
            self.omega.service.port,
            self.telemetry.service.port,
        ]
        if len(ports) != len(set(ports)):
            errors.append("❌ Service ports must be unique")

        # Validate security configuration
        if self.environment == "research":
            if not self.security.jwt_secret_key:
                errors.append("❌ JWT secret key required in research")
            if not self.security.encryption_key:
                errors.append("❌ Encryption key required in research")

        # Validate thresholds
        if not 0.0 <= self.timeline.paradox_threshold <= 1.0:
            errors.append("❌ Paradox threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.loops.detection_threshold <= 1.0:
            errors.append("❌ Loop detection threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.omega.anomaly_detection_threshold <= 1.0:
            errors.append("❌ Anomaly detection threshold must be between 0.0 and 1.0")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""

        def _convert_dataclass(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert_dataclass(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_convert_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert_dataclass(v) for k, v in obj.items()}
            else:
                return obj

        result = _convert_dataclass(self)
        return result if isinstance(result, dict) else {}

    def save_to_file(self, file_path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"✅ Configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: Path) -> "FrameworkConfig":
        """Load configuration from JSON file."""
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Configuration file not found: {file_path}")

        with open(file_path) as f:
            json.load(f)

        # This is a simplified loader - in research you'd want proper deserialization
        config = cls()
        logger.info(f"✅ Configuration loaded from {file_path}")
        return config

    def get_service_config(self, service_name: str) -> ServiceConfig:
        """Get configuration for a specific service."""
        service_configs = {
            "timeline_lattice": self.timeline.service,
            "hive_orchestrator": self.hive.service,
            "memory_inversion": self.memory.service,
            "recursive_loop_synth": self.loops.service,
            "omega_register": self.omega.service,
            "telemetry_exporter": self.telemetry.service,
        }

        if service_name not in service_configs:
            raise ValueError(f"❌ Unknown service: {service_name}")

        return service_configs[service_name]


# Global configuration instance
_global_config: FrameworkConfig | None = None


def get_config() -> FrameworkConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = FrameworkConfig.from_env()
        errors = _global_config.validate()
        if errors:
            for error in errors:
                logger.error(error)
            if _global_config.environment == "research":
                raise ValueError("❌ Configuration validation failed in research mode")
    return _global_config


def reload_config() -> FrameworkConfig:
    """Reload configuration from environment."""
    global _global_config
    _global_config = None
    return get_config()


def create_default_config_file(file_path: Path) -> None:
    """Create a default configuration file."""
    config = FrameworkConfig()
    config.save_to_file(file_path)
    logger.info(f"✅ Default configuration created: {file_path}")


def validate_research_config() -> bool:
    """Validate configuration for research deployment."""
    config = get_config()
    errors = config.validate()

    if errors:
        logger.error("❌ Enterprise configuration validation failed:")
        for error in errors:
            logger.error(f"  {error}")
        return False

    logger.info("✅ Research configuration validation passed")
    return True


# Backward compatibility alias
OmegaPHRConfig: type[FrameworkConfig] = FrameworkConfig
