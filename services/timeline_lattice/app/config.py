"""
Timeline Lattice Service - Enterprise Configuration Management

Advanced configuration system with validation, security hardening,
enterprise deployment support, and comprehensive observability.

Enterprise Features:
- Multi-environment configuration (dev/staging/prod)
- Secure credential management with HashiCorp Vault integration
- Advanced database connection pooling and failover
- Comprehensive observability and telemetry configuration
- Security hardening with TLS, mTLS, and certificate management
- High-availability and disaster recovery settings
- Performance tuning and resource management
- Compliance and audit logging configuration
"""

import logging
import os
import secrets
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Environment(str, Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Supported database backends."""

    SCYLLA = "scylla"
    CASSANDRA = "cassandra"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class TLSMode(str, Enum):
    """TLS configuration modes."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    MUTUAL = "mutual"


@dataclass
class SecurityConfig:
    """Security configuration for enterprise deployment."""

    # TLS Configuration
    enable_tls: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_TLS_ENABLED", "true").lower()
        == "true"
    )
    tls_mode: TLSMode = field(
        default_factory=lambda: TLSMode(os.getenv("TIMELINE_TLS_MODE", "enabled"))
    )
    cert_file: str | None = field(
        default_factory=lambda: os.getenv("TIMELINE_TLS_CERT")
    )
    key_file: str | None = field(default_factory=lambda: os.getenv("TIMELINE_TLS_KEY"))
    ca_file: str | None = field(default_factory=lambda: os.getenv("TIMELINE_TLS_CA"))

    # JWT Configuration
    jwt_secret_key: str = field(
        default_factory=lambda: os.getenv(
            "TIMELINE_JWT_SECRET", secrets.token_urlsafe(32)
        )
    )
    jwt_algorithm: str = field(
        default_factory=lambda: os.getenv("TIMELINE_JWT_ALGORITHM", "HS256")
    )
    jwt_expiration_hours: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_JWT_EXPIRATION", "24"))
    )

    # API Security
    enable_api_key_auth: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_API_KEY_AUTH", "false").lower()
        == "true"
    )
    api_key_header: str = field(
        default_factory=lambda: os.getenv("TIMELINE_API_KEY_HEADER", "X-API-Key")
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_RATE_LIMIT", "1000"))
    )

    # Encryption
    encryption_key: str = field(
        default_factory=lambda: os.getenv(
            "TIMELINE_ENCRYPTION_KEY", secrets.token_urlsafe(32)
        )
    )
    enable_event_encryption: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_ENCRYPT_EVENTS", "false").lower()
        == "true"
    )


@dataclass
class DatabaseConfig:
    """Enterprise database configuration with connection pooling and failover."""

    # Database Type and Connection
    database_type: DatabaseType = field(
        default_factory=lambda: DatabaseType(os.getenv("TIMELINE_DB_TYPE", "scylla"))
    )
    hosts: list[str] = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_HOSTS", "127.0.0.1").split(",")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_DB_PORT", "9042"))
    )
    keyspace: str = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_KEYSPACE", "omega_timeline")
    )

    # Authentication
    username: str | None = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_USERNAME")
    )
    password: str | None = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_PASSWORD")
    )

    # Connection Pooling
    max_connections: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_DB_MAX_CONNECTIONS", "100"))
    )
    min_connections: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_DB_MIN_CONNECTIONS", "10"))
    )
    connection_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_DB_TIMEOUT", "30"))
    )

    # High Availability
    enable_load_balancing: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_LOAD_BALANCING", "true").lower()
        == "true"
    )
    dc_preference: str | None = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_DC_PREFERENCE")
    )

    # Performance Tuning
    read_consistency: str = field(
        default_factory=lambda: os.getenv(
            "TIMELINE_DB_READ_CONSISTENCY", "LOCAL_QUORUM"
        )
    )
    write_consistency: str = field(
        default_factory=lambda: os.getenv(
            "TIMELINE_DB_WRITE_CONSISTENCY", "LOCAL_QUORUM"
        )
    )

    # Backup and Recovery
    enable_backup: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_BACKUP", "false").lower()
        == "true"
    )
    backup_schedule: str = field(
        default_factory=lambda: os.getenv("TIMELINE_DB_BACKUP_SCHEDULE", "0 2 * * *")
    )  # Daily at 2 AM

    # SQLite fallback for development
    sqlite_path: str = field(
        default_factory=lambda: os.getenv(
            "TIMELINE_SQLITE_PATH", "/tmp/omega_timeline.db"  # noqa: S108
        )
    )
    sqlite_wal_mode: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_SQLITE_WAL", "true").lower()
        == "true"
    )


@dataclass
class ServerConfig:
    """gRPC server configuration with enterprise features."""

    # Basic Server Settings
    host: str = field(
        default_factory=lambda: os.getenv("TIMELINE_HOST", "0.0.0.0")  # noqa: S104
    )  # noqa: S104
    port: int = field(default_factory=lambda: int(os.getenv("TIMELINE_PORT", "50051")))
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_MAX_WORKERS", "10"))
    )

    # Performance Configuration
    max_receive_message_length: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_MAX_MESSAGE_SIZE", "4194304"))
    )  # 4MB
    max_send_message_length: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_MAX_SEND_SIZE", "4194304"))
    )  # 4MB
    keepalive_time_ms: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_KEEPALIVE_TIME", "30000"))
    )  # 30s
    keepalive_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_KEEPALIVE_TIMEOUT", "5000"))
    )  # 5s

    # Health Check Configuration
    enable_health_check: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_HEALTH_CHECK", "true").lower()
        == "true"
    )
    health_check_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_HEALTH_INTERVAL", "30"))
    )

    # Reflection and Development
    enable_reflection: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_REFLECTION", "false").lower()
        == "true"
    )


@dataclass
class ObservabilityConfig:
    """Comprehensive observability and monitoring configuration."""

    # Logging Configuration
    log_level: LogLevel = field(
        default_factory=lambda: LogLevel(os.getenv("TIMELINE_LOG_LEVEL", "INFO"))
    )
    log_format: str = field(
        default_factory=lambda: os.getenv("TIMELINE_LOG_FORMAT", "json")
    )
    log_file: str | None = field(default_factory=lambda: os.getenv("TIMELINE_LOG_FILE"))
    enable_structured_logging: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_STRUCTURED_LOGS", "true").lower()
        == "true"
    )

    # Metrics Configuration
    enable_metrics: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_METRICS", "true").lower() == "true"
    )
    metrics_port: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_METRICS_PORT", "9090"))
    )
    metrics_path: str = field(
        default_factory=lambda: os.getenv("TIMELINE_METRICS_PATH", "/metrics")
    )

    # Tracing Configuration
    enable_tracing: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_TRACING", "false").lower() == "true"
    )
    jaeger_endpoint: str | None = field(
        default_factory=lambda: os.getenv("TIMELINE_JAEGER_ENDPOINT")
    )
    trace_sample_rate: float = field(
        default_factory=lambda: float(os.getenv("TIMELINE_TRACE_SAMPLE_RATE", "0.1"))
    )

    # Performance Monitoring
    enable_profiling: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_PROFILING", "false").lower()
        == "true"
    )
    profile_cpu: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_PROFILE_CPU", "false").lower()
        == "true"
    )
    profile_memory: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_PROFILE_MEMORY", "false").lower()
        == "true"
    )


@dataclass
class TimelineConfig:
    """Timeline-specific configuration and business logic settings."""

    # Timeline Limits
    max_timelines: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_MAX_TIMELINES", "10000"))
    )
    max_events_per_timeline: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_MAX_EVENTS", "1000000"))
    )
    max_timeline_depth: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_MAX_DEPTH", "100"))
    )

    # Paradox Detection
    paradox_threshold: float = field(
        default_factory=lambda: float(os.getenv("TIMELINE_PARADOX_THRESHOLD", "0.1"))
    )
    enable_auto_paradox_detection: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_AUTO_PARADOX", "true").lower()
        == "true"
    )
    paradox_analysis_window: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_PARADOX_WINDOW", "100"))
    )

    # Temporal Operations
    enable_timeline_branching: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_BRANCHING", "true").lower()
        == "true"
    )
    enable_timeline_merging: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_MERGING", "true").lower() == "true"
    )
    enable_timeline_rewind: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_REWIND", "true").lower() == "true"
    )

    # Caching and Performance
    enable_event_cache: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_CACHE_EVENTS", "true").lower()
        == "true"
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_CACHE_TTL", "300"))
    )
    cache_max_size: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_CACHE_SIZE", "10000"))
    )


@dataclass
class ComplianceConfig:
    """Compliance and audit configuration for enterprise deployments."""

    # Audit Logging
    enable_audit_logging: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_AUDIT_LOGGING", "false").lower()
        == "true"
    )
    audit_log_file: str | None = field(
        default_factory=lambda: os.getenv("TIMELINE_AUDIT_LOG_FILE")
    )
    audit_log_format: str = field(
        default_factory=lambda: os.getenv("TIMELINE_AUDIT_FORMAT", "json")
    )

    # Data Retention
    data_retention_days: int = field(
        default_factory=lambda: int(os.getenv("TIMELINE_DATA_RETENTION", "365"))
    )
    enable_auto_cleanup: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_AUTO_CLEANUP", "false").lower()
        == "true"
    )

    # Privacy and GDPR
    enable_data_anonymization: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_ANONYMIZATION", "false").lower()
        == "true"
    )
    pii_detection_enabled: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_PII_DETECTION", "false").lower()
        == "true"
    )


@dataclass
class TimelineServiceConfig:
    """Master configuration class for Timeline Lattice service."""

    # Environment
    environment: Environment = field(
        default_factory=lambda: Environment(
            os.getenv("TIMELINE_ENVIRONMENT", "development")
        )
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("TIMELINE_DEBUG", "false").lower() == "true"
    )

    # Configuration Components
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

    # Advanced Configuration
    config_file: str | None = field(
        default_factory=lambda: os.getenv("TIMELINE_CONFIG_FILE")
    )
    secrets_backend: str = field(
        default_factory=lambda: os.getenv("TIMELINE_SECRETS_BACKEND", "environment")
    )

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Load configuration from file if specified
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file()

        # Validate configuration
        self._validate_configuration()

        # Setup logging based on configuration
        self._setup_logging()

    def _load_from_file(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_file:
            return

        try:
            with open(self.config_file) as f:
                config_data = yaml.safe_load(f)

            # Update configuration with file values
            for section, values in config_data.items():
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)

        except Exception as e:
            logging.error(f"Failed to load configuration from {self.config_file}: {e}")
            raise

    def _validate_configuration(self) -> None:
        """Validate configuration values."""
        errors = []

        # Validate server configuration
        if not (1 <= self.server.port <= 65535):
            errors.append(f"Invalid server port: {self.server.port}")

        if self.server.max_workers < 1:
            errors.append(f"Invalid max_workers: {self.server.max_workers}")

        # Validate database configuration
        if not self.database.hosts:
            errors.append("Database hosts cannot be empty")

        # Validate timeline configuration
        if self.timeline.paradox_threshold < 0 or self.timeline.paradox_threshold > 1:
            errors.append(
                f"Paradox threshold must be between 0 and 1: {self.timeline.paradox_threshold}"  # noqa: E501
            )

        # Validate security configuration
        if self.security.enable_tls and not all(
            [self.security.cert_file, self.security.key_file]
        ):
            errors.append("TLS enabled but cert_file or key_file not specified")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = getattr(logging, self.observability.log_level.value)

        if self.observability.enable_structured_logging:
            # Setup structured logging with JSON format
            import structlog

            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

        # Configure standard logging
        handlers: list[Any] = [logging.StreamHandler()]
        if self.observability.log_file:
            handlers.append(logging.FileHandler(self.observability.log_file))

        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    @classmethod
    def from_env(cls) -> "TimelineServiceConfig":
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_file(cls, config_file: str) -> "TimelineServiceConfig":
        """Create configuration from file."""
        config = cls()
        config.config_file = config_file
        config._load_from_file()
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""

        def _convert_dataclass(obj) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert_dataclass(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [_convert_dataclass(item) for item in obj]
            else:
                return obj

        result = _convert_dataclass(self)
        return result if isinstance(result, dict) else {}

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        try:
            self._validate_configuration()
            return []
        except ValueError as e:
            return [str(e)]

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT


def create_research_config() -> TimelineServiceConfig:
    """Create configuration optimized for research environments."""
    config = TimelineServiceConfig()
    config.environment = Environment.DEVELOPMENT
    config.debug = True
    config.timeline.enable_auto_paradox_detection = True
    config.observability.enable_tracing = True
    config.observability.enable_profiling = True
    config.observability.log_level = LogLevel.DEBUG
    return config


# Configuration validation helpers
def validate_ssl_config(
    use_ssl: bool, ssl_cert_file: str | None, ssl_key_file: str | None
) -> list[str]:
    """Validate SSL configuration."""
    errors = []
    if use_ssl and (not ssl_cert_file or not ssl_key_file):
        errors.append("SSL cert and key files required when SSL is enabled")

    if use_ssl:
        cert_path = Path(ssl_cert_file) if ssl_cert_file else None
        key_path = Path(ssl_key_file) if ssl_key_file else None

        if cert_path and not cert_path.exists():
            errors.append(f"SSL cert file not found: {ssl_cert_file}")

        if key_path and not key_path.exists():
            errors.append(f"SSL key file not found: {ssl_key_file}")

    return errors


def validate_paradox_threshold(threshold: float) -> list[str]:
    """Validate paradox threshold."""
    errors = []
    if threshold < 0.0 or threshold > 1.0:
        errors.append("Paradox threshold must be between 0.0 and 1.0")
    return errors


def validate_timeline_limits(
    max_timelines: int, max_events_per_timeline: int
) -> list[str]:
    """Validate timeline limits."""
    errors = []
    if max_timelines < 1:
        errors.append("Max timelines must be at least 1")

    if max_events_per_timeline < 1:
        errors.append("Max events per timeline must be at least 1")

    return errors


def validate_database_config(
    scylla_offline: bool, scylla_hosts: list[str]
) -> list[str]:
    """Validate database configuration."""
    errors = []
    if not scylla_offline and not scylla_hosts:
        errors.append("Scylla hosts required when not in offline mode")

    return errors


def validate_auth_config(enable_auth: bool, jwt_secret: str | None) -> list[str]:
    """Validate authentication configuration."""
    errors = []
    if enable_auth and not jwt_secret:
        errors.append("JWT secret required when authentication is enabled")

    return errors


def config_to_dict(config: TimelineServiceConfig) -> dict[str, Any]:
    """Convert configuration to dictionary."""
    return {
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "max_workers": config.server.max_workers,
        },
        "database": {
            "environment": config.environment.value,
        },
        "timeline": {
            "debug": config.debug,
        },
        "observability": {
            "log_level": config.observability.log_level.value,
        },
    }

    @classmethod
    def from_env(cls) -> "TimelineServiceConfig":
        """Create configuration from environment variables."""
        return cls()

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def get_config() -> TimelineServiceConfig:
    """Get service configuration from environment."""
    config = TimelineServiceConfig.from_env()

    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)

    return config


def load_config_from_file(config_path: str) -> TimelineServiceConfig:
    """Load configuration from YAML file."""
    try:
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Map YAML structure to environment variables
        env_mapping = {
            "server.host": "TIMELINE_HOST",
            "server.port": "TIMELINE_PORT",
            "server.max_workers": "TIMELINE_MAX_WORKERS",
            "database.scylla_hosts": "SCYLLA_HOSTS",
            "database.scylla_keyspace": "SCYLLA_KEYSPACE",
            "database.scylla_offline": "SCYLLA_OFFLINE",
            "timeline.max_timelines": "MAX_TIMELINES",
            "timeline.paradox_threshold": "PARADOX_THRESHOLD",
        }

        # Set environment variables from YAML
        for yaml_path, env_var in env_mapping.items():
            value = data
            for key in yaml_path.split("."):
                value = value.get(key)
                if value is None:
                    break

            if value is not None:
                if isinstance(value, list):
                    os.environ[env_var] = ",".join(map(str, value))
                else:
                    os.environ[env_var] = str(value)

        return get_config()

    except ImportError:
        raise ImportError("PyYAML required for YAML configuration files") from None
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        ) from None
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}") from e


def create_development_config() -> TimelineServiceConfig:
    """Create configuration optimized for development."""
    os.environ.update(
        {
            "TIMELINE_HOST": "127.0.0.1",
            "TIMELINE_PORT": "50051",
            "SCYLLA_OFFLINE": "true",
            "SQLITE_PATH": "/tmp/omega_timeline_dev.db",  # noqa: S108
            "LOG_LEVEL": "DEBUG",
            "ENABLE_METRICS": "true",
            "PARADOX_DETECTION_ENABLED": "true",
        }
    )

    return get_config()
