"""
Configuration for Telemetry Exporter Service.
Research-grade stability settings.
"""

import os
from dataclasses import dataclass


@dataclass
class TelemetryExporterConfig:
    """Configuration for Telemetry Exporter Service."""

    # Service settings
    host: str = "localhost"
    port: int = 50056
    log_level: str = "INFO"

    # Buffer settings
    max_buffer_size: int = 10000
    buffer_flush_interval: int = 60  # seconds
    max_memory_usage_mb: int = 500

    # Export settings
    export_interval: int = 60  # seconds
    export_timeout: int = 30  # seconds
    max_export_retries: int = 3
    export_batch_size: int = 1000

    # Aggregation settings
    aggregation_window: int = 60  # seconds
    histogram_buckets: str = "0.1,0.5,1,2.5,5,10"  # comma-separated
    enable_percentiles: bool = True

    # Storage settings
    temp_storage_path: str = "/tmp/telemetry"
    max_disk_usage_mb: int = 1000
    compression_enabled: bool = True

    # Cleanup settings
    cleanup_interval: int = 3600  # seconds
    data_retention_hours: int = 24

    @classmethod
    def from_env(cls) -> "TelemetryExporterConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("TELEMETRY_HOST", "localhost"),
            port=int(os.getenv("TELEMETRY_PORT", "50056")),
            log_level=os.getenv("TELEMETRY_LOG_LEVEL", "INFO"),
            max_buffer_size=int(os.getenv("TELEMETRY_MAX_BUFFER_SIZE", "10000")),
            buffer_flush_interval=int(os.getenv("TELEMETRY_FLUSH_INTERVAL", "60")),
            max_memory_usage_mb=int(os.getenv("TELEMETRY_MAX_MEMORY_MB", "500")),
            export_interval=int(os.getenv("TELEMETRY_EXPORT_INTERVAL", "60")),
            export_timeout=int(os.getenv("TELEMETRY_EXPORT_TIMEOUT", "30")),
            max_export_retries=int(os.getenv("TELEMETRY_MAX_RETRIES", "3")),
            export_batch_size=int(os.getenv("TELEMETRY_BATCH_SIZE", "1000")),
            aggregation_window=int(os.getenv("TELEMETRY_AGGREGATION_WINDOW", "60")),
            histogram_buckets=os.getenv(
                "TELEMETRY_HISTOGRAM_BUCKETS", "0.1,0.5,1,2.5,5,10"
            ),
            enable_percentiles=os.getenv("TELEMETRY_ENABLE_PERCENTILES", "true").lower()
            == "true",
            temp_storage_path=os.getenv("TELEMETRY_TEMP_PATH", "/tmp/telemetry"),
            max_disk_usage_mb=int(os.getenv("TELEMETRY_MAX_DISK_MB", "1000")),
            compression_enabled=os.getenv("TELEMETRY_COMPRESSION", "true").lower()
            == "true",
            cleanup_interval=int(os.getenv("TELEMETRY_CLEANUP_INTERVAL", "3600")),
            data_retention_hours=int(os.getenv("TELEMETRY_RETENTION_HOURS", "24")),
        )
