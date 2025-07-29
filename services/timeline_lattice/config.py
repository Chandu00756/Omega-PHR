"""
Configuration management for Timeline Lattice service.
"""

import os
from dataclasses import dataclass


@dataclass
class TimelineConfig:
    """Configuration for Timeline Lattice service."""

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 9000
    max_workers: int = 10

    # Database configuration
    from dataclasses import field

    scylla_hosts: list[str] = field(default_factory=list)
    keyspace: str = "timeline_lattice"

    # Timeline configuration
    max_timelines: int = 1000
    paradox_threshold: float = 0.1

    # Performance configuration
    max_events_per_timeline: int = 10000
    consistency_cache_size: int = 1000

    # Security configuration
    require_signatures: bool = True
    max_event_size: int = 1024 * 1024  # 1MB

    # Monitoring configuration
    metrics_enabled: bool = True
    metrics_port: int = 9001

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if not self.scylla_hosts:
            self.scylla_hosts = ["127.0.0.1"]
            self.scylla_hosts = ["127.0.0.1"]


def get_config() -> TimelineConfig:
    """Get configuration from environment variables or defaults."""

    config = TimelineConfig()

    # Server configuration
    config.host = os.getenv("TIMELINE_HOST", config.host)
    config.port = int(os.getenv("TIMELINE_PORT", config.port))
    config.max_workers = int(os.getenv("TIMELINE_MAX_WORKERS", config.max_workers))

    # Database configuration
    scylla_hosts_env = os.getenv("SCYLLA_HOSTS")
    if scylla_hosts_env:
        config.scylla_hosts = scylla_hosts_env.split(",")

    config.keyspace = os.getenv("SCYLLA_KEYSPACE", config.keyspace)

    # Timeline configuration
    config.max_timelines = int(os.getenv("MAX_TIMELINES", config.max_timelines))
    config.paradox_threshold = float(
        os.getenv("PARADOX_THRESHOLD", config.paradox_threshold)
    )

    # Performance configuration
    config.max_events_per_timeline = int(
        os.getenv("MAX_EVENTS_PER_TIMELINE", config.max_events_per_timeline)
    )
    config.consistency_cache_size = int(
        os.getenv("CONSISTENCY_CACHE_SIZE", config.consistency_cache_size)
    )

    # Security configuration
    config.require_signatures = (
        os.getenv("REQUIRE_SIGNATURES", "true").lower() == "true"
    )
    config.max_event_size = int(os.getenv("MAX_EVENT_SIZE", config.max_event_size))

    # Monitoring configuration
    config.metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    config.metrics_port = int(os.getenv("METRICS_PORT", config.metrics_port))

    return config
