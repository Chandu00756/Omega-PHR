"""
gRPC server implementation for the Hive Orchestrator service.

This module provides the main gRPC server setup and service implementations
for coordinating distributed adversarial attack agents.
"""

import asyncio
import logging
from typing import Optional

try:
    import grpc
    from grpc import aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

from .config import HiveServiceConfig

logger = logging.getLogger(__name__)


class HiveOrchestratorServicer:
    """Main servicer for Hive Orchestrator gRPC operations."""

    def __init__(self, config: HiveServiceConfig):
        """Initialize the Hive Orchestrator servicer."""
        self.config = config
        self.logger = logger

    async def start(self):
        """Start the hive orchestrator service."""
        self.logger.info("Starting Hive Orchestrator servicer")

    async def stop(self):
        """Stop the hive orchestrator service."""
        self.logger.info("Stopping Hive Orchestrator servicer")


def create_server(config: HiveServiceConfig) -> Optional[aio.Server]:
    """Create and configure the gRPC server."""
    if not GRPC_AVAILABLE:
        logger.error("gRPC not available, cannot create server")
        return None

    server = aio.server()

    # Add servicer to server
    servicer = HiveOrchestratorServicer(config)

    # Configure server options
    options = [
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 5000),
        ('grpc.keepalive_permit_without_calls', True),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 10000),
        ('grpc.http2.min_ping_interval_without_data_ms', 300000),
    ]

    for option in options:
        server.add_generic_rpc_handlers([])

    # Add port
    if config.use_ssl and config.ssl_cert_file and config.ssl_key_file:
        with open(config.ssl_cert_file, 'rb') as f:
            cert_data = f.read()
        with open(config.ssl_key_file, 'rb') as f:
            key_data = f.read()

        credentials = grpc.ssl_server_credentials([(key_data, cert_data)])
        server.add_secure_port(f"{config.host}:{config.port}", credentials)
        logger.info(f"Added secure port {config.host}:{config.port}")
    else:
        server.add_insecure_port(f"{config.host}:{config.port}")
        logger.info(f"Added insecure port {config.host}:{config.port}")

    return server


async def serve(config: HiveServiceConfig):
    """Start the gRPC server and serve requests."""
    if not GRPC_AVAILABLE:
        logger.error("gRPC not available, cannot serve")
        return

    server = create_server(config)
    if not server:
        logger.error("Failed to create server")
        return

    await server.start()
    logger.info(f"Hive Orchestrator server started on {config.host}:{config.port}")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await server.stop(grace=5.0)
        logger.info("Hive Orchestrator server stopped")
