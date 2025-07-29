"""
Main entry point for the Hive Orchestrator service.

This module provides the main server setup and lifecycle management
for the distributed adversarial attack coordination service.
"""

import asyncio
import logging
import signal
import sys

try:
    import structlog
    from structlog import get_logger

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logger = get_logger(__name__)
except ImportError:
    # Fallback to standard logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

try:
    import grpc  # noqa: F401

    GRPC_AVAILABLE = True
except ImportError:
    logger.warning("gRPC not available, service will run in limited mode")
    GRPC_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray not available, distributed features disabled")
    RAY_AVAILABLE = False

from typing import TYPE_CHECKING

from .config import HiveServiceConfig

if TYPE_CHECKING:
    from grpc import aio

if GRPC_AVAILABLE:
    from .server import create_server


class HiveService:
    """
    Main Hive Orchestrator service.

    This service coordinates distributed adversarial attack agents
    and manages the overall attack strategy and intelligence gathering.
    """

    def __init__(self, config: HiveServiceConfig):
        """
        Initialize the Hive service.

        Args:
            config: Service configuration
        """
        self.config = config
        self.server: aio.Server | None = None
        self._shutdown_event = asyncio.Event()

        # Simple validation - just check if config exists
        if not self.config:
            logger.error("No configuration provided")
            raise ValueError("Invalid configuration")

        logger.info("Hive service initialized")

    async def start(self) -> None:
        """Start the Hive service."""
        try:
            logger.info("Starting Hive Orchestrator service")

            # Initialize Ray if available
            await self._init_ray()

            # Start gRPC server if available
            if GRPC_AVAILABLE:
                await self._start_grpc_server()
            else:
                logger.warning("gRPC not available, starting in standalone mode")
                await self._start_standalone_mode()

            logger.info(
                f"Hive service started successfully on {self.config.host}:{self.config.port}"  # noqa: E501
            )

        except Exception as e:
            logger.error(f"Failed to start Hive service: {str(e)}")
            raise

    async def stop(self) -> None:
        """Stop the Hive service."""
        try:
            logger.info("Stopping Hive service")

            # Signal shutdown
            self._shutdown_event.set()

            # Stop gRPC server
            if self.server:
                logger.info("Stopping gRPC server")
                await self.server.stop(grace=30)
                self.server = None

            # Shutdown Ray
            if RAY_AVAILABLE and ray.is_initialized():
                logger.info("Shutting down Ray")
                ray.shutdown()

            logger.info("Hive service stopped")

        except Exception as e:
            logger.error(f"Error stopping Hive service: {str(e)}")
            raise

    async def wait_for_termination(self) -> None:
        """Wait for the service to be terminated."""
        try:
            if self.server:
                await self.server.wait_for_termination()
            else:
                await self._shutdown_event.wait()
        except Exception as e:
            logger.error(f"Error waiting for termination: {str(e)}")

    async def _init_ray(self) -> None:
        """Initialize Ray cluster."""
        if not RAY_AVAILABLE:
            return

        try:
            logger.info(f"Initializing Ray at {self.config.ray_address}")

            if self.config.ray_address:
                ray.init(address=self.config.ray_address)
            else:
                ray.init()

            logger.info(f"Ray cluster ready with {len(ray.nodes())} nodes")

        except Exception as e:
            logger.error(f"Failed to initialize Ray: {str(e)}")
            raise

    async def _start_grpc_server(self) -> None:
        """Start the gRPC server."""
        try:
            self.server = create_server(self.config)

            if self.server:
                await self.server.start()

            logger.info(f"gRPC server started on {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {str(e)}")
            raise

    async def _start_standalone_mode(self) -> None:
        """Start in standalone mode without gRPC."""
        logger.info("Starting in standalone mode")

        # In standalone mode, we would run core hive logic
        # without gRPC endpoints
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1.0)


async def main() -> None:
    """Main async entry point."""
    config = HiveServiceConfig()
    service = HiveService(config)

    # Setup signal handlers
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(service.stop())

    # Register signal handlers
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    try:
        await service.start()
        await service.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        raise
    finally:
        await service.stop()


def sync_main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    sync_main()
