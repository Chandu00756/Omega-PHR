"""
Timeline Lattice Service - Main Entry Point

This service implements the Timeline Lattice gRPC server for temporal
paradox testing and timeline manipulation.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.timeline_lattice.config import get_config  # noqa: E402
from services.timeline_lattice.server import serve  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("timeline_lattice.log")],
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the Timeline Lattice service."""
    logger.info("Starting Timeline Lattice Service")

    # Load configuration
    config = get_config()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the gRPC server
        await serve(host=config.host, port=config.port)
        logger.info("Timeline Lattice Service started successfully")

    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Timeline Lattice Service stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
