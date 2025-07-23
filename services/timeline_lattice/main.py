"""
Timeline Lattice Service - Main Entry Point

This service implements the Timeline Lattice gRPC server for temporal
paradox testing and timeline manipulation.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.timeline_lattice.server import TimelineLatticeServer  # Ensure TimelineLatticeServer is defined in server.py
from services.timeline_lattice.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('timeline_lattice.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the Timeline Lattice service."""
    logger.info("Starting Timeline Lattice Service")

    # Load configuration
    config = get_config()

    # Create and start server
    server = TimelineLatticeServer(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(server.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await server.start()
        logger.info("Timeline Lattice Service started successfully")

        # Keep the server running
        await server.wait_for_termination()

    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await server.stop()
        logger.info("Timeline Lattice Service stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
