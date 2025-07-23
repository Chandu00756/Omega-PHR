"""
Omega-Paradox Hive Recursion (Ω-PHR) Framework v0.9.3
Timeline Lattice gRPC Server Implementation

This module implements the research gRPC server for the Timeline Lattice service,
providing temporal event management, paradox detection, and causality analysis
through high-performance remote procedure calls.
"""

import asyncio
import logging
import sys

from grpc import aio

from ..timeline_pb2 import (
    AddEventRequest,
    AddEventResponse,
    AnalyzeParadoxRequest,
    AnalyzeParadoxResponse,
    CreateTimelineRequest,
    CreateTimelineResponse,
    GetTimelineStatusRequest,
    GetTimelineStatusResponse,
)
from ..timeline_pb2_grpc import (
    TimelineLatticeServicer,
    add_TimelineLatticeServicer_to_server,
)
from .config import TimelineServiceConfig

# Import service implementation
from .models import TimelineEvent


class SimpleTimelineRepository:
    """Simple in-memory repository for development."""

    def __init__(self, config: TimelineServiceConfig):
        self.config = config
        self.timelines = {}
        self.events = {}

    async def create_timeline(self, name: str, description: str = "") -> str:
        """Create a new timeline."""
        import uuid

        timeline_id = str(uuid.uuid4())
        self.timelines[timeline_id] = {
            "id": timeline_id,
            "name": name,
            "description": description,
            "events": [],
        }
        return timeline_id

    async def add_event(self, timeline_id: str, event: TimelineEvent) -> str:
        """Add an event to a timeline."""
        if timeline_id in self.timelines:
            event_id = str(event.event_id) if event.event_id else str(len(self.events))
            self.events[event_id] = event
            self.timelines[timeline_id]["events"].append(event_id)
            return event_id
        return ""


class TimelineLatticeServer(TimelineLatticeServicer):
    """
    Enterprise gRPC server for Timeline Lattice operations.

    Handles temporal event management, paradox detection, and causality
    analysis through efficient gRPC-based remote procedure calls.
    """

    def __init__(self, config: TimelineServiceConfig):
        """Initialize the Timeline Lattice server."""
        self.config = config
        self.repository = SimpleTimelineRepository(config)
        self.logger = logging.getLogger(__name__)
        self._shutdown_event = asyncio.Event()

    async def CreateTimeline(
        self, request: CreateTimelineRequest, context
    ) -> CreateTimelineResponse:
        """Create a new temporal timeline."""
        try:
            timeline_id = await self.repository.create_timeline(
                name=request.name,
                description=request.description,
                initial_timestamp=request.initial_timestamp,
            )

            self.logger.info(f"Created timeline: {timeline_id}")
            return CreateTimelineResponse(
                timeline_id=timeline_id,
                success=True,
                message="Timeline created successfully",
            )
        except Exception as e:
            self.logger.error(f"Failed to create timeline: {e}")
            return CreateTimelineResponse(
                success=False, message=f"Timeline creation failed: {str(e)}"
            )

    async def AddEvent(self, request: AddEventRequest, context) -> AddEventResponse:
        """Add a temporal event to a timeline."""
        try:
            event = TimelineEvent(
                timeline_id=request.timeline_id,
                event_type=request.event_type,
                timestamp=request.timestamp,
                data=request.data,
                causality_vector=request.causality_vector,
            )

            event_id = await self.repository.add_event(event)

            # Check for paradoxes
            paradox_risk = await self.repository.analyze_paradox_risk(
                request.timeline_id, event
            )

            self.logger.info(
                f"Added event {event_id} to timeline {request.timeline_id}"
            )
            return AddEventResponse(
                event_id=event_id,
                success=True,
                paradox_risk=paradox_risk,
                message="Event added successfully",
            )
        except Exception as e:
            self.logger.error(f"Failed to add event: {e}")
            return AddEventResponse(
                success=False, message=f"Event addition failed: {str(e)}"
            )

    async def AnalyzeParadox(
        self, request: AnalyzeParadoxRequest, context
    ) -> AnalyzeParadoxResponse:
        """Analyze timeline for temporal paradoxes."""
        try:
            paradoxes = await self.repository.detect_paradoxes(request.timeline_id)

            self.logger.info(
                f"Analyzed {len(paradoxes)} paradoxes in timeline {request.timeline_id}"
            )
            return AnalyzeParadoxResponse(
                timeline_id=request.timeline_id,
                paradox_count=len(paradoxes),
                paradoxes=[p.to_proto() for p in paradoxes],
                success=True,
                message="Paradox analysis completed",
            )
        except Exception as e:
            self.logger.error(f"Failed to analyze paradoxes: {e}")
            return AnalyzeParadoxResponse(
                success=False, message=f"Paradox analysis failed: {str(e)}"
            )

    async def GetTimelineStatus(
        self, request: GetTimelineStatusRequest, context
    ) -> GetTimelineStatusResponse:
        """Get comprehensive timeline status and metrics."""
        try:
            status = await self.repository.get_timeline_status(request.timeline_id)

            return GetTimelineStatusResponse(
                timeline_id=request.timeline_id,
                status=status.get("status", "unknown"),
                event_count=status.get("event_count", 0),
                paradox_count=status.get("paradox_count", 0),
                last_event_timestamp=status.get("last_event_timestamp", ""),
                success=True,
                message="Status retrieved successfully",
            )
        except Exception as e:
            self.logger.error(f"Failed to get timeline status: {e}")
            return GetTimelineStatusResponse(
                success=False, message=f"Status retrieval failed: {str(e)}"
            )


async def serve(config: TimelineServiceConfig):
    """Start the Timeline Lattice gRPC server."""
    server = aio.server()

    # Add Timeline Lattice service
    timeline_service = TimelineLatticeServer(config)
    add_TimelineLatticeServicer_to_server(timeline_service, server)

    # Configure server
    listen_addr = f"{config.host}:{config.port}"
    server.add_insecure_port(listen_addr)

    # Start server
    await server.start()
    logging.info(f"Timeline Lattice server started on {listen_addr}")

    # Wait for shutdown
    await timeline_service._shutdown_event.wait()
    await server.stop(30)


def main():
    """Main entry point for Timeline Lattice server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = TimelineServiceConfig()

    try:
        asyncio.run(serve(config))
    except KeyboardInterrupt:
        logging.info("Timeline Lattice server shutdown requested")
    except Exception as e:
        logging.error(f"Timeline Lattice server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
