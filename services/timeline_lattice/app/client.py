"""
Omega-Paradox Hive Recursion (Ω-PHR) Framework v0.9.3
Timeline Lattice gRPC Client

This module provides a research gRPC client for connecting to and interacting
with the Timeline Lattice service, enabling remote temporal oper                    paradox = TemporalParadox(
                        paradox_id=p.get('paradox_id', ''),
                        timeline_id=timeline_id,
                        paradox_type=p.get('paradox_type', ParadoxType.TEMPORAL_INCONSISTENCY),
                        severity=p.get('severity', 0.0),
                        description=p.get('description', ''),
                        affected_events=p.get('affected_events', []),
                        detected_at=datetime.now(timezone.utc)
                    )d
paradox analysis.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

try:
    import grpc
    from grpc import aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    grpc = None
    aio = None

from .models import TimelineEvent, TemporalParadox, EventType
from .config import TimelineConfig


class TimelineLatticeClient:
    """
    Enterprise gRPC client for Timeline Lattice service.

    Provides high-level interface for temporal operations including
    timeline management, event creation, and paradox analysis.
    """

    def __init__(self, host: str = "localhost", port: int = 50051, timeout: float = 30.0):
        """
        Initialize Timeline Lattice client.

        Args:
            host: Server hostname
            port: Server port
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.address = f"{host}:{port}"
        self.channel: Optional[Any] = None
        self.stub: Optional[Any] = None
        self.logger = logging.getLogger(__name__)

        if not GRPC_AVAILABLE:
            self.logger.warning("gRPC not available, client will operate in mock mode")

    async def connect(self) -> None:
        """Establish connection to Timeline Lattice server."""
        if not GRPC_AVAILABLE:
            self.logger.info("Mock connection established to Timeline Lattice")
            return

        try:
            if aio is not None:
                self.channel = aio.insecure_channel(self.address)
            else:
                self.logger.warning("gRPC not available, using mock connection")
                return

            # Import here to avoid circular imports
            from ..timeline_pb2_grpc import TimelineLatticeStub
            self.stub = TimelineLatticeStub(self.channel)

            # Test connection
            await self.health_check()
            self.logger.info(f"Connected to Timeline Lattice at {self.address}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Timeline Lattice: {e}")
            raise ConnectionError(f"Timeline Lattice connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to Timeline Lattice server."""
        if self.channel and GRPC_AVAILABLE:
            await self.channel.close()
            self.logger.info("Disconnected from Timeline Lattice")

    async def health_check(self) -> bool:
        """Check server health status."""
        if not GRPC_AVAILABLE:
            return True

        try:
            # This would call a health check RPC in a real implementation
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def create_timeline(self,
                            name: str,
                            description: str = "",
                            initial_timestamp: Optional[datetime] = None) -> str:
        """
        Create a new temporal timeline.

        Args:
            name: Timeline name
            description: Timeline description
            initial_timestamp: Initial timeline timestamp

        Returns:
            Timeline ID
        """
        if initial_timestamp is None:
            initial_timestamp = datetime.now(timezone.utc)

        # Generate timeline ID
        timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"

        if not GRPC_AVAILABLE:
            # Mock implementation
            self.logger.info(f"Mock: Created timeline {timeline_id} - {name}")
            return timeline_id

        try:
            from ..timeline_pb2 import CreateTimelineRequest

            request = CreateTimelineRequest(
                name=name,
                description=description,
                initial_timestamp=initial_timestamp.isoformat()
            )

            if self.stub is not None:
                response = await self.stub.CreateTimeline(request, timeout=self.timeout)
            else:
                # Mock response when gRPC is not available
                mock_timeline_id = timeline_id  # Capture the timeline_id in local scope
                class MockResponse:
                    success = True
                    timeline_id = mock_timeline_id
                    message = "Mock timeline created"
                response = MockResponse()

            if response.success:
                self.logger.info(f"Created timeline: {response.timeline_id}")
                return response.timeline_id
            else:
                raise RuntimeError(f"Timeline creation failed: {response.message}")

        except Exception as e:
            self.logger.error(f"Failed to create timeline: {e}")
            raise

    async def add_event(self,
                       timeline_id: str,
                       event_type: EventType,
                       timestamp: datetime,
                       data: Dict[str, Any],
                       causality_vector: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a temporal event to a timeline.

        Args:
            timeline_id: Target timeline ID
            event_type: Type of temporal event
            timestamp: Event timestamp
            data: Event data payload
            causality_vector: Causal relationship data

        Returns:
            Event ID
        """
        if not GRPC_AVAILABLE:
            # Mock implementation
            import uuid
            event_id = str(uuid.uuid4())
            self.logger.info(f"Mock: Added event {event_id} to timeline {timeline_id}")
            return event_id

        try:
            from ..timeline_pb2 import AddEventRequest

            request = AddEventRequest(
                timeline_id=timeline_id,
                event_type=event_type.value,
                timestamp=timestamp.isoformat(),
                data=str(data),  # Convert to string for protobuf
                causality_vector=str(causality_vector) if causality_vector else ""
            )

            if self.stub is not None:
                response = await self.stub.AddEvent(request, timeout=self.timeout)
            else:
                # Mock response when gRPC is not available
                import uuid
                class MockResponse:
                    success = True
                    event_id = f"event_{uuid.uuid4().hex[:8]}"
                    message = "Mock event added"
                    paradox_risk = 0.0
                response = MockResponse()

            if response.success:
                self.logger.info(f"Added event: {response.event_id}")
                if response.paradox_risk > 0.1:
                    self.logger.warning(f"High paradox risk detected: {response.paradox_risk}")
                return response.event_id
            else:
                raise RuntimeError(f"Event addition failed: {response.message}")

        except Exception as e:
            self.logger.error(f"Failed to add event: {e}")
            raise

    async def analyze_paradoxes(self, timeline_id: str) -> List[TemporalParadox]:
        """
        Analyze timeline for temporal paradoxes.

        Args:
            timeline_id: Timeline to analyze

        Returns:
            List of detected paradoxes
        """
        if not GRPC_AVAILABLE:
            # Mock implementation
            from .models import ParadoxType
            mock_paradox = TemporalParadox(
                paradox_id="mock_paradox_001",
                timeline_id=timeline_id,
                paradox_type=ParadoxType.CAUSAL_LOOP,
                severity=0.5,
                description="Mock paradox for testing",
                affected_events=["event_001", "event_002"],
                detected_at=datetime.now(timezone.utc)
            )
            self.logger.info(f"Mock: Analyzed paradoxes in timeline {timeline_id}")
            return [mock_paradox]

        try:
            from ..timeline_pb2 import AnalyzeParadoxRequest

            request = AnalyzeParadoxRequest(timeline_id=timeline_id)

            if self.stub is not None:
                response = await self.stub.AnalyzeParadox(request, timeout=self.timeout)
            else:
                # Mock response when gRPC is not available
                class MockResponse:
                    success = True
                    paradoxes = []
                    message = "Mock paradox analysis"
                response = MockResponse()

            if response.success:
                # Convert protobuf paradoxes to model objects
                paradoxes = []
                for p in response.paradoxes:
                    # This would properly deserialize protobuf in real implementation
                    paradox = TemporalParadox(
                        paradox_id=p.get('paradox_id', ''),
                        timeline_id=timeline_id,
                        paradox_type=p.get('paradox_type', 'unknown'),
                        severity=p.get('severity', 0.0),
                        description=p.get('description', ''),
                        affected_events=p.get('affected_events', []),
                        detected_at=datetime.now(timezone.utc)
                    )
                    paradoxes.append(paradox)

                self.logger.info(f"Analyzed {len(paradoxes)} paradoxes in timeline {timeline_id}")
                return paradoxes
            else:
                raise RuntimeError(f"Paradox analysis failed: {response.message}")

        except Exception as e:
            self.logger.error(f"Failed to analyze paradoxes: {e}")
            raise

    async def get_timeline_status(self, timeline_id: str) -> Dict[str, Any]:
        """
        Get comprehensive timeline status and metrics.

        Args:
            timeline_id: Timeline to query

        Returns:
            Timeline status information
        """
        if not GRPC_AVAILABLE:
            # Mock implementation
            status = {
                'timeline_id': timeline_id,
                'status': 'stable',
                'event_count': 42,
                'paradox_count': 3,
                'last_event_timestamp': datetime.now(timezone.utc).isoformat(),
                'entropy_level': 0.25,
                'stability_index': 0.87
            }
            self.logger.info(f"Mock: Retrieved status for timeline {timeline_id}")
            return status

        try:
            from ..timeline_pb2 import GetTimelineStatusRequest

            request = GetTimelineStatusRequest(timeline_id=timeline_id)

            if self.stub is not None:
                response = await self.stub.GetTimelineStatus(request, timeout=self.timeout)
            else:
                # Mock response when gRPC is not available
                class MockResponse:
                    def __init__(self, tid):
                        self.success = True
                        self.timeline_id = tid
                        self.status = "active"
                        self.event_count = 0
                        self.paradox_count = 0
                        self.last_event_timestamp = ""
                        self.message = "Mock timeline status"
                response = MockResponse(timeline_id)

            if response.success:
                status = {
                    'timeline_id': response.timeline_id,
                    'status': response.status,
                    'event_count': response.event_count,
                    'paradox_count': response.paradox_count,
                    'last_event_timestamp': response.last_event_timestamp
                }
                self.logger.info(f"Retrieved status for timeline {timeline_id}")
                return status
            else:
                raise RuntimeError(f"Status retrieval failed: {response.message}")

        except Exception as e:
            self.logger.error(f"Failed to get timeline status: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class TimelineClientPool:
    """Pool of Timeline Lattice clients for high-throughput operations."""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 50051,
                 pool_size: int = 10,
                 timeout: float = 30.0):
        """Initialize client pool."""
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.timeout = timeout
        self.clients: List[TimelineLatticeClient] = []
        self.available_clients: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize client pool."""
        for _ in range(self.pool_size):
            client = TimelineLatticeClient(self.host, self.port, self.timeout)
            await client.connect()
            self.clients.append(client)
            await self.available_clients.put(client)

        self.logger.info(f"Initialized Timeline client pool with {self.pool_size} clients")

    async def get_client(self) -> TimelineLatticeClient:
        """Get an available client from the pool."""
        return await self.available_clients.get()

    async def return_client(self, client: TimelineLatticeClient) -> None:
        """Return a client to the pool."""
        await self.available_clients.put(client)

    async def close(self) -> None:
        """Close all clients in the pool."""
        for client in self.clients:
            await client.disconnect()
        self.logger.info("Closed Timeline client pool")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience function for single operations
async def create_timeline_client(host: str = "localhost", port: int = 50051) -> TimelineLatticeClient:
    """Create and connect a Timeline Lattice client."""
    client = TimelineLatticeClient(host, port)
    await client.connect()
    return client


if __name__ == "__main__":
    # Client testing and validation
    async def test_client():
        """Test Timeline Lattice client operations."""
        logging.basicConfig(level=logging.INFO)

        async with TimelineLatticeClient() as client:
            # Create timeline
            timeline_id = await client.create_timeline(
                name="test_timeline",
                description="Enterprise test timeline"
            )

            # Add events
            event_id = await client.add_event(
                timeline_id=timeline_id,
                event_type=EventType.TEMPORAL_SHIFT,
                timestamp=datetime.now(timezone.utc),
                data={"test": "research_event"}
            )

            # Analyze paradoxes
            paradoxes = await client.analyze_paradoxes(timeline_id)
            print(f"Detected {len(paradoxes)} paradoxes")

            # Get status
            status = await client.get_timeline_status(timeline_id)
            print(f"Timeline status: {status}")

    # Run test
    asyncio.run(test_client())
