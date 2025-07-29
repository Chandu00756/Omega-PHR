"""
Timeline Lattice gRPC service server implementation.

This module implements the gRPC server for the Timeline Lattice service,
connecting the protobuf interface with the core TimelineLattice engine.
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

import grpc
import timeline_pb2_grpc
from config import TimelineServiceConfig
from grpc import aio
from models import EventModel, ParadoxResult, TimelineInfo
from structlog import get_logger
from timeline_pb2 import (
    AppendEventRequest,
    AppendEventResponse,
    BranchTimelineRequest,
    BranchTimelineResponse,
    Event,
    GetTimelineInfoRequest,
    GetTimelineInfoResponse,
    ListTimelinesRequest,
    ListTimelinesResponse,
    MergeTimelinesRequest,
    MergeTimelinesResponse,
)
from timeline_pb2 import ParadoxResult as ProtoParadoxResult
from timeline_pb2 import (
    RewindTimelineRequest,
    RewindTimelineResponse,
    TestParadoxRequest,
    TestParadoxResponse,
)
from timeline_pb2 import TimelineInfo as ProtoTimelineInfo

from omega_phr.models import Event as CoreEvent
from omega_phr.models import ParadoxResult as CoreParadoxResult
from omega_phr.timeline import TimelineLattice

logger = get_logger(__name__)


class TimelineService(timeline_pb2_grpc.TimelineServiceServicer):
    """
    gRPC service implementation for Timeline Lattice operations.

    This service provides gRPC endpoints for temporal paradox testing
    and timeline manipulation operations.
    """

    def __init__(self, config: TimelineServiceConfig):
        """Initialize the Timeline service."""
        self.config = config
        self.timeline_lattice = TimelineLattice()
        self.active_timelines: Dict[str, TimelineInfo] = {}
        logger.info("Timeline service initialized", config=config.__dict__)

    async def AppendEvent(
        self, request: AppendEventRequest, context: grpc.aio.ServicerContext
    ) -> AppendEventResponse:
        """
        Append an event to a timeline.

        Args:
            request: The append event request
            context: gRPC service context

        Returns:
            AppendEventResponse with success status
        """
        try:
            logger.info(
                "Processing append event request",
                timeline_id=request.timeline_id,
                event_id=request.event.event_id,
            )

            # Convert protobuf event to internal model
            event_model = EventModel.from_proto(request.event)

            # Verify event signature
            if not event_model.verify_signature():
                logger.warning(
                    "Event signature verification failed", event_id=event_model.event_id
                )
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Event signature verification failed",
                )

            # Convert to core event model
            core_event = CoreEvent(
                event_id=event_model.event_id,
                timeline_id=event_model.timeline_id,
                data={"payload": event_model.payload.decode("utf-8", errors="ignore")},
                timestamp=event_model.valid_at_us / 1_000_000,  # Convert to seconds
                actor_id=event_model.actor_id,
                event_type=event_model.event_type,
                metadata=event_model.metadata,
            )

            # Append to timeline lattice
            await self.timeline_lattice.append_event(core_event)

            # Update timeline info
            if request.timeline_id not in self.active_timelines:
                self.active_timelines[request.timeline_id] = TimelineInfo(
                    timeline_id=request.timeline_id,
                    event_count=0,
                    created_at=int(time.time() * 1_000_000),
                )

            timeline_info = self.active_timelines[request.timeline_id]
            timeline_info.event_count += 1

            # Check for paradoxes
            paradox_result = await self.timeline_lattice.test_paradox(
                request.timeline_id
            )
            if paradox_result.has_paradox:
                timeline_info.has_paradoxes = True
                timeline_info.entropy_level = paradox_result.entropy_score

                logger.warning(
                    "Paradox detected after event append",
                    timeline_id=request.timeline_id,
                    event_id=event_model.event_id,
                    paradox_type=paradox_result.paradox_type,
                )

            response = AppendEventResponse()
            response.success = True
            response.message = "Event appended successfully"

            logger.info(
                "Event appended successfully",
                timeline_id=request.timeline_id,
                event_id=event_model.event_id,
            )

            return response

        except Exception as e:
            logger.error(
                "Error appending event", error=str(e), timeline_id=request.timeline_id
            )
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    async def BranchTimeline(
        self, request: BranchTimelineRequest, context: grpc.aio.ServicerContext
    ) -> BranchTimelineResponse:
        """
        Create a branch from an existing timeline.

        Args:
            request: The branch timeline request
            context: gRPC service context

        Returns:
            BranchTimelineResponse with new branch ID
        """
        try:
            logger.info(
                "Processing branch timeline request",
                source_timeline_id=request.source_timeline_id,
                branch_point_us=request.branch_point_us,
            )

            # Create branch in timeline lattice
            branch_id = await self.timeline_lattice.create_branch(
                source_timeline_id=request.source_timeline_id,
                branch_point=request.branch_point_us / 1_000_000,  # Convert to seconds
                branch_id=request.new_timeline_id or None,
            )

            # Create timeline info for new branch
            source_info = self.active_timelines.get(request.source_timeline_id)
            self.active_timelines[branch_id] = TimelineInfo(
                timeline_id=branch_id,
                event_count=source_info.event_count if source_info else 0,
                created_at=int(time.time() * 1_000_000),
                metadata={"source_timeline": request.source_timeline_id},
            )

            response = BranchTimelineResponse()
            response.new_timeline_id = branch_id
            response.success = True
            response.message = "Timeline branched successfully"

            logger.info(
                "Timeline branched successfully",
                source_timeline_id=request.source_timeline_id,
                new_timeline_id=branch_id,
            )

            return response

        except Exception as e:
            logger.error(
                "Error branching timeline",
                error=str(e),
                source_timeline_id=request.source_timeline_id,
            )
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    async def RewindTimeline(
        self, request: RewindTimelineRequest, context: grpc.aio.ServicerContext
    ) -> RewindTimelineResponse:
        """
        Rewind timeline to a specific point in time.

        Args:
            request: The rewind timeline request
            context: gRPC service context

        Returns:
            RewindTimelineResponse with rewind status
        """
        try:
            logger.info(
                "Processing rewind timeline request",
                timeline_id=request.timeline_id,
                rewind_to_us=request.rewind_to_us,
            )

            # Rewind timeline in lattice
            await self.timeline_lattice.rewind_timeline(
                timeline_id=request.timeline_id,
                target_time=request.rewind_to_us / 1_000_000,  # Convert to seconds
            )

            # Update timeline info
            if request.timeline_id in self.active_timelines:
                timeline_info = self.active_timelines[request.timeline_id]
                # Estimate event count after rewind (would need proper implementation)
                timeline_info.event_count = max(0, timeline_info.event_count - 1)
                timeline_info.entropy_level += 0.1  # Rewind increases entropy

            response = RewindTimelineResponse()
            response.success = True
            response.message = "Timeline rewound successfully"

            logger.info(
                "Timeline rewound successfully",
                timeline_id=request.timeline_id,
                rewind_to_us=request.rewind_to_us,
            )

            return response

        except Exception as e:
            logger.error(
                "Error rewinding timeline",
                error=str(e),
                timeline_id=request.timeline_id,
            )
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    async def MergeTimelines(
        self, request: MergeTimelinesRequest, context: grpc.aio.ServicerContext
    ) -> MergeTimelinesResponse:
        """
        Merge multiple timelines into one.

        Args:
            request: The merge timelines request
            context: gRPC service context

        Returns:
            MergeTimelinesResponse with merge status
        """
        try:
            logger.info(
                "Processing merge timelines request",
                target_timeline_id=request.target_timeline_id,
                source_timeline_ids=list(request.source_timeline_ids),
            )

            # Merge timelines in lattice
            result = await self.timeline_lattice.merge_timelines(
                target_timeline_id=request.target_timeline_id,
                source_timeline_ids=list(request.source_timeline_ids),
                strategy=request.merge_strategy or "CHRONOLOGICAL",
            )

            # Update timeline info
            target_info = self.active_timelines.get(request.target_timeline_id)
            if target_info:
                # Add event counts from source timelines
                for source_id in request.source_timeline_ids:
                    source_info = self.active_timelines.get(source_id)
                    if source_info:
                        target_info.event_count += source_info.event_count

                # Increase entropy due to merge complexity
                target_info.entropy_level += len(request.source_timeline_ids) * 0.2

            response = MergeTimelinesResponse()
            response.success = True
            response.message = "Timelines merged successfully"
            response.conflicts_detected = len(result.get("conflicts", []))

            logger.info(
                "Timelines merged successfully",
                target_timeline_id=request.target_timeline_id,
                conflicts_detected=response.conflicts_detected,
            )

            return response

        except Exception as e:
            logger.error(
                "Error merging timelines",
                error=str(e),
                target_timeline_id=request.target_timeline_id,
            )
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    async def TestParadox(
        self, request: TestParadoxRequest, context: grpc.aio.ServicerContext
    ) -> TestParadoxResponse:
        """
        Test for temporal paradoxes in a timeline.

        Args:
            request: The test paradox request
            context: gRPC service context

        Returns:
            TestParadoxResponse with paradox detection results
        """
        try:
            logger.info(
                "Processing test paradox request", timeline_id=request.timeline_id
            )

            # Test for paradoxes using timeline lattice
            paradox_result = await self.timeline_lattice.test_paradox(
                request.timeline_id
            )

            # Convert to protobuf result
            proto_result = ProtoParadoxResult()
            proto_result.has_paradox = paradox_result.has_paradox
            proto_result.paradox_type = paradox_result.paradox_type
            proto_result.severity = paradox_result.severity
            proto_result.timeline_conflicts.extend(paradox_result.timeline_conflicts)
            proto_result.causal_loops.extend(paradox_result.causal_loops)
            proto_result.containment_actions.extend(paradox_result.containment_actions)
            proto_result.entropy_score = paradox_result.entropy_score

            response = TestParadoxResponse()
            response.result.CopyFrom(proto_result)

            # Update timeline info
            if request.timeline_id in self.active_timelines:
                timeline_info = self.active_timelines[request.timeline_id]
                timeline_info.has_paradoxes = paradox_result.has_paradox
                timeline_info.entropy_level = paradox_result.entropy_score

            logger.info(
                "Paradox test completed",
                timeline_id=request.timeline_id,
                has_paradox=paradox_result.has_paradox,
                paradox_type=paradox_result.paradox_type,
            )

            return response

        except Exception as e:
            logger.error(
                "Error testing paradox", error=str(e), timeline_id=request.timeline_id
            )
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    async def GetTimelineInfo(
        self, request: GetTimelineInfoRequest, context: grpc.aio.ServicerContext
    ) -> GetTimelineInfoResponse:
        """
        Get information about a specific timeline.

        Args:
            request: The get timeline info request
            context: gRPC service context

        Returns:
            GetTimelineInfoResponse with timeline details
        """
        try:
            logger.info(
                "Processing get timeline info request", timeline_id=request.timeline_id
            )

            timeline_info = self.active_timelines.get(request.timeline_id)
            if not timeline_info:
                await context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Timeline {request.timeline_id} not found",
                )

            # Convert to protobuf
            proto_info = ProtoTimelineInfo()
            proto_info.timeline_id = timeline_info.timeline_id
            proto_info.event_count = timeline_info.event_count
            proto_info.created_at = timeline_info.created_at or 0
            proto_info.consistency_score = timeline_info.consistency_score
            proto_info.entropy_level = timeline_info.entropy_level
            proto_info.has_paradoxes = timeline_info.has_paradoxes
            proto_info.metadata.update(timeline_info.metadata)

            response = GetTimelineInfoResponse()
            response.info.CopyFrom(proto_info)

            logger.info(
                "Timeline info retrieved successfully",
                timeline_id=request.timeline_id,
                event_count=timeline_info.event_count,
            )

            return response

        except Exception as e:
            logger.error(
                "Error getting timeline info",
                error=str(e),
                timeline_id=request.timeline_id,
            )
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")

    async def ListTimelines(
        self, request: ListTimelinesRequest, context: grpc.aio.ServicerContext
    ) -> ListTimelinesResponse:
        """
        List all active timelines with pagination.

        Args:
            request: The list timelines request
            context: gRPC service context

        Returns:
            ListTimelinesResponse with timeline list
        """
        try:
            logger.info(
                "Processing list timelines request",
                page_size=request.page_size,
                page_token=request.page_token,
            )

            # Simple pagination implementation
            all_timelines = list(self.active_timelines.values())

            # Apply filters if specified
            if request.filter_pattern:
                all_timelines = [
                    t for t in all_timelines if request.filter_pattern in t.timeline_id
                ]

            # Pagination
            start_idx = 0
            if request.page_token:
                try:
                    start_idx = int(request.page_token)
                except ValueError:
                    start_idx = 0

            page_size = request.page_size if request.page_size > 0 else 50
            end_idx = start_idx + page_size

            page_timelines = all_timelines[start_idx:end_idx]

            # Convert to protobuf
            response = ListTimelinesResponse()
            for timeline_info in page_timelines:
                proto_info = ProtoTimelineInfo()
                proto_info.timeline_id = timeline_info.timeline_id
                proto_info.event_count = timeline_info.event_count
                proto_info.created_at = timeline_info.created_at or 0
                proto_info.consistency_score = timeline_info.consistency_score
                proto_info.entropy_level = timeline_info.entropy_level
                proto_info.has_paradoxes = timeline_info.has_paradoxes
                proto_info.metadata.update(timeline_info.metadata)
                response.timelines.append(proto_info)

            # Set next page token if there are more results
            if end_idx < len(all_timelines):
                response.next_page_token = str(end_idx)

            logger.info(
                "Timeline list retrieved successfully",
                total_count=len(all_timelines),
                page_count=len(page_timelines),
                has_next=bool(response.next_page_token),
            )

            return response

        except Exception as e:
            logger.error("Error listing timelines", error=str(e))
            await context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {str(e)}")


async def create_server(config: TimelineServiceConfig) -> aio.Server:
    """
    Create and configure the gRPC server.

    Args:
        config: Service configuration

    Returns:
        Configured gRPC server
    """
    server = aio.server()

    # Add the Timeline service
    timeline_service = TimelineService(config)
    timeline_pb2_grpc.add_TimelineServiceServicer_to_server(timeline_service, server)

    # Configure server options
    server_options = [
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 5000),
        ("grpc.keepalive_permit_without_calls", True),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.http2.min_ping_interval_without_data_ms", 300000),
        ("grpc.max_connection_idle_ms", 10000),
        ("grpc.max_message_length", 100 * 1024 * 1024),  # 100MB
    ]

    for option, value in server_options:
        server.add_option(option, value)

    # Add listening port
    listen_addr = f"{config.host}:{config.port}"
    if config.use_ssl:
        # SSL configuration would go here
        logger.warning("SSL requested but not implemented in this prototype")
        server.add_insecure_port(listen_addr)
    else:
        server.add_insecure_port(listen_addr)

    logger.info(
        "Timeline service server created",
        listen_addr=listen_addr,
        ssl_enabled=config.use_ssl,
    )

    return server
