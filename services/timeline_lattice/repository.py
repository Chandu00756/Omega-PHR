"""
Repository layer for Timeline Lattice service.

This module provides data persistence and retrieval functionality
for timeline events and metadata.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from structlog import get_logger
except ImportError:
    import logging

    get_logger = logging.getLogger

from models import EventModel, TimelineInfo

logger = get_logger(__name__)


class TimelineRepository:
    """
    Repository for persisting timeline data.

    This implementation uses in-memory storage for the prototype,
    but can be extended to use databases like ScyllaDB or Cassandra.
    """

    def __init__(self):
        """Initialize the repository."""
        self.events: Dict[str, List[EventModel]] = {}
        self.timelines: Dict[str, TimelineInfo] = {}
        self.indexes: Dict[str, Dict[str, Any]] = {
            "by_actor": {},
            "by_timestamp": {},
            "by_event_type": {},
        }
        logger.info("Timeline repository initialized")

    async def store_event(self, event: EventModel) -> bool:
        """
        Store an event in the repository.

        Args:
            event: The event to store

        Returns:
            True if successfully stored
        """
        try:
            timeline_id = event.timeline_id

            # Initialize timeline if not exists
            if timeline_id not in self.events:
                self.events[timeline_id] = []
                self.timelines[timeline_id] = TimelineInfo(
                    timeline_id=timeline_id,
                    event_count=0,
                    created_at=int(time.time() * 1_000_000),
                )

            # Store the event
            self.events[timeline_id].append(event)

            # Update timeline info
            timeline_info = self.timelines[timeline_id]
            timeline_info.event_count += 1

            # Update indexes
            await self._update_indexes(event)

            logger.info(
                "Event stored successfully",
                event_id=event.event_id,
                timeline_id=timeline_id,
            )

            return True

        except Exception as e:
            logger.error("Error storing event", error=str(e), event_id=event.event_id)
            return False

    async def get_events(
        self,
        timeline_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[EventModel]:
        """
        Retrieve events from a timeline.

        Args:
            timeline_id: The timeline to query
            limit: Maximum number of events to return
            offset: Number of events to skip
            start_time: Start time filter (microseconds)
            end_time: End time filter (microseconds)

        Returns:
            List of events matching criteria
        """
        try:
            if timeline_id not in self.events:
                return []

            events = self.events[timeline_id]

            # Apply time filters
            if start_time is not None or end_time is not None:
                filtered_events = []
                for event in events:
                    if start_time and event.valid_at_us < start_time:
                        continue
                    if end_time and event.valid_at_us > end_time:
                        continue
                    filtered_events.append(event)
                events = filtered_events

            # Sort by timestamp
            events.sort(key=lambda e: e.valid_at_us)

            # Apply pagination
            if offset > 0:
                events = events[offset:]
            if limit is not None:
                events = events[:limit]

            logger.info(
                "Events retrieved successfully",
                timeline_id=timeline_id,
                count=len(events),
            )

            return events

        except Exception as e:
            logger.error(
                "Error retrieving events", error=str(e), timeline_id=timeline_id
            )
            return []

    async def get_event_by_id(self, event_id: str) -> Optional[EventModel]:
        """
        Retrieve a specific event by ID.

        Args:
            event_id: The event ID to find

        Returns:
            The event if found, None otherwise
        """
        try:
            for timeline_events in self.events.values():
                for event in timeline_events:
                    if event.event_id == event_id:
                        return event

            logger.warning("Event not found", event_id=event_id)
            return None

        except Exception as e:
            logger.error(
                "Error retrieving event by ID", error=str(e), event_id=event_id
            )
            return None

    async def get_timeline_info(self, timeline_id: str) -> Optional[TimelineInfo]:
        """
        Get information about a timeline.

        Args:
            timeline_id: The timeline ID

        Returns:
            Timeline information if found
        """
        return self.timelines.get(timeline_id)

    async def list_timelines(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        filter_pattern: Optional[str] = None,
    ) -> List[TimelineInfo]:
        """
        List all timelines with optional filtering.

        Args:
            limit: Maximum number of timelines to return
            offset: Number of timelines to skip
            filter_pattern: Pattern to filter timeline IDs

        Returns:
            List of timeline information
        """
        try:
            timelines = list(self.timelines.values())

            # Apply filter
            if filter_pattern:
                timelines = [
                    t
                    for t in timelines
                    if filter_pattern.lower() in t.timeline_id.lower()
                ]

            # Sort by creation time
            timelines.sort(key=lambda t: t.created_at or 0)

            # Apply pagination
            if offset > 0:
                timelines = timelines[offset:]
            if limit is not None:
                timelines = timelines[:limit]

            logger.info("Timelines listed successfully", count=len(timelines))

            return timelines

        except Exception as e:
            logger.error("Error listing timelines", error=str(e))
            return []

    async def delete_timeline(self, timeline_id: str) -> bool:
        """
        Delete a timeline and all its events.

        Args:
            timeline_id: The timeline to delete

        Returns:
            True if successfully deleted
        """
        try:
            if timeline_id in self.events:
                # Remove from indexes first
                events = self.events[timeline_id]
                for event in events:
                    await self._remove_from_indexes(event)

                # Remove timeline data
                del self.events[timeline_id]
                del self.timelines[timeline_id]

                logger.info("Timeline deleted successfully", timeline_id=timeline_id)
                return True

            logger.warning("Timeline not found for deletion", timeline_id=timeline_id)
            return False

        except Exception as e:
            logger.error(
                "Error deleting timeline", error=str(e), timeline_id=timeline_id
            )
            return False

    async def search_events(
        self,
        actor_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
    ) -> List[EventModel]:
        """
        Search events across all timelines.

        Args:
            actor_id: Filter by actor ID
            event_type: Filter by event type
            start_time: Start time filter (microseconds)
            end_time: End time filter (microseconds)
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        try:
            matching_events = []

            for timeline_events in self.events.values():
                for event in timeline_events:
                    # Apply filters
                    if actor_id and event.actor_id != actor_id:
                        continue
                    if event_type and event.event_type != event_type:
                        continue
                    if start_time and event.valid_at_us < start_time:
                        continue
                    if end_time and event.valid_at_us > end_time:
                        continue

                    matching_events.append(event)

            # Sort by timestamp and apply limit
            matching_events.sort(key=lambda e: e.valid_at_us)
            matching_events = matching_events[:limit]

            logger.info("Event search completed", results_count=len(matching_events))

            return matching_events

        except Exception as e:
            logger.error("Error searching events", error=str(e))
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            total_events = sum(len(events) for events in self.events.values())
            total_timelines = len(self.timelines)

            # Calculate average events per timeline
            avg_events_per_timeline = (
                total_events / total_timelines if total_timelines > 0 else 0
            )

            # Get actor statistics
            actor_counts = {}
            event_type_counts = {}

            for timeline_events in self.events.values():
                for event in timeline_events:
                    actor_counts[event.actor_id] = (
                        actor_counts.get(event.actor_id, 0) + 1
                    )
                    event_type_counts[event.event_type] = (
                        event_type_counts.get(event.event_type, 0) + 1
                    )

            stats = {
                "total_events": total_events,
                "total_timelines": total_timelines,
                "average_events_per_timeline": avg_events_per_timeline,
                "unique_actors": len(actor_counts),
                "unique_event_types": len(event_type_counts),
                "top_actors": sorted(
                    actor_counts.items(), key=lambda x: x[1], reverse=True
                )[:10],
                "event_type_distribution": event_type_counts,
            }

            logger.info(
                "Repository statistics calculated",
                total_events=total_events,
                total_timelines=total_timelines,
            )

            return stats

        except Exception as e:
            logger.error("Error calculating statistics", error=str(e))
            return {}

    async def _update_indexes(self, event: EventModel) -> None:
        """Update search indexes for an event."""
        try:
            # Index by actor
            actor_id = event.actor_id
            if actor_id not in self.indexes["by_actor"]:
                self.indexes["by_actor"][actor_id] = []
            self.indexes["by_actor"][actor_id].append(event.event_id)

            # Index by event type
            event_type = event.event_type
            if event_type not in self.indexes["by_event_type"]:
                self.indexes["by_event_type"][event_type] = []
            self.indexes["by_event_type"][event_type].append(event.event_id)

            # Index by timestamp (hour buckets)
            timestamp_hour = event.valid_at_us // (3600 * 1_000_000)  # Hour bucket
            if timestamp_hour not in self.indexes["by_timestamp"]:
                self.indexes["by_timestamp"][timestamp_hour] = []
            self.indexes["by_timestamp"][timestamp_hour].append(event.event_id)

        except Exception as e:
            logger.error(
                "Error updating indexes", error=str(e), event_id=event.event_id
            )

    async def _remove_from_indexes(self, event: EventModel) -> None:
        """Remove an event from search indexes."""
        try:
            # Remove from actor index
            actor_events = self.indexes["by_actor"].get(event.actor_id, [])
            if event.event_id in actor_events:
                actor_events.remove(event.event_id)

            # Remove from event type index
            type_events = self.indexes["by_event_type"].get(event.event_type, [])
            if event.event_id in type_events:
                type_events.remove(event.event_id)

            # Remove from timestamp index
            timestamp_hour = event.valid_at_us // (3600 * 1_000_000)
            time_events = self.indexes["by_timestamp"].get(timestamp_hour, [])
            if event.event_id in time_events:
                time_events.remove(event.event_id)

        except Exception as e:
            logger.error(
                "Error removing from indexes", error=str(e), event_id=event.event_id
            )
