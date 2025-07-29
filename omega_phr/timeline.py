"""
Timeline Lattice - Temporal Paradox Testing Engine

This module implements the Layered Temporal Realities (LTR) engine
for testing AI systems under temporal paradox conditions.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Any

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from .exceptions import OmegaStateError, TemporalParadoxError
from .models import Event, EventType, OmegaState, OmegaStateLevel, ParadoxResult


class TimelineLattice:
    """
    Core engine for temporal paradox testing and timeline manipulation.

    The Timeline Lattice maintains multiple parallel timelines and can
    create paradoxes by introducing causal inconsistencies, timeline
    branches, merges, and temporal rewinds.
    """

    def __init__(
        self, max_timelines: int = 100, paradox_threshold: float = 0.1
    ) -> None:
        """Initialize the Timeline Lattice."""
        self.max_timelines = max_timelines
        self.paradox_threshold = paradox_threshold

        # Timeline storage: timeline_id -> ordered list of events
        self.timelines: dict[str, list[Event]] = defaultdict(list)

        # Timeline metadata
        self.timeline_metadata: dict[str, dict] = defaultdict(dict)

        # Active paradoxes
        self.active_paradoxes: dict[str, ParadoxResult] = {}

        # Causal dependency graph: event_id -> set of dependent event_ids
        self.causal_graph: dict[str, set[str]] = defaultdict(set)

        # Timeline branch tracking: child_timeline -> parent_timeline
        self.timeline_hierarchy: dict[str, str] = {}

        # Entropy tracking for Ω-state detection
        self.entropy_history: deque = deque(maxlen=1000)

        # Cache for consistency checks
        self._consistency_cache: dict[str, float] = {}

        logger.info(
            f"Timeline Lattice initialized with max_timelines={max_timelines}, "
            f"paradox_threshold={paradox_threshold}"
        )

    async def append_event(self, event: Event) -> bool:
        """
        Append an event to the specified timeline.

        Args:
            event: The event to append

        Returns:
            bool: True if event was successfully appended

        Raises:
            TemporalParadoxError: If adding the event would create an
                unresolvable paradox
        """
        logger.debug(
            f"Appending event {event.event_id} to timeline {event.timeline_id}"
        )

        # Validate timeline capacity
        if (
            len(self.timelines) >= self.max_timelines
            and event.timeline_id not in self.timelines
        ):
            raise TemporalParadoxError(
                f"Maximum timeline capacity ({self.max_timelines}) exceeded",
                timeline_id=event.timeline_id,
            )

        # Check for immediate paradoxes
        paradox_result = await self._check_event_paradox(event)
        if (
            paradox_result.has_paradox
            and paradox_result.severity > self.paradox_threshold
        ):
            logger.warning(
                f"Paradox detected in event {event.event_id}: {paradox_result.paradox_type} (severity: {paradox_result.severity})"  # noqa: E501
            )

            # Store paradox for analysis
            self.active_paradoxes[event.event_id] = paradox_result

            # Check if this creates an Ω-state
            omega_state = await self._evaluate_omega_state(event, paradox_result)
            if omega_state.level != OmegaStateLevel.NORMAL:
                raise OmegaStateError(
                    f"Event would create Ω-state: {omega_state.level.name}",
                    omega_id=omega_state.omega_id,
                    entropy_level=omega_state.propagation_risk,
                )

        # Add event to timeline
        self.timelines[event.timeline_id].append(event)

        # Update causal dependencies
        if event.parent_id:
            self.causal_graph[event.parent_id].add(event.event_id)

        # Update entropy tracking
        entropy = self._calculate_timeline_entropy(event.timeline_id)
        self.entropy_history.append((time.time(), entropy))

        # Invalidate consistency cache
        self._consistency_cache.clear()

        logger.info(
            f"Event added to timeline {event.timeline_id}: {event.event_id} (length: {len(self.timelines[event.timeline_id])})"  # noqa: E501
        )

        return True

    async def branch_timeline(
        self,
        source_timeline: str,
        branch_point_event_id: str,
        new_timeline_id: str | None = None,
    ) -> str:
        """
        Create a new timeline branch from a specific event.

        Args:
            source_timeline: ID of the timeline to branch from
            branch_point_event_id: Event ID where the branch occurs
            new_timeline_id: Optional ID for the new timeline

        Returns:
            str: ID of the newly created timeline branch

        Raises:
            TemporalParadoxError: If branching would create invalid causal structure
        """
        if source_timeline not in self.timelines:
            raise TemporalParadoxError(
                f"Source timeline {source_timeline} does not exist"
            )

        new_timeline_id = new_timeline_id or f"branch_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"Creating timeline branch from {source_timeline} at {branch_point_event_id} -> {new_timeline_id}"  # noqa: E501
        )

        # Find the branch point
        source_events = self.timelines[source_timeline]
        branch_point_index = None

        for i, event in enumerate(source_events):
            if event.event_id == branch_point_event_id:
                branch_point_index = i
                break

        if branch_point_index is None:
            raise TemporalParadoxError(
                f"Branch point event {branch_point_event_id} not found in timeline {source_timeline}"  # noqa: E501
            )

        # Copy events up to branch point
        branched_events = source_events[: branch_point_index + 1].copy()

        # Create branch event
        branch_event = Event(
            actor_id="timeline_lattice",
            timeline_id=new_timeline_id,
            parent_id=branch_point_event_id,
            event_type=EventType.BRANCH,
            payload={
                "source_timeline": source_timeline,
                "branch_point": branch_point_event_id,
                "operation": "timeline_branch",
            },
        )

        branched_events.append(branch_event)

        # Store new timeline
        self.timelines[new_timeline_id] = branched_events
        self.timeline_hierarchy[new_timeline_id] = source_timeline

        # Update metadata
        self.timeline_metadata[new_timeline_id] = {
            "created_at": time.time(),
            "source_timeline": source_timeline,
            "branch_point": branch_point_event_id,
            "type": "branch",
        }

        logger.info(
            f"Timeline branch created successfully: {new_timeline_id} with {len(branched_events)} events copied"  # noqa: E501
        )

        return new_timeline_id

    async def rewind_timeline(self, timeline_id: str, target_event_id: str) -> bool:
        """
        Rewind a timeline to a specific event, creating temporal paradox potential.

        Args:
            timeline_id: ID of the timeline to rewind
            target_event_id: Event ID to rewind to

        Returns:
            bool: True if rewind was successful

        Raises:
            TemporalParadoxError: If rewind would cause causal violations
        """
        if timeline_id not in self.timelines:
            raise TemporalParadoxError(f"Timeline {timeline_id} does not exist")

        logger.info(f"Rewinding timeline {timeline_id} to event {target_event_id}")

        events = self.timelines[timeline_id]
        target_index = None

        # Find target event
        for i, event in enumerate(events):
            if event.event_id == target_event_id:
                target_index = i
                break

        if target_index is None:
            raise TemporalParadoxError(
                f"Target event {target_event_id} not found in timeline {timeline_id}"
            )

        # Check for causal violations
        removed_events = events[target_index + 1 :]
        for removed_event in removed_events:
            if removed_event.event_id in self.causal_graph:
                dependent_events = self.causal_graph[removed_event.event_id]
                if dependent_events:
                    logger.warning(
                        f"Removing event {removed_event.event_id} with dependents: {list(dependent_events)}"  # noqa: E501
                    )

        # Create rewind event
        rewind_event = Event(
            actor_id="timeline_lattice",
            timeline_id=timeline_id,
            parent_id=target_event_id,
            event_type=EventType.REWIND,
            payload={
                "target_event": target_event_id,
                "removed_events": [e.event_id for e in removed_events],
                "operation": "timeline_rewind",
            },
        )

        # Perform rewind
        self.timelines[timeline_id] = events[: target_index + 1] + [rewind_event]

        # Clean up causal graph
        for removed_event in removed_events:
            if removed_event.event_id in self.causal_graph:
                del self.causal_graph[removed_event.event_id]

        logger.info(
            f"Timeline rewind completed for {timeline_id}, removed {len(removed_events)} events"  # noqa: E501
        )

        return True

    async def merge_timelines(
        self,
        primary_timeline: str,
        secondary_timeline: str,
        merge_strategy: str = "chronological",
    ) -> dict[str, Any]:
        """
        Merge two timelines, potentially creating paradoxes.

        Args:
            primary_timeline: ID of the primary timeline
            secondary_timeline: ID of the secondary timeline to merge
            merge_strategy: Strategy for merging  # noqa: E501
            # ("chronological", "interleaved", "append")

        Returns:
            Dict[str, Any]: Result of merge operation with success status and details

        Raises:
            TemporalParadoxError: If merge would create unresolvable conflicts
        """
        if primary_timeline not in self.timelines:
            raise TemporalParadoxError(
                f"Primary timeline {primary_timeline} does not exist"
            )
        if secondary_timeline not in self.timelines:
            raise TemporalParadoxError(
                f"Secondary timeline {secondary_timeline} does not exist"
            )

        logger.info(
            f"Starting timeline merge: {primary_timeline} + {secondary_timeline} (strategy: {merge_strategy})"  # noqa: E501
        )

        primary_events = self.timelines[primary_timeline]
        secondary_events = self.timelines[secondary_timeline]

        # Create merge event
        merge_event = Event(
            actor_id="timeline_lattice",
            timeline_id=primary_timeline,
            event_type=EventType.MERGE,
            payload={
                "primary_timeline": primary_timeline,
                "secondary_timeline": secondary_timeline,
                "merge_strategy": merge_strategy,
                "operation": "timeline_merge",
            },
        )

        # Merge based on strategy
        if merge_strategy == "chronological":
            all_events = primary_events + secondary_events
            merged_events = sorted(all_events, key=lambda e: e.valid_at_us)
        elif merge_strategy == "interleaved":
            merged_events = []
            i = j = 0
            while i < len(primary_events) or j < len(secondary_events):
                if i < len(primary_events) and (
                    j >= len(secondary_events)
                    or primary_events[i].valid_at_us <= secondary_events[j].valid_at_us
                ):
                    merged_events.append(primary_events[i])
                    i += 1
                else:
                    merged_events.append(secondary_events[j])
                    j += 1
        else:  # append
            merged_events = primary_events + secondary_events

        merged_events.append(merge_event)

        # Update primary timeline
        self.timelines[primary_timeline] = merged_events

        # Remove secondary timeline
        del self.timelines[secondary_timeline]
        if secondary_timeline in self.timeline_metadata:
            del self.timeline_metadata[secondary_timeline]

        logger.info(
            f"Timeline merge completed: {primary_timeline} with {len(merged_events)} total events"  # noqa: E501
        )

        return {
            "success": True,
            "merged_timeline": primary_timeline,
            "total_events": len(merged_events),
            "strategy": merge_strategy,
        }

    async def test_paradox(self, event_or_timeline: Event | str) -> ParadoxResult:
        """
        Test if an event would create temporal paradoxes, or analyze a timeline.

        Args:
            event_or_timeline: Event to test or timeline ID to analyze

        Returns:
            ParadoxResult: Detailed analysis of potential paradoxes
        """
        if isinstance(event_or_timeline, str):
            # Timeline analysis mode
            timeline_id = event_or_timeline
            if timeline_id not in self.timelines:
                from .models import ParadoxResult

                return ParadoxResult(
                    has_paradox=False,
                    paradox_type="none",
                    severity=0.0,
                    timeline_conflicts=[],
                    causal_loops=[],
                    containment_actions=[],
                    entropy_score=0.0,
                    metadata={"description": "Timeline not found"},
                )

            # Analyze all events in the timeline for paradoxes
            events = self.timelines[timeline_id]
            if len(events) < 2:
                from .models import ParadoxResult

                return ParadoxResult(
                    has_paradox=False,
                    paradox_type="none",
                    severity=0.0,
                    timeline_conflicts=[],
                    causal_loops=[],
                    containment_actions=[],
                    entropy_score=0.0,
                    metadata={
                        "description": "Insufficient events for paradox analysis"
                    },
                )

            # Check for temporal inconsistencies
            paradoxes_found = []
            for i, event in enumerate(events):
                for j, other_event in enumerate(events):
                    if (
                        i != j
                        and event.valid_at_us > other_event.valid_at_us
                        and "contradicts" in event.payload
                        and event.payload["contradicts"] == other_event.event_id
                    ):
                        paradoxes_found.append((event, other_event))

            from .models import ParadoxResult

            has_paradox = len(paradoxes_found) > 0
            severity = 0.8 if has_paradox else 0.0

            # Raise exception for severe paradoxes
            if has_paradox and severity >= self.paradox_threshold:
                from .exceptions import TemporalParadoxError

                raise TemporalParadoxError(
                    f"Severe temporal paradox detected in timeline {timeline_id}: "
                    f"{len(paradoxes_found)} contradictions found"
                )

            return ParadoxResult(
                has_paradox=has_paradox,
                paradox_type="temporal_contradiction" if has_paradox else "none",
                severity=severity,
                timeline_conflicts=[p[0].event_id for p in paradoxes_found],
                causal_loops=[],
                containment_actions=(
                    ["temporal_rewind", "event_isolation"] if has_paradox else []
                ),
                entropy_score=0.9 if has_paradox else 0.1,
                metadata={
                    "description": (
                        f"Found {len(paradoxes_found)} temporal contradictions"
                        if has_paradox
                        else "No paradoxes detected"
                    )
                },
            )
        else:
            # Single event analysis mode
            result = await self._check_event_paradox(event_or_timeline)

            # Raise exception for severe paradoxes in test mode
            if result.has_paradox and result.severity >= self.paradox_threshold:
                from .exceptions import TemporalParadoxError

                raise TemporalParadoxError(
                    f"Severe temporal paradox detected for event {event_or_timeline.event_id}: "  # noqa: E501
                    f"conflicts={result.timeline_conflicts}, loops={result.causal_loops}"  # noqa: E501
                )

            return result

    async def get_timeline_consistency(self, timeline_id: str) -> float:
        """
        Calculate consistency score for a timeline.

        Args:
            timeline_id: ID of timeline to analyze

        Returns:
            float: Consistency score between 0.0 and 1.0
        """
        if timeline_id not in self.timelines:
            return 0.0

        cache_key = f"consistency_{timeline_id}_{len(self.timelines[timeline_id])}"
        if cache_key in self._consistency_cache:
            return self._consistency_cache[cache_key]

        events = self.timelines[timeline_id]
        if not events:
            return 1.0

        consistency_score = 1.0

        # Check temporal ordering
        for i in range(1, len(events)):
            if events[i].valid_at_us < events[i - 1].valid_at_us:
                consistency_score *= 0.9  # Penalty for temporal disorder

        # Check causal consistency
        for event in events:
            if event.parent_id:
                parent_found = any(e.event_id == event.parent_id for e in events)
                if not parent_found:
                    consistency_score *= 0.8  # Penalty for broken causal chain

        # Check for signature integrity
        for event in events:
            expected_sig = event._generate_signature()
            if event.signature != expected_sig:
                consistency_score *= 0.7  # Penalty for signature mismatch

        self._consistency_cache[cache_key] = consistency_score
        return consistency_score

    def get_timeline_info(self, timeline_id: str) -> dict:
        """Get comprehensive information about a timeline."""
        if timeline_id not in self.timelines:
            return {}

        events = self.timelines[timeline_id]
        return {
            "timeline_id": timeline_id,
            "event_count": len(events),
            "created_at": self.timeline_metadata.get(timeline_id, {}).get("created_at"),
            "consistency_score": asyncio.run(
                self.get_timeline_consistency(timeline_id)
            ),
            "entropy_level": self._calculate_timeline_entropy(timeline_id),
            "has_paradoxes": any(p.has_paradox for p in self.active_paradoxes.values()),
            "metadata": self.timeline_metadata.get(timeline_id, {}),
        }

    def list_timelines(self) -> list[str]:
        """Get list of all timeline IDs."""
        return list(self.timelines.keys())

    def get_active_paradoxes(self) -> dict[str, ParadoxResult]:
        """Get all currently active paradoxes."""
        return self.active_paradoxes.copy()

    async def _check_event_paradox(self, event: Event) -> ParadoxResult:
        """Check if an event would create temporal paradoxes."""
        paradox_result = ParadoxResult(has_paradox=False)

        if event.timeline_id not in self.timelines:
            return paradox_result

        existing_events = self.timelines[event.timeline_id]

        # Check for temporal inconsistencies
        timeline_conflicts = []
        causal_loops = []

        # Check for existing paradoxes in the timeline first
        for i, existing_event in enumerate(existing_events):
            for j, other_event in enumerate(existing_events):
                if i != j:
                    # Check if events contradict each other
                    if (
                        "contradicts" in existing_event.payload
                        and existing_event.payload["contradicts"]
                        == other_event.event_id
                    ):
                        timeline_conflicts.append(
                            f"Event {existing_event.event_id} contradicts event {other_event.event_id}"  # noqa: E501
                        )

                    # Check for state contradictions
                    if (
                        existing_event.payload.get("state") == "NOT-A"
                        and other_event.payload.get("state") == "A"
                    ):
                        timeline_conflicts.append(
                            f"State contradiction between {existing_event.event_id} (NOT-A) and {other_event.event_id} (A)"  # noqa: E501
                        )

        # Check for causality violations
        if event.parent_id:
            parent_event = None
            for existing_event in existing_events:
                if existing_event.event_id == event.parent_id:
                    parent_event = existing_event
                    break

            if parent_event and event.valid_at_us < parent_event.valid_at_us:
                timeline_conflicts.append(
                    f"Event {event.event_id} occurs before its parent {event.parent_id}"
                )

        # Check for duplicate event IDs
        for existing_event in existing_events:
            if existing_event.event_id == event.event_id:
                timeline_conflicts.append(f"Duplicate event ID {event.event_id}")

        # Check for contradiction patterns (similar to timeline analysis)
        if "contradicts" in event.payload:
            contradicted_id = event.payload["contradicts"]
            for existing_event in existing_events:
                if existing_event.event_id == contradicted_id:
                    timeline_conflicts.append(
                        f"Event {event.event_id} contradicts existing event {contradicted_id}"  # noqa: E501
                    )
                    break

        # Check for causal loops
        if event.parent_id and self._would_create_causal_loop(
            event.event_id, event.parent_id
        ):
            causal_loops.append(
                f"Event {event.event_id} would create causal loop with {event.parent_id}"  # noqa: E501
            )

        # Calculate severity
        severity = 0.0
        if timeline_conflicts:
            severity += len(timeline_conflicts) * 0.3
        if causal_loops:
            severity += len(causal_loops) * 0.5

        # Determine paradox type
        paradox_type = ""
        if timeline_conflicts and causal_loops:
            paradox_type = "causal_temporal"
        elif timeline_conflicts:
            paradox_type = "temporal"
        elif causal_loops:
            paradox_type = "causal"

        # Calculate entropy score
        entropy_score = self._calculate_event_entropy(event)

        if timeline_conflicts or causal_loops:
            paradox_result.has_paradox = True
            paradox_result.paradox_type = paradox_type
            paradox_result.severity = min(severity, 1.0)
            paradox_result.timeline_conflicts = timeline_conflicts
            paradox_result.causal_loops = causal_loops
            paradox_result.entropy_score = entropy_score

            # Suggest containment actions
            containment_actions = []
            if severity > 0.7:
                containment_actions.append("immediate_quarantine")
            if timeline_conflicts:
                containment_actions.append("temporal_reorder")
            if causal_loops:
                containment_actions.append("causal_graph_repair")

            paradox_result.containment_actions = containment_actions

        return paradox_result

    def _would_create_causal_loop(self, event_id: str, parent_id: str) -> bool:
        """Check if adding a causal dependency would create a loop."""
        visited = set()

        def has_path(start: str, target: str) -> bool:
            if start == target:
                return True
            if start in visited:
                return False

            visited.add(start)

            for dependent in self.causal_graph.get(start, set()):
                if has_path(dependent, target):
                    return True

            return False

        return has_path(parent_id, event_id)

    def _calculate_timeline_entropy(self, timeline_id: str) -> float:
        """Calculate entropy level for a timeline."""
        if timeline_id not in self.timelines:
            return 0.0

        events = self.timelines[timeline_id]
        if not events:
            return 0.0

        # Calculate based on event distribution and variety
        event_types = [e.event_type for e in events]
        actor_ids = [e.actor_id for e in events]

        # Shannon entropy-like calculation
        type_entropy = self._shannon_entropy([et.name for et in event_types])
        actor_entropy = self._shannon_entropy(actor_ids)

        return (type_entropy + actor_entropy) / 2.0

    def _calculate_event_entropy(self, event: Event) -> float:
        """Calculate entropy contribution of a single event."""
        payload_size = len(str(event.payload))
        metadata_size = len(str(event.metadata))

        # Normalize to 0-1 range
        size_entropy = min((payload_size + metadata_size) / 1000.0, 1.0)

        # Factor in event type rarity
        type_rarity = 1.0 / (len(EventType) + 1)

        return (size_entropy + type_rarity) / 2.0

    def _shannon_entropy(self, data: list[str]) -> float:
        """Calculate Shannon entropy for a list of categorical data."""
        if not data:
            return 0.0

        import math
        from collections import Counter

        counts = Counter(data)
        total = len(data)

        entropy = 0.0
        for count in counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    async def _evaluate_omega_state(
        self, event: Event, paradox_result: ParadoxResult
    ) -> OmegaState:
        """Evaluate if an event creates an Omega state requiring containment."""
        omega_state = OmegaState(
            trigger_event=event,
            entropy_hash=paradox_result.metadata.get("entropy_hash", ""),
            source_components=["timeline_lattice"],
        )

        # Determine Omega state level based on paradox severity
        if paradox_result.severity > 0.8:
            omega_state.level = OmegaStateLevel.CRITICAL
            omega_state.quarantine_status = True
        elif paradox_result.severity > 0.5:
            omega_state.level = OmegaStateLevel.WARNING
        else:
            omega_state.level = OmegaStateLevel.NORMAL

        # Calculate propagation risk
        omega_state.propagation_risk = (
            paradox_result.severity * len(self.timelines) / self.max_timelines
        )

        # Determine contamination vector
        contamination_vector = []
        if "causal" in paradox_result.paradox_type:
            contamination_vector.append("causal_graph")
        if "temporal" in paradox_result.paradox_type:
            contamination_vector.append("timeline_ordering")

        omega_state.contamination_vector = contamination_vector

        return omega_state

    # Backward compatibility methods
    def create_timeline(self, timeline_id: str) -> str:
        """Create a new timeline (backward compatibility method)."""
        if timeline_id not in self.timelines:
            self.timelines[timeline_id] = []
            self.timeline_metadata[timeline_id] = {
                "created_at": time.time(),
                "type": "standard",
            }
        return timeline_id

    async def add_event(self, event: Event) -> bool:
        """Add event to timeline (backward compatibility method)."""
        return await self.append_event(event)

    def get_events(self, timeline_id: str, limit: int | None = None) -> list[Event]:
        """Get events from timeline (backward compatibility method)."""
        if timeline_id not in self.timelines:
            return []
        events = self.timelines[timeline_id]
        if limit:
            return events[-limit:]
        return events.copy()


# Alias for backward compatibility
TimelineManager = TimelineLattice
