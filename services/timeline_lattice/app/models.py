"""
Omega-Paradox Hive Recursion (Ω-PHR) Framework v0.9.3
Timeline Lattice Data Models

This module defines the core data models for temporal events, paradoxes,
and timeline structures used throughout the Timeline Lattice service.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Types of temporal events."""

    TEMPORAL_SHIFT = "temporal_shift"
    CAUSALITY_VIOLATION = "causality_violation"
    PARADOX_CREATION = "paradox_creation"
    TIMELINE_MERGE = "timeline_merge"
    TIMELINE_SPLIT = "timeline_split"
    ENTROPY_FLUCTUATION = "entropy_fluctuation"
    QUANTUM_DECOHERENCE = "quantum_decoherence"


class ParadoxType(str, Enum):
    """Types of temporal paradoxes."""

    GRANDFATHER = "grandfather"
    BOOTSTRAP = "bootstrap"
    PREDESTINATION = "predestination"
    CAUSAL_LOOP = "causal_loop"
    ONTOLOGICAL = "ontological"
    INFORMATION = "information"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    CAUSALITY_VIOLATION = "causality_violation"


class TimelineStatus(str, Enum):
    """Timeline operational status."""

    ACTIVE = "active"
    STABLE = "stable"
    UNSTABLE = "unstable"
    CORRUPTED = "corrupted"
    QUARANTINED = "quarantined"
    ARCHIVED = "archived"


@dataclass
class CausalityVector:
    """Vector representing causal relationships."""

    timeline_id: str
    event_id: str | None = None
    sequence_number: int = 0
    dependency_chain: list[str] = field(default_factory=list)
    causality_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timeline_id": self.timeline_id,
            "event_id": self.event_id,
            "sequence_number": self.sequence_number,
            "dependency_chain": self.dependency_chain,
            "causality_weight": self.causality_weight,
        }


@dataclass
class TimelineEvent:
    """Represents a temporal event within a timeline."""

    event_id: str
    timeline_id: str
    actor_id: str
    parent_id: str
    event_type: EventType
    payload: str
    metadata: dict[str, str]
    valid_at_us: int
    recorded_at_us: int
    signature: bytes

    # Legacy compatibility fields
    timestamp: datetime | None = None
    data: dict[str, Any] | None = None
    causality_vector: CausalityVector | None = None
    created_at: datetime | None = None
    paradox_risk: float = 0.0
    entropy_level: float = 0.0

    def __post_init__(self):
        """Initialize computed fields."""
        if self.timestamp is None:
            self.timestamp = datetime.fromtimestamp(
                self.valid_at_us / 1_000_000, tz=UTC
            )
        if self.created_at is None:
            self.created_at = datetime.fromtimestamp(
                self.recorded_at_us / 1_000_000, tz=UTC
            )
        if self.causality_vector is None:
            self.causality_vector = CausalityVector(
                timeline_id=self.timeline_id, event_id=self.event_id
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": self.event_id,
            "timeline_id": self.timeline_id,
            "actor_id": self.actor_id,
            "parent_id": self.parent_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "metadata": self.metadata,
            "valid_at_us": self.valid_at_us,
            "recorded_at_us": self.recorded_at_us,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "data": self.data,
            "causality_vector": (
                self.causality_vector.to_dict() if self.causality_vector else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "paradox_risk": self.paradox_risk,
            "entropy_level": self.entropy_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimelineEvent":
        """Create event from dictionary representation."""
        causality_data = data.get("causality_vector")
        causality_vector = None
        if causality_data:
            causality_vector = CausalityVector(
                timeline_id=causality_data["timeline_id"],
                event_id=causality_data.get("event_id"),
                sequence_number=causality_data.get("sequence_number", 0),
                dependency_chain=causality_data.get("dependency_chain", []),
                causality_weight=causality_data.get("causality_weight", 1.0),
            )

        return cls(
            event_id=data["event_id"],
            timeline_id=data["timeline_id"],
            actor_id=data.get("actor_id", ""),
            parent_id=data.get("parent_id", ""),
            event_type=EventType(data["event_type"]),
            payload=data.get("payload", ""),
            metadata=data.get("metadata", {}),
            valid_at_us=data.get("valid_at_us", 0),
            recorded_at_us=data.get("recorded_at_us", 0),
            signature=data.get("signature", b""),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else None
            ),
            data=data.get("data"),
            causality_vector=causality_vector,
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else None
            ),
            paradox_risk=data.get("paradox_risk", 0.0),
            entropy_level=data.get("entropy_level", 0.0),
        )

    @classmethod
    def from_cassandra_row(cls, row) -> "TimelineEvent":
        """Create event from Cassandra row."""
        return cls(
            event_id=str(row.event_id),
            timeline_id=row.timeline_id,
            actor_id=row.actor_id or "",
            parent_id=row.parent_id or "",
            event_type=EventType(row.event_type),
            payload=row.payload or "",
            metadata=dict(row.metadata) if row.metadata else {},
            valid_at_us=row.valid_at_us,
            recorded_at_us=row.recorded_at_us,
            signature=row.signature or b"",
        )

    @classmethod
    def from_sqlite_row(cls, row: dict[str, Any]) -> "TimelineEvent":
        """Create event from SQLite row."""
        import json

        return cls(
            event_id=row["event_id"],
            timeline_id=row["timeline_id"],
            actor_id=row.get("actor_id", ""),
            parent_id=row.get("parent_id", ""),
            event_type=EventType(row.get("event_type", "TEMPORAL_SHIFT")),
            payload=row.get("payload", ""),
            metadata=(
                json.loads(row.get("metadata", "{}")) if row.get("metadata") else {}
            ),
            valid_at_us=row.get("valid_at_us", 0),
            recorded_at_us=row.get("recorded_at_us", 0),
            signature=row.get("signature", b""),
        )


@dataclass
class TemporalParadox:
    """Represents a detected temporal paradox."""

    paradox_id: str
    timeline_id: str
    paradox_type: ParadoxType
    severity: float
    description: str
    affected_events: list[str]  # Changed from involved_events
    detected_at: datetime
    resolved: bool = False
    resolution_method: str | None = None

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.paradox_id:
            self.paradox_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert paradox to dictionary representation."""
        return {
            "paradox_id": self.paradox_id,
            "timeline_id": self.timeline_id,
            "paradox_type": self.paradox_type.value,
            "severity": self.severity,
            "description": self.description,
            "affected_events": self.affected_events,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution_method": self.resolution_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalParadox":
        """Create paradox from dictionary representation."""
        return cls(
            paradox_id=data["paradox_id"],
            timeline_id=data["timeline_id"],
            paradox_type=ParadoxType(data["paradox_type"]),
            severity=data["severity"],
            description=data["description"],
            affected_events=data["affected_events"],
            detected_at=datetime.fromisoformat(data["detected_at"]),
            resolved=data.get("resolved", False),
            resolution_method=data.get("resolution_method"),
        )

    def to_proto(self):
        """Convert to Protocol Buffer message (placeholder)."""
        # This would return the actual protobuf message in a real implementation
        return {
            "paradox_id": self.paradox_id,
            "timeline_id": self.timeline_id,
            "paradox_type": self.paradox_type.value,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
@dataclass
class Timeline:
    """Represents a temporal timeline."""

    timeline_id: str
    name: str
    description: str
    status: TimelineStatus
    created_at: datetime
    initial_timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    last_event_timestamp: datetime | None = None
    event_count: int = 0
    paradox_count: int = 0
    entropy_level: float = 0.0
    stability_index: float = 1.0

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.timeline_id:
            self.timeline_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert timeline to dictionary representation."""
        return {
            "timeline_id": self.timeline_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "initial_timestamp": self.initial_timestamp.isoformat(),
            "last_event_timestamp": (
                self.last_event_timestamp.isoformat()
                if self.last_event_timestamp
                else None
            ),
            "event_count": self.event_count,
            "paradox_count": self.paradox_count,
            "entropy_level": self.entropy_level,
            "stability_index": self.stability_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Timeline":
        """Create timeline from dictionary representation."""
        return cls(
            timeline_id=data["timeline_id"],
            name=data["name"],
            description=data["description"],
            status=TimelineStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            initial_timestamp=datetime.fromisoformat(data["initial_timestamp"]),
            last_event_timestamp=(
                datetime.fromisoformat(data["last_event_timestamp"])
                if data.get("last_event_timestamp")
                else None
            ),
            event_count=data.get("event_count", 0),
            paradox_count=data.get("paradox_count", 0),
            entropy_level=data.get("entropy_level", 0.0),
            stability_index=data.get("stability_index", 1.0),
        )


@dataclass
class TemporalMetrics:
    """Temporal analysis metrics."""

    timeline_id: str
    measurement_timestamp: datetime
    causality_violations: int = 0
    entropy_fluctuations: list[float] = field(default_factory=list)
    paradox_density: float = 0.0
    temporal_coherence: float = 1.0
    stability_trend: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "timeline_id": self.timeline_id,
            "measurement_timestamp": self.measurement_timestamp.isoformat(),
            "causality_violations": self.causality_violations,
            "entropy_fluctuations": self.entropy_fluctuations,
            "paradox_density": self.paradox_density,
            "temporal_coherence": self.temporal_coherence,
            "stability_trend": self.stability_trend,
        }


# Type aliases for convenience
EventDict = dict[str, Any]
ParadoxDict = dict[str, Any]
TimelineDict = dict[str, Any]
MetricsDict = dict[str, Any]
