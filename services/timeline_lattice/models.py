"""
Data models for Timeline Lattice service.
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from timeline_pb2 import Event as ProtoEvent


@dataclass
class EventModel:
    """
    Internal event model for Timeline Lattice service.

    This model provides conversion between protobuf and internal representations.
    """
    event_id: str
    actor_id: str
    timeline_id: str
    parent_id: str
    payload: bytes
    valid_at_us: int
    recorded_at_us: int
    signature: bytes
    event_type: str = "NORMAL"
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_proto(cls, proto_event: ProtoEvent) -> "EventModel":
        """Create EventModel from protobuf Event."""
        return cls(
            event_id=proto_event.event_id,
            actor_id=proto_event.actor_id,
            timeline_id=proto_event.timeline_id,
            parent_id=proto_event.parent_id,
            payload=proto_event.payload,
            valid_at_us=proto_event.valid_at_us,
            recorded_at_us=proto_event.recorded_at_us,
            signature=proto_event.signature,
            event_type=proto_event.event_type,
            metadata=dict(proto_event.metadata)
        )

    def to_proto(self) -> ProtoEvent:
        """Convert EventModel to protobuf Event."""
        proto_event = ProtoEvent()
        proto_event.event_id = self.event_id
        proto_event.actor_id = self.actor_id
        proto_event.timeline_id = self.timeline_id
        proto_event.parent_id = self.parent_id
        proto_event.payload = self.payload
        proto_event.valid_at_us = self.valid_at_us
        proto_event.recorded_at_us = self.recorded_at_us
        proto_event.signature = self.signature
        proto_event.event_type = self.event_type
        proto_event.metadata.update(self.metadata)
        return proto_event

    def calculate_signature(self) -> bytes:
        """Calculate cryptographic signature for event integrity."""
        content = f"{self.event_id}{self.actor_id}{self.timeline_id}{self.payload.decode('utf-8', errors='ignore')}{self.valid_at_us}"
        return hashlib.sha256(content.encode()).digest()

    def verify_signature(self) -> bool:
        """Verify event signature integrity."""
        expected_signature = self.calculate_signature()
        return self.signature == expected_signature

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": self.event_id,
            "actor_id": self.actor_id,
            "timeline_id": self.timeline_id,
            "parent_id": self.parent_id,
            "payload": self.payload.decode('utf-8', errors='ignore'),
            "valid_at_us": self.valid_at_us,
            "recorded_at_us": self.recorded_at_us,
            "signature": self.signature.hex(),
            "event_type": self.event_type,
            "metadata": self.metadata
        }


@dataclass
class TimelineInfo:
    """Information about a timeline."""
    timeline_id: str
    event_count: int
    created_at: Optional[int] = None
    consistency_score: float = 1.0
    entropy_level: float = 0.0
    has_paradoxes: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ParadoxResult:
    """Result of paradox testing."""
    has_paradox: bool
    paradox_type: str = ""
    severity: float = 0.0
    timeline_conflicts: list = field(default_factory=list)
    causal_loops: list = field(default_factory=list)
    containment_actions: list = field(default_factory=list)
    entropy_score: float = 0.0
