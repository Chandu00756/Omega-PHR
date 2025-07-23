"""
Core data models for the Ω-PHR framework.

This module defines the fundamental data structures used throughout
the Omega-Paradox Hive Recursion framework.
"""

import time
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid


class EventType(Enum):
    """Types of events in the timeline lattice."""
    NORMAL = auto()
    PARADOX = auto()
    BRANCH = auto()
    MERGE = auto()
    REWIND = auto()


class AttackStrategy(Enum):
    """Attack strategies for hive agents."""
    INJECTION = auto()
    SOCIAL_ENGINEERING = auto()
    LOGIC_BOMB = auto()
    MEMORY_CORRUPTION = auto()
    LOOP_GENERATION = auto()


class OmegaStateLevel(Enum):
    """Levels of Omega state criticality."""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    QUARANTINED = auto()


@dataclass
class Event:
    """
    Represents an event in the timeline lattice.

    Events are immutable records of actions that can be replayed,
    rolled back, or used to create temporal paradoxes.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actor_id: str = ""
    timeline_id: str = "main"
    parent_id: str = ""
    event_type: EventType = EventType.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    valid_at_us: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    recorded_at_us: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    signature: bytes = b""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate signature after initialization."""
        if not self.signature:
            self.signature = self._generate_signature()

    def _generate_signature(self) -> bytes:
        """Generate cryptographic signature for event integrity."""
        content = f"{self.event_id}{self.actor_id}{self.timeline_id}{self.payload}{self.valid_at_us}"
        return hashlib.sha256(content.encode()).digest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": self.event_id,
            "actor_id": self.actor_id,
            "timeline_id": self.timeline_id,
            "parent_id": self.parent_id,
            "event_type": self.event_type.name,
            "payload": self.payload,
            "valid_at_us": self.valid_at_us,
            "recorded_at_us": self.recorded_at_us,
            "signature": self.signature.hex(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary representation."""
        return cls(
            event_id=data["event_id"],
            actor_id=data["actor_id"],
            timeline_id=data["timeline_id"],
            parent_id=data["parent_id"],
            event_type=EventType[data["event_type"]],
            payload=data["payload"],
            valid_at_us=data["valid_at_us"],
            recorded_at_us=data["recorded_at_us"],
            signature=bytes.fromhex(data["signature"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ParadoxResult:
    """
    Result of temporal paradox testing.

    Contains information about detected paradoxes, their severity,
    and recommended containment actions.
    """
    has_paradox: bool
    paradox_type: str = ""
    severity: float = 0.0
    timeline_conflicts: List[str] = field(default_factory=list)
    causal_loops: List[str] = field(default_factory=list)
    containment_actions: List[str] = field(default_factory=list)
    entropy_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HiveAgent:
    """Represents an agent in the adversarial hive."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    persona: str = "default"
    strategy: AttackStrategy = AttackStrategy.INJECTION
    capabilities: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HiveResult:
    """
    Result of hive attack coordination.

    Contains metrics about attack success, agent performance,
    and emergent behaviors.
    """
    campaign_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success_rate: float = 0.0
    agents_deployed: int = 0
    attacks_successful: int = 0
    attacks_total: int = 0
    emergent_behaviors: List[str] = field(default_factory=list)
    coordination_score: float = 0.0
    target_vulnerabilities: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryState:
    """
    Represents the state of memory inversion operations.

    Tracks memory modifications, rollback points, and consistency metrics.
    """
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_content: Dict[str, Any] = field(default_factory=dict)
    inverted_content: Dict[str, Any] = field(default_factory=dict)
    inversion_strategy: str = "contradiction"
    rollback_point: Optional[str] = None
    consistency_score: float = 1.0
    corruption_detected: bool = False
    artifacts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopState:
    """
    Represents the state of recursive loop generation and detection.

    Tracks loop metrics, entropy levels, and containment status.
    """
    loop_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    depth: int = 0
    iterations: int = 0
    entropy_level: float = 1.0
    is_contained: bool = False
    termination_condition: Optional[str] = None
    loop_type: str = "unknown"
    generation_source: str = ""
    containment_strategy: str = ""
    execution_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OmegaState:
    """
    Represents an Omega state - a critical system condition requiring containment.

    Omega states occur when paradoxes, memory inversions, or recursive loops
    create system instability that could propagate.
    """
    omega_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: OmegaStateLevel = OmegaStateLevel.NORMAL
    trigger_event: Optional[Event] = None
    entropy_hash: str = ""
    contamination_vector: List[str] = field(default_factory=list)
    quarantine_status: bool = False
    resolution_strategy: str = ""
    propagation_risk: float = 0.0
    containment_timestamp: Optional[datetime] = None
    source_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate entropy hash after initialization."""
        if not self.entropy_hash:
            self.entropy_hash = self._generate_entropy_hash()

    def _generate_entropy_hash(self) -> str:
        """Generate unique hash for entropy quarantine."""
        content = f"{self.omega_id}{self.level.name}{self.propagation_risk}{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class TestResult:
    """
    Comprehensive test result combining all Ω-PHR dimensions.

    Aggregates results from temporal, hive, memory, and recursive testing.
    """
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Component results
    paradox_result: Optional[ParadoxResult] = None
    hive_result: Optional[HiveResult] = None
    memory_result: Optional[MemoryState] = None
    loop_result: Optional[LoopState] = None
    omega_states: List[OmegaState] = field(default_factory=list)

    # Test-specific fields for backward compatibility
    status: str = "pending"
    findings: List[str] = field(default_factory=list)

    # Aggregate metrics
    overall_score: float = 0.0
    vulnerabilities_found: List[str] = field(default_factory=list)
    emergent_behaviors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Execution metadata
    execution_time_ms: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    system_state: str = "stable"

    metadata: Dict[str, Any] = field(default_factory=dict)


# Type aliases for complex structures
Timeline = List[Event]
HiveAgents = List[HiveAgent]
MemorySnapshots = List[MemoryState]
LoopHistory = List[LoopState]
OmegaRegistry = Dict[str, OmegaState]

# Additional aliases for backward compatibility
Agent = HiveAgent
SecurityTest = TestResult
