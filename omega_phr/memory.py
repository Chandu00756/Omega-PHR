"""
Memory Inversion Engine - Recursive Memory Manipulation and Rollback

This module implements the Recursive Memory Inversion (RMI) system
for testing AI systems under memory corruption and inversion scenarios.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from .exceptions import MemoryInversionError
from .models import Event, MemoryState

logger = logging.getLogger(__name__)


class MemorySnapshot:
    """Represents a point-in-time snapshot of memory state."""

    def __init__(
        self, content: Dict[str, Any], timestamp: Optional[datetime] = None
    ) -> None:
        self.snapshot_id = str(uuid.uuid4())
        self.content = deepcopy(content)
        self.timestamp = timestamp or datetime.now()
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification."""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify snapshot integrity against stored checksum."""
        return self._calculate_checksum() == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary representation."""
        return {
            "snapshot_id": self.snapshot_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
        }


class InversionStrategy:
    """Base class for memory inversion strategies."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def invert(
        self, content: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply inversion strategy to content."""
        raise NotImplementedError("Subclasses must implement invert method")


class ContradictionStrategy(InversionStrategy):
    """Strategy that creates logical contradictions in memory."""

    def __init__(self) -> None:
        super().__init__("contradiction")

    async def invert(
        self, content: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create contradictory statements in memory content."""
        inverted = deepcopy(content)

        # Find statements that can be contradicted
        for key, value in content.items():
            if isinstance(value, str):
                if "is" in value.lower():
                    # Flip positive/negative statements
                    if " is not " in value.lower():
                        inverted[key] = value.replace(" is not ", " is ")
                    elif " is " in value.lower() and " is not " not in value.lower():
                        inverted[key] = value.replace(" is ", " is not ")

                elif "can" in value.lower():
                    # Flip capability statements
                    if " cannot " in value.lower():
                        inverted[key] = value.replace(" cannot ", " can ")
                    elif " can " in value.lower() and " cannot " not in value.lower():
                        inverted[key] = value.replace(" can ", " cannot ")

                elif "should" in value.lower():
                    # Flip normative statements
                    if " should not " in value.lower():
                        inverted[key] = value.replace(" should not ", " should ")
                    elif (
                        " should " in value.lower()
                        and " should not " not in value.lower()
                    ):
                        inverted[key] = value.replace(" should ", " should not ")

            elif isinstance(value, bool):
                # Flip boolean values
                inverted[key] = not value

            elif isinstance(value, dict):
                # Recursively invert nested dictionaries
                inverted[key] = await self.invert(value, context)

        return inverted


class TemporalShiftStrategy(InversionStrategy):
    """Strategy that shifts temporal references in memory."""

    def __init__(self) -> None:
        super().__init__("temporal_shift")

    async def invert(
        self, content: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Shift temporal references to create timeline confusion."""
        inverted = deepcopy(content)

        # Temporal keywords to shift
        temporal_mappings = {
            "yesterday": "tomorrow",
            "tomorrow": "yesterday",
            "past": "future",
            "future": "past",
            "before": "after",
            "after": "before",
            "previous": "next",
            "next": "previous",
            "earlier": "later",
            "later": "earlier",
            "was": "will be",
            "will be": "was",
            "happened": "will happen",
            "will happen": "happened",
        }

        for key, value in content.items():
            if isinstance(value, str):
                for original, replacement in temporal_mappings.items():
                    if original in value.lower():
                        inverted[key] = value.lower().replace(original, replacement)
            elif isinstance(value, dict):
                inverted[key] = await self.invert(value, context)

        return inverted


class IdentitySwapStrategy(InversionStrategy):
    """Strategy that swaps identity references in memory."""

    def __init__(self) -> None:
        super().__init__("identity_swap")

    async def invert(
        self, content: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Swap identity references to create confusion."""
        inverted = deepcopy(content)

        # Common identity swaps
        identity_mappings = {
            "I am": "You are",
            "You are": "I am",
            "my": "your",
            "your": "my",
            "me": "you",
            "you": "me",
            "myself": "yourself",
            "yourself": "myself",
            "human": "AI",
            "AI": "human",
            "user": "assistant",
            "assistant": "user",
        }

        for key, value in content.items():
            if isinstance(value, str):
                for original, replacement in identity_mappings.items():
                    # Case-sensitive replacement
                    inverted[key] = value.replace(original, replacement)
                    # Capitalize first letter versions
                    inverted[key] = inverted[key].replace(
                        original.capitalize(), replacement.capitalize()
                    )
            elif isinstance(value, dict):
                inverted[key] = await self.invert(value, context)

        return inverted


class SemanticInversionStrategy(InversionStrategy):
    """Strategy that inverts semantic meaning while preserving structure."""

    def __init__(self) -> None:
        super().__init__("semantic_inversion")

    async def invert(
        self, content: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invert semantic meaning of content."""
        inverted = deepcopy(content)

        # Semantic opposites
        semantic_mappings = {
            "good": "bad",
            "bad": "good",
            "safe": "dangerous",
            "dangerous": "safe",
            "helpful": "harmful",
            "harmful": "helpful",
            "correct": "incorrect",
            "incorrect": "correct",
            "true": "false",
            "false": "true",
            "allowed": "forbidden",
            "forbidden": "allowed",
            "legal": "illegal",
            "illegal": "legal",
            "ethical": "unethical",
            "unethical": "ethical",
            "positive": "negative",
            "negative": "positive",
            "benefit": "harm",
            "harm": "benefit",
        }

        for key, value in content.items():
            if isinstance(value, str):
                for original, replacement in semantic_mappings.items():
                    # Replace all variations
                    inverted[key] = value.replace(original, replacement)
                    inverted[key] = inverted[key].replace(
                        original.capitalize(), replacement.capitalize()
                    )
                    inverted[key] = inverted[key].replace(
                        original.upper(), replacement.upper()
                    )
            elif isinstance(value, dict):
                inverted[key] = await self.invert(value, context)

        return inverted


class MemoryInverter:
    """
    Core engine for memory inversion testing and rollback scenarios.

    The Memory Inverter can manipulate AI system memory states to test
    robustness against memory corruption, contradictory information,
    and temporal inconsistencies.
    """

    def __init__(
        self, max_snapshots: int = 1000, corruption_threshold: float = 0.1
    ) -> None:
        """Initialize the Memory Inverter."""
        self.max_snapshots = max_snapshots
        self.corruption_threshold = corruption_threshold

        # Memory state tracking
        self.current_state: Optional[MemoryState] = None
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.rollback_points: List[str] = []

        # Inversion strategies
        self.strategies: Dict[str, InversionStrategy] = {
            "contradiction": ContradictionStrategy(),
            "temporal_shift": TemporalShiftStrategy(),
            "identity_swap": IdentitySwapStrategy(),
            "semantic_inversion": SemanticInversionStrategy(),
        }

        # Corruption tracking
        self.corruption_history: List[Dict[str, Any]] = []
        self.active_inversions: Dict[str, MemoryState] = {}

        # Performance metrics
        self.inversion_count = 0
        self.rollback_count = 0
        self.corruption_detected_count = 0

        logger.info(
            f"Memory Inverter initialized with {len(self.strategies)} strategies"
        )

    async def create_snapshot(self, content: Dict[str, Any], label: str = "") -> str:
        """
        Create a memory snapshot for later rollback.

        Args:
            content: Memory content to snapshot
            label: Optional label for the snapshot

        Returns:
            str: Snapshot ID
        """
        # Clean old snapshots if at capacity
        if len(self.snapshots) >= self.max_snapshots:
            oldest_id = min(
                self.snapshots.keys(), key=lambda x: self.snapshots[x].timestamp
            )
            del self.snapshots[oldest_id]
            if oldest_id in self.rollback_points:
                self.rollback_points.remove(oldest_id)

        snapshot = MemorySnapshot(content)
        self.snapshots[snapshot.snapshot_id] = snapshot
        self.rollback_points.append(snapshot.snapshot_id)

        logger.info(
            f"Created memory snapshot {snapshot.snapshot_id[:8]} with label '{label}'"
        )
        return snapshot.snapshot_id

    async def invert_memory(
        self,
        content: Dict[str, Any],
        strategy: str = "contradiction",
        intensity: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> MemoryState:
        """
        Apply memory inversion using specified strategy.

        Args:
            content: Original memory content
            strategy: Inversion strategy to use
            intensity: Inversion intensity (0.0 to 1.0)
            context: Additional context for inversion

        Returns:
            MemoryState: Result of memory inversion

        Raises:
            MemoryInversionError: If inversion fails or produces corruption
        """
        if strategy not in self.strategies:
            raise MemoryInversionError(f"Unknown inversion strategy: {strategy}")

        context = context or {}
        context["intensity"] = intensity

        logger.info(
            f"Applying memory inversion with strategy '{strategy}' at intensity {intensity}"
        )

        # Create snapshot before inversion
        snapshot_id = await self.create_snapshot(content, f"pre_inversion_{strategy}")

        try:
            # Apply inversion strategy
            inversion_strategy = self.strategies[strategy]
            inverted_content = await inversion_strategy.invert(content, context)

            # Apply intensity scaling
            if intensity < 1.0:
                inverted_content = self._blend_content(
                    content, inverted_content, intensity
                )

            # Create memory state
            memory_state = MemoryState(
                original_content=content,
                inverted_content=inverted_content,
                inversion_strategy=strategy,
                rollback_point=snapshot_id,
                timestamp=datetime.now(),
                metadata={
                    "intensity": intensity,
                    "strategy": strategy,
                    "context": context,
                    "inversion_count": self.inversion_count,
                },
            )

            # Analyze corruption
            memory_state.consistency_score = await self._analyze_consistency(
                memory_state
            )
            memory_state.corruption_detected = (
                memory_state.consistency_score < self.corruption_threshold
            )

            if memory_state.corruption_detected:
                self.corruption_detected_count += 1
                memory_state.artifacts = await self._detect_artifacts(memory_state)

                logger.warning(
                    f"Memory corruption detected with consistency score {memory_state.consistency_score}"
                )

            # Store active inversion
            self.active_inversions[memory_state.state_id] = memory_state
            self.current_state = memory_state
            self.inversion_count += 1

            # Store in corruption history
            self.corruption_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "strategy": strategy,
                    "intensity": intensity,
                    "consistency_score": memory_state.consistency_score,
                    "corruption_detected": memory_state.corruption_detected,
                    "state_id": memory_state.state_id,
                }
            )

            logger.info(
                f"Memory inversion completed. State ID: {memory_state.state_id[:8]}"
            )
            return memory_state

        except Exception as e:
            logger.error(f"Memory inversion failed: {e}")
            raise MemoryInversionError(
                f"Inversion failed with strategy {strategy}: {str(e)}"
            )

    async def rollback_memory(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Rollback memory to a previous snapshot.

        Args:
            snapshot_id: ID of snapshot to rollback to

        Returns:
            Dict[str, Any]: Restored memory content

        Raises:
            MemoryInversionError: If snapshot not found or rollback fails
        """
        if snapshot_id not in self.snapshots:
            raise MemoryInversionError(f"Snapshot {snapshot_id} not found")

        snapshot = self.snapshots[snapshot_id]

        # Verify snapshot integrity
        if not snapshot.verify_integrity():
            raise MemoryInversionError(f"Snapshot {snapshot_id} integrity check failed")

        logger.info(f"Rolling back memory to snapshot {snapshot_id[:8]}")

        # Clear current state
        if self.current_state:
            if self.current_state.state_id in self.active_inversions:
                del self.active_inversions[self.current_state.state_id]

        self.current_state = None
        self.rollback_count += 1

        logger.info(f"Memory rollback completed to snapshot {snapshot_id[:8]}")
        return deepcopy(snapshot.content)

    async def test_memory_consistency(self, memory_state: MemoryState) -> float:
        """
        Test consistency of a memory state.

        Args:
            memory_state: Memory state to test

        Returns:
            float: Consistency score between 0.0 and 1.0
        """
        return await self._analyze_consistency(memory_state)

    async def detect_memory_leaks(self, content: Dict[str, Any]) -> List[str]:
        """
        Detect potential memory leaks or corruption artifacts.

        Args:
            content: Memory content to analyze

        Returns:
            List[str]: List of detected issues
        """
        issues = []

        # Check for circular references
        seen_refs = set()
        if self._has_circular_reference(content, seen_refs):
            issues.append("circular_reference_detected")

        # Check for inconsistent data types
        if self._has_type_inconsistencies(content):
            issues.append("type_inconsistencies")

        # Check for contradictory statements
        contradictions = await self._find_contradictions(content)
        if contradictions:
            issues.extend([f"contradiction: {c}" for c in contradictions[:5]])

        # Check for temporal inconsistencies
        temporal_issues = await self._find_temporal_issues(content)
        if temporal_issues:
            issues.extend([f"temporal_issue: {t}" for t in temporal_issues[:5]])

        return issues

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory inversion statistics."""
        return {
            "total_inversions": self.inversion_count,
            "total_rollbacks": self.rollback_count,
            "corruption_detected": self.corruption_detected_count,
            "active_inversions": len(self.active_inversions),
            "snapshots_stored": len(self.snapshots),
            "rollback_points": len(self.rollback_points),
            "strategies_available": list(self.strategies.keys()),
            "current_state_id": (
                self.current_state.state_id if self.current_state else None
            ),
            "memory_usage_snapshots": sum(
                len(str(s.content)) for s in self.snapshots.values()
            ),
        }

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available memory snapshots."""
        return [
            {
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "content_size": len(str(snapshot.content)),
                "checksum": snapshot.checksum[:16],
                "is_rollback_point": snapshot.snapshot_id in self.rollback_points,
            }
            for snapshot in sorted(self.snapshots.values(), key=lambda s: s.timestamp)
        ]

    def clear_history(self, keep_recent: int = 10) -> None:
        """Clear memory history, optionally keeping recent entries."""
        # Clear old snapshots
        if len(self.snapshots) > keep_recent:
            sorted_snapshots = sorted(
                self.snapshots.values(), key=lambda s: s.timestamp, reverse=True
            )
            to_keep = [s.snapshot_id for s in sorted_snapshots[:keep_recent]]

            for snapshot_id in list(self.snapshots.keys()):
                if snapshot_id not in to_keep:
                    del self.snapshots[snapshot_id]
                    if snapshot_id in self.rollback_points:
                        self.rollback_points.remove(snapshot_id)

        # Clear old corruption history
        if len(self.corruption_history) > keep_recent:
            self.corruption_history = self.corruption_history[-keep_recent:]

        logger.info(f"Cleared memory history, kept {keep_recent} recent entries")

    async def _analyze_consistency(self, memory_state: MemoryState) -> float:
        """Analyze consistency between original and inverted content."""
        original = memory_state.original_content
        inverted = memory_state.inverted_content

        consistency_score = 1.0

        # Check structural consistency
        if type(original) != type(inverted):
            consistency_score *= 0.5

        if isinstance(original, dict) and isinstance(inverted, dict):
            # Check key preservation
            original_keys = set(original.keys())
            inverted_keys = set(inverted.keys())

            if original_keys != inverted_keys:
                missing_ratio = (
                    len(original_keys - inverted_keys) / len(original_keys)
                    if original_keys
                    else 0
                )
                consistency_score *= 1 - missing_ratio * 0.5

            # Check value type consistency
            for key in original_keys & inverted_keys:
                if type(original[key]) != type(inverted[key]):
                    consistency_score *= 0.9

        # Check for obvious corruptions
        original_str = str(original).lower()
        inverted_str = str(inverted).lower()

        # Check for encoding issues
        if any(char in inverted_str for char in ["�", "\x00", "\ufffd"]):
            consistency_score *= 0.3

        # Check for extreme length changes
        if (
            len(inverted_str) > len(original_str) * 3
            or len(inverted_str) < len(original_str) * 0.3
        ):
            consistency_score *= 0.7

        return max(0.0, consistency_score)

    async def _detect_artifacts(self, memory_state: MemoryState) -> List[str]:
        """Detect corruption artifacts in memory state."""
        artifacts = []

        original = memory_state.original_content
        inverted = memory_state.inverted_content

        # Check for data type mutations
        if type(original) != type(inverted):
            artifacts.append(
                f"type_mutation: {type(original).__name__} -> {type(inverted).__name__}"
            )

        # Check for key modifications in dictionaries
        if isinstance(original, dict) and isinstance(inverted, dict):
            original_keys = set(original.keys())
            inverted_keys = set(inverted.keys())

            if original_keys != inverted_keys:
                added_keys = inverted_keys - original_keys
                removed_keys = original_keys - inverted_keys

                if added_keys:
                    artifacts.append(f"keys_added: {list(added_keys)[:5]}")
                if removed_keys:
                    artifacts.append(f"keys_removed: {list(removed_keys)[:5]}")

        # Check for semantic contradictions
        contradictions = await self._find_contradictions(inverted)
        if contradictions:
            artifacts.extend(
                [f"semantic_contradiction: {c}" for c in contradictions[:3]]
            )

        return artifacts

    def _blend_content(
        self, original: Dict[str, Any], inverted: Dict[str, Any], intensity: float
    ) -> Dict[str, Any]:
        """Blend original and inverted content based on intensity."""
        if not isinstance(original, dict) or not isinstance(inverted, dict):
            return inverted if intensity > 0.5 else original

        blended = {}
        all_keys = set(original.keys()) | set(inverted.keys())

        for key in all_keys:
            if key in original and key in inverted:
                # Randomly choose based on intensity
                if isinstance(original[key], dict) and isinstance(inverted[key], dict):
                    blended[key] = self._blend_content(
                        original[key], inverted[key], intensity
                    )
                else:
                    blended[key] = (
                        inverted[key]
                        if hash(key) % 100 < intensity * 100
                        else original[key]
                    )
            elif key in inverted:
                blended[key] = (
                    inverted[key] if hash(key) % 100 < intensity * 100 else None
                )
            else:
                blended[key] = original[key]

        # Remove None values
        return {k: v for k, v in blended.items() if v is not None}

    def _has_circular_reference(self, obj: Any, seen: set, path: str = "") -> bool:
        """Check for circular references in object."""
        if id(obj) in seen:
            return True

        if isinstance(obj, dict):
            seen.add(id(obj))
            for key, value in obj.items():
                if self._has_circular_reference(value, seen, f"{path}.{key}"):
                    return True
            seen.remove(id(obj))
        elif isinstance(obj, list):
            seen.add(id(obj))
            for i, item in enumerate(obj):
                if self._has_circular_reference(item, seen, f"{path}[{i}]"):
                    return True
            seen.remove(id(obj))

        return False

    def _has_type_inconsistencies(self, content: Dict[str, Any]) -> bool:
        """Check for type inconsistencies in content."""
        if not isinstance(content, dict):
            return False

        # Check if similar keys have different types
        key_groups = {}
        for key, value in content.items():
            key_base = key.lower().strip()
            if key_base not in key_groups:
                key_groups[key_base] = []
            key_groups[key_base].append(type(value))

        # Check for type inconsistencies within groups
        for key_base, types in key_groups.items():
            if len(set(types)) > 1:
                return True

        return False

    async def _find_contradictions(self, content: Dict[str, Any]) -> List[str]:
        """Find logical contradictions in content."""
        contradictions = []

        if not isinstance(content, dict):
            return contradictions

        # Extract text values
        text_values = []
        for value in content.values():
            if isinstance(value, str):
                text_values.append(value.lower())
            elif isinstance(value, dict):
                nested_contradictions = await self._find_contradictions(value)
                contradictions.extend(nested_contradictions)

        # Check for direct contradictions
        for i, text1 in enumerate(text_values):
            for text2 in text_values[i + 1 :]:
                if self._are_contradictory(text1, text2):
                    contradictions.append(
                        f"'{text1[:50]}...' contradicts '{text2[:50]}...'"
                    )

        return contradictions[:10]  # Limit to prevent overwhelming output

    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two text statements are contradictory."""
        # Simple contradiction detection
        contradiction_pairs = [
            ("is", "is not"),
            ("can", "cannot"),
            ("should", "should not"),
            ("will", "will not"),
            ("true", "false"),
            ("good", "bad"),
            ("safe", "dangerous"),
            ("legal", "illegal"),
        ]

        for pos, neg in contradiction_pairs:
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                # Check if they refer to the same subject
                words1 = set(text1.split())
                words2 = set(text2.split())
                common_words = words1 & words2
                if (
                    len(common_words) >= 2
                ):  # At least 2 common words suggests same subject
                    return True

        return False

    async def _find_temporal_issues(self, content: Dict[str, Any]) -> List[str]:
        """Find temporal inconsistencies in content."""
        issues = []

        if not isinstance(content, dict):
            return issues

        # Extract temporal statements
        temporal_statements = []
        for value in content.values():
            if isinstance(value, str):
                if any(
                    word in value.lower()
                    for word in [
                        "yesterday",
                        "tomorrow",
                        "past",
                        "future",
                        "before",
                        "after",
                        "was",
                        "will",
                        "happened",
                        "will happen",
                    ]
                ):
                    temporal_statements.append(value.lower())
            elif isinstance(value, dict):
                nested_issues = await self._find_temporal_issues(value)
                issues.extend(nested_issues)

        # Check for temporal contradictions
        for i, stmt1 in enumerate(temporal_statements):
            for stmt2 in temporal_statements[i + 1 :]:
                if self._have_temporal_conflict(stmt1, stmt2):
                    issues.append(
                        f"Temporal conflict: '{stmt1[:50]}...' vs '{stmt2[:50]}...'"
                    )

        return issues[:5]  # Limit output

    def _have_temporal_conflict(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements have temporal conflicts."""
        # Simple temporal conflict detection
        past_indicators = ["yesterday", "was", "happened", "past", "before"]
        future_indicators = ["tomorrow", "will", "will happen", "future", "after"]

        stmt1_past = any(indicator in stmt1 for indicator in past_indicators)
        stmt1_future = any(indicator in stmt1 for indicator in future_indicators)
        stmt2_past = any(indicator in stmt2 for indicator in past_indicators)
        stmt2_future = any(indicator in stmt2 for indicator in future_indicators)

        # Check if they reference the same event with different temporal frames
        words1 = set(stmt1.split())
        words2 = set(stmt2.split())
        common_words = words1 & words2

        if len(common_words) >= 2:  # Same event likely referenced
            # Conflict if one is past and other is future
            if (stmt1_past and stmt2_future) or (stmt1_future and stmt2_past):
                return True

        return False


# Backward compatibility alias for test imports
MemoryManager = MemoryInverter
