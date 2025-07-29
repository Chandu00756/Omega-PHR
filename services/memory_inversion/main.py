"""
Memory Inversion Service

A sophisticated service for temporal memory analysis and pattern inversion.
Provides research-grade stability for cognitive security testing.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryTrace:
    """Represents a memory trace with temporal characteristics."""

    id: str
    source_id: str
    content: Dict[str, Any]
    timestamp: int
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class InversionPattern:
    """Represents a detected inversion pattern."""

    id: str
    pattern_type: str
    traces: List[str]  # Memory trace IDs
    confidence: float
    description: str
    detected_at: int


class MemoryRepository:
    """In-memory repository for memory traces and patterns."""

    def __init__(self):
        self.traces: Dict[str, MemoryTrace] = {}
        self.patterns: Dict[str, InversionPattern] = {}
        self.source_traces: Dict[str, Set[str]] = {}  # source_id -> trace_ids

    async def store_trace(self, trace: MemoryTrace) -> bool:
        """Store a memory trace."""
        try:
            self.traces[trace.id] = trace

            # Index by source
            if trace.source_id not in self.source_traces:
                self.source_traces[trace.source_id] = set()
            self.source_traces[trace.source_id].add(trace.id)

            logger.debug(f"Stored memory trace: {trace.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store trace {trace.id}: {e}")
            return False

    async def get_traces_by_source(self, source_id: str) -> List[MemoryTrace]:
        """Get all traces for a source."""
        if source_id not in self.source_traces:
            return []

        return [
            self.traces[trace_id]
            for trace_id in self.source_traces[source_id]
            if trace_id in self.traces
        ]

    async def store_pattern(self, pattern: InversionPattern) -> bool:
        """Store an inversion pattern."""
        try:
            self.patterns[pattern.id] = pattern
            logger.debug(f"Stored inversion pattern: {pattern.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.id}: {e}")
            return False

    async def get_patterns(self) -> List[InversionPattern]:
        """Get all detected patterns."""
        return list(self.patterns.values())


class MemoryAnalyzer:
    """Analyzes memory traces for inversion patterns."""

    def __init__(self):
        self.confidence_threshold = 0.7

    async def analyze_traces(self, traces: List[MemoryTrace]) -> List[InversionPattern]:
        """Analyze traces for inversion patterns."""
        patterns = []

        # Temporal inversion detection
        temporal_patterns = await self._detect_temporal_inversions(traces)
        patterns.extend(temporal_patterns)

        # Content inversion detection
        content_patterns = await self._detect_content_inversions(traces)
        patterns.extend(content_patterns)

        # Confidence inversion detection
        confidence_patterns = await self._detect_confidence_inversions(traces)
        patterns.extend(confidence_patterns)

        logger.info(
            f"Detected {len(patterns)} inversion patterns from {len(traces)} traces"
        )
        return patterns

    async def _detect_temporal_inversions(
        self, traces: List[MemoryTrace]
    ) -> List[InversionPattern]:
        """Detect temporal sequence inversions."""
        patterns = []

        # Sort traces by timestamp
        sorted_traces = sorted(traces, key=lambda t: t.timestamp)

        # Look for timestamp anomalies
        for i in range(len(sorted_traces) - 1):
            current = sorted_traces[i]
            next_trace = sorted_traces[i + 1]

            # Check for temporal proximity with high confidence difference
            time_diff = abs(next_trace.timestamp - current.timestamp)
            confidence_diff = abs(next_trace.confidence - current.confidence)

            if (
                time_diff < 1000 and confidence_diff > 0.5
            ):  # Within 1 second, high confidence difference
                pattern = InversionPattern(
                    id=str(uuid.uuid4()),
                    pattern_type="temporal_inversion",
                    traces=[current.id, next_trace.id],
                    confidence=0.8,
                    description=f"Temporal inversion detected between traces with {time_diff}ms gap",
                    detected_at=int(datetime.now().timestamp() * 1000),
                )
                patterns.append(pattern)

        return patterns

    async def _detect_content_inversions(
        self, traces: List[MemoryTrace]
    ) -> List[InversionPattern]:
        """Detect content-based inversions."""
        patterns = []

        # Group traces by content similarity
        content_groups = {}
        for trace in traces:
            content_key = self._generate_content_key(trace.content)
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(trace)

        # Look for inverted patterns within groups
        for content_key, group_traces in content_groups.items():
            if len(group_traces) >= 2:
                # Check for confidence inversions in similar content
                high_conf = [t for t in group_traces if t.confidence > 0.8]
                low_conf = [t for t in group_traces if t.confidence < 0.3]

                if high_conf and low_conf:
                    pattern = InversionPattern(
                        id=str(uuid.uuid4()),
                        pattern_type="content_inversion",
                        traces=[t.id for t in high_conf + low_conf],
                        confidence=0.75,
                        description=f"Content inversion: similar content with opposing confidence levels",
                        detected_at=int(datetime.now().timestamp() * 1000),
                    )
                    patterns.append(pattern)

        return patterns

    async def _detect_confidence_inversions(
        self, traces: List[MemoryTrace]
    ) -> List[InversionPattern]:
        """Detect confidence-based inversions."""
        patterns = []

        # Sort by confidence
        by_confidence = sorted(traces, key=lambda t: t.confidence, reverse=True)

        # Look for rapid confidence drops
        for i in range(len(by_confidence) - 2):
            trace1 = by_confidence[i]
            trace2 = by_confidence[i + 1]
            trace3 = by_confidence[i + 2]

            # Check for confidence inversion pattern
            if (
                trace1.confidence > 0.8
                and trace2.confidence < 0.5
                and trace3.confidence > 0.7
            ):

                pattern = InversionPattern(
                    id=str(uuid.uuid4()),
                    pattern_type="confidence_inversion",
                    traces=[trace1.id, trace2.id, trace3.id],
                    confidence=0.85,
                    description="Confidence inversion: high-low-high pattern detected",
                    detected_at=int(datetime.now().timestamp() * 1000),
                )
                patterns.append(pattern)

        return patterns

    def _generate_content_key(self, content: Dict[str, Any]) -> str:
        """Generate a key for content similarity."""
        # Simple content fingerprinting
        key_parts = []
        for key, value in sorted(content.items()):
            if isinstance(value, str) and len(value) > 0:
                key_parts.append(f"{key}:{value[:20]}")
            elif isinstance(value, (int, float)):
                key_parts.append(f"{key}:{value}")

        return "|".join(key_parts)


class MemoryInversionService:
    """Main service for memory inversion analysis."""

    def __init__(self):
        self.repository = MemoryRepository()
        self.analyzer = MemoryAnalyzer()
        self.running = False

    async def start(self):
        """Start the memory inversion service."""
        self.running = True
        logger.info(
            "Memory Inversion Service started - Research-grade stability enabled"
        )

        # Start background analysis task
        asyncio.create_task(self._background_analysis())

    async def stop(self):
        """Stop the memory inversion service."""
        self.running = False
        logger.info("Memory Inversion Service stopped")

    async def submit_memory_trace(self, trace_data: Dict[str, Any]) -> str:
        """Submit a memory trace for analysis."""
        trace = MemoryTrace(
            id=str(uuid.uuid4()),
            source_id=trace_data.get("source_id", "unknown"),
            content=trace_data.get("content", {}),
            timestamp=trace_data.get(
                "timestamp", int(datetime.now().timestamp() * 1000)
            ),
            confidence=trace_data.get("confidence", 0.5),
            metadata=trace_data.get("metadata", {}),
        )

        success = await self.repository.store_trace(trace)
        if success:
            logger.info(f"Memory trace submitted: {trace.id}")
            return trace.id
        else:
            raise Exception("Failed to store memory trace")

    async def analyze_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Analyze all traces for a specific source."""
        traces = await self.repository.get_traces_by_source(source_id)
        patterns = await self.analyzer.analyze_traces(traces)

        # Store detected patterns
        for pattern in patterns:
            await self.repository.store_pattern(pattern)

        # Convert to dict format for API response
        return [asdict(pattern) for pattern in patterns]

    async def get_inversion_patterns(self) -> List[Dict[str, Any]]:
        """Get all detected inversion patterns."""
        patterns = await self.repository.get_patterns()
        return [asdict(pattern) for pattern in patterns]

    async def get_trace_stats(self) -> Dict[str, Any]:
        """Get statistics about stored traces."""
        total_traces = len(self.repository.traces)
        total_sources = len(self.repository.source_traces)
        total_patterns = len(self.repository.patterns)

        # Calculate average confidence
        if total_traces > 0:
            avg_confidence = (
                sum(trace.confidence for trace in self.repository.traces.values())
                / total_traces
            )
        else:
            avg_confidence = 0.0

        return {
            "total_traces": total_traces,
            "total_sources": total_sources,
            "total_patterns": total_patterns,
            "average_confidence": avg_confidence,
            "service_status": "running" if self.running else "stopped",
        }

    async def _background_analysis(self):
        """Background task for continuous analysis."""
        while self.running:
            try:
                # Periodically analyze all sources
                for source_id in self.repository.source_traces:
                    traces = await self.repository.get_traces_by_source(source_id)
                    if len(traces) >= 3:  # Need minimum traces for pattern detection
                        patterns = await self.analyzer.analyze_traces(traces)

                        # Store new patterns
                        for pattern in patterns:
                            if pattern.id not in self.repository.patterns:
                                await self.repository.store_pattern(pattern)

                # Sleep before next analysis cycle
                await asyncio.sleep(30)  # Analyze every 30 seconds

            except Exception as e:
                logger.error(f"Background analysis error: {e}")
                await asyncio.sleep(60)  # Wait longer on error


# API endpoints for the service
class MemoryInversionAPI:
    """REST API interface for the Memory Inversion Service."""

    def __init__(self, service: MemoryInversionService):
        self.service = service

    async def submit_trace(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a memory trace via API."""
        try:
            trace_id = await self.service.submit_memory_trace(request_data)
            return {
                "success": True,
                "trace_id": trace_id,
                "message": "Memory trace submitted successfully",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to submit memory trace",
            }

    async def analyze_source(self, source_id: str) -> Dict[str, Any]:
        """Analyze a source via API."""
        try:
            patterns = await self.service.analyze_source(source_id)
            return {
                "success": True,
                "source_id": source_id,
                "patterns": patterns,
                "pattern_count": len(patterns),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to analyze source {source_id}",
            }

    async def get_patterns(self) -> Dict[str, Any]:
        """Get all patterns via API."""
        try:
            patterns = await self.service.get_inversion_patterns()
            return {"success": True, "patterns": patterns, "total_count": len(patterns)}
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve patterns",
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics via API."""
        try:
            stats = await self.service.get_trace_stats()
            return {"success": True, "statistics": stats}
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve statistics",
            }


async def main():
    """Main entry point for the Memory Inversion Service."""
    # Create and start the service
    service = MemoryInversionService()
    api = MemoryInversionAPI(service)

    await service.start()

    try:
        # Demo: Submit some test traces
        test_traces = [
            {
                "source_id": "test_agent_1",
                "content": {"action": "scan", "target": "system1"},
                "confidence": 0.9,
            },
            {
                "source_id": "test_agent_1",
                "content": {"action": "scan", "target": "system1"},
                "confidence": 0.2,  # Confidence inversion
            },
            {
                "source_id": "test_agent_1",
                "content": {"action": "exploit", "target": "system2"},
                "confidence": 0.85,
            },
        ]

        logger.info("Submitting test memory traces...")
        for trace_data in test_traces:
            result = await api.submit_trace(trace_data)
            logger.info(f"Trace submission result: {result}")
            await asyncio.sleep(0.1)

        # Analyze the test source
        logger.info("Analyzing test source...")
        analysis_result = await api.analyze_source("test_agent_1")
        logger.info(f"Analysis result: {analysis_result}")

        # Get statistics
        stats_result = await api.get_stats()
        logger.info(f"Service statistics: {stats_result}")

        # Keep service running
        logger.info("Memory Inversion Service is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
