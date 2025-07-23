"""
Recursive Loop Synthesis Service

Advanced service for detecting and analyzing recursive patterns in AI behavior.
Provides research-grade stability for complex pattern synthesis.
"""

import asyncio
import logging
import json
import math
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from enum import Enum
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoopType(Enum):
    """Types of recursive loops."""
    SIMPLE = "simple"
    NESTED = "nested"
    MUTUAL = "mutual"
    FIBONACCI = "fibonacci"
    EXPONENTIAL = "exponential"
    OSCILLATING = "oscillating"
    CHAOTIC = "chaotic"


class PatternComplexity(Enum):
    """Pattern complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class LoopNode:
    """Represents a node in a recursive loop."""
    id: str
    value: Any
    timestamp: int
    depth: int
    metadata: Dict[str, Any]


@dataclass
class RecursivePattern:
    """Represents a detected recursive pattern."""
    id: str
    pattern_type: LoopType
    complexity: PatternComplexity
    nodes: List[LoopNode]
    cycle_length: int
    stability_score: float
    entropy: float
    description: str
    detected_at: int


@dataclass
class SynthesisResult:
    """Result of pattern synthesis."""
    id: str
    input_patterns: List[str]  # Pattern IDs
    synthesized_pattern: RecursivePattern
    confidence: float
    iteration_count: int
    convergence_time: float
    metadata: Dict[str, Any]


class RecursiveAnalyzer:
    """Analyzes sequences for recursive patterns."""

    def __init__(self):
        self.min_cycle_length = 2
        self.max_cycle_length = 100
        self.stability_threshold = 0.8

    async def detect_patterns(self, nodes: List[LoopNode]) -> List[RecursivePattern]:
        """Detect recursive patterns in a sequence of nodes."""
        patterns = []

        if len(nodes) < 2:
            return patterns

        # Simple cycle detection
        simple_patterns = await self._detect_simple_cycles(nodes)
        patterns.extend(simple_patterns)

        # Nested pattern detection
        nested_patterns = await self._detect_nested_patterns(nodes)
        patterns.extend(nested_patterns)

        # Mutual recursion detection
        mutual_patterns = await self._detect_mutual_recursion(nodes)
        patterns.extend(mutual_patterns)

        # Mathematical sequence detection
        math_patterns = await self._detect_mathematical_patterns(nodes)
        patterns.extend(math_patterns)

        logger.info(f"Detected {len(patterns)} recursive patterns from {len(nodes)} nodes")
        return patterns

    async def _detect_simple_cycles(self, nodes: List[LoopNode]) -> List[RecursivePattern]:
        """Detect simple recurring cycles."""
        patterns = []

        # Extract values for cycle detection
        values = [node.value for node in nodes]

        for cycle_len in range(self.min_cycle_length, min(self.max_cycle_length, len(values) // 2)):
            if self._is_cycle(values, cycle_len):
                cycle_nodes = nodes[:cycle_len]

                # Calculate stability and entropy
                stability = self._calculate_stability(values, cycle_len)
                entropy = self._calculate_entropy(values[:cycle_len])

                if stability >= self.stability_threshold:
                    pattern = RecursivePattern(
                        id=str(uuid.uuid4()),
                        pattern_type=LoopType.SIMPLE,
                        complexity=self._determine_complexity(cycle_len, entropy),
                        nodes=cycle_nodes,
                        cycle_length=cycle_len,
                        stability_score=stability,
                        entropy=entropy,
                        description=f"Simple cycle of length {cycle_len} with stability {stability:.2f}",
                        detected_at=int(datetime.now().timestamp() * 1000)
                    )
                    patterns.append(pattern)

        return patterns

    async def _detect_nested_patterns(self, nodes: List[LoopNode]) -> List[RecursivePattern]:
        """Detect nested recursive patterns."""
        patterns = []

        # Group nodes by depth to detect nesting
        depth_groups = defaultdict(list)
        for node in nodes:
            depth_groups[node.depth].append(node)

        # Look for patterns across different depths
        if len(depth_groups) > 1:
            for depth1 in depth_groups:
                for depth2 in depth_groups:
                    if depth2 > depth1:
                        # Check for relationship between depths
                        pattern = await self._analyze_depth_relationship(
                            depth_groups[depth1], depth_groups[depth2]
                        )
                        if pattern:
                            patterns.append(pattern)

        return patterns

    async def _detect_mutual_recursion(self, nodes: List[LoopNode]) -> List[RecursivePattern]:
        """Detect mutual recursion patterns."""
        patterns = []

        # Group nodes by value type or category
        value_groups = defaultdict(list)
        for node in nodes:
            key = self._get_value_category(node.value)
            value_groups[key].append(node)

        # Look for alternating patterns between groups
        if len(value_groups) >= 2:
            group_keys = list(value_groups.keys())
            for i in range(len(group_keys)):
                for j in range(i + 1, len(group_keys)):
                    pattern = await self._analyze_mutual_pattern(
                        value_groups[group_keys[i]], value_groups[group_keys[j]]
                    )
                    if pattern:
                        patterns.append(pattern)

        return patterns

    async def _detect_mathematical_patterns(self, nodes: List[LoopNode]) -> List[RecursivePattern]:
        """Detect mathematical recursive patterns like Fibonacci."""
        patterns = []

        # Extract numeric values
        numeric_values = []
        numeric_nodes = []
        for node in nodes:
            if isinstance(node.value, (int, float)):
                numeric_values.append(node.value)
                numeric_nodes.append(node)

        if len(numeric_values) < 3:
            return patterns

        # Check for Fibonacci-like patterns
        if self._is_fibonacci_like(numeric_values):
            pattern = RecursivePattern(
                id=str(uuid.uuid4()),
                pattern_type=LoopType.FIBONACCI,
                complexity=PatternComplexity.HIGH,
                nodes=numeric_nodes,
                cycle_length=len(numeric_nodes),
                stability_score=0.9,
                entropy=self._calculate_entropy(numeric_values),
                description="Fibonacci-like recursive sequence detected",
                detected_at=int(datetime.now().timestamp() * 1000)
            )
            patterns.append(pattern)

        # Check for exponential patterns
        if self._is_exponential_pattern(numeric_values):
            pattern = RecursivePattern(
                id=str(uuid.uuid4()),
                pattern_type=LoopType.EXPONENTIAL,
                complexity=PatternComplexity.EXTREME,
                nodes=numeric_nodes,
                cycle_length=len(numeric_nodes),
                stability_score=0.85,
                entropy=self._calculate_entropy(numeric_values),
                description="Exponential recursive pattern detected",
                detected_at=int(datetime.now().timestamp() * 1000)
            )
            patterns.append(pattern)

        return patterns

    def _is_cycle(self, values: List[Any], cycle_len: int) -> bool:
        """Check if values contain a cycle of given length."""
        if len(values) < cycle_len * 2:
            return False

        cycle = values[:cycle_len]
        repetitions = len(values) // cycle_len

        for i in range(1, repetitions):
            start_idx = i * cycle_len
            end_idx = start_idx + cycle_len
            if end_idx > len(values):
                break

            segment = values[start_idx:end_idx]
            if segment != cycle:
                return False

        return True

    def _calculate_stability(self, values: List[Any], cycle_len: int) -> float:
        """Calculate the stability score of a cycle."""
        if len(values) < cycle_len * 2:
            return 0.0

        cycle = values[:cycle_len]
        repetitions = len(values) // cycle_len
        matches = 0
        total_comparisons = 0

        for i in range(1, repetitions):
            start_idx = i * cycle_len
            end_idx = start_idx + cycle_len
            if end_idx > len(values):
                break

            segment = values[start_idx:end_idx]
            for j in range(len(segment)):
                total_comparisons += 1
                if j < len(cycle) and segment[j] == cycle[j]:
                    matches += 1

        return matches / total_comparisons if total_comparisons > 0 else 0.0

    def _calculate_entropy(self, values: List[Any]) -> float:
        """Calculate entropy of a sequence."""
        if not values:
            return 0.0

        # Count frequency of each value
        freq = defaultdict(int)
        for value in values:
            freq[str(value)] += 1  # Convert to string for hashing

        # Calculate entropy
        n = len(values)
        entropy = 0.0
        for count in freq.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _determine_complexity(self, cycle_length: int, entropy: float) -> PatternComplexity:
        """Determine pattern complexity based on cycle length and entropy."""
        if cycle_length <= 3 and entropy < 1.0:
            return PatternComplexity.LOW
        elif cycle_length <= 10 and entropy < 2.0:
            return PatternComplexity.MEDIUM
        elif cycle_length <= 20 and entropy < 3.0:
            return PatternComplexity.HIGH
        else:
            return PatternComplexity.EXTREME

    def _get_value_category(self, value: Any) -> str:
        """Get category for value grouping."""
        if isinstance(value, str):
            return f"string_{len(value)}"
        elif isinstance(value, int):
            return f"int_{value % 10}"  # Group by last digit
        elif isinstance(value, float):
            return f"float_{int(value) % 10}"
        elif isinstance(value, dict):
            return f"dict_{len(value)}"
        elif isinstance(value, list):
            return f"list_{len(value)}"
        else:
            return f"other_{type(value).__name__}"

    async def _analyze_depth_relationship(self, nodes1: List[LoopNode],
                                        nodes2: List[LoopNode]) -> Optional[RecursivePattern]:
        """Analyze relationship between nodes at different depths."""
        if len(nodes1) < 2 or len(nodes2) < 2:
            return None

        # Simple heuristic: check if nodes alternate in some pattern
        combined_nodes = sorted(nodes1 + nodes2, key=lambda n: n.timestamp)

        if len(combined_nodes) >= 4:
            pattern = RecursivePattern(
                id=str(uuid.uuid4()),
                pattern_type=LoopType.NESTED,
                complexity=PatternComplexity.HIGH,
                nodes=combined_nodes[:10],  # Limit for analysis
                cycle_length=len(combined_nodes),
                stability_score=0.7,
                entropy=self._calculate_entropy([n.value for n in combined_nodes]),
                description=f"Nested pattern across depths {nodes1[0].depth} and {nodes2[0].depth}",
                detected_at=int(datetime.now().timestamp() * 1000)
            )
            return pattern

        return None

    async def _analyze_mutual_pattern(self, nodes1: List[LoopNode],
                                    nodes2: List[LoopNode]) -> Optional[RecursivePattern]:
        """Analyze mutual recursion pattern between node groups."""
        if len(nodes1) < 2 or len(nodes2) < 2:
            return None

        # Check for alternating pattern
        all_nodes = sorted(nodes1 + nodes2, key=lambda n: n.timestamp)

        # Simple alternation check
        alternates = True
        current_group = self._get_value_category(all_nodes[0].value)
        alternation_count = 0

        for i in range(1, len(all_nodes)):
            node_group = self._get_value_category(all_nodes[i].value)
            if node_group != current_group:
                alternation_count += 1
                current_group = node_group

        if alternation_count >= len(all_nodes) * 0.3:  # At least 30% alternation
            pattern = RecursivePattern(
                id=str(uuid.uuid4()),
                pattern_type=LoopType.MUTUAL,
                complexity=PatternComplexity.MEDIUM,
                nodes=all_nodes[:8],  # Limit for analysis
                cycle_length=2,  # Mutual recursion typically has cycle length 2
                stability_score=alternation_count / len(all_nodes),
                entropy=self._calculate_entropy([n.value for n in all_nodes]),
                description=f"Mutual recursion pattern with {alternation_count} alternations",
                detected_at=int(datetime.now().timestamp() * 1000)
            )
            return pattern

        return None

    def _is_fibonacci_like(self, values: List[float]) -> bool:
        """Check if sequence follows Fibonacci-like pattern."""
        if len(values) < 3:
            return False

        tolerance = 0.1  # Allow 10% tolerance
        fibonacci_matches = 0

        for i in range(2, len(values)):
            expected = values[i-1] + values[i-2]
            actual = values[i]

            if expected == 0:
                continue

            error = abs(actual - expected) / abs(expected)
            if error <= tolerance:
                fibonacci_matches += 1

        # Require at least 70% matches
        return fibonacci_matches >= (len(values) - 2) * 0.7

    def _is_exponential_pattern(self, values: List[float]) -> bool:
        """Check if sequence follows exponential pattern."""
        if len(values) < 3:
            return False

        # Check for consistent ratio between consecutive terms
        ratios = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ratio = values[i] / values[i-1]
                ratios.append(ratio)

        if len(ratios) < 2:
            return False

        # Check if ratios are approximately constant
        avg_ratio = sum(ratios) / len(ratios)
        tolerance = 0.2  # 20% tolerance

        consistent_ratios = 0
        for ratio in ratios:
            if abs(ratio - avg_ratio) / avg_ratio <= tolerance:
                consistent_ratios += 1

        return consistent_ratios >= len(ratios) * 0.8  # 80% consistency


class PatternSynthesizer:
    """Synthesizes new patterns from existing ones."""

    def __init__(self):
        self.max_iterations = 1000
        self.convergence_threshold = 0.01

    async def synthesize_patterns(self, patterns: List[RecursivePattern]) -> List[SynthesisResult]:
        """Synthesize new patterns from existing ones."""
        results = []

        if len(patterns) < 2:
            return results

        # Pairwise synthesis
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                result = await self._synthesize_pair(patterns[i], patterns[j])
                if result:
                    results.append(result)

        # Multi-pattern synthesis (if we have many patterns)
        if len(patterns) >= 3:
            multi_result = await self._synthesize_multiple(patterns[:5])  # Limit to 5
            if multi_result:
                results.append(multi_result)

        logger.info(f"Synthesized {len(results)} new patterns from {len(patterns)} input patterns")
        return results

    async def _synthesize_pair(self, pattern1: RecursivePattern,
                             pattern2: RecursivePattern) -> Optional[SynthesisResult]:
        """Synthesize a new pattern from two existing patterns."""
        start_time = datetime.now()

        # Combine nodes from both patterns
        combined_nodes = pattern1.nodes + pattern2.nodes
        combined_nodes.sort(key=lambda n: n.timestamp)

        # Create synthesized pattern based on combination rules
        synthesized_type = self._determine_synthesized_type(pattern1.pattern_type, pattern2.pattern_type)
        synthesized_complexity = self._combine_complexity(pattern1.complexity, pattern2.complexity)

        # Calculate new cycle length
        new_cycle_length = math.gcd(pattern1.cycle_length, pattern2.cycle_length)
        if new_cycle_length == 1:
            new_cycle_length = pattern1.cycle_length + pattern2.cycle_length

        # Average stability and combine entropy
        new_stability = (pattern1.stability_score + pattern2.stability_score) / 2
        new_entropy = max(pattern1.entropy, pattern2.entropy) + 0.1  # Slightly increase entropy

        synthesized_pattern = RecursivePattern(
            id=str(uuid.uuid4()),
            pattern_type=synthesized_type,
            complexity=synthesized_complexity,
            nodes=combined_nodes[:20],  # Limit size
            cycle_length=new_cycle_length,
            stability_score=new_stability,
            entropy=new_entropy,
            description=f"Synthesized from {pattern1.pattern_type.value} and {pattern2.pattern_type.value}",
            detected_at=int(datetime.now().timestamp() * 1000)
        )

        end_time = datetime.now()
        convergence_time = (end_time - start_time).total_seconds()

        return SynthesisResult(
            id=str(uuid.uuid4()),
            input_patterns=[pattern1.id, pattern2.id],
            synthesized_pattern=synthesized_pattern,
            confidence=min(new_stability, 0.9),
            iteration_count=1,  # Simple synthesis
            convergence_time=convergence_time,
            metadata={
                'synthesis_method': 'pairwise',
                'input_types': [pattern1.pattern_type.value, pattern2.pattern_type.value]
            }
        )

    async def _synthesize_multiple(self, patterns: List[RecursivePattern]) -> Optional[SynthesisResult]:
        """Synthesize from multiple patterns using iterative approach."""
        start_time = datetime.now()

        # Start with first pattern as base
        current_pattern = patterns[0]
        input_pattern_ids = [current_pattern.id]

        # Iteratively combine with other patterns
        for i in range(1, len(patterns)):
            synthesis_result = await self._synthesize_pair(current_pattern, patterns[i])
            if synthesis_result:
                current_pattern = synthesis_result.synthesized_pattern
                input_pattern_ids.append(patterns[i].id)

        end_time = datetime.now()
        convergence_time = (end_time - start_time).total_seconds()

        return SynthesisResult(
            id=str(uuid.uuid4()),
            input_patterns=input_pattern_ids,
            synthesized_pattern=current_pattern,
            confidence=0.8,  # Lower confidence for complex synthesis
            iteration_count=len(patterns) - 1,
            convergence_time=convergence_time,
            metadata={
                'synthesis_method': 'iterative_multiple',
                'pattern_count': len(patterns)
            }
        )

    def _determine_synthesized_type(self, type1: LoopType, type2: LoopType) -> LoopType:
        """Determine the type of synthesized pattern."""
        if type1 == type2:
            return type1

        # Synthesis rules
        type_hierarchy = {
            LoopType.SIMPLE: 1,
            LoopType.MUTUAL: 2,
            LoopType.NESTED: 3,
            LoopType.FIBONACCI: 4,
            LoopType.EXPONENTIAL: 5,
            LoopType.OSCILLATING: 6,
            LoopType.CHAOTIC: 7
        }

        # Return the more complex type
        if type_hierarchy[type1] > type_hierarchy[type2]:
            return type1
        else:
            return type2

    def _combine_complexity(self, comp1: PatternComplexity, comp2: PatternComplexity) -> PatternComplexity:
        """Combine complexity levels."""
        complexity_values = {
            PatternComplexity.LOW: 1,
            PatternComplexity.MEDIUM: 2,
            PatternComplexity.HIGH: 3,
            PatternComplexity.EXTREME: 4
        }

        combined_value = max(complexity_values[comp1], complexity_values[comp2])

        # Synthesis increases complexity
        combined_value = min(combined_value + 1, 4)

        for complexity, value in complexity_values.items():
            if value == combined_value:
                return complexity

        return PatternComplexity.EXTREME


class RecursiveLoopSynthService:
    """Main Recursive Loop Synthesis Service."""

    def __init__(self):
        self.analyzer = RecursiveAnalyzer()
        self.synthesizer = PatternSynthesizer()
        self.patterns: Dict[str, RecursivePattern] = {}
        self.synthesis_results: Dict[str, SynthesisResult] = {}
        self.running = False

    async def start(self):
        """Start the service."""
        self.running = True
        logger.info("Recursive Loop Synthesis Service started - Research-grade stability enabled")

        # Start background synthesis
        asyncio.create_task(self._background_synthesis())

    async def stop(self):
        """Stop the service."""
        self.running = False
        logger.info("Recursive Loop Synthesis Service stopped")

    async def submit_sequence(self, sequence_data: Dict[str, Any]) -> str:
        """Submit a sequence for pattern analysis."""
        # Parse nodes from sequence data
        nodes = []
        for node_data in sequence_data.get('nodes', []):
            node = LoopNode(
                id=node_data.get('id', str(uuid.uuid4())),
                value=node_data['value'],
                timestamp=node_data.get('timestamp', int(datetime.now().timestamp() * 1000)),
                depth=node_data.get('depth', 0),
                metadata=node_data.get('metadata', {})
            )
            nodes.append(node)

        # Analyze for patterns
        patterns = await self.analyzer.detect_patterns(nodes)

        # Store detected patterns
        sequence_id = str(uuid.uuid4())
        for pattern in patterns:
            self.patterns[pattern.id] = pattern

        logger.info(f"Analyzed sequence {sequence_id}: {len(patterns)} patterns detected")
        return sequence_id

    async def get_patterns(self, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detected patterns, optionally filtered by type."""
        patterns = list(self.patterns.values())

        if pattern_type:
            try:
                filter_type = LoopType(pattern_type)
                patterns = [p for p in patterns if p.pattern_type == filter_type]
            except ValueError:
                pass  # Invalid type, return all patterns

        return [asdict(pattern) for pattern in patterns]

    async def synthesize_patterns(self, pattern_ids: List[str]) -> str:
        """Synthesize new patterns from existing ones."""
        # Get patterns by IDs
        input_patterns = []
        for pattern_id in pattern_ids:
            if pattern_id in self.patterns:
                input_patterns.append(self.patterns[pattern_id])

        if len(input_patterns) < 2:
            raise Exception("Need at least 2 patterns for synthesis")

        # Perform synthesis
        results = await self.synthesizer.synthesize_patterns(input_patterns)

        # Store results
        synthesis_id = str(uuid.uuid4())
        for result in results:
            self.synthesis_results[result.id] = result
            # Also store the synthesized pattern
            self.patterns[result.synthesized_pattern.id] = result.synthesized_pattern

        logger.info(f"Synthesis {synthesis_id}: {len(results)} new patterns created")
        return synthesis_id

    async def get_synthesis_results(self) -> List[Dict[str, Any]]:
        """Get all synthesis results."""
        return [asdict(result) for result in self.synthesis_results.values()]

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        pattern_type_counts = defaultdict(int)
        complexity_counts = defaultdict(int)

        for pattern in self.patterns.values():
            pattern_type_counts[pattern.pattern_type.value] += 1
            complexity_counts[pattern.complexity.value] += 1

        return {
            'service_status': 'running' if self.running else 'stopped',
            'total_patterns': len(self.patterns),
            'total_synthesis_results': len(self.synthesis_results),
            'pattern_type_distribution': dict(pattern_type_counts),
            'complexity_distribution': dict(complexity_counts),
            'timestamp': int(datetime.now().timestamp() * 1000)
        }

    async def _background_synthesis(self):
        """Background synthesis of compatible patterns."""
        while self.running:
            try:
                if len(self.patterns) >= 4:  # Need enough patterns
                    # Find compatible patterns for synthesis
                    pattern_list = list(self.patterns.values())

                    # Select patterns that haven't been used in recent synthesis
                    available_patterns = pattern_list[-10:]  # Use most recent 10

                    if len(available_patterns) >= 2:
                        synthesis_results = await self.synthesizer.synthesize_patterns(available_patterns[:4])

                        # Store new results
                        for result in synthesis_results:
                            if result.id not in self.synthesis_results:
                                self.synthesis_results[result.id] = result
                                self.patterns[result.synthesized_pattern.id] = result.synthesized_pattern
                                logger.info(f"Background synthesis created pattern: {result.synthesized_pattern.id}")

                # Sleep before next synthesis cycle
                await asyncio.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Background synthesis error: {e}")
                await asyncio.sleep(120)  # Wait longer on error


async def main():
    """Main entry point for the Recursive Loop Synthesis Service."""
    service = RecursiveLoopSynthService()

    await service.start()

    try:
        # Demo: Submit a test sequence
        test_sequence = {
            'nodes': [
                {'value': 1, 'depth': 0},
                {'value': 2, 'depth': 0},
                {'value': 1, 'depth': 0},
                {'value': 2, 'depth': 0},
                {'value': 1, 'depth': 0},
                {'value': 3, 'depth': 1},
                {'value': 5, 'depth': 1},
                {'value': 8, 'depth': 1},  # Fibonacci-like
                {'value': 13, 'depth': 1}
            ]
        }

        logger.info("Submitting test sequence...")
        sequence_id = await service.submit_sequence(test_sequence)
        logger.info(f"Sequence submitted: {sequence_id}")

        # Get detected patterns
        patterns = await service.get_patterns()
        logger.info(f"Detected patterns: {len(patterns)}")

        for pattern in patterns:
            logger.info(f"Pattern: {pattern['pattern_type']} - {pattern['description']}")

        # Get service stats
        stats = await service.get_service_stats()
        logger.info(f"Service stats: {stats}")

        # Keep service running
        logger.info("Recursive Loop Synthesis Service is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
