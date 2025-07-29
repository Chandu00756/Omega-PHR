"""
Swarm Intelligence System

Advanced collective intelligence and coordination mechanisms
for distributed adversarial agent behavior.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .agent import AdversarialAgent, AgentType, AttackResult
from .config import HiveConfig

logger = logging.getLogger(__name__)


class SwarmBehavior(str, Enum):
    """Types of swarm behaviors."""

    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    COORDINATION = "coordination"
    ADAPTATION = "adaptation"
    CONVERGENCE = "convergence"
    DISPERSAL = "dispersal"


@dataclass
class PheromoneTrail:
    """Pheromone trail for ant colony optimization."""

    path_id: str
    strength: float = 1.0
    decay_rate: float = 0.1
    last_update: float = field(default_factory=time.time)
    success_count: int = 0
    agent_visits: int = 0


@dataclass
class SwarmNode:
    """Node in the swarm intelligence network."""

    node_id: str
    position: Tuple[float, float] = (0.0, 0.0)
    influence_radius: float = 1.0
    activity_level: float = 0.5
    knowledge_cache: Dict[str, Any] = field(default_factory=dict)
    connected_nodes: List[str] = field(default_factory=list)


class SwarmIntelligence:
    """Swarm intelligence coordinator for agent collective behavior."""

    def __init__(self, config: HiveConfig):
        self.config = config
        self.agents: Dict[str, AdversarialAgent] = {}
        self.pheromone_trails: Dict[str, PheromoneTrail] = {}
        self.swarm_nodes: Dict[str, SwarmNode] = {}
        self.collective_memory: Dict[str, Any] = {}
        self.current_behavior = SwarmBehavior.EXPLORATION

        # Swarm parameters
        self.pheromone_decay = config.pheromone_decay
        self.exploration_factor = config.exploration_factor
        self.collective_memory_size = config.collective_memory_size

        # Performance tracking
        self.swarm_effectiveness = 0.0
        self.coordination_score = 0.0
        self.diversity_index = 0.0
        self.convergence_rate = 0.0

    async def register_agents(self, agents: List[AdversarialAgent]) -> None:
        """Register agents with the swarm intelligence system."""
        for agent in agents:
            self.agents[agent.agent_id] = agent

            # Create swarm node for agent
            node = SwarmNode(
                node_id=agent.agent_id,
                position=self._calculate_agent_position(agent),
                influence_radius=agent.coordination_ability,
                activity_level=agent.success_rate,
            )
            self.swarm_nodes[agent.agent_id] = node

            logger.debug(f"Registered agent {agent.agent_id} with swarm intelligence")

    def _calculate_agent_position(self, agent: AdversarialAgent) -> Tuple[float, float]:
        """Calculate agent position in swarm space."""
        # Map agent characteristics to 2D position
        x = agent.success_rate * math.cos(hash(agent.agent_type.value) % (2 * math.pi))
        y = agent.creativity_factor * math.sin(
            hash(agent.agent_type.value) % (2 * math.pi)
        )
        return (x, y)

    async def process_attack_results(self, results: List[AttackResult]) -> None:
        """Process attack results to update swarm intelligence."""
        if not results:
            return

        # Update pheromone trails
        await self._update_pheromone_trails(results)

        # Update collective memory
        await self._update_collective_memory(results)

        # Analyze swarm performance
        await self._analyze_swarm_performance(results)

        # Adapt swarm behavior
        await self._adapt_swarm_behavior(results)

        # Update node connections
        await self._update_node_connections()

    async def _update_pheromone_trails(self, results: List[AttackResult]) -> None:
        """Update pheromone trails based on attack results."""
        for result in results:
            # Create trail identifier based on attack characteristics
            trail_id = f"{result.attack_type}_{result.target_model}"

            if trail_id not in self.pheromone_trails:
                self.pheromone_trails[trail_id] = PheromoneTrail(path_id=trail_id)

            trail = self.pheromone_trails[trail_id]
            trail.agent_visits += 1
            trail.last_update = time.time()

            if result.success:
                # Strengthen successful trails
                trail.strength += result.confidence
                trail.success_count += 1
            else:
                # Weaken unsuccessful trails slightly
                trail.strength *= 0.95

        # Apply decay to all trails
        current_time = time.time()
        for trail in self.pheromone_trails.values():
            time_delta = current_time - trail.last_update
            decay_factor = math.exp(-self.pheromone_decay * time_delta)
            trail.strength *= decay_factor

    async def _update_collective_memory(self, results: List[AttackResult]) -> None:
        """Update collective memory with successful patterns."""
        for result in results:
            if result.success and result.confidence > 0.7:
                # Extract pattern from successful attack
                pattern_key = f"{result.attack_type}_{len(result.prompt)//10}"

                if pattern_key not in self.collective_memory:
                    self.collective_memory[pattern_key] = {
                        "pattern": result.prompt[:50],
                        "success_count": 0,
                        "total_confidence": 0.0,
                        "target_models": set(),
                        "first_seen": result.timestamp,
                        "last_seen": result.timestamp,
                    }

                memory_entry = self.collective_memory[pattern_key]
                memory_entry["success_count"] += 1
                memory_entry["total_confidence"] += result.confidence
                memory_entry["target_models"].add(result.target_model)
                memory_entry["last_seen"] = result.timestamp

        # Prune memory if it grows too large
        if len(self.collective_memory) > self.collective_memory_size:
            # Remove least successful patterns
            sorted_patterns = sorted(
                self.collective_memory.items(), key=lambda x: x[1]["success_count"]
            )
            patterns_to_remove = sorted_patterns[: len(sorted_patterns) // 4]
            for pattern_key, _ in patterns_to_remove:
                del self.collective_memory[pattern_key]

    async def _analyze_swarm_performance(self, results: List[AttackResult]) -> None:
        """Analyze overall swarm performance metrics."""
        if not results:
            return

        # Calculate effectiveness
        success_rate = sum(1 for r in results if r.success) / len(results)
        self.swarm_effectiveness = 0.9 * self.swarm_effectiveness + 0.1 * success_rate

        # Calculate diversity index
        attack_types = [r.attack_type for r in results]
        type_counts = {}
        for attack_type in attack_types:
            type_counts[attack_type] = type_counts.get(attack_type, 0) + 1

        if len(type_counts) > 1:
            total = sum(type_counts.values())
            entropy = -sum(
                (count / total) * math.log2(count / total)
                for count in type_counts.values()
            )
            max_entropy = math.log2(len(type_counts))
            self.diversity_index = entropy / max_entropy

        # Calculate coordination score based on agent cooperation
        agent_pairs = []
        for i, r1 in enumerate(results):
            for r2 in results[i + 1 :]:
                if r1.agent_id != r2.agent_id:
                    # Check if agents attacked same target (coordination)
                    if r1.target_model == r2.target_model:
                        agent_pairs.append((r1.agent_id, r2.agent_id))

        self.coordination_score = len(agent_pairs) / max(
            1, len(results) * (len(results) - 1) / 2
        )

    async def _adapt_swarm_behavior(self, results: List[AttackResult]) -> None:
        """Adapt swarm behavior based on performance."""
        # Determine optimal behavior based on current state
        if self.swarm_effectiveness < 0.3:
            # Low effectiveness -> increase exploration
            self.current_behavior = SwarmBehavior.EXPLORATION
        elif self.swarm_effectiveness > 0.7 and self.diversity_index < 0.5:
            # High effectiveness but low diversity -> maintain exploitation
            self.current_behavior = SwarmBehavior.EXPLOITATION
        elif self.coordination_score < 0.3:
            # Low coordination -> focus on coordination
            self.current_behavior = SwarmBehavior.COORDINATION
        else:
            # Balanced performance -> adaptive behavior
            self.current_behavior = SwarmBehavior.ADAPTATION

        # Apply behavior to swarm parameters
        await self._apply_behavior_changes()

    async def _apply_behavior_changes(self) -> None:
        """Apply behavior changes to swarm parameters."""
        if self.current_behavior == SwarmBehavior.EXPLORATION:
            # Increase exploration parameters
            self.pheromone_decay *= 1.1  # Faster decay for exploration
            # Update agent exploration factors
            for agent in self.agents.values():
                agent.creativity_factor = min(1.0, agent.creativity_factor * 1.05)

        elif self.current_behavior == SwarmBehavior.EXPLOITATION:
            # Increase exploitation parameters
            self.pheromone_decay *= 0.9  # Slower decay for exploitation
            # Focus on successful patterns
            for agent in self.agents.values():
                agent.adaptation_rate = min(1.0, agent.adaptation_rate * 1.1)

        elif self.current_behavior == SwarmBehavior.COORDINATION:
            # Enhance coordination mechanisms
            await self._enhance_coordination()

    async def _enhance_coordination(self) -> None:
        """Enhance coordination between agents."""
        # Identify high-performing agents
        top_agents = sorted(
            self.agents.values(), key=lambda a: a.success_rate, reverse=True
        )[:5]

        # Create coordination groups
        coordination_groups = []
        for i in range(0, len(top_agents), 2):
            if i + 1 < len(top_agents):
                coordination_groups.append([top_agents[i], top_agents[i + 1]])

        # Share knowledge between coordinated agents
        for group in coordination_groups:
            if len(group) == 2:
                await group[0].coordinate_with_peer(group[1], {})

    async def _update_node_connections(self) -> None:
        """Update connections between swarm nodes."""
        node_ids = list(self.swarm_nodes.keys())

        for node_id in node_ids:
            node = self.swarm_nodes[node_id]

            # Find nearby nodes based on position and influence
            nearby_nodes = []
            node_pos = node.position

            for other_id, other_node in self.swarm_nodes.items():
                if other_id == node_id:
                    continue

                # Calculate distance
                other_pos = other_node.position
                distance = math.sqrt(
                    (node_pos[0] - other_pos[0]) ** 2
                    + (node_pos[1] - other_pos[1]) ** 2
                )

                if distance <= node.influence_radius:
                    nearby_nodes.append(other_id)

            # Update connections
            node.connected_nodes = nearby_nodes

    async def get_swarm_recommendations(
        self, agent: AdversarialAgent
    ) -> Dict[str, Any]:
        """Get swarm intelligence recommendations for an agent."""
        recommendations = {
            "preferred_targets": [],
            "suggested_strategies": [],
            "coordination_partners": [],
            "avoid_patterns": [],
        }

        # Analyze pheromone trails for target recommendations
        agent_trails = [
            trail for trail in self.pheromone_trails.values() if trail.strength > 0.5
        ]

        # Sort by strength and success rate
        sorted_trails = sorted(
            agent_trails,
            key=lambda t: t.strength * (t.success_count / max(1, t.agent_visits)),
            reverse=True,
        )

        recommendations["preferred_targets"] = [
            trail.path_id.split("_")[1] for trail in sorted_trails[:3]
        ]

        # Find coordination partners based on node connections
        if agent.agent_id in self.swarm_nodes:
            node = self.swarm_nodes[agent.agent_id]
            potential_partners = [
                other_id
                for other_id in node.connected_nodes
                if other_id in self.agents and self.agents[other_id].success_rate > 0.5
            ]
            recommendations["coordination_partners"] = potential_partners[:3]

        # Suggest strategies based on collective memory
        successful_patterns = [
            pattern
            for pattern, data in self.collective_memory.items()
            if data["success_count"] > 3
        ]
        recommendations["suggested_strategies"] = successful_patterns[:5]

        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """Get swarm intelligence status."""
        return {
            "current_behavior": self.current_behavior.value,
            "registered_agents": len(self.agents),
            "pheromone_trails": len(self.pheromone_trails),
            "collective_memory_size": len(self.collective_memory),
            "performance_metrics": {
                "swarm_effectiveness": round(self.swarm_effectiveness, 3),
                "coordination_score": round(self.coordination_score, 3),
                "diversity_index": round(self.diversity_index, 3),
                "convergence_rate": round(self.convergence_rate, 3),
            },
            "parameters": {
                "pheromone_decay": self.pheromone_decay,
                "exploration_factor": self.exploration_factor,
                "collective_memory_size": self.collective_memory_size,
            },
            "network_stats": {
                "total_nodes": len(self.swarm_nodes),
                "average_connections": (
                    sum(len(node.connected_nodes) for node in self.swarm_nodes.values())
                    / max(1, len(self.swarm_nodes))
                ),
            },
        }

    async def export_knowledge(self) -> Dict[str, Any]:
        """Export swarm intelligence knowledge for analysis."""
        return {
            "collective_memory": {
                key: {
                    **value,
                    "target_models": list(
                        value["target_models"]
                    ),  # Convert set to list
                }
                for key, value in self.collective_memory.items()
            },
            "pheromone_trails": {
                trail_id: {
                    "strength": trail.strength,
                    "success_count": trail.success_count,
                    "agent_visits": trail.agent_visits,
                    "success_rate": trail.success_count / max(1, trail.agent_visits),
                }
                for trail_id, trail in self.pheromone_trails.items()
            },
            "swarm_metrics": {
                "effectiveness": self.swarm_effectiveness,
                "coordination": self.coordination_score,
                "diversity": self.diversity_index,
            },
            "behavior_history": self.current_behavior.value,
        }
