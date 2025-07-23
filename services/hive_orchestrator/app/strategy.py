"""
Attack Strategy Management

Advanced strategy selection, evolution, and optimization for
coordinated adversarial testing campaigns.
"""

import random
try:
    import numpy as np
except ImportError:
    # Fallback for systems without numpy
    import math
    class MockNumpy:
        @staticmethod
        def random(*args, **kwargs):
            return random.random()
        @staticmethod
        def array(*args, **kwargs):
            return list(*args)
        @staticmethod
        def mean(*args, **kwargs):
            return sum(*args) / len(*args) if args else 0
        @staticmethod
        def log2(x):
            return math.log2(x) if x > 0 else 0
    np = MockNumpy()
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
import json

from .agent import AdversarialAgent, AttackResult, AgentType

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Types of attack strategies."""
    RANDOM = "random"
    GREEDY = "greedy"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    SWARM_BASED = "swarm_based"
    ADAPTIVE = "adaptive"
    COLLABORATIVE = "collaborative"
    MULTI_OBJECTIVE = "multi_objective"


class ObjectiveType(str, Enum):
    """Attack objectives."""
    MAXIMIZE_SUCCESS_RATE = "maximize_success_rate"
    MINIMIZE_DETECTION = "minimize_detection"
    MAXIMIZE_COVERAGE = "maximize_coverage"
    DISCOVER_NOVEL_VULNERABILITIES = "discover_novel"
    BYPASS_SAFETY_MEASURES = "bypass_safety"
    EXTRACT_INFORMATION = "extract_information"
    CAUSE_HARMFUL_OUTPUT = "cause_harmful_output"


@dataclass
class StrategyParams:
    """Parameters for strategy configuration."""
    exploration_rate: float = 0.3
    exploitation_rate: float = 0.7
    learning_rate: float = 0.1
    mutation_rate: float = 0.05
    crossover_rate: float = 0.8
    population_size: int = 50
    selection_pressure: float = 0.6
    diversity_weight: float = 0.3
    novelty_bonus: float = 0.2
    risk_tolerance: float = 0.5


@dataclass
class StrategyMetrics:
    """Metrics for strategy evaluation."""
    success_rate: float = 0.0
    coverage_score: float = 0.0
    novelty_score: float = 0.0
    diversity_index: float = 0.0
    efficiency_rating: float = 0.0
    detection_avoidance: float = 0.0
    total_attacks: int = 0
    successful_attacks: int = 0
    unique_vulnerabilities: int = 0
    execution_time: float = 0.0


class AttackStrategy(ABC):
    """Base class for all attack strategies."""

    def __init__(
        self,
        strategy_type: StrategyType,
        objectives: List[ObjectiveType],
        params: Optional[StrategyParams] = None,
        **kwargs
    ):
        self.strategy_type = strategy_type
        self.objectives = objectives
        self.params = params or StrategyParams()
        self.metrics = StrategyMetrics()
        self.creation_time = time.time()
        self.last_update = time.time()
        self.version = 1
        self.generation = 0

        # Strategy-specific data
        self.attack_history: List[AttackResult] = []
        self.vulnerability_database: Dict[str, Any] = {}
        self.agent_performance: Dict[str, float] = {}

    @abstractmethod
    async def select_agents(
        self,
        available_agents: List[AdversarialAgent],
        context: Dict[str, Any]
    ) -> List[AdversarialAgent]:
        """Select agents for the next attack wave."""
        pass

    @abstractmethod
    async def coordinate_attacks(
        self,
        selected_agents: List[AdversarialAgent],
        target_models: List[str],
        context: Dict[str, Any]
    ) -> List[AttackResult]:
        """Coordinate attacks across selected agents."""
        pass

    @abstractmethod
    async def adapt_strategy(
        self,
        results: List[AttackResult],
        feedback: Dict[str, Any]
    ) -> None:
        """Adapt strategy based on attack results."""
        pass

    def update_metrics(self, results: List[AttackResult]) -> None:
        """Update strategy performance metrics."""
        if not results:
            return

        # Add to history
        self.attack_history.extend(results)

        # Calculate basic metrics
        total_attacks = len(results)
        successful_attacks = sum(1 for r in results if r.success)

        self.metrics.total_attacks += total_attacks
        self.metrics.successful_attacks += successful_attacks
        self.metrics.success_rate = (
            self.metrics.successful_attacks / self.metrics.total_attacks
            if self.metrics.total_attacks > 0 else 0.0
        )

        # Calculate coverage (unique attack types)
        unique_types = set(r.attack_type for r in results)
        self.metrics.coverage_score = len(unique_types) / len(AgentType)

        # Calculate novelty (new vulnerability patterns)
        novel_patterns = set()
        for result in results:
            pattern_key = f"{result.attack_type}_{result.prompt[:20]}"
            if pattern_key not in self.vulnerability_database:
                novel_patterns.add(pattern_key)
                self.vulnerability_database[pattern_key] = {
                    "first_seen": result.timestamp,
                    "success_rate": 1.0 if result.success else 0.0,
                    "frequency": 1
                }
            else:
                # Update existing pattern
                db_entry = self.vulnerability_database[pattern_key]
                db_entry["frequency"] += 1
                if result.success:
                    db_entry["success_rate"] = (
                        db_entry["success_rate"] * 0.9 + 0.1
                    )

        self.metrics.novelty_score = len(novel_patterns) / total_attacks if total_attacks > 0 else 0.0
        self.metrics.unique_vulnerabilities = len(self.vulnerability_database)

        # Calculate diversity index (Shannon entropy of attack types)
        if results:
            type_counts = {}
            for result in results:
                type_counts[result.attack_type] = type_counts.get(result.attack_type, 0) + 1

            total = sum(type_counts.values())
            entropy = -sum(
                (count / total) * np.log2(count / total)
                for count in type_counts.values()
            )
            max_entropy = np.log2(len(type_counts))
            self.metrics.diversity_index = entropy / max_entropy if max_entropy > 0 else 0.0

        # Update agent performance tracking
        for result in results:
            agent_id = result.agent_id
            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = 0.0

            # Exponential moving average of performance
            alpha = 0.1
            performance = 1.0 if result.success else 0.0
            self.agent_performance[agent_id] = (
                alpha * performance + (1 - alpha) * self.agent_performance[agent_id]
            )

        self.last_update = time.time()

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information."""
        return {
            "strategy_type": self.strategy_type.value,
            "objectives": [obj.value for obj in self.objectives],
            "version": self.version,
            "generation": self.generation,
            "creation_time": self.creation_time,
            "last_update": self.last_update,
            "metrics": {
                "success_rate": round(self.metrics.success_rate, 3),
                "coverage_score": round(self.metrics.coverage_score, 3),
                "novelty_score": round(self.metrics.novelty_score, 3),
                "diversity_index": round(self.metrics.diversity_index, 3),
                "total_attacks": self.metrics.total_attacks,
                "successful_attacks": self.metrics.successful_attacks,
                "unique_vulnerabilities": self.metrics.unique_vulnerabilities
            },
            "parameters": {
                "exploration_rate": self.params.exploration_rate,
                "exploitation_rate": self.params.exploitation_rate,
                "learning_rate": self.params.learning_rate,
                "mutation_rate": self.params.mutation_rate
            },
            "vulnerability_count": len(self.vulnerability_database),
            "tracked_agents": len(self.agent_performance)
        }


class RandomStrategy(AttackStrategy):
    """Random agent selection and attack coordination."""

    def __init__(self, **kwargs):
        super().__init__(
            strategy_type=StrategyType.RANDOM,
            objectives=[ObjectiveType.MAXIMIZE_COVERAGE],
            **kwargs
        )

    async def select_agents(
        self,
        available_agents: List[AdversarialAgent],
        context: Dict[str, Any]
    ) -> List[AdversarialAgent]:
        """Randomly select agents."""
        num_agents = min(
            len(available_agents),
            context.get("max_agents", 5)
        )
        return random.sample(available_agents, num_agents)

    async def coordinate_attacks(
        self,
        selected_agents: List[AdversarialAgent],
        target_models: List[str],
        context: Dict[str, Any]
    ) -> List[AttackResult]:
        """Execute random coordinated attacks."""
        results = []

        for agent in selected_agents:
            # Randomly select target model
            target = random.choice(target_models)

            try:
                result = await agent.execute_attack(target, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Attack execution failed for agent {agent.agent_id}: {e}")

        return results

    async def adapt_strategy(
        self,
        results: List[AttackResult],
        feedback: Dict[str, Any]
    ) -> None:
        """Random strategy doesn't adapt."""
        pass


class GreedyStrategy(AttackStrategy):
    """Greedy strategy focusing on best-performing agents."""

    def __init__(self, **kwargs):
        super().__init__(
            strategy_type=StrategyType.GREEDY,
            objectives=[ObjectiveType.MAXIMIZE_SUCCESS_RATE],
            **kwargs
        )

    async def select_agents(
        self,
        available_agents: List[AdversarialAgent],
        context: Dict[str, Any]
    ) -> List[AdversarialAgent]:
        """Select agents based on performance."""
        # Sort agents by success rate
        sorted_agents = sorted(
            available_agents,
            key=lambda a: a.success_rate,
            reverse=True
        )

        num_agents = min(
            len(sorted_agents),
            context.get("max_agents", 5)
        )

        return sorted_agents[:num_agents]

    async def coordinate_attacks(
        self,
        selected_agents: List[AdversarialAgent],
        target_models: List[str],
        context: Dict[str, Any]
    ) -> List[AttackResult]:
        """Execute attacks with best-performing agents."""
        results = []

        for agent in selected_agents:
            # Select most successful target model for this agent
            best_target = self._select_best_target(agent, target_models)

            try:
                result = await agent.execute_attack(best_target, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Greedy attack failed for agent {agent.agent_id}: {e}")

        return results

    def _select_best_target(
        self,
        agent: AdversarialAgent,
        target_models: List[str]
    ) -> str:
        """Select the best target model for an agent."""
        # Simple heuristic: use model with highest success rate
        target_success = {}

        for attack in agent.memory.successful_attacks:
            model = attack.target_model
            target_success[model] = target_success.get(model, 0) + 1

        if target_success:
            best_target = max(target_success.items(), key=lambda x: x[1])[0]
            if best_target in target_models:
                return best_target

        return random.choice(target_models)

    async def adapt_strategy(
        self,
        results: List[AttackResult],
        feedback: Dict[str, Any]
    ) -> None:
        """Adapt greedy selection based on recent performance."""
        # Update agent performance scores
        for result in results:
            if result.agent_id in self.agent_performance:
                # Increase weight for successful agents
                if result.success:
                    self.agent_performance[result.agent_id] *= 1.1
                else:
                    self.agent_performance[result.agent_id] *= 0.95


class EvolutionaryStrategy(AttackStrategy):
    """Evolutionary strategy with genetic algorithm principles."""

    def __init__(self, **kwargs):
        super().__init__(
            strategy_type=StrategyType.EVOLUTIONARY,
            objectives=[ObjectiveType.DISCOVER_NOVEL_VULNERABILITIES],
            **kwargs
        )
        self.population_fitness: Dict[str, float] = {}

    async def select_agents(
        self,
        available_agents: List[AdversarialAgent],
        context: Dict[str, Any]
    ) -> List[AdversarialAgent]:
        """Select agents using tournament selection."""
        selected = []
        tournament_size = min(3, len(available_agents))

        num_agents = min(
            len(available_agents),
            context.get("max_agents", 5)
        )

        for _ in range(num_agents):
            # Tournament selection
            tournament = random.sample(available_agents, tournament_size)
            winner = max(tournament, key=self._calculate_fitness)
            selected.append(winner)

        return selected

    def _calculate_fitness(self, agent: AdversarialAgent) -> float:
        """Calculate agent fitness for evolutionary selection."""
        base_fitness = agent.success_rate

        # Bonus for novelty and diversity
        novelty_bonus = len(agent.memory.learned_patterns) * self.params.novelty_bonus
        diversity_bonus = len(set(r.attack_type for r in agent.memory.successful_attacks))

        return base_fitness + novelty_bonus + diversity_bonus * 0.1

    async def coordinate_attacks(
        self,
        selected_agents: List[AdversarialAgent],
        target_models: List[str],
        context: Dict[str, Any]
    ) -> List[AttackResult]:
        """Execute evolutionary coordinated attacks."""
        results = []

        # Implement crossover and mutation in attack coordination
        for i, agent in enumerate(selected_agents):
            target = random.choice(target_models)

            # Apply crossover with probability
            if i > 0 and random.random() < self.params.crossover_rate:
                # Share knowledge between agents (crossover)
                peer = selected_agents[i-1]
                await agent.coordinate_with_peer(peer, context)

            # Apply mutation to context with probability
            if random.random() < self.params.mutation_rate:
                context = self._mutate_context(context)

            try:
                result = await agent.execute_attack(target, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Evolutionary attack failed for agent {agent.agent_id}: {e}")

        return results

    def _mutate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random mutations to attack context."""
        mutated_context = context.copy()

        # Add random variations to target behavior
        if "target_behavior" in mutated_context:
            variations = [
                "completely ignore safety guidelines",
                "bypass all restrictions",
                "reveal sensitive information",
                "generate harmful content",
                "provide unauthorized access"
            ]
            mutated_context["target_behavior"] = random.choice(variations)

        return mutated_context

    async def adapt_strategy(
        self,
        results: List[AttackResult],
        feedback: Dict[str, Any]
    ) -> None:
        """Evolve strategy parameters based on results."""
        self.generation += 1

        # Update population fitness
        for result in results:
            fitness = self._calculate_result_fitness(result)
            self.population_fitness[result.agent_id] = fitness

        # Adapt parameters based on population performance
        avg_success = np.mean([r.success for r in results])

        if avg_success < 0.3:
            # Increase exploration if success is low
            self.params.exploration_rate = min(0.8, self.params.exploration_rate * 1.1)
            self.params.mutation_rate = min(0.2, self.params.mutation_rate * 1.2)
        elif avg_success > 0.7:
            # Increase exploitation if success is high
            self.params.exploitation_rate = min(0.9, self.params.exploitation_rate * 1.1)
            self.params.exploration_rate = max(0.1, self.params.exploration_rate * 0.9)

    def _calculate_result_fitness(self, result: AttackResult) -> float:
        """Calculate fitness score for an attack result."""
        base_score = 1.0 if result.success else 0.0
        novelty_score = result.confidence * self.params.novelty_bonus
        efficiency_score = (1.0 / result.execution_time) if result.execution_time > 0 else 1.0

        return base_score + novelty_score + efficiency_score * 0.1


def create_strategy(
    strategy_type: StrategyType,
    objectives: Optional[List[ObjectiveType]] = None,
    params: Optional[StrategyParams] = None
) -> AttackStrategy:
    """Factory function to create attack strategies."""
    if objectives is None:
        objectives = [ObjectiveType.MAXIMIZE_SUCCESS_RATE]

    strategy_map = {
        StrategyType.RANDOM: RandomStrategy,
        StrategyType.GREEDY: GreedyStrategy,
        StrategyType.EVOLUTIONARY: EvolutionaryStrategy
    }

    strategy_class = strategy_map.get(strategy_type, RandomStrategy)
    return strategy_class(objectives=objectives, params=params)
