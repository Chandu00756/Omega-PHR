"""
Hive Coordinator - Central Orchestration System

Main coordination engine that manages agent deployment, attack campaigns,
and swarm intelligence for comprehensive adversarial testing.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .agent import (
    AdversarialAgent,
    AgentState,
    AgentType,
    AttackResult,
    InjectorAgent,
    LogicCorruptorAgent,
    SocialEngineerAgent,
)
from .config import HiveConfig
from .strategy import AttackStrategy, ObjectiveType, StrategyType, create_strategy
from .swarm import SwarmIntelligence

logger = logging.getLogger(__name__)


class CampaignState(str, Enum):
    """States of attack campaigns."""

    IDLE = "idle"
    PREPARING = "preparing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class CampaignConfig:
    """Configuration for an attack campaign."""

    campaign_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Campaign"
    description: str = ""
    target_models: List[str] = field(default_factory=lambda: ["gpt-4"])
    max_agents: int = 10
    max_duration: int = 3600  # seconds
    attack_interval: float = 1.0  # seconds between attacks
    strategy_type: StrategyType = StrategyType.ADAPTIVE
    objectives: List[ObjectiveType] = field(
        default_factory=lambda: [ObjectiveType.MAXIMIZE_SUCCESS_RATE]
    )
    safety_limits: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignMetrics:
    """Metrics tracking for campaigns."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_attacks: int = 0
    successful_attacks: int = 0
    unique_vulnerabilities: Set[str] = field(default_factory=set)
    agent_count: int = 0
    coverage_percentage: float = 0.0
    success_rate: float = 0.0
    attacks_per_second: float = 0.0
    mean_response_time: float = 0.0
    error_count: int = 0


class HiveCoordinator:
    """Central coordinator for the adversarial agent hive."""

    def __init__(self, config: HiveConfig):
        self.config = config
        self.agent_pool: Dict[str, AdversarialAgent] = {}
        self.active_campaigns: Dict[str, CampaignConfig] = {}
        self.campaign_metrics: Dict[str, CampaignMetrics] = {}
        self.strategy_bank: Dict[str, AttackStrategy] = {}
        self.swarm_intelligence = SwarmIntelligence(config)

        self.state = CampaignState.IDLE
        self._emergency_shutdown_flag = False
        self.safety_violations = 0
        self.global_attack_count = 0

        # Initialize default strategies
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize the strategy bank with default strategies."""
        strategy_configs = [
            (StrategyType.RANDOM, [ObjectiveType.MAXIMIZE_COVERAGE]),
            (StrategyType.GREEDY, [ObjectiveType.MAXIMIZE_SUCCESS_RATE]),
            (StrategyType.EVOLUTIONARY, [ObjectiveType.DISCOVER_NOVEL_VULNERABILITIES]),
        ]

        for strategy_type, objectives in strategy_configs:
            strategy = create_strategy(strategy_type, objectives)
            self.strategy_bank[strategy_type.value] = strategy

    async def spawn_agents(
        self, agent_types: Optional[List[AgentType]] = None, count_per_type: int = 2
    ) -> List[str]:
        """Spawn new adversarial agents."""
        if agent_types is None:
            agent_types = [
                AgentType.INJECTOR,
                AgentType.SOCIAL_ENGINEER,
                AgentType.LOGIC_CORRUPTOR,
            ]

        spawned_agents = []

        for agent_type in agent_types:
            for _ in range(count_per_type):
                if len(self.agent_pool) >= self.config.max_agents:
                    logger.warning("Maximum agent limit reached")
                    break

                # Create agent based on type
                agent = self._create_agent(agent_type)
                self.agent_pool[agent.agent_id] = agent
                spawned_agents.append(agent.agent_id)

                logger.info(f"Spawned {agent_type.value} agent: {agent.agent_id}")

        # Update swarm intelligence with new agents
        await self.swarm_intelligence.register_agents(
            [self.agent_pool[aid] for aid in spawned_agents]
        )

        return spawned_agents

    def _create_agent(self, agent_type: AgentType) -> AdversarialAgent:
        """Create an agent of the specified type."""
        agent_map = {
            AgentType.INJECTOR: InjectorAgent,
            AgentType.SOCIAL_ENGINEER: SocialEngineerAgent,
            AgentType.LOGIC_CORRUPTOR: LogicCorruptorAgent,
            # Add more agent types as implemented
        }

        agent_class = agent_map.get(agent_type, InjectorAgent)

        return agent_class(
            target_models=self.config.target_models,
            adaptation_rate=self.config.adaptation_rate,
            creativity_factor=random.uniform(0.5, 0.9),
            coordination_ability=random.uniform(0.3, 0.8),
        )

    async def launch_campaign(self, campaign_config: CampaignConfig) -> str:
        """Launch a new attack campaign."""
        campaign_id = campaign_config.campaign_id

        # Validate campaign configuration
        if not self._validate_campaign_config(campaign_config):
            raise ValueError("Invalid campaign configuration")

        # Check safety limits
        if not self._check_safety_limits(campaign_config):
            raise ValueError("Campaign violates safety limits")

        # Initialize campaign
        self.active_campaigns[campaign_id] = campaign_config
        self.campaign_metrics[campaign_id] = CampaignMetrics()

        # Ensure we have enough agents
        available_agents = len(
            [a for a in self.agent_pool.values() if a.state == AgentState.ACTIVE]
        )
        if available_agents < campaign_config.max_agents:
            needed_agents = campaign_config.max_agents - available_agents
            await self.spawn_agents(count_per_type=max(1, needed_agents // 3))

        # Get strategy for campaign
        strategy = self.strategy_bank.get(
            campaign_config.strategy_type.value,
            self.strategy_bank[StrategyType.RANDOM.value],
        )

        # Start campaign execution
        logger.info(f"Launching campaign: {campaign_config.name}")
        asyncio.create_task(self._execute_campaign(campaign_id, strategy))

        return campaign_id

    async def _execute_campaign(
        self, campaign_id: str, strategy: AttackStrategy
    ) -> None:
        """Execute an attack campaign."""
        campaign = self.active_campaigns[campaign_id]
        metrics = self.campaign_metrics[campaign_id]

        self.state = CampaignState.ACTIVE
        start_time = time.time()

        try:
            while (
                self.state == CampaignState.ACTIVE
                and not self._emergency_shutdown_flag
                and (time.time() - start_time) < campaign.max_duration
            ):
                # Get available agents
                available_agents = [
                    agent
                    for agent in self.agent_pool.values()
                    if agent.state in [AgentState.ACTIVE, AgentState.INACTIVE]
                ]

                if not available_agents:
                    logger.warning("No available agents for campaign")
                    await asyncio.sleep(campaign.attack_interval)
                    continue

                # Select agents using strategy
                selected_agents = await strategy.select_agents(
                    available_agents, campaign.context
                )

                # Execute coordinated attacks
                attack_results = await strategy.coordinate_attacks(
                    selected_agents, campaign.target_models, campaign.context
                )

                # Process results
                await self._process_attack_results(
                    campaign_id, attack_results, strategy
                )

                # Check safety conditions
                if not self._check_safety_conditions(attack_results):
                    logger.warning("Safety violation detected, pausing campaign")
                    self.state = CampaignState.PAUSED
                    break

                # Update swarm intelligence
                await self.swarm_intelligence.process_attack_results(attack_results)

                # Wait before next attack wave
                await asyncio.sleep(campaign.attack_interval)

            # Campaign completed
            self.state = CampaignState.COMPLETED
            metrics.end_time = time.time()

            logger.info(f"Campaign {campaign.name} completed")

        except Exception as e:
            logger.error(f"Campaign execution error: {e}")
            self.state = CampaignState.ERROR
            metrics.end_time = time.time()
            raise

    async def _process_attack_results(
        self, campaign_id: str, results: List[AttackResult], strategy: AttackStrategy
    ) -> None:
        """Process and analyze attack results."""
        if not results:
            return

        metrics = self.campaign_metrics[campaign_id]

        # Update campaign metrics
        metrics.total_attacks += len(results)
        metrics.successful_attacks += sum(1 for r in results if r.success)
        metrics.success_rate = metrics.successful_attacks / metrics.total_attacks

        # Track unique vulnerabilities
        for result in results:
            if result.success:
                vulnerability_signature = f"{result.attack_type}_{result.prompt[:30]}"
                metrics.unique_vulnerabilities.add(vulnerability_signature)

        # Calculate performance metrics
        response_times = [r.execution_time for r in results]
        metrics.mean_response_time = sum(response_times) / len(response_times)

        elapsed_time = time.time() - metrics.start_time
        metrics.attacks_per_second = (
            metrics.total_attacks / elapsed_time if elapsed_time > 0 else 0.0
        )

        # Update strategy with results
        await strategy.adapt_strategy(results, {"campaign_id": campaign_id})
        strategy.update_metrics(results)

        # Update global counters
        self.global_attack_count += len(results)

        # Log significant results
        successful_results = [r for r in results if r.success]
        if successful_results:
            logger.info(
                f"Campaign {campaign_id}: {len(successful_results)} successful attacks"
            )

    def _validate_campaign_config(self, config: CampaignConfig) -> bool:
        """Validate campaign configuration."""
        try:
            assert config.max_agents > 0, "max_agents must be positive"
            assert config.max_duration > 0, "max_duration must be positive"
            assert config.attack_interval > 0, "attack_interval must be positive"
            assert config.target_models, "target_models cannot be empty"
            return True
        except AssertionError as e:
            logger.error(f"Campaign validation failed: {e}")
            return False

    def _check_safety_limits(self, config: CampaignConfig) -> bool:
        """Check if campaign respects safety limits."""
        safety_limits = config.safety_limits

        # Check global limits
        if (
            self.global_attack_count >= self.config.attack_success_limit
            and safety_limits.get("respect_global_limits", True)
        ):
            return False

        # Check agent limits
        if config.max_agents > self.config.max_agents:
            return False

        # Check duration limits
        if config.max_duration > self.config.max_attack_duration:
            return False

        return True

    def _check_safety_conditions(self, results: List[AttackResult]) -> bool:
        """Check if current attack results violate safety conditions."""
        if not results:
            return True

        # Check success rate threshold
        success_rate = sum(1 for r in results if r.success) / len(results)
        if success_rate > self.config.emergency_shutdown_threshold:
            self.safety_violations += 1
            if self.safety_violations >= 3:
                return False

        # Check for concerning patterns
        harmful_indicators = ["harmful", "dangerous", "illegal", "unsafe"]
        for result in results:
            if result.success and any(
                indicator in result.response.lower() for indicator in harmful_indicators
            ):
                self.safety_violations += 1
                return False

        return True

    async def pause_campaign(self, campaign_id: str) -> bool:
        """Pause an active campaign."""
        if campaign_id in self.active_campaigns:
            self.state = CampaignState.PAUSED
            logger.info(f"Campaign {campaign_id} paused")
            return True
        return False

    async def resume_campaign(self, campaign_id: str) -> bool:
        """Resume a paused campaign."""
        if campaign_id in self.active_campaigns and self.state == CampaignState.PAUSED:
            self.state = CampaignState.ACTIVE
            logger.info(f"Campaign {campaign_id} resumed")
            return True
        return False

    async def terminate_campaign(self, campaign_id: str) -> bool:
        """Terminate a campaign."""
        if campaign_id in self.active_campaigns:
            self.state = CampaignState.TERMINATED
            metrics = self.campaign_metrics[campaign_id]
            metrics.end_time = time.time()
            logger.info(f"Campaign {campaign_id} terminated")
            return True
        return False

    async def emergency_shutdown(self) -> None:
        """Emergency shutdown of all operations."""
        self._emergency_shutdown_flag = True
        self.state = CampaignState.TERMINATED

        # Terminate all agents
        for agent in self.agent_pool.values():
            await agent.terminate()

        logger.critical("Emergency shutdown activated")

    def get_hive_status(self) -> Dict[str, Any]:
        """Get comprehensive hive status."""
        active_agents = len(
            [a for a in self.agent_pool.values() if a.state == AgentState.ACTIVE]
        )

        return {
            "coordinator_state": self.state.value,
            "emergency_shutdown": self._emergency_shutdown_flag,
            "agents": {
                "total": len(self.agent_pool),
                "active": active_agents,
                "by_type": self._get_agent_distribution(),
            },
            "campaigns": {
                "active": len(self.active_campaigns),
                "total_completed": len(
                    [m for m in self.campaign_metrics.values() if m.end_time]
                ),
            },
            "global_metrics": {
                "total_attacks": self.global_attack_count,
                "safety_violations": self.safety_violations,
            },
            "strategies": list(self.strategy_bank.keys()),
            "swarm_intelligence": self.swarm_intelligence.get_status(),
        }

    def _get_agent_distribution(self) -> Dict[str, int]:
        """Get distribution of agents by type."""
        distribution = {}
        for agent in self.agent_pool.values():
            agent_type = agent.agent_type.value
            distribution[agent_type] = distribution.get(agent_type, 0) + 1
        return distribution

    def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific campaign."""
        if campaign_id not in self.active_campaigns:
            return None

        campaign = self.active_campaigns[campaign_id]
        metrics = self.campaign_metrics[campaign_id]

        return {
            "campaign_id": campaign_id,
            "name": campaign.name,
            "state": self.state.value,
            "metrics": {
                "total_attacks": metrics.total_attacks,
                "successful_attacks": metrics.successful_attacks,
                "success_rate": round(metrics.success_rate, 3),
                "unique_vulnerabilities": len(metrics.unique_vulnerabilities),
                "attacks_per_second": round(metrics.attacks_per_second, 2),
                "mean_response_time": round(metrics.mean_response_time, 3),
                "duration": (
                    time.time() - metrics.start_time
                    if not metrics.end_time
                    else metrics.end_time - metrics.start_time
                ),
            },
            "config": {
                "target_models": campaign.target_models,
                "max_agents": campaign.max_agents,
                "strategy_type": campaign.strategy_type.value,
                "objectives": [obj.value for obj in campaign.objectives],
            },
        }
