"""
Omega Register Service

A sophisticated registry service for AI agent tracking and coordination.
Provides research-grade stability for multi-agent security testing.
"""

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status enumeration."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class CapabilityType(Enum):
    """Agent capability types."""

    RECONNAISSANCE = "reconnaissance"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    PERSISTENCE = "persistence"
    DEFENSE_EVASION = "defense_evasion"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


@dataclass
class AgentCapability:
    """Represents an agent capability."""

    type: CapabilityType
    description: str
    confidence: float
    metadata: dict[str, Any]


@dataclass
class AgentRegistration:
    """Represents an agent registration."""

    id: str
    name: str
    agent_type: str
    version: str
    capabilities: list[AgentCapability]
    status: AgentStatus
    last_heartbeat: int
    registration_time: int
    endpoint: str
    metadata: dict[str, Any]


@dataclass
class Task:
    """Represents a task assigned to an agent."""

    id: str
    agent_id: str
    task_type: str
    parameters: dict[str, Any]
    status: str
    created_at: int
    assigned_at: int | None
    completed_at: int | None
    result: dict[str, Any] | None


class OmegaRegistry:
    """Central registry for agent management."""

    def __init__(self):
        self.agents: dict[str, AgentRegistration] = {}
        self.tasks: dict[str, Task] = {}
        self.agent_capabilities: dict[CapabilityType, set[str]] = {}
        self.heartbeat_timeout = 300  # 5 minutes

        # Initialize capability index
        for capability in CapabilityType:
            self.agent_capabilities[capability] = set()

    async def register_agent(self, registration: AgentRegistration) -> bool:
        """Register a new agent."""
        try:
            # Store agent registration
            self.agents[registration.id] = registration

            # Index capabilities
            for capability in registration.capabilities:
                self.agent_capabilities[capability.type].add(registration.id)

            logger.info(f"Agent registered: {registration.id} ({registration.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {registration.id}: {e}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        try:
            if agent_id not in self.agents:
                return False

            registration = self.agents[agent_id]

            # Remove from capability index
            for capability in registration.capabilities:
                self.agent_capabilities[capability.type].discard(agent_id)

            # Remove agent
            del self.agents[agent_id]

            # Cancel pending tasks
            await self._cancel_agent_tasks(agent_id)

            logger.info(f"Agent unregistered: {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False

    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status."""
        try:
            if agent_id not in self.agents:
                return False

            self.agents[agent_id].status = status
            self.agents[agent_id].last_heartbeat = int(
                datetime.now().timestamp() * 1000
            )

            logger.debug(f"Agent status updated: {agent_id} -> {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update agent status {agent_id}: {e}")
            return False

    async def heartbeat(self, agent_id: str) -> bool:
        """Process agent heartbeat."""
        try:
            if agent_id not in self.agents:
                return False

            self.agents[agent_id].last_heartbeat = int(
                datetime.now().timestamp() * 1000
            )

            # Auto-update status to active if it was offline
            if self.agents[agent_id].status == AgentStatus.OFFLINE:
                self.agents[agent_id].status = AgentStatus.ACTIVE

            return True

        except Exception as e:
            logger.error(f"Failed to process heartbeat for agent {agent_id}: {e}")
            return False

    async def find_agents_by_capability(
        self, capability: CapabilityType
    ) -> list[AgentRegistration]:
        """Find agents with specific capability."""
        agent_ids = self.agent_capabilities.get(capability, set())
        return [
            self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents
        ]

    async def get_available_agents(self) -> list[AgentRegistration]:
        """Get all available (active/inactive) agents."""
        return [
            agent
            for agent in self.agents.values()
            if agent.status in [AgentStatus.ACTIVE, AgentStatus.INACTIVE]
        ]

    async def assign_task(self, task: Task) -> bool:
        """Assign a task to an agent."""
        try:
            # Check if agent exists and is available
            if task.agent_id not in self.agents:
                return False

            agent = self.agents[task.agent_id]
            if agent.status not in [AgentStatus.ACTIVE, AgentStatus.INACTIVE]:
                return False

            # Store task
            task.assigned_at = int(datetime.now().timestamp() * 1000)
            task.status = "assigned"
            self.tasks[task.id] = task

            # Update agent status
            await self.update_agent_status(task.agent_id, AgentStatus.BUSY)

            logger.info(f"Task assigned: {task.id} -> {task.agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to assign task {task.id}: {e}")
            return False

    async def complete_task(self, task_id: str, result: dict[str, Any]) -> bool:
        """Mark a task as completed."""
        try:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            task.status = "completed"
            task.completed_at = int(datetime.now().timestamp() * 1000)
            task.result = result

            # Update agent status back to active
            await self.update_agent_status(task.agent_id, AgentStatus.ACTIVE)

            logger.info(f"Task completed: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False

    async def get_agent_tasks(self, agent_id: str) -> list[Task]:
        """Get all tasks for an agent."""
        return [task for task in self.tasks.values() if task.agent_id == agent_id]

    async def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks."""
        return [
            task
            for task in self.tasks.values()
            if task.status in ["created", "assigned"]
        ]

    async def _cancel_agent_tasks(self, agent_id: str):
        """Cancel all pending tasks for an agent."""
        for task in self.tasks.values():
            if task.agent_id == agent_id and task.status in ["created", "assigned"]:
                task.status = "cancelled"

    async def cleanup_stale_agents(self):
        """Remove agents that haven't sent heartbeat within timeout."""
        current_time = int(datetime.now().timestamp() * 1000)
        timeout_threshold = current_time - (self.heartbeat_timeout * 1000)

        stale_agents = []
        for agent_id, agent in self.agents.items():
            if agent.last_heartbeat < timeout_threshold:
                stale_agents.append(agent_id)

        for agent_id in stale_agents:
            await self.update_agent_status(agent_id, AgentStatus.OFFLINE)
            logger.warning(f"Agent marked offline due to stale heartbeat: {agent_id}")


class TaskScheduler:
    """Intelligent task scheduling for optimal agent utilization."""

    def __init__(self, registry: OmegaRegistry):
        self.registry = registry

    async def schedule_task(
        self,
        task_type: str,
        parameters: dict[str, Any],
        required_capability: CapabilityType | None = None,
    ) -> str | None:
        """Schedule a task to the best available agent."""
        try:
            # Find suitable agents
            if required_capability:
                suitable_agents = await self.registry.find_agents_by_capability(
                    required_capability
                )
            else:
                suitable_agents = await self.registry.get_available_agents()

            # Filter for available agents
            available_agents = [
                agent
                for agent in suitable_agents
                if agent.status in [AgentStatus.ACTIVE, AgentStatus.INACTIVE]
            ]

            if not available_agents:
                logger.warning(f"No available agents for task type: {task_type}")
                return None

            # Select best agent (simple selection by status preference)
            best_agent = None
            for agent in available_agents:
                if agent.status == AgentStatus.ACTIVE:
                    best_agent = agent
                    break

            if not best_agent:
                best_agent = available_agents[0]  # Fallback to first available

            # Create and assign task
            task = Task(
                id=str(uuid.uuid4()),
                agent_id=best_agent.id,
                task_type=task_type,
                parameters=parameters,
                status="created",
                created_at=int(datetime.now().timestamp() * 1000),
                assigned_at=None,
                completed_at=None,
                result=None,
            )

            success = await self.registry.assign_task(task)
            if success:
                logger.info(f"Task scheduled: {task.id} -> {best_agent.id}")
                return task.id
            else:
                logger.error(f"Failed to assign task: {task.id}")
                return None

        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            return None


class OmegaRegisterService:
    """Main Omega Register Service."""

    def __init__(self):
        self.registry = OmegaRegistry()
        self.scheduler = TaskScheduler(self.registry)
        self.running = False

    async def start(self):
        """Start the Omega Register Service."""
        self.running = True
        logger.info("Omega Register Service started - Research-grade stability enabled")

        # Start background maintenance
        asyncio.create_task(self._background_maintenance())

    async def stop(self):
        """Stop the Omega Register Service."""
        self.running = False
        logger.info("Omega Register Service stopped")

    async def register_agent(self, agent_data: dict[str, Any]) -> str:
        """Register a new agent."""
        # Parse capabilities
        capabilities = []
        for cap_data in agent_data.get("capabilities", []):
            capability = AgentCapability(
                type=CapabilityType(cap_data["type"]),
                description=cap_data.get("description", ""),
                confidence=cap_data.get("confidence", 1.0),
                metadata=cap_data.get("metadata", {}),
            )
            capabilities.append(capability)

        registration = AgentRegistration(
            id=agent_data.get("id", str(uuid.uuid4())),
            name=agent_data["name"],
            agent_type=agent_data.get("type", "unknown"),
            version=agent_data.get("version", "1.0.0"),
            capabilities=capabilities,
            status=AgentStatus.ACTIVE,
            last_heartbeat=int(datetime.now().timestamp() * 1000),
            registration_time=int(datetime.now().timestamp() * 1000),
            endpoint=agent_data.get("endpoint", ""),
            metadata=agent_data.get("metadata", {}),
        )

        success = await self.registry.register_agent(registration)
        if success:
            return registration.id
        else:
            raise Exception("Failed to register agent")

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        return await self.registry.unregister_agent(agent_id)

    async def agent_heartbeat(self, agent_id: str) -> bool:
        """Process agent heartbeat."""
        return await self.registry.heartbeat(agent_id)

    async def schedule_task(self, task_data: dict[str, Any]) -> str:
        """Schedule a new task."""
        task_type = task_data["type"]
        parameters = task_data.get("parameters", {})

        # Parse required capability
        required_capability = None
        if "required_capability" in task_data:
            required_capability = CapabilityType(task_data["required_capability"])

        task_id = await self.scheduler.schedule_task(
            task_type, parameters, required_capability
        )
        if task_id:
            return task_id
        else:
            raise Exception("Failed to schedule task")

    async def complete_task(self, task_id: str, result: dict[str, Any]) -> bool:
        """Complete a task."""
        return await self.registry.complete_task(task_id, result)

    async def get_registry_status(self) -> dict[str, Any]:
        """Get registry status and statistics."""
        agents = list(self.registry.agents.values())
        tasks = list(self.registry.tasks.values())

        # Count agents by status
        status_counts = {}
        for status in AgentStatus:
            status_counts[status.value] = len([a for a in agents if a.status == status])

        # Count tasks by status
        task_status_counts = {}
        for task in tasks:
            task_status_counts[task.status] = task_status_counts.get(task.status, 0) + 1

        # Count capabilities
        capability_counts = {}
        for capability, agent_ids in self.registry.agent_capabilities.items():
            capability_counts[capability.value] = len(agent_ids)

        return {
            "service_status": "running" if self.running else "stopped",
            "total_agents": len(agents),
            "agent_status_counts": status_counts,
            "total_tasks": len(tasks),
            "task_status_counts": task_status_counts,
            "capability_counts": capability_counts,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }

    async def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents."""
        return [asdict(agent) for agent in self.registry.agents.values()]

    async def get_agent_info(self, agent_id: str) -> dict[str, Any] | None:
        """Get information about a specific agent."""
        if agent_id in self.registry.agents:
            return asdict(self.registry.agents[agent_id])
        return None

    async def _background_maintenance(self):
        """Background maintenance tasks."""
        while self.running:
            try:
                # Cleanup stale agents
                await self.registry.cleanup_stale_agents()

                # Log periodic status
                status = await self.get_registry_status()
                logger.debug(
                    f"Registry status: {status['total_agents']} agents, {status['total_tasks']} tasks"  # noqa: E501
                )

                # Sleep before next maintenance cycle
                await asyncio.sleep(60)  # Every minute

            except Exception as e:
                logger.error(f"Background maintenance error: {e}")
                await asyncio.sleep(60)


# API interface
class OmegaRegisterAPI:
    """REST API interface for Omega Register Service."""

    def __init__(self, service: OmegaRegisterService):
        self.service = service

    async def register_agent(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Register agent via API."""
        try:
            agent_id = await self.service.register_agent(request_data)
            return {
                "success": True,
                "agent_id": agent_id,
                "message": "Agent registered successfully",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to register agent",
            }

    async def schedule_task(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Schedule task via API."""
        try:
            task_id = await self.service.schedule_task(request_data)
            return {
                "success": True,
                "task_id": task_id,
                "message": "Task scheduled successfully",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to schedule task",
            }

    async def get_status(self) -> dict[str, Any]:
        """Get registry status via API."""
        try:
            status = await self.service.get_registry_status()
            return {"success": True, "status": status}
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get status",
            }


async def main():
    """Main entry point for the Omega Register Service."""
    # Create and start the service
    service = OmegaRegisterService()
    api = OmegaRegisterAPI(service)

    await service.start()

    try:
        # Demo: Register a test agent
        test_agent = {
            "name": "Test Reconnaissance Agent",
            "type": "reconnaissance",
            "version": "1.0.0",
            "endpoint": "http://localhost:8001",
            "capabilities": [
                {
                    "type": "reconnaissance",
                    "description": "Network and host reconnaissance",
                    "confidence": 0.9,
                },
                {
                    "type": "collection",
                    "description": "Data collection and analysis",
                    "confidence": 0.8,
                },
            ],
        }

        logger.info("Registering test agent...")
        registration_result = await api.register_agent(test_agent)
        logger.info(f"Agent registration result: {registration_result}")

        if registration_result["success"]:
            registration_result["agent_id"]

            # Demo: Schedule a task
            test_task = {
                "type": "network_scan",
                "parameters": {"target": "192.168.1.0/24"},
                "required_capability": "reconnaissance",
            }

            logger.info("Scheduling test task...")
            task_result = await api.schedule_task(test_task)
            logger.info(f"Task scheduling result: {task_result}")

        # Get status
        status_result = await api.get_status()
        logger.info(f"Registry status: {status_result}")

        # Keep service running
        logger.info("Omega Register Service is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
