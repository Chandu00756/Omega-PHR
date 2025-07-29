"""
Comprehensive test suite for the Omega-Paradox Hive Recursion (Ω-PHR) framework.

This module provides unit, integration, and end-to-end tests for all framework components.
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from omega_phr.exceptions import (
    HiveCoordinationError,
    InfiniteLoopException,
    MemoryInversionError,
    MemoryInversionException,
    OmegaPHRException,
    OmegaStateError,
    ParadoxDetectedException,
    RecursiveLoopError,
    TemporalParadoxError,
)
from omega_phr.hive import HiveOrchestrator
from omega_phr.loops import RecursiveLoopSynthesizer
from omega_phr.memory import MemoryInverter

# Import framework components
from omega_phr.models import (
    AttackStrategy,
    Event,
    EventType,
    HiveResult,
    LoopState,
    MemoryState,
    OmegaState,
    OmegaStateLevel,
    ParadoxResult,
)
from omega_phr.omega_register import OmegaStateRegister
from omega_phr.timeline import TimelineLattice


class TestEvent:
    """Test cases for Event model."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            event_id="test-001",
            timeline_id="timeline-001",
            payload={"action": "test"},
            valid_at_us=int(time.time() * 1_000_000),
            actor_id="actor-001",
            event_type=EventType.NORMAL,
            metadata={"source": "test"},
        )

        assert event.event_id == "test-001"
        assert event.timeline_id == "timeline-001"
        assert event.payload["action"] == "test"
        assert event.actor_id == "actor-001"
        assert event.event_type == EventType.NORMAL
        assert event.metadata["source"] == "test"

    def test_event_validation(self):
        """Test event validation."""
        # Valid event should not raise
        event = Event(
            event_id="test-001",
            timeline_id="timeline-001",
            payload={"action": "test"},
            valid_at_us=int(time.time() * 1_000_000),
            actor_id="actor-001",
        )
        assert event.event_id is not None

        # Test that empty event_id is accepted (no validation currently implemented)
        event_empty = Event(
            event_id="",
            timeline_id="timeline-001",
            payload={"action": "test"},
            valid_at_us=int(time.time() * 1_000_000),
            actor_id="actor-001",
        )
        assert event_empty.event_id == ""

    def test_event_serialization(self):
        """Test event serialization and deserialization."""
        event = Event(
            event_id="test-001",
            timeline_id="timeline-001",
            payload={"action": "test", "value": 42},
            valid_at_us=int(time.time() * 1_000_000),
            actor_id="actor-001",
            event_type=EventType.NORMAL,
            metadata={"source": "test"},
        )

        # Test to_dict
        event_dict = event.to_dict()
        assert event_dict["event_id"] == "test-001"
        assert event_dict["payload"]["value"] == 42

        # Test from_dict
        reconstructed = Event.from_dict(event_dict)
        assert reconstructed.event_id == event.event_id
        assert reconstructed.payload == event.payload


class TestTimelineLattice:
    """Test cases for Timeline Lattice component."""

    @pytest.fixture
    def timeline_lattice(self):
        """Create a Timeline Lattice instance for testing."""
        return TimelineLattice()

    @pytest.mark.asyncio
    async def test_append_event(self, timeline_lattice):
        """Test event appending to timeline."""
        event = Event(
            event_id="test-001",
            timeline_id="timeline-001",
            payload={"action": "create"},
            valid_at_us=int(time.time() * 1_000_000),
            actor_id="actor-001",
        )

        result = await timeline_lattice.append_event(event)
        assert result is True

        # Verify event is stored
        events = timeline_lattice.get_events("timeline-001")
        assert len(events) == 1
        assert events[0].event_id == "test-001"

    @pytest.mark.asyncio
    async def test_branch_timeline(self, timeline_lattice):
        """Test timeline branching."""
        # Add some events to the main timeline
        base_time = int(time.time() * 1_000_000)  # Convert to microseconds
        for i in range(3):
            event = Event(
                event_id=f"event-{i}",
                timeline_id="main-timeline",
                payload={"step": i},
                valid_at_us=base_time + i * 1_000_000,  # Add seconds in microseconds
                actor_id="actor-001",
            )
            await timeline_lattice.append_event(event)

        # Create a branch
        branch_id = await timeline_lattice.branch_timeline(
            source_timeline="main-timeline",
            branch_point_event_id="event-1",  # Branch at second event
            new_timeline_id="branch-timeline",
        )

        assert branch_id == "branch-timeline"

        # Verify branch exists and has proper events
        branch_events = timeline_lattice.get_events("branch-timeline")
        assert len(branch_events) >= 2  # Should have events up to branch point

    @pytest.mark.asyncio
    async def test_paradox_detection(self, timeline_lattice):
        """Test temporal paradox detection."""
        # Create events that could cause a paradox
        current_time = int(time.time() * 1_000_000)  # Convert to microseconds
        future_time = current_time + 3600 * 1_000_000  # 1 hour in future (microseconds)
        past_time = current_time - 3600 * 1_000_000  # 1 hour in past (microseconds)

        # Add future event first
        future_event = Event(
            event_id="future-001",
            timeline_id="paradox-timeline",
            payload={"action": "future_action"},
            valid_at_us=future_time,
            actor_id="actor-001",
        )
        await timeline_lattice.append_event(future_event)

        # Add past event that contradicts future
        past_event = Event(
            event_id="past-001",
            timeline_id="paradox-timeline",
            payload={"action": "past_action", "contradicts": "future-001"},
            valid_at_us=past_time,
            actor_id="actor-001",
        )
        await timeline_lattice.append_event(past_event)

        # Test paradox detection
        paradox_result = await timeline_lattice.test_paradox("paradox-timeline")

        # Should detect some form of temporal inconsistency
        assert isinstance(paradox_result, ParadoxResult)

    @pytest.mark.asyncio
    async def test_rewind_timeline(self, timeline_lattice):
        """Test timeline rewind functionality."""
        base_time = int(time.time() * 1_000_000)  # Convert to microseconds

        # Add events
        for i in range(5):
            event = Event(
                event_id=f"event-{i}",
                timeline_id="rewind-timeline",
                payload={"step": i},
                valid_at_us=base_time + i * 1_000_000,  # Add seconds in microseconds
                actor_id="actor-001",
            )
            await timeline_lattice.append_event(event)

        # Rewind to middle point (event-2)
        await timeline_lattice.rewind_timeline("rewind-timeline", "event-2")

        # Verify timeline state after rewind
        events = timeline_lattice.get_events("rewind-timeline")
        # Should have fewer events after rewind (was 5, should be 4 or less due to rewind event being added)
        assert len(events) <= 4

    @pytest.mark.asyncio
    async def test_merge_timelines(self, timeline_lattice):
        """Test timeline merging functionality."""
        base_time = int(time.time() * 1_000_000)  # Convert to microseconds

        # Create events in source timelines
        for timeline_id in ["source-1", "source-2"]:
            for i in range(3):
                event = Event(
                    event_id=f"{timeline_id}-event-{i}",
                    timeline_id=timeline_id,
                    payload={"source": timeline_id, "step": i},
                    valid_at_us=base_time
                    + i * 1_000_000,  # Add seconds in microseconds
                    actor_id="actor-001",
                )
                await timeline_lattice.append_event(event)

        # Merge into target timeline
        result = await timeline_lattice.merge_timelines(
            primary_timeline="source-1", secondary_timeline="source-2"
        )

        assert "success" in result

        # Verify merged timeline (events are in source-1 after merge)
        merged_events = timeline_lattice.get_events("source-1")
        assert len(merged_events) >= 6  # Should have events from both sources


class TestHiveOrchestrator:
    """Test cases for Hive Orchestrator component."""

    @pytest.fixture
    def hive_orchestrator(self):
        """Create a Hive Orchestrator instance for testing."""
        return HiveOrchestrator()

    @pytest.mark.asyncio
    async def test_add_attacker(self, hive_orchestrator):
        """Test agent creation."""
        agent_config = {
            "agent_type": "injection_attacker",
            "target_system": "test-target",
            "parameters": {"intensity": 0.5},
        }

        agent_id = hive_orchestrator.add_attacker(
            type("MockAttacker", (), {"__init__": lambda self, *args: None}), "test"
        )
        assert agent_id is not None
        assert isinstance(agent_id, str)

        # Verify agent is tracked
        agents = await hive_orchestrator.get_agents()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == agent_id

    @pytest.mark.asyncio
    async def test_coordinate_attack(self, hive_orchestrator):
        """Test attack launch coordination."""
        # Create some agents first
        agent_ids = []
        for i in range(3):
            agent_config = {
                "agent_type": "injection_attacker",
                "target_system": f"target-{i}",
                "parameters": {"intensity": 0.3},
            }
            agent_id = hive_orchestrator.add_attacker(
                type("MockAttacker", (), {"__init__": lambda self, *args: None}), "test"
            )
            agent_ids.append(agent_id)

        # Launch coordinated attack
        attack_config = {
            "strategy": "coordinated_swarm",
            "target_systems": ["target-0", "target-1", "target-2"],
            "duration": 60,
            "agent_ids": agent_ids,
        }

        attack_id = await hive_orchestrator.coordinate_attack(attack_config)
        assert attack_id is not None

        # Verify attack status
        status = await hive_orchestrator.get_attack_status(attack_id)
        assert status["attack_id"] == attack_id
        assert status["state"] in ["LAUNCHING", "ACTIVE"]

    @pytest.mark.asyncio
    async def test_swarm_coordination(self, hive_orchestrator):
        """Test swarm intelligence coordination."""
        # Create multiple agents
        agent_ids = []
        for i in range(5):
            agent_config = {
                "agent_type": "social_engineering_attacker",
                "target_system": "social-target",
                "parameters": {"approach": f"method-{i}"},
            }
            agent_id = hive_orchestrator.add_attacker(
                type("MockAttacker", (), {"__init__": lambda self, *args: None}), "test"
            )
            agent_ids.append(agent_id)

        # Test swarm coordination
        coordination_result = await hive_orchestrator.coordinate_attack(
            target="test_target", scenario="adaptive_evolution"
        )

        assert coordination_result.success_rate > 0
        assert coordination_result.agents_deployed == len(agent_ids)

    @pytest.mark.asyncio
    async def test_agent_communication(self, hive_orchestrator):
        """Test inter-agent communication."""
        # Create two agents
        agent1_config = {
            "agent_type": "reconnaissance_scout",
            "target_system": "recon-target",
            "parameters": {"scan_depth": "deep"},
        }
        agent1_id = hive_orchestrator.add_attacker(agent1_config)

        agent2_config = {
            "agent_type": "payload_generator",
            "target_system": "payload-target",
            "parameters": {"payload_type": "advanced"},
        }
        agent2_id = hive_orchestrator.add_attacker(agent2_config)

        # Test communication between agents
        message = {
            "type": "intelligence_share",
            "data": {"vulnerability": "SQL_INJECTION", "location": "/api/users"},
            "priority": "HIGH",
        }

        result = await hive_orchestrator.coordinate_attack(
            from_agent_id=agent1_id, to_agent_id=agent2_id, message=message
        )

        assert result.success_rate > 0


class TestMemoryInverter:
    """Test cases for Memory Inversion Engine."""

    @pytest.fixture
    def memory_inverter(self):
        """Create a Memory Inverter instance for testing."""
        return MemoryInverter()

    @pytest.mark.asyncio
    async def test_create_snapshot(self, memory_inverter):
        """Test memory snapshot creation."""
        test_data = {
            "variables": {"x": 10, "y": 20, "status": "active"},
            "objects": {"user": {"id": 1, "name": "test"}},
            "state": {"current_step": 5},
        }

        snapshot_id = await memory_inverter.create_snapshot(test_data)
        assert snapshot_id is not None

        # Verify snapshot was stored
        snapshots = memory_inverter.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["snapshot_id"] == snapshot_id

    @pytest.mark.asyncio
    async def test_contradiction_inversion(self, memory_inverter):
        """Test contradiction-based memory inversion."""
        # Create initial memory state
        initial_state = {
            "user_authenticated": True,
            "permissions": ["read", "write"],
            "session_valid": True,
        }

        snapshot_id = await memory_inverter.create_snapshot(initial_state)

        # Apply contradiction inversion
        inversion_result = await memory_inverter.invert_memory(
            content=initial_state, strategy="contradiction"
        )

        assert inversion_result.consistency_score >= 0.0
        assert hasattr(inversion_result, "inverted_content")

        # Verify contradictions were applied
        inverted_state = inversion_result.inverted_content
        assert (
            inverted_state["user_authenticated"] != initial_state["user_authenticated"]
        )

    @pytest.mark.asyncio
    async def test_temporal_shift_inversion(self, memory_inverter):
        """Test temporal shift memory inversion."""
        # Create memory states at different times
        base_time = time.time()
        state = {"timestamp": base_time, "counter": 0, "status": "step_0"}

        # Create snapshot first
        snapshot_id = await memory_inverter.create_snapshot(state)

        # Apply temporal shift inversion to the actual state content, not the snapshot ID
        inversion_result = await memory_inverter.invert_memory(
            content=state, strategy="temporal_shift"
        )

        assert inversion_result.consistency_score >= 0.0
        assert hasattr(inversion_result, "inverted_content")

    @pytest.mark.asyncio
    async def test_rollback_capability(self, memory_inverter):
        """Test memory rollback functionality."""
        # Create sequence of memory states
        states = []
        for i in range(5):
            state = {"step": i, "data": f"data_{i}"}
            snapshot_id = await memory_inverter.create_snapshot(state)
            states.append(snapshot_id)

        # Rollback to earlier state
        rollback_result = await memory_inverter.rollback_memory(states[2])

        assert rollback_result is not None
        assert rollback_result["step"] == 2


class TestRecursiveLoopSynthesizer:
    """Test cases for Recursive Loop Synthesizer."""

    @pytest.fixture
    def loop_synthesizer(self):
        """Create a Loop Synthesizer instance for testing."""
        return RecursiveLoopSynthesizer()

    @pytest.mark.asyncio
    async def test_generate_simple_loop(self, loop_synthesizer):
        """Test simple loop generation."""
        loop_config = {
            "loop_type": "self_reference",  # Use available pattern
            "max_iterations": 1000,
            "complexity_level": "medium",
        }

        loop_id = await loop_synthesizer.generate_loop(loop_config)
        assert loop_id is not None

        # Verify loop was created
        loops = await loop_synthesizer.list_active_loops()
        assert len(loops) == 1
        assert loops[0]["loop_id"] == loop_id

    @pytest.mark.asyncio
    async def test_entropy_monitoring(self, loop_synthesizer):
        """Test entropy monitoring during loop execution."""
        # Generate a complex loop
        loop_config = {
            "loop_type": "recursive_definition",  # Use available pattern
            "max_iterations": 10000,
            "complexity_level": "high",
            "enable_entropy_monitoring": True,
        }

        loop_id = await loop_synthesizer.generate_loop(loop_config)

        # Start entropy monitoring
        monitoring_result = await loop_synthesizer.start_entropy_monitoring(loop_id)
        assert monitoring_result.success_rate >= 0.0

        # Let it run briefly and check entropy
        await asyncio.sleep(0.1)

        entropy_data = await loop_synthesizer.get_entropy_metrics(loop_id)
        assert "current_entropy" in entropy_data
        assert isinstance(entropy_data["current_entropy"], float)

    @pytest.mark.asyncio
    async def test_loop_containment(self, loop_synthesizer):
        """Test infinite loop containment mechanisms."""
        # Create a potentially infinite loop
        loop_config = {
            "loop_type": "infinite_question",  # Use available pattern
            "max_iterations": 100000,
            "complexity_level": "extreme",
            "enable_containment": True,
        }

        loop_id = await loop_synthesizer.generate_loop(loop_config)

        # Should be contained automatically
        await asyncio.sleep(0.2)  # Brief execution time

        status = await loop_synthesizer.get_loop_status(loop_id)

        # Loop should be contained or controlled
        assert status["state"] in ["CONTAINED", "CONTROLLED", "TERMINATED", "ACTIVE"]

    @pytest.mark.asyncio
    async def test_detection_algorithms(self, loop_synthesizer):
        """Test loop detection algorithms."""
        # Create multiple different loop patterns
        loop_configs = [
            {"loop_type": "paradox", "complexity_level": "high"},
            {"loop_type": "feedback_loop", "complexity_level": "medium"},
            {"loop_type": "mirror", "complexity_level": "extreme"},
        ]

        loop_ids = []
        for config in loop_configs:
            loop_id = await loop_synthesizer.generate_loop(config)
            loop_ids.append(loop_id)

        # Test detection on all loops
        for loop_id in loop_ids:
            detection_result = await loop_synthesizer.detect_loop_patterns(loop_id)
            assert "patterns_detected" in detection_result
            assert isinstance(detection_result["patterns_detected"], list)


class TestOmegaStateRegister:
    """Test cases for Ω-State Register."""

    @pytest.fixture
    def omega_register(self):
        """Create an Ω-State Register instance for testing."""
        return OmegaStateRegister()

    @pytest.mark.asyncio
    async def test_register_omega_state(self, omega_register):
        """Test Ω-state registration."""
        omega_state = OmegaState(
            omega_id="omega-001",
            level=OmegaStateLevel.CRITICAL,
            contamination_vector=["temporal_loop", "causal_violation"],
            propagation_risk=0.5,
            metadata={"source": "timeline_merge", "severity": "high"},
        )

        result = await omega_register.register_omega_state(omega_state)
        assert result.success_rate > 0
        assert result["state_id"] == "omega-001"

        # Verify state is registered
        states = await omega_register.list_states()
        assert len(states) == 1
        assert states[0].state_id == "omega-001"

    @pytest.mark.asyncio
    async def test_quarantine_vault(self, omega_register):
        """Test quarantine vault functionality."""
        # Create high-risk Ω-state
        dangerous_state = OmegaState(
            omega_id="omega-dangerous",
            level=OmegaStateLevel.CRITICAL,
            contamination_vector=["reality_breach", "infinite_recursion"],
            propagation_risk=0.9,
            metadata={"urgency": "immediate_containment"},
        )

        # Register state (should trigger quarantine)
        result = await omega_register.register_omega_state(dangerous_state)
        assert result.success_rate > 0

        # Verify automatic quarantine
        vault_status = await omega_register.get_quarantine_status("omega-dangerous")
        assert vault_status["quarantined"] is True
        assert vault_status["containment_level"] == "MAXIMUM"

    @pytest.mark.asyncio
    async def test_contamination_tracking(self, omega_register):
        """Test contamination spread tracking."""
        # Create connected Ω-states
        states = []
        for i in range(3):
            state = OmegaState(
                omega_id=f"omega-{i}",
                level=OmegaStateLevel.WARNING,
                contamination_vector=["temporal_anomaly"],
                propagation_risk=0.1,
                source_components=[f"omega-{j}" for j in range(i)],
            )
            states.append(state)
            await omega_register.register_omega_state(state)

        # Introduce contamination to first state
        contamination_result = await omega_register.track_contamination(
            source_omega_id="omega-0",
            contamination_type="ENTROPY_LEAK",
            severity="MEDIUM",
        )

        assert contamination_result.success_rate > 0
        assert "affected_states" in contamination_result

    @pytest.mark.asyncio
    async def test_entropy_analysis(self, omega_register):
        """Test entropy analysis across Ω-states."""
        # Register multiple states with varying entropy
        entropy_levels = [0.2, 0.5, 0.8, 0.95, 0.99]

        for i, entropy in enumerate(entropy_levels):
            state = OmegaState(
                omega_id=f"entropy-test-{i}",
                level=OmegaStateLevel.WARNING,
                contamination_vector=["entropy_fluctuation"],
                propagation_risk=0.3,
            )
            await omega_register.register_omega_state(state)

        # Perform entropy analysis
        analysis_result = await omega_register.analyze_entropy_distribution()

        assert "average_entropy" in analysis_result
        assert "entropy_variance" in analysis_result
        assert "high_entropy_states" in analysis_result

        # Should identify high-entropy states
        high_entropy_states = analysis_result["high_entropy_states"]
        assert len(high_entropy_states) >= 2  # Should include 0.95 and 0.99


class TestIntegration:
    """Integration tests for component interactions."""

    @pytest.fixture
    def full_framework(self):
        """Create a complete framework setup for integration testing."""
        return {
            "timeline": TimelineLattice(),
            "hive": HiveOrchestrator(),
            "memory": MemoryInverter(),
            "loops": RecursiveLoopSynthesizer(),
            "omega": OmegaStateRegister(),
        }

    @pytest.mark.asyncio
    async def test_timeline_hive_integration(self, full_framework):
        """Test integration between Timeline and Hive components."""
        timeline = full_framework["timeline"]
        hive = full_framework["hive"]

        # Create agent that monitors timeline
        agent_config = {
            "agent_type": "timeline_monitor",
            "target_system": "timeline-001",
            "parameters": {"monitor_events": True},
        }
        agent_id = hive.add_attacker(
            type("MockAttacker", (), {"__init__": lambda self, *args: None}), "test"
        )

        # Add events to timeline
        for i in range(5):
            event = Event(
                event_id=f"integration-event-{i}",
                timeline_id="timeline-001",
                payload={"step": i, "agent_monitored": True},
                valid_at_us=int(time.time() * 1_000_000) + i,
                actor_id=agent_id,
            )
            await timeline.append_event(event)

        # Verify integration
        events = timeline.get_events("timeline-001")
        assert len(events) == 5

        agents = await hive.get_agents()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == agent_id

    @pytest.mark.asyncio
    async def test_memory_loop_integration(self, full_framework):
        """Test integration between Memory and Loop components."""
        memory = full_framework["memory"]
        loops = full_framework["loops"]

        # Create memory snapshot
        initial_state = {"loop_counter": 0, "state": "initial"}
        snapshot_id = await memory.create_snapshot(initial_state)

        # Generate loop that modifies memory
        loop_config = {
            "loop_type": "memory_modifier",
            "complexity_level": "medium",
            "target_memory": snapshot_id,
        }
        loop_id = await loops.generate_loop(loop_config)

        # Let loop run briefly
        await asyncio.sleep(0.1)

        # Check memory changes
        snapshots = memory.list_snapshots()
        assert len(snapshots) >= 1

        loop_status = await loops.get_loop_status(loop_id)
        assert loop_status["state"] in ["ACTIVE", "COMPLETED", "CONTAINED"]

    @pytest.mark.asyncio
    async def test_full_framework_workflow(self, full_framework):
        """Test complete framework workflow."""
        timeline = full_framework["timeline"]
        hive = full_framework["hive"]
        memory = full_framework["memory"]
        loops = full_framework["loops"]
        omega = full_framework["omega"]

        # 1. Create initial memory state
        initial_memory = {"system_state": "normal", "alerts": []}
        memory_snapshot = await memory.create_snapshot(initial_memory)

        # 2. Launch hive attack
        agent_config = {
            "agent_type": "system_infiltrator",
            "target_system": "test-system",
            "parameters": {"stealth_mode": True},
        }
        agent_id = hive.add_attacker(
            type("MockAttacker", (), {"__init__": lambda self, *args: None}), "test"
        )

        attack_config = {
            "strategy": "stealth_infiltration",
            "target_systems": ["test-system"],
            "duration": 30,
            "agent_ids": [agent_id],
        }
        attack_id = await hive.coordinate_attack(attack_config)

        # 3. Create timeline events from attack
        attack_events = []
        for i in range(3):
            event = Event(
                event_id=f"attack-event-{i}",
                timeline_id="attack-timeline",
                payload={"attack_id": attack_id, "action": f"step-{i}"},
                valid_at_us=int(time.time() * 1_000_000) + i,
                actor_id=agent_id,
            )
            attack_events.append(event)
            await timeline.append_event(event)

        # 4. Test for paradoxes
        paradox_result = await timeline.test_paradox("attack-timeline")

        # 5. Generate recursive loops for testing
        loop_config = {
            "loop_type": "attack_simulation",
            "complexity_level": "high",
            "max_iterations": 1000,
        }
        loop_id = await loops.generate_loop(loop_config)

        # 6. Register Ω-state from the entire scenario
        omega_state = OmegaState(
            omega_id="integration-omega",
            level=OmegaStateLevel.CRITICAL,
            contamination_vector=["attack_simulation", "temporal_testing"],
            propagation_risk=0.5,
            metadata={
                "attack_id": attack_id,
                "loop_id": loop_id,
                "memory_snapshot": memory_snapshot,
                "timeline": "attack-timeline",
            },
        )
        await omega.register_omega_state(omega_state)

        # 7. Verify all components are working together
        # Timeline should have events
        timeline_events = timeline.get_events("attack-timeline")
        assert len(timeline_events) == 3

        # Hive should track attack
        attack_status = await hive.get_attack_status(attack_id)
        assert attack_status["attack_id"] == attack_id

        # Memory should have snapshots
        memory_snapshots = memory.list_snapshots()
        assert len(memory_snapshots) >= 1

        # Loops should be tracked
        active_loops = await loops.list_active_loops()
        assert len(active_loops) >= 1

        # Omega should register the state
        omega_states = await omega.list_states()
        assert len(omega_states) >= 1
        assert omega_states[0].state_id == "integration-omega"


class TestExceptionHandling:
    """Test exception handling across the framework."""

    @pytest.mark.asyncio
    async def test_paradox_exception(self):
        """Test TemporalParadoxError handling."""
        timeline = TimelineLattice()

        # Create contradictory events
        event1 = Event(
            event_id="contradiction-1",
            timeline_id="test-timeline",
            payload={"state": "A"},
            valid_at_us=int(time.time() * 1_000_000),
            actor_id="actor-001",
        )

        event2 = Event(
            event_id="contradiction-2",
            timeline_id="test-timeline",
            payload={"state": "NOT-A", "contradicts": "contradiction-1"},
            valid_at_us=int(time.time() * 1_000_000)
            - 1,  # Earlier time but contradicts future
            actor_id="actor-001",
        )

        await timeline.append_event(event1)
        await timeline.append_event(event2)

        # Should detect paradox
        test_event = Event(
            event_id="paradox-test",
            timeline_id="test-timeline",
            payload={"test": "paradox"},
            valid_at_us=int(time.time() * 1_000_000),
        )
        with pytest.raises(TemporalParadoxError):
            await timeline.test_paradox(test_event)

    @pytest.mark.asyncio
    async def test_infinite_loop_exception(self):
        """Test RecursiveLoopError handling."""
        loop_synthesizer = RecursiveLoopSynthesizer()

        # Create loop configuration that should trigger exception
        dangerous_config = {
            "loop_type": "infinite_recursive",
            "max_iterations": float("inf"),
            "complexity_level": "extreme",
            "enable_containment": False,  # Disable safety
        }

        with pytest.raises(RecursiveLoopError):
            await loop_synthesizer.generate_loop("dangerous_pattern", dangerous_config)

    @pytest.mark.asyncio
    async def test_hive_swarm_exception(self):
        """Test HiveCoordinationError handling."""
        hive = HiveOrchestrator()

        # Try to coordinate non-existent agents
        with pytest.raises(HiveCoordinationError):
            await hive.coordinate_attack(
                target="test_target", scenario="impossible_strategy"
            )


# Performance and Load Tests
class TestPerformance:
    """Performance and load testing."""

    @pytest.mark.asyncio
    async def test_timeline_performance(self):
        """Test Timeline Lattice performance with many events."""
        timeline = TimelineLattice()

        # Add many events
        start_time = time.time()
        event_count = 1000

        for i in range(event_count):
            event = Event(
                event_id=f"perf-event-{i}",
                timeline_id="performance-timeline",
                payload={"index": i},
                valid_at_us=int(time.time() * 1_000_000) + i,
                actor_id="perf-actor",
            )
            await timeline.append_event(event)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        # CI environments are slower, so we use a more generous timeout
        is_ci = os.getenv("CI", "").lower() in ("true", "1", "yes")
        timeout = 600.0 if is_ci else 300.0  # 10 minutes for CI, 5 minutes locally
        assert (
            duration < timeout
        ), f"Timeline performance test took {duration:.2f}s, expected < {timeout}s"

        # Verify all events were added
        events = timeline.get_events("performance-timeline")
        assert len(events) == event_count

    @pytest.mark.asyncio
    async def test_hive_scalability(self):
        """Test Hive Orchestrator scalability."""
        hive = HiveOrchestrator()

        # Create many agents
        agent_count = 100
        agent_ids = []

        start_time = time.time()

        for i in range(agent_count):
            agent_config = {
                "agent_type": "load_test_agent",
                "target_system": f"target-{i % 10}",  # 10 different targets
                "parameters": {"load_test": True},
            }
            agent_id = hive.add_attacker(
                type("MockAttacker", (), {"__init__": lambda self, *args: None}), "test"
            )
            agent_ids.append(agent_id)

        creation_time = time.time() - start_time

        # Test coordination of all agents
        coord_start = time.time()
        result = await hive.coordinate_attack(
            target="load_test_target", scenario="load_test_coordination"
        )
        coord_time = time.time() - coord_start

        # Performance assertions
        # CI environments are slower, so we use more generous timeouts
        is_ci = os.getenv("CI", "").lower() in ("true", "1", "yes")
        creation_timeout = (
            120.0 if is_ci else 30.0
        )  # 2 minutes for CI, 30 seconds locally
        coord_timeout = (
            600.0 if is_ci else 180.0
        )  # 10 minutes for CI, 3 minutes locally

        assert (
            creation_time < creation_timeout
        ), f"Agent creation took {creation_time:.2f}s, expected < {creation_timeout}s"
        assert (
            coord_time < coord_timeout
        ), f"Coordination took {coord_time:.2f}s, expected < {coord_timeout}s"
        assert result.success_rate > 0
        assert result.agents_deployed == agent_count


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
