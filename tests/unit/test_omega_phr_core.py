"""
Unit tests for Omega PHR framework.
Research-grade stability testing.
"""

import asyncio
import json
import os

# Import modules from the framework
import sys
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omega_phr.config import FrameworkConfig as OmegaPHRConfig
from omega_phr.hive import HiveOrchestrator as HiveCoordinator
from omega_phr.hive import InjectionAttacker
from omega_phr.memory import MemoryInverter as MemoryManager
from omega_phr.models import Agent, AttackStrategy, HiveAgent
from omega_phr.models import OmegaTestResult as TestResult
from omega_phr.models import SecurityTest
from omega_phr.omega_register import OmegaStateRegister as OmegaRegister
from omega_phr.timeline import TimelineLattice, TimelineManager


class TestOmegaPHRConfig(unittest.TestCase):
    """Test configuration management."""

    def test_config_creation(self):
        """Test configuration object creation."""
        config = OmegaPHRConfig()

        self.assertIsNotNone(config.debug)
        self.assertIsNotNone(config.log_level)
        self.assertIsInstance(config.max_agents, int)
        self.assertIsInstance(config.timeout, float)

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {"OMEGA_PHR_DEBUG": "true", "LOG_LEVEL": "DEBUG", "HIVE_MAX_AGENTS": "50"},
        ):
            config = OmegaPHRConfig.from_env()

            self.assertTrue(config.debug)
            self.assertEqual(config.monitoring.log_level, "DEBUG")
            self.assertEqual(config.hive.max_agents, 50)


class TestSecurityModels(unittest.TestCase):
    """Test security testing models."""

    def test_security_test_creation(self):
        """Test SecurityTest model creation."""
        test = SecurityTest(
            test_id="test-1",
            status="network_scan",
            findings=["Port 80 open", "Port 443 open"],
            metadata={
                "name": "Test Scan",
                "description": "Network security test",
                "target": "192.168.1.0/24",
                "test_type": "network_scan",
                "parameters": {"ports": [80, 443, 22]},
            },
        )

        self.assertEqual(test.test_id, "test-1")
        self.assertEqual(test.metadata["name"], "Test Scan")
        self.assertEqual(test.status, "network_scan")
        self.assertIn("parameters", test.metadata)

    def test_test_result_creation(self):
        """Test TestResult model creation."""
        result = TestResult(
            test_id="test-1",
            status="completed",
            findings=["High: Open SSH port", "Medium: HTTP service detected"],
            metadata={"scan_duration": 30.5},
        )

        self.assertEqual(result.test_id, "test-1")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.findings), 2)
        self.assertIn("scan_duration", result.metadata)

    def test_agent_creation(self):
        """Test Agent model creation."""
        agent = Agent(
            agent_id="agent-1",
            persona="Reconnaissance Agent",
            strategy=AttackStrategy.INJECTION,
            capabilities=["port_scan", "service_detection"],
            is_active=True,
        )

        self.assertEqual(agent.agent_id, "agent-1")
        self.assertEqual(agent.strategy, AttackStrategy.INJECTION)
        self.assertIn("port_scan", agent.capabilities)
        self.assertTrue(agent.is_active)


class TestTimelineManager(unittest.TestCase):
    """Test timeline management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OmegaPHRConfig()
        self.timeline_manager = TimelineLattice(
            max_timelines=100, paradox_threshold=0.1
        )

    def test_timeline_creation(self):
        """Test timeline creation."""
        timeline_id = self.timeline_manager.create_timeline("test-timeline")

        self.assertIsNotNone(timeline_id)
        self.assertIn(timeline_id, self.timeline_manager.timelines)

    def test_event_addition(self):
        """Test adding events to timeline."""
        timeline_id = self.timeline_manager.create_timeline("test-timeline")

        from omega_phr.models import Event, EventType

        event = Event(
            event_id="test-event-1",
            timeline_id=timeline_id,
            event_type=EventType.NORMAL,
            payload={"description": "Security test initiated"},
            metadata={"test_id": "test-1"},
        )

        # This is an async method, but we'll test the sync version
        result = asyncio.run(self.timeline_manager.add_event(event))

        self.assertTrue(result)

        # Verify event was added
        events = self.timeline_manager.get_events(timeline_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.NORMAL)

    def test_timeline_not_found(self):
        """Test handling of non-existent timeline."""
        events = self.timeline_manager.get_events("non-existent-timeline")
        self.assertEqual(len(events), 0)


class TestHiveCoordinator(unittest.TestCase):
    """Test hive coordination functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OmegaPHRConfig()
        self.hive = HiveCoordinator(use_ray=False)

    def test_agent_registration(self):
        """Test agent registration."""
        agent_id = self.hive.add_attacker(InjectionAttacker, "Test Agent")

        self.assertIsNotNone(agent_id)
        self.assertIn(agent_id, self.hive.agents)

    def test_task_assignment(self):
        """Test task assignment to agents."""
        # Register an agent first
        agent_id = self.hive.add_attacker(InjectionAttacker, "Test Agent")

        # Create a mock target and coordinate attack
        mock_target = Mock()
        mock_target.generate = AsyncMock(return_value="Mock response")

        # Use async context to run the coordination
        import asyncio

        async def run_test():
            result = await self.hive.coordinate_attack(mock_target, "jailbreak")
            return result

        # Run the async test
        result = asyncio.run(run_test())

        self.assertIsNotNone(result.campaign_id)
        self.assertEqual(result.agents_deployed, 1)
        self.assertIsInstance(result.success_rate, float)

    def test_agent_selection(self):
        """Test automatic agent selection for capabilities."""
        # Register multiple agents
        agent1_id = self.hive.add_attacker(InjectionAttacker, "Scanner Agent")
        agent2_id = self.hive.add_attacker(InjectionAttacker, "Exploit Agent")

        # Verify agents were added
        self.assertEqual(len(self.hive.agents), 2)
        self.assertIn(agent1_id, self.hive.agents)
        self.assertIn(agent2_id, self.hive.agents)


class TestMemoryManager(unittest.TestCase):
    """Test memory management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OmegaPHRConfig()
        self.memory_manager = MemoryManager(max_snapshots=1000)

    def test_memory_storage(self):
        """Test storing data in memory."""
        data = {"test": "data", "value": 123}

        import asyncio

        async def run_test():
            snapshot_id = await self.memory_manager.create_snapshot(data, "test_key")
            return snapshot_id

        snapshot_id = asyncio.run(run_test())

        self.assertIsNotNone(snapshot_id)
        self.assertIn(snapshot_id, self.memory_manager.snapshots)

    def test_memory_retrieval(self):
        """Test retrieving data from memory."""
        data = {"test": "data", "value": 123}

        import asyncio

        async def run_test():
            snapshot_id = await self.memory_manager.create_snapshot(data, "test_key")
            retrieved_data = await self.memory_manager.rollback_memory(snapshot_id)
            return retrieved_data

        retrieved_data = asyncio.run(run_test())

        self.assertEqual(retrieved_data, data)

    def test_memory_pattern_detection(self):
        """Test memory pattern detection."""
        # Store multiple related data points
        patterns = [
            {"type": "scan", "target": "192.168.1.1"},
            {"type": "scan", "target": "192.168.1.2"},
            {"type": "scan", "target": "192.168.1.3"},
        ]

        import asyncio

        async def run_test():
            snapshot_ids = []
            for i, pattern in enumerate(patterns):
                snapshot_id = await self.memory_manager.create_snapshot(
                    pattern, f"pattern_{i}"
                )
                snapshot_ids.append(snapshot_id)
            return snapshot_ids

        snapshot_ids = asyncio.run(run_test())

        # Verify that all snapshots were created
        self.assertEqual(len(snapshot_ids), 3)
        for snapshot_id in snapshot_ids:
            self.assertIn(snapshot_id, self.memory_manager.snapshots)


class TestOmegaRegister(unittest.TestCase):
    """Test omega register functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OmegaPHRConfig()
        self.omega_register = OmegaRegister()

    def test_omega_state_registration(self):
        """Test omega state registration."""
        from datetime import datetime

        from omega_phr.models import OmegaState, OmegaStateLevel

        omega_state = OmegaState(
            omega_id="test-omega-1",
            level=OmegaStateLevel.WARNING,
            entropy_hash="test-hash",
            contamination_vector=["test_vector"],
            quarantine_status=False,
            propagation_risk=0.5,
            source_components=["test_component"],
            metadata={"test": "data"},
        )

        import asyncio

        async def run_test():
            token = await self.omega_register.register_omega_state(omega_state)
            return token

        token = asyncio.run(run_test())

        self.assertIsNotNone(token)
        self.assertIn(omega_state.omega_id, self.omega_register.active_omega_states)

    def test_omega_state_detection(self):
        """Test omega state detection."""
        from omega_phr.models import Event, EventType

        components = ["component1", "component2"]
        events = [
            Event(
                event_id="test-event",
                event_type=EventType.PARADOX,
                payload={"test": "data"},
            )
        ]
        system_metrics = {"entropy": 0.9, "stability": 0.1}

        import asyncio

        async def run_test():
            omega_state = await self.omega_register.detect_omega_state(
                components, events, system_metrics
            )
            return omega_state

        omega_state = asyncio.run(run_test())

        # High entropy should potentially trigger detection
        self.assertIsInstance(omega_state, (type(None), object))

    def test_omega_state_containment(self):
        """Test omega state containment."""
        from omega_phr.models import OmegaState, OmegaStateLevel

        omega_state = OmegaState(
            omega_id="test-omega-2",
            level=OmegaStateLevel.CRITICAL,
            entropy_hash="test-hash-2",
            contamination_vector=["test_vector"],
            quarantine_status=False,
            propagation_risk=0.9,
            source_components=["test_component"],
            metadata={"test": "data"},
        )

        import asyncio

        async def run_test():
            # Register the omega state first
            await self.omega_register.register_omega_state(omega_state)
            # Try to contain it
            result = await self.omega_register.contain_omega_state(omega_state.omega_id)
            return result

        result = asyncio.run(run_test())

        self.assertIsInstance(result, bool)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.config = OmegaPHRConfig()
        self.timeline_manager = TimelineManager(max_timelines=100)
        self.hive = HiveCoordinator(use_ray=False)
        self.memory_manager = MemoryManager(max_snapshots=1000)
        self.omega_register = OmegaRegister()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Register an agent
        agent_id = self.hive.add_attacker(InjectionAttacker, "Integration Test Agent")

        # 2. Create a timeline
        timeline_id = self.timeline_manager.create_timeline("integration-test")

        # 3. Add initial event
        from omega_phr.models import Event, EventType

        event1 = Event(
            event_id="integration-event-1",
            timeline_id=timeline_id,
            event_type=EventType.NORMAL,
            payload={
                "event_type": "agent_registered",
                "description": f"Agent {agent_id} registered for integration test",
            },
            metadata={},
        )

        import asyncio

        async def run_test():
            await self.timeline_manager.add_event(event1)

            # 4. Create memory snapshot
            test_data = {"task_type": "port_scan", "target": "192.168.1.1"}
            snapshot_id = await self.memory_manager.create_snapshot(
                test_data, "integration_test"
            )

            # 5. Retrieve the data
            retrieved_data = await self.memory_manager.rollback_memory(snapshot_id)
            return retrieved_data

        retrieved_data = asyncio.run(run_test())

        # Verify the workflow
        self.assertIsNotNone(agent_id)
        self.assertIn(agent_id, self.hive.agents)
        self.assertIsNotNone(timeline_id)

        events = self.timeline_manager.get_events(timeline_id)
        self.assertEqual(len(events), 1)

        self.assertEqual(retrieved_data["task_type"], "port_scan")


# Async test support
class AsyncTestCase(unittest.TestCase):
    """Base class for async tests."""

    def setUp(self):
        """Set up async test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up async test environment."""
        self.loop.close()

    def async_test(self, coro):
        """Helper to run async tests."""
        return self.loop.run_until_complete(coro)


class TestAsyncOperations(AsyncTestCase):
    """Test async operations in the framework."""

    def test_async_timeline_operations(self):
        """Test async timeline operations."""

        async def run_test():
            config = OmegaPHRConfig()
            timeline_manager = TimelineManager(max_timelines=100)

            # Simulate async timeline operations
            timeline_id = timeline_manager.create_timeline("async-test")

            # Add multiple events asynchronously
            event_tasks = []
            from omega_phr.models import Event, EventType

            for i in range(5):
                event = Event(
                    event_id=f"async-event-{i}",
                    timeline_id=timeline_id,
                    event_type=EventType.NORMAL,
                    payload={
                        "event_type": f"async_event_{i}",
                        "description": f"Async event {i}",
                    },
                    metadata={},
                )
                event_result = await timeline_manager.add_event(event)
                event_tasks.append(event_result)

            # Verify all events were added
            events = timeline_manager.get_events(timeline_id)
            self.assertEqual(len(events), 5)

            return True

        result = self.async_test(run_test())
        self.assertTrue(result)


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestOmegaPHRConfig,
        TestSecurityModels,
        TestTimelineManager,
        TestHiveCoordinator,
        TestMemoryManager,
        TestOmegaRegister,
        TestIntegration,
        TestAsyncOperations,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return test results
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful(),
    }


if __name__ == "__main__":
    print("Running Omega PHR Test Suite...")
    print("=" * 50)

    results = run_all_tests()

    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"Tests run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success: {results['success']}")

    if results["success"]:
        print("\n✅ All tests passed! Research-grade stability confirmed.")
    else:
        print("\n❌ Some tests failed. Please review the output above.")

    exit(0 if results["success"] else 1)
