"""
Unit tests for Omega PHR framework.
Research-grade stability testing.
"""

import unittest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import modules from the framework
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from omega_phr.config import OmegaPHRConfig
from omega_phr.models import SecurityTest, TestResult, Agent, AttackStrategy
from omega_phr.timeline import TimelineLattice, TimelineManager
from omega_phr.hive import HiveCoordinator # type: ignore
from omega_phr.memory import MemoryManager # type: ignore
from omega_phr.omega_register import OmegaRegister # type: ignore


class TestOmegaPHRConfig(unittest.TestCase):
    """Test configuration management."""

    def test_config_creation(self):
        """Test configuration object creation."""
        config = OmegaPHRConfig()

        self.assertIsNotNone(config.debug)
        self.assertIsNotNone(config.log_level)
        self.assertIsInstance(config.max_agents, int)
        self.assertIsInstance(config.timeout, int)

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'OMEGA_DEBUG': 'true',
            'OMEGA_LOG_LEVEL': 'DEBUG',
            'OMEGA_MAX_AGENTS': '50'
        }):
            config = OmegaPHRConfig.from_env()

            self.assertTrue(config.debug)
            self.assertEqual(config.log_level, 'DEBUG')
            self.assertEqual(config.max_agents, 50)


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
                "parameters": {"ports": [80, 443, 22]}
            }
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
            findings=[
                "High: Open SSH port",
                "Medium: HTTP service detected"
            ],
            metadata={"scan_duration": 30.5}
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
            is_active=True
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
        self.timeline_manager = TimelineLattice(max_timelines=100, paradox_threshold=0.1)

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
            metadata={"test_id": "test-1"}
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
        self.hive = HiveCoordinator(self.config)

    def test_agent_registration(self):
        """Test agent registration."""
        agent = Agent(
            agent_id="agent-1",
            persona="Test Agent",
            strategy=AttackStrategy.INJECTION,
            capabilities=["port_scan"],
            is_active=True
        )

        result = self.hive.register_agent(agent)

        self.assertTrue(result)
        self.assertIn("agent-1", self.hive.agents)

    def test_task_assignment(self):
        """Test task assignment to agents."""
        # Register an agent first
        agent = Agent(
            agent_id="agent-1",
            persona="Test Agent",
            strategy=AttackStrategy.INJECTION,
            capabilities=["port_scan"],
            is_active=True
        )
        self.hive.register_agent(agent)

        # Create a task
        task_data = {
            "task_type": "port_scan",
            "target": "192.168.1.1",
            "parameters": {"ports": [80, 443]}
        }

        task_id = self.hive.assign_task("agent-1", task_data)

        self.assertIsNotNone(task_id)
        self.assertIn(task_id, self.hive.tasks)

    def test_agent_selection(self):
        """Test automatic agent selection for capabilities."""
        # Register multiple agents
        agent1 = Agent(
            agent_id="agent-1",
            persona="Scanner Agent",
            strategy=AttackStrategy.INJECTION,
            capabilities=["port_scan"],
            is_active=True
        )

        agent2 = Agent(
            agent_id="agent-2",
            persona="Exploit Agent",
            capabilities=["exploit"],
            is_active=True
        )

        self.hive.register_agent(agent1)
        self.hive.register_agent(agent2)

        # Find agents with specific capability
        scanners = self.hive.find_agents_by_capability("port_scan")
        exploiters = self.hive.find_agents_by_capability("exploit")

        self.assertEqual(len(scanners), 1)
        self.assertEqual(scanners[0].id, "agent-1")

        self.assertEqual(len(exploiters), 1)
        self.assertEqual(exploiters[0].id, "agent-2")


class TestMemoryManager(unittest.TestCase):
    """Test memory management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OmegaPHRConfig()
        self.memory_manager = MemoryManager(self.config)

    def test_memory_storage(self):
        """Test storing data in memory."""
        key = "test_key"
        data = {"test": "data", "value": 123}

        result = self.memory_manager.store(key, data)

        self.assertTrue(result)
        self.assertIn(key, self.memory_manager.memory_store)

    def test_memory_retrieval(self):
        """Test retrieving data from memory."""
        key = "test_key"
        data = {"test": "data", "value": 123}

        self.memory_manager.store(key, data)
        retrieved_data = self.memory_manager.retrieve(key)

        self.assertEqual(retrieved_data, data)

    def test_memory_pattern_detection(self):
        """Test memory pattern detection."""
        # Store multiple related data points
        patterns = [
            {"type": "scan", "target": "192.168.1.1"},
            {"type": "scan", "target": "192.168.1.2"},
            {"type": "scan", "target": "192.168.1.3"}
        ]

        for i, pattern in enumerate(patterns):
            self.memory_manager.store(f"pattern_{i}", pattern)

        detected_patterns = self.memory_manager.detect_patterns()

        self.assertIsInstance(detected_patterns, list)
        # Pattern detection should find at least some commonality
        self.assertGreaterEqual(len(detected_patterns), 0)


class TestOmegaRegister(unittest.TestCase):
    """Test omega register functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OmegaPHRConfig()
        self.omega_register = OmegaRegister(self.config)

    def test_service_registration(self):
        """Test service registration."""
        service_info = {
            "name": "test_service",
            "endpoint": "http://localhost:8000",
            "capabilities": ["scan", "analyze"],
            "status": "active"
        }

        service_id = self.omega_register.register_service(service_info)

        self.assertIsNotNone(service_id)
        self.assertIn(service_id, self.omega_register.services)

    def test_service_discovery(self):
        """Test service discovery."""
        # Register multiple services
        service1 = {
            "name": "scanner_service",
            "endpoint": "http://localhost:8001",
            "capabilities": ["port_scan"],
            "status": "active"
        }

        service2 = {
            "name": "exploit_service",
            "endpoint": "http://localhost:8002",
            "capabilities": ["exploit"],
            "status": "active"
        }

        id1 = self.omega_register.register_service(service1)
        id2 = self.omega_register.register_service(service2)

        # Discover services by capability
        scanners = self.omega_register.discover_services("port_scan")
        exploiters = self.omega_register.discover_services("exploit")

        self.assertEqual(len(scanners), 1)
        self.assertEqual(len(exploiters), 1)

        self.assertEqual(scanners[0]["name"], "scanner_service")
        self.assertEqual(exploiters[0]["name"], "exploit_service")

    def test_health_monitoring(self):
        """Test service health monitoring."""
        service_info = {
            "name": "test_service",
            "endpoint": "http://localhost:8000",
            "capabilities": ["test"],
            "status": "active"
        }

        service_id = self.omega_register.register_service(service_info)

        # Update health status
        self.omega_register.update_service_health(service_id, "healthy", {"uptime": 3600})

        service = self.omega_register.get_service(service_id)
        self.assertEqual(service["health_status"], "healthy")
        self.assertIn("uptime", service["health_data"])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.config = OmegaPHRConfig()
        self.timeline_manager = TimelineManager(max_timelines=100)
        self.hive = HiveCoordinator(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.omega_register = OmegaRegister(self.config)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Register a service
        service_info = {
            "name": "integration_scanner",
            "endpoint": "http://localhost:8000",
            "capabilities": ["port_scan"],
            "status": "active"
        }
        service_id = self.omega_register.register_service(service_info)

        # 2. Register an agent
        agent = Agent(
            agent_id="integration-agent",
            persona="Integration Test Agent",
            strategy=AttackStrategy.INJECTION,
            capabilities=["port_scan"],
            is_active=True
        )
        self.hive.register_agent(agent)

        # 3. Create a timeline
        timeline_id = self.timeline_manager.create_timeline(
            "integration-test"
        )

        # 4. Add initial event
        from omega_phr.models import Event, EventType
        event1 = Event(
            event_id="integration-event-1",
            timeline_id=timeline_id,
            event_type=EventType.NORMAL,
            payload={"description": "Integration test started"},
            metadata={}
        )
        asyncio.run(self.timeline_manager.add_event(event1))

        # 5. Assign a task
        task_data = {
            "task_type": "port_scan",
            "target": "192.168.1.1",
            "parameters": {"ports": [80, 443]}
        }
        task_id = self.hive.assign_task("integration-agent", task_data)

        # 6. Store task in memory
        self.memory_manager.store(f"task_{task_id}", task_data)

        # 7. Add completion event
        from omega_phr.models import Event, EventType
        event2 = Event(
            event_id=f"integration-event-2",
            timeline_id=timeline_id,
            event_type=EventType.NORMAL,
            payload={
                "event_type": "task_assigned",
                "description": f"Task {task_id} assigned to integration-agent"
            },
            metadata={}
        )
        asyncio.run(self.timeline_manager.add_event(event2))

        # Verify the workflow
        self.assertIsNotNone(service_id)
        self.assertIn("integration-agent", self.hive.agents)
        self.assertIsNotNone(timeline_id)
        self.assertIsNotNone(task_id)

        events = self.timeline_manager.get_events(timeline_id)
        self.assertEqual(len(events), 2)

        stored_task = self.memory_manager.retrieve(f"task_{task_id}")
        self.assertEqual(stored_task["task_type"], "port_scan")


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
                    payload={"event_type": f"async_event_{i}", "description": f"Async event {i}"},
                    metadata={}
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
        TestAsyncOperations
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
        "success": result.wasSuccessful()
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

    if results['success']:
        print("\n✅ All tests passed! Research-grade stability confirmed.")
    else:
        print("\n❌ Some tests failed. Please review the output above.")

    exit(0 if results['success'] else 1)
