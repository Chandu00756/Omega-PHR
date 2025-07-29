"""
Minimal working tests for CI.
"""

import os
import sys
import unittest

# Import modules from the framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omega_phr.config import OmegaPHRConfig
from omega_phr.exceptions import OmegaPHRError, TemporalParadoxError


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests that should pass."""

    def test_imports_work(self):
        """Test that basic imports work."""

        # If we get here, imports worked
        self.assertTrue(True)

    def test_config_creation(self):
        """Test configuration object creation."""
        config = OmegaPHRConfig()
        self.assertIsNotNone(config)

    def test_exception_inheritance(self):
        """Test exception hierarchy."""
        self.assertTrue(issubclass(TemporalParadoxError, OmegaPHRError))
        self.assertTrue(issubclass(OmegaPHRError, Exception))

    def test_exception_creation(self):
        """Test exception creation."""
        exc = OmegaPHRError("Test error", "TEST_ERROR", {"context": "test"})
        self.assertEqual(exc.message, "Test error")
        self.assertEqual(exc.error_code, "TEST_ERROR")

    def test_temporal_paradox_error(self):
        """Test TemporalParadoxError creation."""
        exc = TemporalParadoxError("Paradox detected", "timeline-001", "causal_loop")
        self.assertEqual(exc.timeline_id, "timeline-001")
        self.assertEqual(exc.paradox_type, "causal_loop")


if __name__ == "__main__":
    unittest.main()
