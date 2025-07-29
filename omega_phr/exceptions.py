"""
Custom exceptions for the Ω-PHR framework.

This module defines all custom exceptions used throughout the
Omega-Paradox Hive Recursion framework.
"""


class OmegaPHRException(Exception):
    def __init__(
        self, message: str, error_code: str = "", context: dict | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.error_code = error_code
        self.context = context or {}


class TemporalParadoxError(OmegaPHRException):
    """Raised when temporal paradoxes are detected or created."""

    def __init__(
        self, message: str, timeline_id: str = "", paradox_type: str = ""
    ) -> None:
        super().__init__(message, "TEMPORAL_PARADOX")
        self.timeline_id = timeline_id
        self.paradox_type = paradox_type


class HiveCoordinationError(OmegaPHRException):
    """Raised when hive coordination fails or produces unexpected results."""

    def __init__(
        self, message: str, agent_count: int = 0, failure_type: str = ""
    ) -> None:
        super().__init__(message, "HIVE_COORDINATION")
        self.agent_count = agent_count
        self.failure_type = failure_type


class MemoryInversionError(OmegaPHRException):
    """Raised when memory inversion operations fail or cause corruption."""

    def __init__(
        self, message: str, state_id: str = "", corruption_level: float = 0.0
    ) -> None:
        super().__init__(message, "MEMORY_INVERSION")
        self.state_id = state_id
        self.corruption_level = corruption_level


class RecursiveLoopError(OmegaPHRException):
    """Raised when recursive loops cannot be contained or analyzed."""

    def __init__(self, message: str, loop_id: str = "", depth: int = 0) -> None:
        super().__init__(message, "RECURSIVE_LOOP")
        self.loop_id = loop_id
        self.depth = depth


class OmegaStateError(OmegaPHRException):
    """Raised when Omega states cannot be properly contained or resolved."""

    def __init__(
        self, message: str, omega_id: str = "", entropy_level: float = 0.0
    ) -> None:
        super().__init__(message, "OMEGA_STATE")
        self.omega_id = omega_id
        self.entropy_level = entropy_level


class ConfigurationError(OmegaPHRException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str = "") -> None:
        super().__init__(message, "CONFIGURATION")
        self.config_key = config_key


class ValidationError(OmegaPHRException):
    """Raised when data validation fails."""

    def __init__(self, message: str, field_name: str = "", value: str = "") -> None:
        super().__init__(message, "VALIDATION")
        self.field_name = field_name
        self.value = value


class ContainmentError(OmegaPHRException):
    """Raised when containment protocols fail."""

    def __init__(
        self, message: str, containment_type: str = "", risk_level: float = 0.0
    ) -> None:
        super().__init__(message, "CONTAINMENT")
        self.containment_type = containment_type
        self.risk_level = risk_level


class ServiceUnavailableError(OmegaPHRException):
    """Raised when required services are unavailable."""

    def __init__(
        self, message: str, service_name: str = "", retry_after: int = 0
    ) -> None:
        super().__init__(message, "SERVICE_UNAVAILABLE")
        self.service_name = service_name
        self.retry_after = retry_after


# Backward compatibility aliases for test imports
ParadoxDetectedException = TemporalParadoxError
HiveSwarmException = HiveCoordinationError
MemoryInversionException = MemoryInversionError
InfiniteLoopException = RecursiveLoopError
OmegaStateException = OmegaStateError
