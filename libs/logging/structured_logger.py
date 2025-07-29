"""
Structured logger for Ω-PHR framework.

Provides centralized logging configuration with trace correlation,
performance monitoring, and security audit capabilities.
"""

import functools
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from .formatter import configure_logging, get_logger


class StructuredLogger:
    """Centralized structured logger with context management."""

    def __init__(self, service_name: str = "omega-phr"):
        """Initialize structured logger."""
        self.service_name = service_name
        self._initialized = False

    def initialize(
        self,
        log_level: str = "INFO",
        environment: str = "development",
        enable_console: bool = True,
        enable_file: bool = False,
        log_file: str | None = None,
    ) -> None:
        """Initialize logging configuration."""
        if self._initialized:
            return

        configure_logging(
            service_name=self.service_name,
            log_level=log_level,
            environment=environment,
            enable_console=enable_console,
            enable_file=enable_file,
            log_file=log_file,
        )

        self._initialized = True

    def get_logger(self, name: str, **context: Any) -> Any:
        """Get logger with context."""
        return get_logger(name, **context)

    @contextmanager
    def trace_context(self, operation: str, **context: Any):
        """Context manager for distributed tracing."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        logger = self.get_logger(
            "trace", trace_id=trace_id, span_id=span_id, operation=operation, **context
        )

        start_time = time.time()
        logger.info(f"Starting operation: {operation}")

        try:
            yield logger
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"Operation failed: {operation}", duration=duration, error=str(e)
            )
            raise
        else:
            duration = (time.time() - start_time) * 1000
            logger.info(f"Operation completed: {operation}", duration=duration)

    def performance_monitor(self, operation: str):
        """Decorator for performance monitoring."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.trace_context(operation, function=func.__name__):
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.trace_context(operation, function=func.__name__):
                    return func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def audit_log(
        self, action: str, resource: str, user_id: str | None = None, **context: Any
    ) -> None:
        """Log audit event."""
        logger = logging.getLogger("audit")
        extra = {"action": action, "resource": resource, "audit": True, **context}
        if user_id:
            extra["user_id"] = user_id

        logger.info(f"Audit: {action} on {resource}", extra=extra)

    def security_alert(
        self, threat_type: str, severity: str = "LOW", **context: Any
    ) -> None:
        """Log security alert."""
        logger = logging.getLogger("security")
        extra = {
            "threat_type": threat_type,
            "severity": severity,
            "security_alert": True,
            **context,
        }

        if severity in ("HIGH", "CRITICAL"):
            logger.error(f"Security Alert: {threat_type}", extra=extra)
        elif severity == "MEDIUM":
            logger.warning(f"Security Alert: {threat_type}", extra=extra)
        else:
            logger.info(f"Security Alert: {threat_type}", extra=extra)


# Global instance
_structured_logger = None


def get_structured_logger(service_name: str = "omega-phr") -> StructuredLogger:
    """Get global structured logger instance."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger(service_name)
    return _structured_logger


def initialize_logging(**kwargs) -> None:
    """Initialize global logging configuration."""
    logger = get_structured_logger()
    logger.initialize(**kwargs)
