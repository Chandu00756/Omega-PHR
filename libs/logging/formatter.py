"""
Advanced JSON logging formatter for Ω-PHR framework.

Provides structured logging with consistent format across all services,
including trace correlation, performance metrics, and security audit trails.
"""

import json
import logging
import threading
import traceback
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Enterprise-grade JSON log formatter with security and observability features."""

    def __init__(
        self,
        service_name: str = "omega-phr",
        version: str = "1.0.0",
        environment: str = "development",
        include_trace: bool = True,
        sensitive_fields: set[str] | None = None,
    ):
        """
        Initialize JSON formatter.

        Args:
            service_name: Name of the service generating logs
            version: Service version
            environment: Deployment environment
            include_trace: Whether to include stack traces
            sensitive_fields: Field names to redact from logs
        """
        super().__init__()
        self.service_name = service_name
        self.version = version
        self.environment = environment
        self.include_trace = include_trace
        self.sensitive_fields = sensitive_fields or {
            "password",
            "token",
            "secret",
            "key",
            "credential",
            "authorization",
            "auth",
            "jwt",
            "api_key",
        }

        # Thread-local storage for request context
        self._local = threading.local()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_entry = {
            "timestamp": self._get_timestamp(),
            "level": record.levelname,
            "service": self.service_name,
            "version": self.version,
            "environment": self.environment,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add trace ID if available
        trace_id = self._get_trace_id()
        if trace_id:
            log_entry["trace_id"] = trace_id

        # Add span ID if available
        span_id = self._get_span_id()
        if span_id:
            log_entry["span_id"] = span_id

        # Add performance metrics
        if hasattr(record, "duration"):
            log_entry["duration_ms"] = record.duration

        if hasattr(record, "memory_usage"):
            log_entry["memory_mb"] = record.memory_usage

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
            }:
                extra_fields[key] = self._sanitize_value(key, value)

        if extra_fields:
            log_entry["extra"] = extra_fields

        # Add exception information
        if record.exc_info and self.include_trace:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_entry["stack_trace"] = record.stack_info

        # Add security context
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id

        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id

        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        # Add operational metadata
        if hasattr(record, "component"):
            log_entry["component"] = record.component

        if hasattr(record, "operation"):
            log_entry["operation"] = record.operation

        if hasattr(record, "status"):
            log_entry["status"] = record.status

        # Serialize to JSON
        return json.dumps(log_entry, default=self._json_serializer, ensure_ascii=False)

    def _get_timestamp(self) -> str:
        """Get ISO 8601 timestamp with microseconds."""
        return datetime.now(UTC).isoformat()

    def _get_trace_id(self) -> str | None:
        """Get current trace ID from context."""
        return getattr(self._local, "trace_id", None)

    def _get_span_id(self) -> str | None:
        """Get current span ID from context."""
        return getattr(self._local, "span_id", None)

    def set_trace_context(self, trace_id: str, span_id: str | None = None) -> None:
        """Set trace context for current thread."""
        self._local.trace_id = trace_id
        if span_id:
            self._local.span_id = span_id

    def clear_trace_context(self) -> None:
        """Clear trace context for current thread."""
        self._local.trace_id = None
        self._local.span_id = None

    def _sanitize_value(self, key: str, value: Any) -> Any:
        """Sanitize sensitive values in logs."""
        if isinstance(key, str) and any(
            sensitive in key.lower() for sensitive in self.sensitive_fields
        ):
            if isinstance(value, str) and len(value) > 8:
                return f"{value[:4]}****{value[-4:]}"
            else:
                return "****"

        # Recursively sanitize dictionaries
        if isinstance(value, dict):
            return {k: self._sanitize_value(k, v) for k, v in value.items()}

        # Recursively sanitize lists
        if isinstance(value, list):
            return [
                self._sanitize_value(f"{key}[{i}]", item)
                for i, item in enumerate(value)
            ]

        return value

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Exception):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)


class PerformanceLogFormatter(JSONFormatter):
    """Specialized formatter for performance logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format performance log with additional metrics."""
        # Add performance-specific fields
        if not hasattr(record, "performance_metrics"):
            record.performance_metrics = {}

        # Add system resource usage
        try:
            import psutil

            process = psutil.Process()
            record.performance_metrics.update(
                {
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "open_fds": (
                        process.num_fds() if hasattr(process, "num_fds") else None
                    ),
                    "threads": process.num_threads(),
                }
            )
        except ImportError:
            pass

        return super().format(record)


class AuditLogFormatter(JSONFormatter):
    """Specialized formatter for security audit logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format audit log with security context."""
        # Add audit-specific fields
        if not hasattr(record, "audit"):
            record.audit = True

        if not hasattr(record, "severity"):
            if record.levelno >= logging.ERROR:
                record.severity = "HIGH"
            elif record.levelno >= logging.WARNING:
                record.severity = "MEDIUM"
            else:
                record.severity = "LOW"

        # Add compliance tags
        if hasattr(record, "compliance_tags"):
            record.compliance = record.compliance_tags

        return super().format(record)


class StructuredLogAdapter(logging.LoggerAdapter):
    """Logger adapter that adds structured context to all log messages."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None):
        """Initialize adapter with context."""
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Process log message with additional context."""
        if "extra" in kwargs:
            kwargs["extra"].update(self.extra)
        else:
            kwargs["extra"] = self.extra.copy()

        return msg, kwargs

    def with_context(self, **context: Any) -> "StructuredLogAdapter":
        """Create new adapter with additional context."""
        new_extra = self.extra.copy()
        new_extra.update(context)
        return StructuredLogAdapter(self.logger, new_extra)

    def performance(self, msg: str, duration: float, **kwargs: Any) -> None:
        """Log performance metric."""
        kwargs.setdefault("extra", {})
        kwargs["extra"]["duration"] = duration
        kwargs["extra"]["metric_type"] = "performance"
        self.info(msg, **kwargs)

    def audit(self, msg: str, action: str, resource: str, **kwargs: Any) -> None:
        """Log security audit event."""
        kwargs.setdefault("extra", {})
        kwargs["extra"].update(
            {
                "audit": True,
                "action": action,
                "resource": resource,
                "metric_type": "audit",
            }
        )
        self.info(msg, **kwargs)

    def security(self, msg: str, threat_level: str = "LOW", **kwargs: Any) -> None:
        """Log security event."""
        kwargs.setdefault("extra", {})
        kwargs["extra"].update(
            {"security": True, "threat_level": threat_level, "metric_type": "security"}
        )

        if threat_level in ("HIGH", "CRITICAL"):
            self.error(msg, **kwargs)
        elif threat_level == "MEDIUM":
            self.warning(msg, **kwargs)
        else:
            self.info(msg, **kwargs)


def configure_logging(
    service_name: str,
    log_level: str = "INFO",
    environment: str = "development",
    enable_console: bool = True,
    enable_file: bool = False,
    log_file: str | None = None,
    enable_audit: bool = True,
    enable_performance: bool = True,
) -> None:
    """
    Configure application logging with research standards.

    Args:
        service_name: Name of the service
        log_level: Logging level
        environment: Deployment environment
        enable_console: Enable console logging
        enable_file: Enable file logging
        log_file: Log file path
        enable_audit: Enable audit logging
        enable_performance: Enable performance logging
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_formatter = JSONFormatter(
            service_name=service_name, environment=environment
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if enable_file and log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = JSONFormatter(
            service_name=service_name, environment=environment
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Audit logger
    if enable_audit:
        audit_logger = logging.getLogger("audit")
        audit_handler = logging.StreamHandler()
        audit_formatter = AuditLogFormatter(
            service_name=service_name, environment=environment
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        audit_logger.propagate = False

    # Performance logger
    if enable_performance:
        perf_logger = logging.getLogger("performance")
        perf_handler = logging.StreamHandler()
        perf_formatter = PerformanceLogFormatter(
            service_name=service_name, environment=environment
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False


def get_logger(name: str, **context: Any) -> StructuredLogAdapter:
    """Get a structured logger with context."""
    logger = logging.getLogger(name)
    return StructuredLogAdapter(logger, context)
