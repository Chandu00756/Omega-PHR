"""
Telemetry Exporter Service

Advanced telemetry collection and export service for AI security testing.
Provides research-grade stability for comprehensive system monitoring.
"""

import asyncio
import base64
import gzip
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of telemetry metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"
    ELASTIC = "elastic"


class SeverityLevel(Enum):
    """Severity levels for telemetry events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TelemetryMetric:
    """Represents a telemetry metric."""

    id: str
    name: str
    metric_type: MetricType
    value: float
    unit: str
    labels: Dict[str, str]
    timestamp: int
    source: str
    metadata: Dict[str, Any]


@dataclass
class TelemetryEvent:
    """Represents a telemetry event."""

    id: str
    event_type: str
    severity: SeverityLevel
    message: str
    source: str
    timestamp: int
    duration_ms: Optional[int]
    properties: Dict[str, Any]
    trace_id: Optional[str]
    span_id: Optional[str]


@dataclass
class ExportTarget:
    """Configuration for export target."""

    id: str
    name: str
    format: ExportFormat
    endpoint: str
    credentials: Dict[str, str]
    filters: Dict[str, Any]
    enabled: bool


class MetricsAggregator:
    """Aggregates metrics for efficient export."""

    def __init__(self):
        self.aggregation_window = 60  # seconds
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}

    async def add_metric(self, metric: TelemetryMetric):
        """Add a metric to aggregation."""
        key = f"{metric.source}:{metric.name}"

        if metric.metric_type == MetricType.COUNTER:
            self.counters[key] = self.counters.get(key, 0) + metric.value
        elif metric.metric_type == MetricType.GAUGE:
            self.gauges[key] = metric.value
        elif metric.metric_type == MetricType.HISTOGRAM:
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(metric.value)
        elif metric.metric_type == MetricType.TIMER:
            if key not in self.timers:
                self.timers[key] = []
            self.timers[key].append(metric.value)

    async def get_aggregated_data(self) -> Dict[str, Any]:
        """Get aggregated metrics data."""
        # Calculate histogram statistics
        histogram_stats = {}
        for key, values in self.histograms.items():
            if values:
                histogram_stats[key] = {
                    "count": len(values),
                    "sum": sum(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": self._percentile(values, 50),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99),
                }

        # Calculate timer statistics
        timer_stats = {}
        for key, values in self.timers.items():
            if values:
                timer_stats[key] = {
                    "count": len(values),
                    "total_ms": sum(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "avg_ms": sum(values) / len(values),
                    "p50_ms": self._percentile(values, 50),
                    "p95_ms": self._percentile(values, 95),
                    "p99_ms": self._percentile(values, 99),
                }

        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": histogram_stats,
            "timers": timer_stats,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = max(0, min(index, len(sorted_values) - 1))
        return sorted_values[index]

    async def reset(self):
        """Reset aggregation data."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()


class TelemetryBuffer:
    """Buffer for storing telemetry data before export."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.metrics: List[TelemetryMetric] = []
        self.events: List[TelemetryEvent] = []

    async def add_metric(self, metric: TelemetryMetric):
        """Add metric to buffer."""
        self.metrics.append(metric)
        if len(self.metrics) > self.max_size:
            # Remove oldest metrics
            self.metrics = self.metrics[-self.max_size :]

    async def add_event(self, event: TelemetryEvent):
        """Add event to buffer."""
        self.events.append(event)
        if len(self.events) > self.max_size:
            # Remove oldest events
            self.events = self.events[-self.max_size :]

    async def get_metrics(self, limit: Optional[int] = None) -> List[TelemetryMetric]:
        """Get metrics from buffer."""
        if limit:
            return self.metrics[-limit:]
        return self.metrics.copy()

    async def get_events(self, limit: Optional[int] = None) -> List[TelemetryEvent]:
        """Get events from buffer."""
        if limit:
            return self.events[-limit:]
        return self.events.copy()

    async def clear(self):
        """Clear buffer."""
        self.metrics.clear()
        self.events.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "metrics_count": len(self.metrics),
            "events_count": len(self.events),
            "memory_usage_mb": (len(self.metrics) + len(self.events))
            * 0.001,  # Rough estimate
            "oldest_metric_timestamp": (
                self.metrics[0].timestamp if self.metrics else None
            ),
            "newest_metric_timestamp": (
                self.metrics[-1].timestamp if self.metrics else None
            ),
            "oldest_event_timestamp": self.events[0].timestamp if self.events else None,
            "newest_event_timestamp": (
                self.events[-1].timestamp if self.events else None
            ),
        }


class ExportFormatter:
    """Formats telemetry data for different export targets."""

    async def format_prometheus(self, data: Dict[str, Any]) -> str:
        """Format data for Prometheus export."""
        lines = []
        timestamp = int(time.time() * 1000)

        # Counters
        for key, value in data.get("counters", {}).items():
            metric_name = key.replace(":", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name}_total counter")
            lines.append(f"{metric_name}_total {value} {timestamp}")

        # Gauges
        for key, value in data.get("gauges", {}).items():
            metric_name = key.replace(":", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value} {timestamp}")

        # Histograms
        for key, stats in data.get("histograms", {}).items():
            metric_name = key.replace(":", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} histogram")
            lines.append(f"{metric_name}_count {stats['count']} {timestamp}")
            lines.append(f"{metric_name}_sum {stats['sum']} {timestamp}")
            lines.append(
                f"{metric_name}_bucket{{le=\"+Inf\"}} {stats['count']} {timestamp}"
            )

        return "\n".join(lines)

    async def format_influxdb(self, data: Dict[str, Any]) -> str:
        """Format data for InfluxDB line protocol."""
        lines = []
        timestamp = int(time.time() * 1000000000)  # InfluxDB uses nanoseconds

        # Counters
        for key, value in data.get("counters", {}).items():
            source, name = key.split(":", 1) if ":" in key else ("unknown", key)
            lines.append(f"counters,source={source} {name}={value} {timestamp}")

        # Gauges
        for key, value in data.get("gauges", {}).items():
            source, name = key.split(":", 1) if ":" in key else ("unknown", key)
            lines.append(f"gauges,source={source} {name}={value} {timestamp}")

        # Histograms
        for key, stats in data.get("histograms", {}).items():
            source, name = key.split(":", 1) if ":" in key else ("unknown", key)
            for stat_name, stat_value in stats.items():
                lines.append(
                    f"histograms,source={source},metric={name} {stat_name}={stat_value} {timestamp}"
                )

        return "\n".join(lines)

    async def format_json(self, data: Dict[str, Any]) -> str:
        """Format data as JSON."""
        return json.dumps(data, indent=2)

    async def format_csv(
        self, metrics: List[TelemetryMetric], events: List[TelemetryEvent]
    ) -> str:
        """Format data as CSV."""
        lines = []

        # Metrics CSV
        lines.append("# METRICS")
        lines.append("timestamp,source,name,type,value,unit,labels")
        for metric in metrics:
            labels_str = json.dumps(metric.labels)
            lines.append(
                f"{metric.timestamp},{metric.source},{metric.name},"
                f"{metric.metric_type.value},{metric.value},{metric.unit},{labels_str}"
            )

        lines.append("")
        lines.append("# EVENTS")
        lines.append(
            "timestamp,source,type,severity,message,duration_ms,trace_id,span_id"
        )
        for event in events:
            lines.append(
                f"{event.timestamp},{event.source},{event.event_type},"
                f"{event.severity.value},{event.message},{event.duration_ms or ''},"
                f"{event.trace_id or ''},{event.span_id or ''}"
            )

        return "\n".join(lines)

    async def format_elastic(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format data for Elasticsearch."""
        documents = []
        timestamp = datetime.now().isoformat()

        # Convert aggregated data to documents
        for metric_type in ["counters", "gauges"]:
            for key, value in data.get(metric_type, {}).items():
                source, name = key.split(":", 1) if ":" in key else ("unknown", key)
                doc = {
                    "@timestamp": timestamp,
                    "metric_type": metric_type[:-1],  # Remove 's'
                    "source": source,
                    "name": name,
                    "value": value,
                }
                documents.append(doc)

        # Histogram data
        for key, stats in data.get("histograms", {}).items():
            source, name = key.split(":", 1) if ":" in key else ("unknown", key)
            doc = {
                "@timestamp": timestamp,
                "metric_type": "histogram",
                "source": source,
                "name": name,
                "stats": stats,
            }
            documents.append(doc)

        return documents


class TelemetryExporter:
    """Exports telemetry data to various targets."""

    def __init__(self):
        self.targets: Dict[str, ExportTarget] = {}
        self.formatter = ExportFormatter()
        self.export_queue: List[Dict[str, Any]] = []

    async def add_target(self, target: ExportTarget):
        """Add an export target."""
        self.targets[target.id] = target
        logger.info(f"Added export target: {target.name} ({target.format.value})")

    async def remove_target(self, target_id: str):
        """Remove an export target."""
        if target_id in self.targets:
            del self.targets[target_id]
            logger.info(f"Removed export target: {target_id}")

    async def export_data(
        self,
        aggregated_data: Dict[str, Any],
        metrics: List[TelemetryMetric],
        events: List[TelemetryEvent],
    ):
        """Export data to all configured targets."""
        for target in self.targets.values():
            if not target.enabled:
                continue

            try:
                await self._export_to_target(target, aggregated_data, metrics, events)
            except Exception as e:
                logger.error(f"Failed to export to target {target.name}: {e}")

    async def _export_to_target(
        self,
        target: ExportTarget,
        aggregated_data: Dict[str, Any],
        metrics: List[TelemetryMetric],
        events: List[TelemetryEvent],
    ):
        """Export data to a specific target."""
        # Apply filters
        filtered_metrics = await self._apply_filters(metrics, target.filters)
        filtered_events = await self._apply_filters(events, target.filters)

        # Format data based on target format
        if target.format == ExportFormat.PROMETHEUS:
            formatted_data = await self.formatter.format_prometheus(aggregated_data)
        elif target.format == ExportFormat.INFLUXDB:
            formatted_data = await self.formatter.format_influxdb(aggregated_data)
        elif target.format == ExportFormat.JSON:
            export_data = {
                "aggregated": aggregated_data,
                "metrics": [asdict(m) for m in filtered_metrics],
                "events": [asdict(e) for e in filtered_events],
            }
            formatted_data = await self.formatter.format_json(export_data)
        elif target.format == ExportFormat.CSV:
            formatted_data = await self.formatter.format_csv(
                filtered_metrics, filtered_events
            )
        elif target.format == ExportFormat.ELASTIC:
            formatted_data = await self.formatter.format_elastic(aggregated_data)
        else:
            logger.warning(f"Unsupported export format: {target.format}")
            return

        # For demo purposes, just log the export
        logger.info(
            f"Exported {len(filtered_metrics)} metrics and {len(filtered_events)} events "
            f"to {target.name} in {target.format.value} format"
        )

        # In a real implementation, you would send the data to the actual endpoint
        # await self._send_to_endpoint(target.endpoint, formatted_data, target.credentials)

    async def _apply_filters(self, items: List, filters: Dict[str, Any]) -> List:
        """Apply filters to telemetry items."""
        if not filters:
            return items

        filtered_items = []
        for item in items:
            # Simple filter implementation - match on source
            if "source" in filters:
                if hasattr(item, "source") and item.source in filters["source"]:
                    filtered_items.append(item)
            else:
                filtered_items.append(item)

        return filtered_items


class TelemetryExporterService:
    """Main Telemetry Exporter Service."""

    def __init__(self):
        self.buffer = TelemetryBuffer()
        self.aggregator = MetricsAggregator()
        self.exporter = TelemetryExporter()
        self.running = False
        self.export_interval = 60  # seconds

    async def start(self):
        """Start the telemetry exporter service."""
        self.running = True
        logger.info(
            "Telemetry Exporter Service started - Research-grade stability enabled"
        )

        # Start background export
        asyncio.create_task(self._background_export())

    async def stop(self):
        """Stop the telemetry exporter service."""
        self.running = False
        logger.info("Telemetry Exporter Service stopped")

    async def submit_metric(self, metric_data: Dict[str, Any]) -> str:
        """Submit a telemetry metric."""
        metric = TelemetryMetric(
            id=str(uuid.uuid4()),
            name=metric_data["name"],
            metric_type=MetricType(metric_data["type"]),
            value=metric_data["value"],
            unit=metric_data.get("unit", ""),
            labels=metric_data.get("labels", {}),
            timestamp=metric_data.get(
                "timestamp", int(datetime.now().timestamp() * 1000)
            ),
            source=metric_data.get("source", "unknown"),
            metadata=metric_data.get("metadata", {}),
        )

        await self.buffer.add_metric(metric)
        await self.aggregator.add_metric(metric)

        logger.debug(f"Metric submitted: {metric.name} = {metric.value}")
        return metric.id

    async def submit_event(self, event_data: Dict[str, Any]) -> str:
        """Submit a telemetry event."""
        event = TelemetryEvent(
            id=str(uuid.uuid4()),
            event_type=event_data["type"],
            severity=SeverityLevel(event_data.get("severity", "info")),
            message=event_data["message"],
            source=event_data.get("source", "unknown"),
            timestamp=event_data.get(
                "timestamp", int(datetime.now().timestamp() * 1000)
            ),
            duration_ms=event_data.get("duration_ms"),
            properties=event_data.get("properties", {}),
            trace_id=event_data.get("trace_id"),
            span_id=event_data.get("span_id"),
        )

        await self.buffer.add_event(event)

        logger.debug(f"Event submitted: {event.event_type} - {event.message}")
        return event.id

    async def add_export_target(self, target_config: Dict[str, Any]) -> str:
        """Add an export target."""
        target = ExportTarget(
            id=str(uuid.uuid4()),
            name=target_config["name"],
            format=ExportFormat(target_config["format"]),
            endpoint=target_config["endpoint"],
            credentials=target_config.get("credentials", {}),
            filters=target_config.get("filters", {}),
            enabled=target_config.get("enabled", True),
        )

        await self.exporter.add_target(target)
        return target.id

    async def remove_export_target(self, target_id: str) -> bool:
        """Remove an export target."""
        await self.exporter.remove_target(target_id)
        return True

    async def trigger_export(self) -> Dict[str, Any]:
        """Manually trigger an export."""
        try:
            aggregated_data = await self.aggregator.get_aggregated_data()
            metrics = await self.buffer.get_metrics()
            events = await self.buffer.get_events()

            await self.exporter.export_data(aggregated_data, metrics, events)

            return {
                "success": True,
                "exported_metrics": len(metrics),
                "exported_events": len(events),
                "export_targets": len(self.exporter.targets),
                "timestamp": int(datetime.now().timestamp() * 1000),
            }
        except Exception as e:
            logger.error(f"Manual export failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        buffer_stats = await self.buffer.get_stats()
        aggregated_data = await self.aggregator.get_aggregated_data()

        return {
            "service_status": "running" if self.running else "stopped",
            "buffer_stats": buffer_stats,
            "export_targets": len(self.exporter.targets),
            "aggregated_counters": len(aggregated_data.get("counters", {})),
            "aggregated_gauges": len(aggregated_data.get("gauges", {})),
            "aggregated_histograms": len(aggregated_data.get("histograms", {})),
            "aggregated_timers": len(aggregated_data.get("timers", {})),
            "export_interval": self.export_interval,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }

    async def _background_export(self):
        """Background export task."""
        while self.running:
            try:
                # Export data
                export_result = await self.trigger_export()
                if export_result["success"]:
                    logger.debug(
                        f"Background export completed: {export_result['exported_metrics']} metrics, "
                        f"{export_result['exported_events']} events"
                    )

                # Reset aggregator after export
                await self.aggregator.reset()

                # Sleep until next export
                await asyncio.sleep(self.export_interval)

            except Exception as e:
                logger.error(f"Background export error: {e}")
                await asyncio.sleep(60)  # Wait longer on error


async def main():
    """Main entry point for the Telemetry Exporter Service."""
    service = TelemetryExporterService()

    await service.start()

    try:
        # Demo: Add export targets
        prometheus_target = {
            "name": "Prometheus",
            "format": "prometheus",
            "endpoint": "http://localhost:9090/api/v1/write",
            "enabled": True,
        }

        json_target = {
            "name": "JSON File",
            "format": "json",
            "endpoint": "/tmp/telemetry_export.json",
            "enabled": True,
        }

        logger.info("Adding export targets...")
        prometheus_id = await service.add_export_target(prometheus_target)
        json_id = await service.add_export_target(json_target)

        logger.info(f"Added export targets: {prometheus_id}, {json_id}")

        # Demo: Submit test metrics
        test_metrics = [
            {
                "name": "cpu_usage",
                "type": "gauge",
                "value": 75.5,
                "unit": "percent",
                "source": "system_monitor",
                "labels": {"host": "localhost", "cpu": "0"},
            },
            {
                "name": "requests_total",
                "type": "counter",
                "value": 1,
                "unit": "count",
                "source": "web_server",
                "labels": {"method": "GET", "status": "200"},
            },
            {
                "name": "response_time",
                "type": "histogram",
                "value": 150.0,
                "unit": "ms",
                "source": "web_server",
                "labels": {"endpoint": "/api/status"},
            },
        ]

        # Demo: Submit test events
        test_events = [
            {
                "type": "request_received",
                "severity": "info",
                "message": "HTTP request received",
                "source": "web_server",
                "properties": {"method": "GET", "path": "/api/status"},
                "trace_id": "trace-123",
                "span_id": "span-456",
            },
            {
                "type": "error_occurred",
                "severity": "error",
                "message": "Database connection failed",
                "source": "database",
                "duration_ms": 5000,
            },
        ]

        logger.info("Submitting test metrics and events...")
        for metric_data in test_metrics:
            await service.submit_metric(metric_data)

        for event_data in test_events:
            await service.submit_event(event_data)

        # Get service stats
        stats = await service.get_service_stats()
        logger.info(f"Service stats: {json.dumps(stats, indent=2)}")

        # Trigger manual export
        export_result = await service.trigger_export()
        logger.info(f"Export result: {export_result}")

        # Keep service running
        logger.info("Telemetry Exporter Service is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
