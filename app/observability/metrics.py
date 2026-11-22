"""CloudWatch Custom Metrics publisher strictly capped within AWS Always-Free 10-metric limit."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Literal

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.observability.logger import get_logger

# Strictly 6 metric names (zero dynamic dimensions) to remain well under the 10-metric AWS Free Tier limit.
AllowedMetricName = Literal[
    "QueryLatencyMs",
    "RetrievalHitCount",
    "RefusalCount",
    "CacheHitCount",
    "InjectionBlockedCount",
    "LLMErrorCount",
]

ALLOWED_METRICS: set[str] = {
    "QueryLatencyMs",
    "RetrievalHitCount",
    "RefusalCount",
    "CacheHitCount",
    "InjectionBlockedCount",
    "LLMErrorCount",
}


class MetricsRegistry:
    """In-memory telemetry store with optional CloudWatch zero-dimension publishing."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._latencies: list[float] = []

    def record(
        self,
        metric_name: AllowedMetricName,
        value: float = 1.0,
        unit: str = "Count",
    ) -> None:
        """Record a metric locally and optionally publish to CloudWatch."""
        if metric_name not in ALLOWED_METRICS:
            raise ValueError(f"Metric {metric_name} violates the 6-metric free-tier allowlist.")

        with self._lock:
            if metric_name == "QueryLatencyMs":
                self._latencies.append(value)
                self._counters["QueryCount"] += 1
            else:
                self._counters[metric_name] += value

        settings = get_settings()
        if settings.aws_enabled and settings.enable_cloudwatch_metrics:
            self._publish_cloudwatch(metric_name, value, unit, settings.cloudwatch_namespace, settings.aws_region)

    def _publish_cloudwatch(
        self,
        metric_name: str,
        value: float,
        unit: str,
        namespace: str,
        region: str,
    ) -> None:
        try:
            cw = boto3.client("cloudwatch", region_name=region)
            cw.put_metric_data(
                Namespace=namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": float(value),
                        "Unit": unit,
                        # Intentionally empty Dimensions to prevent metric cardinality explosion!
                        "Dimensions": [],
                    }
                ],
            )
        except (BotoCoreError, ClientError) as exc:
            get_logger().warning("cloudwatch_metric_failed", metric=metric_name, error=str(exc))

    def snapshot(self) -> dict[str, float]:
        """Return current telemetry summary for UI & health checks."""
        with self._lock:
            avg_lat = (
                round(sum(self._latencies) / len(self._latencies), 2)
                if self._latencies
                else 0.0
            )
            data = dict(self._counters)
            data["AvgQueryLatencyMs"] = avg_lat
            return data

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()


metrics_registry = MetricsRegistry()
