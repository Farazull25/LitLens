"""Single-table DynamoDB client strictly adhering to AWS Always-Free Provisioned limits (5 RCU / 5 WCU)."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.observability.logger import get_logger


class DynamoDBStore:
    """Single-table storage for sessions, research memory, query cache, topics, and eval logs.

    Includes an automatic in-memory fallback when AWS is disabled or unreachable so local dev,
    HF Space ephemeral mode, and unit tests work seamlessly.
    """

    def __init__(self, table_name: str | None = None, region: str | None = None) -> None:
        settings = get_settings()
        self.table_name = table_name or settings.dynamodb_table_name
        self.region = region or settings.aws_region
        self.aws_enabled = settings.aws_enabled
        self._memory_fallback: dict[tuple[str, str], dict[str, Any]] = {}

    def _get_table(self) -> Any:
        dynamodb = boto3.resource("dynamodb", region_name=self.region)
        return dynamodb.Table(self.table_name)

    def put_item(
        self,
        pk: str,
        sk: str,
        payload: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> bool:
        """Write an item to the single DynamoDB table with optional TTL."""
        now = int(time.time())
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "updated_at": now,
            "payload": json.dumps(payload, ensure_ascii=False),
        }
        if ttl_seconds is not None:
            item["ttl"] = now + ttl_seconds

        self._memory_fallback[(pk, sk)] = item

        if not self.aws_enabled:
            return True

        try:
            table = self._get_table()
            table.put_item(Item=item)
            return True
        except (BotoCoreError, ClientError) as exc:
            get_logger().warning("dynamodb_put_failed", pk=pk, sk=sk, error=str(exc))
            return False

    def get_item(self, pk: str, sk: str) -> dict[str, Any] | None:
        """Fetch an item by (PK, SK), respecting TTL expiration."""
        now = int(time.time())

        if self.aws_enabled:
            try:
                table = self._get_table()
                resp = table.get_item(Key={"PK": pk, "SK": sk})
                raw = resp.get("Item")
                if raw:
                    if "ttl" in raw and int(raw["ttl"]) < now:
                        return None
                    return json.loads(raw["payload"])
            except (BotoCoreError, ClientError) as exc:
                get_logger().warning("dynamodb_get_failed", pk=pk, sk=sk, error=str(exc))

        raw_local = self._memory_fallback.get((pk, sk))
        if not raw_local:
            return None
        if "ttl" in raw_local and int(raw_local["ttl"]) < now:
            self._memory_fallback.pop((pk, sk), None)
            return None
        return json.loads(raw_local["payload"])

    def query_by_pk(self, pk: str) -> list[dict[str, Any]]:
        """Return all valid items sharing a partition key PK."""
        now = int(time.time())
        results: list[dict[str, Any]] = []

        if self.aws_enabled:
            try:
                from boto3.dynamodb.conditions import Key

                table = self._get_table()
                resp = table.query(KeyConditionExpression=Key("PK").eq(pk))
                for raw in resp.get("Items", []):
                    if "ttl" in raw and int(raw["ttl"]) < now:
                        continue
                    results.append(json.loads(raw["payload"]))
                return results
            except (BotoCoreError, ClientError) as exc:
                get_logger().warning("dynamodb_query_failed", pk=pk, error=str(exc))

        for (k_pk, _), raw in sorted(self._memory_fallback.items(), key=lambda x: x[0][1]):
            if k_pk == pk:
                if "ttl" in raw and int(raw["ttl"]) < now:
                    continue
                results.append(json.loads(raw["payload"]))
        return results

    # --- High-level Domain Helpers ---

    @staticmethod
    def _hash_query(query: str) -> str:
        normalized = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def get_cached_response(self, query: str) -> dict[str, Any] | None:
        """Check exact/normalized query cache."""
        qhash = self._hash_query(query)
        return self.get_item(pk=f"CACHE#{qhash}", sk="V1")

    def set_cached_response(self, query: str, response_data: dict[str, Any]) -> None:
        """Cache a verified response with 24h TTL."""
        settings = get_settings()
        qhash = self._hash_query(query)
        self.put_item(
            pk=f"CACHE#{qhash}",
            sk="V1",
            payload=response_data,
            ttl_seconds=settings.cache_ttl_seconds,
        )

    def save_user_memory(self, user_id: str, key: str, value: str, category: str = "note") -> bool:
        """Persist a research note or topic preference for the user."""
        return self.put_item(
            pk=f"MEMORY#{user_id}",
            sk=f"{category.upper()}#{key}",
            payload={"user_id": user_id, "key": key, "value": value, "category": category},
        )

    def list_user_memories(self, user_id: str) -> list[dict[str, Any]]:
        """List saved research memories for a user."""
        return self.query_by_pk(pk=f"MEMORY#{user_id}")

    def subscribe_topic(self, email: str, topic: str) -> bool:
        """Subscribe an email address to a weekly arXiv digest topic."""
        clean_topic = topic.strip()
        slug = clean_topic.lower().replace(" ", "_")
        return self.put_item(
            pk="SUBSCRIPTION#ACTIVE",
            sk=f"TOPIC#{slug}#{email.lower().strip()}",
            payload={"email": email.strip(), "topic": clean_topic, "subscribed_at": int(time.time())},
        )

    def list_subscriptions(self) -> list[dict[str, Any]]:
        """List all active weekly digest topic subscriptions."""
        return self.query_by_pk(pk="SUBSCRIPTION#ACTIVE")

    def log_eval_run(self, run_id: str, metrics: dict[str, Any]) -> bool:
        """Persist evaluation run metrics in DynamoDB."""
        return self.put_item(
            pk="EVAL#RUNS",
            sk=f"RUN#{int(time.time())}#{run_id}",
            payload={"run_id": run_id, **metrics},
        )
