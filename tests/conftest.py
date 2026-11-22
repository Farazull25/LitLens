"""Pytest fixtures including moto AWS Free-Tier mocks and isolated FAISS indices."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

os.environ["ENVIRONMENT"] = "test"
os.environ["LLM_PROVIDER"] = "heuristic"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from app.config import get_settings  # noqa: E402
from app.observability.metrics import metrics_registry  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402
from app.rag.vector_store import FAISSVectorStore  # noqa: E402
from app.storage.dynamodb import DynamoDBStore  # noqa: E402
from app.storage.s3 import S3Store  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_metrics() -> Generator[None, None, None]:
    get_settings.cache_clear()
    metrics_registry.reset()
    yield
    metrics_registry.reset()


@pytest.fixture()
def temp_vector_store() -> FAISSVectorStore:
    tmp_dir = Path(tempfile.mkdtemp(prefix="litlens_test_faiss_"))
    embedder = get_embedder(force_hash=True)
    s3 = S3Store(bucket_name="test-litlens-bucket", region="us-east-1")
    return FAISSVectorStore(embedder=embedder, index_dir=tmp_dir, s3_store=s3)


@pytest.fixture()
def aws_free_tier_mocks() -> Generator[tuple[S3Store, DynamoDBStore], None, None]:
    """Spin up moto-mocked S3 bucket and Provisioned (5 RCU / 5 WCU) DynamoDB single table."""
    with mock_aws():
        region = "us-east-1"
        bucket_name = "litlens-free-tier-store"
        table_name = "LitLensStore"

        s3_client = boto3.client("s3", region_name=region)
        s3_client.create_bucket(Bucket=bucket_name)

        ddb_client = boto3.client("dynamodb", region_name=region)
        ddb_client.create_table(
            TableName=table_name,
            BillingMode="PROVISIONED",
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
        )

        s3_store = S3Store(bucket_name=bucket_name, region=region)
        s3_store.aws_enabled = True
        ddb_store = DynamoDBStore(table_name=table_name, region=region)
        ddb_store.aws_enabled = True
        yield s3_store, ddb_store
