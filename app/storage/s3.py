"""S3 storage manager optimized for AWS Free Tier (5 GB, 2,000 PUTs, 20,000 GETs).

Stores 1 compressed JSON bundle per paper (`papers/{arxiv_id}.json.gz`) instead of raw PDFs or
per-chunk objects, reducing storage by >95% and S3 PUT requests to 1 per ingested paper.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.observability.logger import get_logger


class S3Store:
    """S3 adapter for compressed paper bundles and FAISS index snapshots."""

    def __init__(self, bucket_name: str | None = None, region: str | None = None) -> None:
        settings = get_settings()
        self.bucket_name = bucket_name or settings.s3_bucket_name
        self.region = region or settings.aws_region
        self.aws_enabled = settings.aws_enabled
        self._local_cache_dir = Path(settings.local_index_dir) / "s3_mirror"
        self._local_cache_dir.mkdir(parents=True, exist_ok=True)

    def _client(self) -> Any:
        return boto3.client("s3", region_name=self.region)

    def save_paper_bundle(self, arxiv_id: str, paper_data: dict[str, Any]) -> str:
        """Compress and store a parsed paper and its chunks as a single gzipped JSON object."""
        safe_id = arxiv_id.replace("/", "_")
        key = f"papers/{safe_id}.json.gz"
        compressed = gzip.compress(json.dumps(paper_data, ensure_ascii=False).encode("utf-8"))

        local_path = self._local_cache_dir / f"{safe_id}.json.gz"
        local_path.write_bytes(compressed)

        if self.aws_enabled:
            try:
                self._client().put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=compressed,
                    ContentType="application/json",
                    ContentEncoding="gzip",
                )
            except (BotoCoreError, ClientError) as exc:
                get_logger().warning("s3_put_paper_failed", arxiv_id=arxiv_id, error=str(exc))
        return key

    def load_paper_bundle(self, arxiv_id: str) -> dict[str, Any] | None:
        """Load a compressed paper bundle from local mirror or S3."""
        safe_id = arxiv_id.replace("/", "_")
        local_path = self._local_cache_dir / f"{safe_id}.json.gz"
        if local_path.exists():
            return json.loads(gzip.decompress(local_path.read_bytes()).decode("utf-8"))

        if not self.aws_enabled:
            return None

        key = f"papers/{safe_id}.json.gz"
        try:
            resp = self._client().get_object(Bucket=self.bucket_name, Key=key)
            raw_bytes = resp["Body"].read()
            local_path.write_bytes(raw_bytes)
            return json.loads(gzip.decompress(raw_bytes).decode("utf-8"))
        except (BotoCoreError, ClientError) as exc:
            get_logger().warning("s3_get_paper_failed", arxiv_id=arxiv_id, error=str(exc))
            return None

    def upload_faiss_snapshot(self, index_file: Path, metadata_file: Path) -> bool:
        """Sync local FAISS index and chunk metadata to S3 in 2 PUT requests."""
        if not self.aws_enabled:
            return False
        try:
            s3 = self._client()
            s3.upload_file(str(index_file), self.bucket_name, "indices/faiss.index")
            s3.upload_file(str(metadata_file), self.bucket_name, "indices/chunks_meta.json.gz")
            return True
        except (BotoCoreError, ClientError) as exc:
            get_logger().warning("s3_upload_faiss_failed", error=str(exc))
            return False

    def download_faiss_snapshot(self, index_file: Path, metadata_file: Path) -> bool:
        """Hydrate local FAISS index from S3 on HF Space cold start."""
        if not self.aws_enabled:
            return False
        try:
            s3 = self._client()
            index_file.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(self.bucket_name, "indices/faiss.index", str(index_file))
            s3.download_file(self.bucket_name, "indices/chunks_meta.json.gz", str(metadata_file))
            return True
        except (BotoCoreError, ClientError):
            return False
