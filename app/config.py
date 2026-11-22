"""Configuration management for LitLens via environment variables and AWS SSM Parameter Store."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings enforcing free-tier resource boundaries."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # App Metadata
    app_name: str = "LitLens"
    environment: Literal["local", "test", "production"] = "local"
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 7860

    # LLM Settings (100% Free Local GGUF + Optional Free HF Inference Fallback)
    llm_provider: Literal["llamacpp", "hf_fallback", "heuristic"] = "llamacpp"
    local_model_size: Literal["1.5B", "3B"] = "1.5B"
    gguf_repo_1_5b: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    gguf_file_1_5b: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    gguf_repo_3b: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    gguf_file_3b: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    models_dir: str = "./models"
    llm_n_ctx: int = 4096
    llm_n_threads: int = 2  # Matches HF Spaces CPU Basic (2 vCPU)
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1

    # Optional Free Hugging Face Serverless Inference Fallback
    enable_hf_fallback: bool = True
    hf_token: str | None = Field(default=None, description="Free HF Hub Read/Inference Token")
    hf_fallback_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Embeddings & Vector Store (Local CPU)
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieval_top_k: int = 5
    min_retrieval_confidence: float = 0.32  # Cosine/hybrid refusal gate threshold
    local_index_dir: str = "./data/faiss_index"

    # AWS Free-Tier Services
    aws_region: str = "us-east-1"
    aws_enabled: bool = False
    s3_bucket_name: str = "litlens-free-tier-store"
    dynamodb_table_name: str = "LitLensStore"
    lambda_arxiv_function_url: str | None = None
    ssm_prefix: str = "/litlens/prod"
    load_from_ssm: bool = False

    # SES Weekly Digest (Free Tier)
    ses_sender_email: str = "digest@example.com"
    ses_default_recipient: str = "researcher@example.com"

    # Guardrails & Rate Limits
    rate_limit_requests_per_minute: int = 20
    max_query_length: int = 1000
    max_papers_in_matrix: int = 5
    cache_ttl_seconds: int = 86400  # 24 hours

    # Observability (CloudWatch Free Tier: max 10 custom metrics)
    cloudwatch_namespace: str = "LitLens/Prod"
    enable_cloudwatch_metrics: bool = False

    @property
    def active_gguf_repo(self) -> str:
        return self.gguf_repo_3b if self.local_model_size == "3B" else self.gguf_repo_1_5b

    @property
    def active_gguf_filename(self) -> str:
        return self.gguf_file_3b if self.local_model_size == "3B" else self.gguf_file_1_5b


def load_ssm_parameters(prefix: str, region: str) -> dict[str, str]:
    """Load free standard parameters from AWS SSM Parameter Store."""
    loaded: dict[str, str] = {}
    try:
        client = boto3.client("ssm", region_name=region)
        paginator = client.get_paginator("get_parameters_by_path")
        for page in paginator.paginate(Path=prefix, Recursive=True, WithDecryption=True):
            for param in page.get("Parameters", []):
                key = param["Name"].replace(f"{prefix}/", "").upper()
                loaded[key] = param["Value"]
    except (BotoCoreError, ClientError):
        pass
    return loaded


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance, optionally hydrating from AWS SSM first."""
    if os.getenv("LOAD_FROM_SSM", "false").lower() == "true":
        region = os.getenv("AWS_REGION", "us-east-1")
        prefix = os.getenv("SSM_PREFIX", "/litlens/prod")
        for k, v in load_ssm_parameters(prefix, region).items():
            os.environ.setdefault(k, v)
    return Settings()
