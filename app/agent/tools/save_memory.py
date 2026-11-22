"""User research memory persistence tool backed by the single-table DynamoDB store."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.guardrails.injection import sanitize_untrusted_text
from app.storage.dynamodb import DynamoDBStore


class SaveMemoryInput(BaseModel):
    """Pydantic schema for save_memory tool input."""

    key: str = Field(..., min_length=1, max_length=80)
    value: str = Field(..., min_length=1, max_length=500)
    category: str = Field(default="topic", max_length=40)
    user_id: str = Field(default="default_researcher", max_length=80)


def run_save_memory(
    raw_args: dict[str, Any],
    ddb_store: DynamoDBStore,
) -> dict[str, Any]:
    """Validate, sanitize, and save user memory entry in DynamoDB."""
    validated = SaveMemoryInput.model_validate(raw_args)
    safe_key = sanitize_untrusted_text(validated.key).cleaned_text
    safe_val = sanitize_untrusted_text(validated.value).cleaned_text
    ok = ddb_store.save_user_memory(
        user_id=validated.user_id,
        key=safe_key,
        value=safe_val,
        category=validated.category,
    )
    return {
        "saved": ok,
        "user_id": validated.user_id,
        "key": safe_key,
        "value": safe_val,
        "category": validated.category,
    }
