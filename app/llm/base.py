"""Provider-agnostic LLM protocol with Pydantic JSON-schema enforcement and automatic fallback."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base interface for all LLM backends (local GGUF, HF Inference API, heuristic test engine)."""

    provider_name: str = "abstract"

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate free-text completion."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[T],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> T:
        """Generate completion strictly conforming to a Pydantic model schema."""


def parse_json_into_model(raw_text: str, schema_model: type[T]) -> T:
    """Extract and validate a JSON object from model output into a Pydantic model."""
    cleaned = raw_text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", cleaned, re.S)
        if brace_match:
            cleaned = brace_match.group(1)

    parsed = json.loads(cleaned)
    return schema_model.model_validate(parsed)
