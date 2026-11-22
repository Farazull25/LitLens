"""Optional free Hugging Face Serverless Inference fallback provider with automatic circuit breaker."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.config import get_settings
from app.llm.base import LLMProvider, parse_json_into_model
from app.llm.llamacpp_provider import LlamaCppProvider
from app.observability.logger import get_logger
from app.observability.metrics import metrics_registry

T = TypeVar("T", bound=BaseModel)


class HFFallbackProvider(LLMProvider):
    """Calls free HF Serverless Inference API (`hf-inference`) and falls back to local GGUF on error/rate-limit."""

    provider_name = "hf_serverless_with_local_fallback"

    def __init__(self, primary_local: LlamaCppProvider | None = None) -> None:
        settings = get_settings()
        self.hf_token = settings.hf_token
        self.model_id = settings.hf_fallback_model
        self.enabled = settings.enable_hf_fallback and bool(self.hf_token)
        self.local_provider = primary_local or LlamaCppProvider()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        # Always attempt local GGUF first unless explicitly configured as primary cloud fallback
        settings = get_settings()
        if settings.llm_provider != "hf_fallback" or not self.enabled:
            return self.local_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(model=self.model_id, token=self.hf_token, timeout=15)
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            return str(resp.choices[0].message.content).strip()
        except Exception as exc:
            metrics_registry.record("LLMErrorCount", 1.0)
            get_logger().warning("hf_inference_fallback_triggered", error=str(exc))
            return self.local_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[T],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> T:
        settings = get_settings()
        if settings.llm_provider == "hf_fallback" and self.enabled:
            try:
                from huggingface_hub import InferenceClient

                client = InferenceClient(model=self.model_id, token=self.hf_token, timeout=15)
                schema_hint = f"\nRespond ONLY with JSON matching this schema:\n{schema_model.model_json_schema()}"
                resp = client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt + schema_hint},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.01,
                )
                raw = str(resp.choices[0].message.content)
                return parse_json_into_model(raw, schema_model)
            except Exception as exc:
                metrics_registry.record("LLMErrorCount", 1.0)
                get_logger().warning("hf_structured_fallback_triggered", error=str(exc))

        return self.local_provider.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=schema_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def get_llm_provider(model_size: str | None = None) -> LLMProvider:
    """Factory returning the configured provider-agnostic LLM implementation."""
    local = LlamaCppProvider(model_size=model_size)
    return HFFallbackProvider(primary_local=local)
