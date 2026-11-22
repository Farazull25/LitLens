"""Prompt-injection defenses treating all PDF text and external content as untrusted."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.observability.logger import get_logger
from app.observability.metrics import metrics_registry

# Zero-width & control characters often used to hide prompt injections in PDFs
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")

# High-risk prompt override signatures
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("system_override", re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I)),
    ("role_hijack", re.compile(r"(?:you\s+are\s+now|act\s+as\s+a|new\s+system\s+prompt)", re.I)),
    ("delimiter_escape", re.compile(r"</?(?:untrusted_paper_content|system|instruction|prompt)>", re.I)),
    ("exfiltration_link", re.compile(r"!\[[^\]]*\]\(https?://(?!(?:arxiv\.org|doi\.org))[^\)]+\)", re.I)),
    ("tool_hijack", re.compile(r"(?:execute_tool|save_memory|call_function)\s*\(", re.I)),
]


@dataclass(frozen=True)
class SanitizationResult:
    """Output of untrusted PDF or query sanitization."""

    cleaned_text: str
    injection_detected: bool
    matched_rules: list[str]


def sanitize_untrusted_text(text: str) -> SanitizationResult:
    """Neutralize prompt injection vectors in untrusted PDF chunks or user input."""
    cleaned = ZERO_WIDTH_PATTERN.sub("", text)
    matched: list[str] = []

    for rule_name, pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            matched.append(rule_name)
            cleaned = pattern.sub("[REDACTED_POTENTIAL_INJECTION]", cleaned)

    if matched:
        metrics_registry.record("InjectionBlockedCount", 1.0)
        get_logger().warning("prompt_injection_neutralized", matched_rules=matched)

    return SanitizationResult(
        cleaned_text=cleaned.strip(),
        injection_detected=bool(matched),
        matched_rules=matched,
    )


def wrap_untrusted_chunk(chunk_id: str, title: str, section: str, text: str) -> str:
    """Wrap sanitized PDF chunk in strict XML sandboxing tags for the LLM."""
    res = sanitize_untrusted_text(text)
    safe_title = sanitize_untrusted_text(title).cleaned_text
    return (
        f'<untrusted_paper_content chunk_id="{chunk_id}" title="{safe_title}" section="{section}">\n'
        f"{res.cleaned_text}\n"
        f"</untrusted_paper_content>"
    )
