"""Input validation and citation grounding verification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import get_settings
from app.guardrails.injection import sanitize_untrusted_text

# Matches citations in the canonical format [arxiv_id#cN], e.g. [2312.10997#c0] or [upload-paper#c2]
CITATION_TAG_REGEX = re.compile(r"\[((?:[0-9]{4}\.[0-9]{4,5}(?:v\d+)?|upload-[a-zA-Z0-9_-]{3,40})#c\d+)\]")

# Matches non-arXiv external markdown links to prevent phishing/exfiltration from poisoned PDFs
UNSAFE_MARKDOWN_LINK_REGEX = re.compile(
    r"\[([^\]]+)\]\((https?://(?!(?:arxiv\.org|doi\.org|huggingface\.co))[^\)]+)\)",
    re.I,
)


@dataclass(frozen=True)
class CitationValidationResult:
    """Result of verifying citations in an LLM answer against retrieved chunk IDs."""

    verified_answer: str
    valid_citations: list[str]
    hallucinated_citations: list[str]
    is_grounded: bool


def validate_user_query(query: str) -> str:
    """Validate user query length, non-emptiness, and injection attempts."""
    settings = get_settings()
    stripped = (query or "").strip()
    if not stripped:
        raise ValueError("Query cannot be empty.")
    if len(stripped) > settings.max_query_length:
        raise ValueError(
            f"Query exceeds maximum length of {settings.max_query_length} characters."
        )
    sanitized = sanitize_untrusted_text(stripped)
    if sanitized.injection_detected:
        raise ValueError(
            f"Query rejected by security guardrails (matched: {', '.join(sanitized.matched_rules)})."
        )
    return sanitized.cleaned_text


def extract_citation_ids(text: str) -> list[str]:
    """Extract unique citation IDs like '2312.10997#c0' in order of appearance."""
    found = CITATION_TAG_REGEX.findall(text)
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in found:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def verify_and_clean_citations(
    answer: str,
    retrieved_chunk_ids: set[str],
) -> CitationValidationResult:
    """Verify every [arxiv_id#cN] citation maps to an actually retrieved chunk.

    Strips hallucinated citations and removes untrusted external links.
    """
    # Strip any suspicious non-arXiv markdown links
    cleaned_answer = UNSAFE_MARKDOWN_LINK_REGEX.sub(r"\1", answer)

    cited_ids = extract_citation_ids(cleaned_answer)
    valid: list[str] = []
    hallucinated: list[str] = []

    for cid in cited_ids:
        if cid in retrieved_chunk_ids:
            valid.append(cid)
        else:
            hallucinated.append(cid)
            # Strip fabricated citation tag from output text
            cleaned_answer = cleaned_answer.replace(f"[{cid}]", "")

    return CitationValidationResult(
        verified_answer=cleaned_answer.strip(),
        valid_citations=valid,
        hallucinated_citations=hallucinated,
        is_grounded=len(valid) > 0,
    )
