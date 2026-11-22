"""Section-aware academic paper chunker with token budgets and bibliography stripping."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.guardrails.injection import sanitize_untrusted_text

SECTION_HEADER_REGEX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+)?"
    r"(Abstract|Introduction|Background|Related Work|Preliminaries|Method(?:s|ology)?|"
    r"Approach|Architecture|System Design|Experiments?|Evaluation|Results?(?:\s+and\s+Discussion)?|"
    r"Discussion|Limitations?|Conclusion(?:s)?|Future Work|References|Bibliography)\s*$",
    re.I | re.M,
)

REFERENCES_TAIL_REGEX = re.compile(
    r"\n\s*(?:\d+\.?\s+|\[?\s*)?(?:References|Bibliography)\s*\n",
    re.I,
)


@dataclass(frozen=True)
class PaperChunk:
    """A single indexed chunk of an arXiv paper."""

    chunk_id: str  # Format: "{arxiv_id}#c{idx}"
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    section: str
    page_number: int
    text: str
    token_estimate: int

    def to_dict(self) -> dict:
        return asdict(self)


def strip_bibliography(full_text: str) -> str:
    """Remove trailing References/Bibliography section to prevent reference noise in retrieval."""
    matches = list(REFERENCES_TAIL_REGEX.finditer(full_text))
    if matches:
        last_match = matches[-1]
        # Only strip if it appears in the second half of the document
        if last_match.start() > len(full_text) * 0.45:
            return full_text[: last_match.start()].strip()
    return full_text.strip()


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split paper text into (section_name, section_body) tuples."""
    cleaned = strip_bibliography(text)
    matches = list(SECTION_HEADER_REGEX.finditer(cleaned))
    if not matches:
        return [("Body", cleaned)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 40:
        preamble = cleaned[: matches[0].start()].strip()
        if preamble:
            sections.append(("Abstract", preamble))

    for idx, match in enumerate(matches):
        sec_name = match.group(1).title()
        if sec_name.lower() in {"references", "bibliography"}:
            continue
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        body = cleaned[start_pos:end_pos].strip()
        if body:
            sections.append((sec_name, body))

    return sections or [("Body", cleaned)]


def chunk_paper(
    arxiv_id: str,
    title: str,
    authors: list[str],
    published: str,
    text: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    page_map: dict[int, str] | None = None,
) -> list[PaperChunk]:
    """Chunk an academic paper by section with sliding token windows (~0.75 words/token)."""
    words_per_chunk = max(int(chunk_size_tokens * 0.75), 32)
    words_overlap = min(max(int(chunk_overlap_tokens * 0.75), 0), words_per_chunk // 2)
    step = max(words_per_chunk - words_overlap, 16)

    sections = split_into_sections(text)
    chunks: list[PaperChunk] = []
    chunk_idx = 0

    for section_name, section_body in sections:
        sanitized = sanitize_untrusted_text(section_body).cleaned_text
        words = sanitized.split()
        if not words:
            continue

        pos = 0
        while pos < len(words):
            window_words = words[pos : pos + words_per_chunk]
            window_text = " ".join(window_words).strip()
            if window_text:
                token_est = max(int(len(window_words) / 0.75), 1)
                page_num = 1
                if page_map:
                    sample_phrase = " ".join(window_words[:8])
                    for pno, ptext in page_map.items():
                        if sample_phrase and sample_phrase in ptext:
                            page_num = pno
                            break
                chunks.append(
                    PaperChunk(
                        chunk_id=f"{arxiv_id}#c{chunk_idx}",
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors,
                        published=published,
                        section=section_name,
                        page_number=page_num,
                        text=window_text,
                        token_estimate=token_est,
                    )
                )
                chunk_idx += 1
            if pos + words_per_chunk >= len(words):
                break
            pos += step

    return chunks
