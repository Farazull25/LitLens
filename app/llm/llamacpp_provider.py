"""Local CPU GGUF provider via llama-cpp-python (Qwen2.5-1.5B/3B-Instruct Q4_K_M) + deterministic extractive fallback."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import get_settings
from app.llm.base import LLMProvider, parse_json_into_model
from app.observability.logger import get_logger

T = TypeVar("T", bound=BaseModel)

CHUNK_BLOCK_RE = re.compile(
    r'<untrusted_paper_content\s+chunk_id="([^"]+)"\s+title="([^"]*)"\s+section="([^"]*)">\s*(.*?)\s*</untrusted_paper_content>',
    re.S,
)
ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")


class ExtractiveGroundedEngine:
    """Deterministic zero-API extractive engine used when GGUF binary is absent or in unit/eval tests.

    Guarantees 100% schema adherence and verifiable `[arxiv_id#cN]` citation grounding directly from
    retrieved `<untrusted_paper_content>` blocks.
    """

    @staticmethod
    def synthesize_grounded_answer(user_prompt: str) -> str:
        blocks = CHUNK_BLOCK_RE.findall(user_prompt)
        if not blocks:
            return "Insufficient grounded evidence in the indexed papers to answer this question."

        q_match = re.search(r"USER QUESTION:\s*(.+)", user_prompt, re.S)
        question = q_match.group(1).strip() if q_match else user_prompt
        q_terms = {w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", question)}

        scored_sentences: list[tuple[float, str, str, str]] = []
        for chunk_id, title, _section, body in blocks:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 20]
            if not sentences and body.strip():
                sentences = [body.strip()]
            for sent in sentences:
                s_terms = {w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", sent)}
                overlap = len(q_terms & s_terms) / max(len(q_terms), 1)
                scored_sentences.append((overlap, sent, chunk_id, title))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_score = scored_sentences[0][0] if scored_sentences else 0.0
        selected = [
            item for item in scored_sentences[:2] if item[0] >= max(top_score * 0.65, 0.12)
        ] or (scored_sentences[:1] if scored_sentences else [])
        if not selected:
            cid, title, _, body = blocks[0]
            return f"Based on *{title}*, {body[:280].strip()} [{cid}]."

        lines: list[str] = []
        used_cids: list[str] = []
        for _, sent, cid, _title in selected:
            clean_sent = sent.rstrip(".")
            lines.append(f"{clean_sent} [{cid}].")
            if cid not in used_cids:
                used_cids.append(cid)

        return " ".join(lines)

    @staticmethod
    def plan_for_query(user_prompt: str, schema_model: type[T]) -> T:
        text_lower = user_prompt.lower()
        arxiv_ids = ARXIV_ID_RE.findall(user_prompt)

        if any(k in text_lower for k in ["matrix", "compare", "comparison table", "literature review table"]):
            intent = "lit_matrix"
            steps = [
                {
                    "tool_name": "build_lit_matrix",
                    "arguments": {"arxiv_ids": arxiv_ids, "topic": user_prompt[:120]},
                    "rationale": "Generate structured literature review comparison table across papers.",
                }
            ]
        elif any(k in text_lower for k in ["remember", "save memory", "follow topic", "note that"]):
            intent = "memory"
            steps = [
                {
                    "tool_name": "save_memory",
                    "arguments": {"key": "research_focus", "value": user_prompt[:200], "category": "topic"},
                    "rationale": "Persist user research preference to DynamoDB memory.",
                }
            ]
        elif any(k in text_lower for k in ["bibtex", "apa", "format citation", "reference list"]):
            intent = "citation"
            steps = [
                {
                    "tool_name": "format_citations",
                    "arguments": {"arxiv_ids": arxiv_ids, "style": "both"},
                    "rationale": "Format verifiable BibTeX and APA citations.",
                }
            ]
        elif any(k in text_lower for k in ["ingest", "download paper", "fetch pdf", "parse pdf"]) and arxiv_ids:
            intent = "ingest"
            steps = [
                {
                    "tool_name": "fetch_and_parse_pdf",
                    "arguments": {"arxiv_id": aid},
                    "rationale": f"Fetch, sanitize, chunk, and index arXiv paper {aid}.",
                }
                for aid in arxiv_ids[:3]
            ]
        elif "search arxiv" in text_lower or "find papers on" in text_lower:
            intent = "search"
            steps = [
                {
                    "tool_name": "arxiv_search",
                    "arguments": {"query": user_prompt, "max_results": 3},
                    "rationale": "Query arXiv API for relevant papers.",
                },
                {
                    "tool_name": "retrieve_chunks",
                    "arguments": {"query": user_prompt, "top_k": 5},
                    "rationale": "Retrieve grounded chunks from indexed papers.",
                },
            ]
        else:
            intent = "qa"
            steps = []
            if arxiv_ids:
                for aid in arxiv_ids[:2]:
                    steps.append(
                        {
                            "tool_name": "fetch_and_parse_pdf",
                            "arguments": {"arxiv_id": aid},
                            "rationale": f"Ensure paper {aid} is indexed before retrieval.",
                        }
                    )
            steps.append(
                {
                    "tool_name": "retrieve_chunks",
                    "arguments": {"query": user_prompt, "top_k": 5},
                    "rationale": "Retrieve top-K hybrid dense+BM25 chunks with confidence gating.",
                }
            )

        payload = {"intent": intent, "steps": steps}
        return schema_model.model_validate(payload)

    @staticmethod
    def extract_matrix_row(user_prompt: str, schema_model: type[T]) -> T:
        blocks = CHUNK_BLOCK_RE.findall(user_prompt)
        if blocks:
            cid, title, _sec, body = blocks[0]
            arxiv_id = cid.split("#")[0]
        else:
            arxiv_id = "0000.00000"
            cid = "0000.00000#c0"
            title = "Unknown Paper"
            body = user_prompt[:300]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        problem = sentences[0] if sentences else body[:120]
        method = sentences[1] if len(sentences) > 1 else body[:140]
        findings = sentences[2] if len(sentences) > 2 else body[-140:]

        # Find dataset/benchmark mentions across all chunks
        full_body = " ".join(b[3] for b in blocks) if blocks else body
        dataset_matches = re.findall(
            r"\b(?:MMLU|GSM8K|HumanEval|Natural Questions|TriviaQA|HotpotQA|MS-MARCO|SQuAD|FEVER|KILT|WikiText|GLUE|ImageNet)\b",
            full_body,
        )
        datasets = ", ".join(dict.fromkeys(dataset_matches)) if dataset_matches else "Standard domain benchmarks"

        payload = {
            "arxiv_id": arxiv_id,
            "title": title,
            "problem": problem[:180],
            "methodology": method[:180],
            "datasets_or_benchmarks": datasets,
            "key_findings": findings[:180],
            "limitations": "Compute/latency trade-offs noted in evaluation.",
            "primary_chunk_id": cid,
        }
        return schema_model.model_validate(payload)


class LlamaCppProvider(LLMProvider):
    """Local quantized GGUF model runner using llama-cpp-python with JSON-schema GBNF constraints."""

    provider_name = "llamacpp_qwen_gguf"

    def __init__(self, model_size: str | None = None) -> None:
        settings = get_settings()
        self.model_size = model_size or settings.local_model_size
        self.repo_id = (
            settings.gguf_repo_3b if self.model_size == "3B" else settings.gguf_repo_1_5b
        )
        self.filename = (
            settings.gguf_file_3b if self.model_size == "3B" else settings.gguf_file_1_5b
        )
        self.models_dir = Path(settings.models_dir)
        self.n_ctx = settings.llm_n_ctx
        self.n_threads = settings.llm_n_threads
        self._llm: Any = None
        self._load_attempted = False

    def _ensure_loaded(self) -> Any:
        if self._load_attempted:
            return self._llm
        self._load_attempted = True

        settings = get_settings()
        if settings.environment == "test" or settings.llm_provider == "heuristic":
            return None

        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]

            local_gguf = self.models_dir / self.filename
            if not local_gguf.exists():
                from huggingface_hub import hf_hub_download

                self.models_dir.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_dir=str(self.models_dir),
                )
                local_gguf = Path(downloaded)

            self._llm = Llama(
                model_path=str(local_gguf),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            get_logger().info(
                "llamacpp_model_loaded",
                model_size=self.model_size,
                path=str(local_gguf),
            )
        except Exception as exc:
            get_logger().warning(
                "llamacpp_unavailable_using_extractive_engine",
                model_size=self.model_size,
                error=str(exc),
            )
            self._llm = None
        return self._llm

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        llm = self._ensure_loaded()
        if llm is None:
            return ExtractiveGroundedEngine.synthesize_grounded_answer(user_prompt)

        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = str(resp["choices"][0]["message"]["content"]).strip()
        return content

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[T],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> T:
        llm = self._ensure_loaded()
        if llm is not None:
            try:
                json_schema = schema_model.model_json_schema()
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_object",
                        "schema": json_schema,
                    },
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw = str(resp["choices"][0]["message"]["content"])
                return parse_json_into_model(raw, schema_model)
            except Exception as exc:
                get_logger().warning("llamacpp_structured_fallback", error=str(exc))

        # Deterministic schema-guaranteed fallback
        schema_keys = set(schema_model.model_fields.keys())
        if "steps" in schema_keys and "intent" in schema_keys:
            return ExtractiveGroundedEngine.plan_for_query(user_prompt, schema_model)
        if "methodology" in schema_keys and "primary_chunk_id" in schema_keys:
            return ExtractiveGroundedEngine.extract_matrix_row(user_prompt, schema_model)
        raise ValueError(f"Unsupported schema fallback for {schema_model.__name__}")
