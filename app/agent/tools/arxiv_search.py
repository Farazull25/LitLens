"""arXiv search tool with local Atom XML client and optional AWS Lambda Function URL (AWS_IAM SigV4) support."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlencode

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.observability.logger import get_logger

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivSearchInput(BaseModel):
    """Pydantic schema for arxiv_search tool input."""

    query: str = Field(..., min_length=2, max_length=400, description="Search query or arXiv ID")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum papers to return")


class ArxivPaperMeta(BaseModel):
    """Normalized metadata for an arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    summary: str
    pdf_url: str


def parse_arxiv_atom_xml(xml_text: str) -> list[ArxivPaperMeta]:
    """Parse arXiv Atom XML response into normalized ArxivPaperMeta objects."""
    root = ET.fromstring(xml_text)  # noqa: S314 (arXiv official Atom feed)
    papers: list[ArxivPaperMeta] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        if not raw_id:
            continue
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]
        title = " ".join(
            entry.findtext("atom:title", default="Untitled", namespaces=ATOM_NS).split()
        )
        summary = " ".join(
            entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split()
        )
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)[:10]
        authors = [
            a.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
            for a in entry.findall("atom:author", ATOM_NS)
            if a.findtext("atom:name", default="", namespaces=ATOM_NS)
        ]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        papers.append(
            ArxivPaperMeta(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors or ["Unknown Author"],
                published=published or "2024-01-01",
                summary=summary,
                pdf_url=pdf_url,
            )
        )
    return papers


def _invoke_lambda_function_url(url: str, payload: dict[str, Any], region: str) -> list[dict[str, Any]]:
    """Invoke AWS Lambda Function URL signed with AWS_IAM SigV4 credentials."""
    session = boto3.Session()
    credentials = session.get_credentials()
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    if credentials is not None:
        frozen_creds = credentials.get_frozen_credentials()
        aws_req = AWSRequest(method="POST", url=url, data=body_bytes, headers=headers)
        SigV4Auth(frozen_creds, "lambda", region).add_auth(aws_req)
        headers = dict(aws_req.headers)

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, content=body_bytes, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("papers", []))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0), reraise=True)
def _query_arxiv_direct(query: str, max_results: int) -> list[ArxivPaperMeta]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urlencode(params)}"
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return parse_arxiv_atom_xml(resp.text)


def run_arxiv_search(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Validate arguments and search arXiv via Lambda Function URL or direct Atom API."""
    validated = ArxivSearchInput.model_validate(raw_args)
    settings = get_settings()

    if settings.aws_enabled and settings.lambda_arxiv_function_url:
        try:
            papers_raw = _invoke_lambda_function_url(
                url=settings.lambda_arxiv_function_url,
                payload={"action": "search", "query": validated.query, "max_results": validated.max_results},
                region=settings.aws_region,
            )
            return {"count": len(papers_raw), "papers": papers_raw, "source": "lambda_function_url"}
        except Exception as exc:
            get_logger().warning("lambda_arxiv_search_fallback_to_direct", error=str(exc))

    try:
        papers = _query_arxiv_direct(validated.query, validated.max_results)
        return {
            "count": len(papers),
            "papers": [p.model_dump() for p in papers],
            "source": "arxiv_direct_api",
        }
    except Exception as exc:
        get_logger().warning("arxiv_search_network_error", error=str(exc))
        return {"count": 0, "papers": [], "error": str(exc)}
