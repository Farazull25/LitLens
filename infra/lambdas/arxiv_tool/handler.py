"""AWS Lambda handler for arXiv search and metadata caching behind a Lambda Function URL (AWS_IAM auth)."""

from __future__ import annotations

import gzip
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

import boto3

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_atom(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)  # noqa: S314
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        if not raw_id:
            continue
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ATOM_NS).split())
        summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split())
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)[:10]
        authors = [
            a.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
            for a in entry.findall("atom:author", ATOM_NS)
            if a.findtext("atom:name", default="", namespaces=ATOM_NS)
        ]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors or ["Unknown"],
                "published": published,
                "summary": summary,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            }
        )
    return papers


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Handle Lambda Function URL POST requests for arXiv searches."""
    raw_body = event.get("body") or "{}"
    body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body

    query = str(body.get("query", "retrieval augmented generation")).strip()
    max_results = min(max(int(body.get("max_results", 5)), 1), 20)

    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    req_url = f"{ARXIV_API_URL}?{params}"
    with urllib.request.urlopen(req_url, timeout=15) as response:  # noqa: S310
        xml_payload = response.read().decode("utf-8")

    papers = _parse_atom(xml_payload)

    bucket = os.getenv("S3_BUCKET_NAME")
    if bucket and body.get("cache_metadata", False):
        s3 = boto3.client("s3")
        for p in papers:
            key = f"metadata/{p['arxiv_id']}.json.gz"
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=gzip.compress(json.dumps(p).encode("utf-8")),
                ContentType="application/json",
                ContentEncoding="gzip",
            )

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"count": len(papers), "papers": papers}),
    }
