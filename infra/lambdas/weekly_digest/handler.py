"""AWS EventBridge Weekly Digest Lambda: reads followed topics from DynamoDB, queries arXiv, and sends HTML digest via SES."""

from __future__ import annotations

import html
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_latest_papers_for_topic(topic: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Fetch most recent arXiv papers for a subscribed topic."""
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{topic}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API_URL}?{params}"
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
        xml_text = resp.read().decode("utf-8")

    root = ET.fromstring(xml_text)  # noqa: S314
    results: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        if not raw_id:
            continue
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]
        title = " ".join(entry.findtext("atom:title", default="", namespaces=ATOM_NS).split())
        summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split())
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)[:10]
        results.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "published": published,
                "summary": summary[:360] + ("..." if len(summary) > 360 else ""),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return results


def render_digest_html(email: str, topics_map: dict[str, list[dict[str, Any]]]) -> str:
    """Render clean HTML email body for the weekly arXiv digest."""
    sections: list[str] = []
    for topic, papers in topics_map.items():
        items_html = []
        for p in papers:
            items_html.append(
                f"<li><strong><a href='{html.escape(p['url'])}'>{html.escape(p['title'])}</a></strong> "
                f"(<code>arXiv:{html.escape(p['arxiv_id'])}</code>, {html.escape(p['published'])})<br/>"
                f"<span>{html.escape(p['summary'])}</span></li>"
            )
        joined_items = "\n".join(items_html) or "<li>No new papers this week.</li>"
        sections.append(f"<h3>Topic: {html.escape(topic)}</h3><ul>{joined_items}</ul>")

    return (
        "<html><body style='font-family: sans-serif; line-height: 1.5;'>"
        "<h2>LitLens Weekly arXiv Research Digest</h2>"
        f"<p>Prepared for <strong>{html.escape(email)}</strong></p>"
        + "\n".join(sections)
        + "<hr/><p style='font-size:12px;color:#666;'>Sent by LitLens Serverless Digest (AWS Free Tier)</p>"
        "</body></html>"
    )


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Query active DynamoDB subscriptions, compile per-recipient digest, and send via SES."""
    table_name = os.getenv("DYNAMODB_TABLE_NAME", "LitLensStore")
    sender = os.getenv("SES_SENDER_EMAIL", "digest@example.com")
    region = os.getenv("AWS_REGION", "us-east-1")

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)
    resp = table.query(KeyConditionExpression=Key("PK").eq("SUBSCRIPTION#ACTIVE"))

    by_email: dict[str, list[str]] = defaultdict(list)
    for item in resp.get("Items", []):
        payload = json.loads(item.get("payload", "{}"))
        email = payload.get("email")
        topic = payload.get("topic")
        if email and topic and topic not in by_email[email]:
            by_email[email].append(topic)

    # Allow dry-run override in event payload for testing without live arXiv HTTP call
    mock_papers: list[dict[str, Any]] | None = event.get("mock_papers") if isinstance(event, dict) else None

    ses = boto3.client("ses", region_name=region)
    sent_count = 0

    for email, topics in list(by_email.items())[:50]:  # Hard cap 50 emails/run for Free Tier safety
        topics_map: dict[str, list[dict[str, Any]]] = {}
        for t in topics:
            topics_map[t] = (
                mock_papers
                if mock_papers is not None
                else fetch_latest_papers_for_topic(t, max_results=3)
            )
        html_body = render_digest_html(email, topics_map)
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": [email]},
            Message={
                "Subject": {"Data": f"LitLens Weekly arXiv Digest ({len(topics)} topics)"},
                "Body": {"Html": {"Data": html_body}},
            },
        )
        sent_count += 1

    return {"statusCode": 200, "emails_sent": sent_count}
