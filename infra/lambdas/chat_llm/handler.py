"""AWS Free-Tier Lambda Function URL handler for fast Qwen3.8-27B / GPT-OSS-20B inference via Groq."""

from __future__ import annotations

import json
import os
import pathlib
import re
import urllib.error
import urllib.request
from typing import Any

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
INDEX_HTML_PATH = pathlib.Path(__file__).parent / "index.html"


def _strip_think_tags(text: str) -> str:
    """Remove internal <think>...</think> blocks if emitted by Qwen3 reasoning mode."""
    cleaned = THINK_BLOCK_RE.sub("", text).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].replace("<think>", "").strip()
    return cleaned


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Handle GET (serve LitLens HTML UI over AWS HTTPS), OPTIONS (CORS), and POST (Groq LPU chat)."""
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "POST"
    ).upper()
    if method == "OPTIONS":
        return {"statusCode": 200, "body": ""}
    if method == "GET":
        html_body = (
            INDEX_HTML_PATH.read_text(encoding="utf-8")
            if INDEX_HTML_PATH.exists()
            else "<html><body><h1>LitLens AWS HTTPS Endpoint Ready</h1></body></html>"
        )
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/html; charset=utf-8",
                "Cache-Control": "public, max-age=60",
            },
            "body": html_body,
        }

    raw_body = event.get("body") or "{}"
    body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body

    messages = body.get("messages") or []
    if not messages:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "messages array is required"}),
        }

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    primary_model = body.get("model") or os.getenv("GROQ_MODEL", "allam-2-7b")
    models_to_try = [primary_model, "allam-2-7b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

    last_err = "Unknown error"
    for model_id in dict.fromkeys(models_to_try):
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": float(body.get("temperature", 0.2)),
            "max_tokens": min(int(body.get("max_tokens", 700)), 1200),
        }
        if "qwen3" in model_id:
            payload["reasoning_effort"] = "none"

        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "LitLens-Lambda/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                raw_content = data["choices"][0]["message"]["content"]
                answer = _strip_think_tags(raw_content)
                return {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(
                        {
                            "answer": answer,
                            "model": model_id,
                        }
                    ),
                }
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            # Retry without reasoning_effort if model doesn't accept that param
            if exc.code == 400 and "reasoning_effort" in payload:
                payload.pop("reasoning_effort", None)
                req2 = urllib.request.Request(
                    GROQ_API_URL,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "LitLens-Lambda/1.0",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req2, timeout=15) as resp2:  # noqa: S310
                        data2 = json.loads(resp2.read().decode("utf-8"))
                        raw_content2 = data2["choices"][0]["message"]["content"]
                        answer2 = _strip_think_tags(raw_content2)
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": json.dumps({"answer": answer2, "model": model_id}),
                        }
                except Exception as exc2:
                    last_err = str(exc2)
                    continue
            last_err = f"HTTP {exc.code}: {err_body[:200]}"
        except Exception as exc:
            last_err = str(exc)

    return {
        "statusCode": 502,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": last_err}),
    }
