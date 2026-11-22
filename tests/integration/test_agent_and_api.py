"""Integration tests for LangGraph Plan-then-Execute agent, Moto AWS Free-Tier storage, Lambdas, and FastAPI."""

from __future__ import annotations

from typing import Any

import boto3
from app.agent.graph import LitLensCopilotAgent
from app.agent.tools.fetch_parse_pdf import run_fetch_and_parse_pdf
from app.main import app
from app.rag.vector_store import FAISSVectorStore
from app.storage.dynamodb import DynamoDBStore
from app.storage.s3 import S3Store
from eval.runner import run_evaluation
from fastapi.testclient import TestClient
from infra.lambdas.weekly_digest.handler import lambda_handler as digest_lambda_handler
from moto import mock_aws


def test_aws_free_tier_s3_and_dynamodb_single_table(
    aws_free_tier_mocks: tuple[S3Store, DynamoDBStore],
    temp_vector_store: FAISSVectorStore,
) -> None:
    s3_store, ddb_store = aws_free_tier_mocks

    # 1. Ingest paper and verify single compressed .json.gz bundle in S3
    res = run_fetch_and_parse_pdf(
        {
            "arxiv_id": "2401.10001",
            "raw_text_override": "Abstract\nHybrid retrieval uses Reciprocal Rank Fusion with k=60.",
            "title_override": "Hybrid RRF Retrieval",
        },
        vector_store=temp_vector_store,
        s3_store=s3_store,
    )
    assert res["s3_key"] == "papers/2401.10001.json.gz"
    loaded_bundle = s3_store.load_paper_bundle("2401.10001")
    assert loaded_bundle is not None
    assert loaded_bundle["title"] == "Hybrid RRF Retrieval"

    # 2. Verify DynamoDB single-table memory, cache, and topic subscriptions
    assert ddb_store.save_user_memory("u1", "focus", "RAG evaluation", category="topic") is True
    memories = ddb_store.list_user_memories("u1")
    assert len(memories) == 1
    assert memories[0]["value"] == "RAG evaluation"

    ddb_store.set_cached_response("test query", {"final_answer": "Cached [2401.10001#c0]"})
    cached = ddb_store.get_cached_response("  TEST   query ")
    assert cached is not None
    assert cached["final_answer"] == "Cached [2401.10001#c0]"


def test_langgraph_agent_grounded_qa_lit_matrix_and_refusal(
    temp_vector_store: FAISSVectorStore,
) -> None:
    ddb = DynamoDBStore()
    for pid, title, text in [
        (
            "2401.10001",
            "Hybrid RRF Architecture",
            "Abstract\nHybrid RRF uses k=60 to merge dense and BM25 scores.\n\nResults\nImproves SciFact recall by 14.2 percent.",
        ),
        (
            "2401.10002",
            "GBNF Grammar Planning",
            "Abstract\nGBNF grammars constrain 1.5B Q4_K_M models to 4 steps.\n\nResults\nAchieves 100.0 percent JSON validity on HotpotQA.",
        ),
    ]:
        run_fetch_and_parse_pdf(
            {"arxiv_id": pid, "raw_text_override": text, "title_override": title},
            vector_store=temp_vector_store,
        )

    agent = LitLensCopilotAgent(vector_store=temp_vector_store, ddb_store=ddb)

    # 1. Grounded Q&A with verifiable citation
    qa_state = agent.run("What RRF constant k is used to merge dense and BM25 scores?")
    assert qa_state["low_confidence_refusal"] is False
    assert any(c.startswith("2401.10001#c") for c in qa_state["citations"])
    assert "[2401.10001#c" in qa_state["final_answer"]

    # 2. Second identical call hits DynamoDB query cache
    cached_state = agent.run("What RRF constant k is used to merge dense and BM25 scores?")
    assert cached_state["cached_hit"] is True

    # 3. Literature Review Comparison Matrix
    matrix_state = agent.run("Build a literature review comparison matrix for 2401.10001 and 2401.10002")
    assert "| Paper (arXiv ID) |" in matrix_state["final_answer"]
    assert "2401.10001" in matrix_state["final_answer"]
    assert "2401.10002" in matrix_state["final_answer"]
    assert len(matrix_state["citations"]) == 2

    # 4. Out-of-domain query triggers low-confidence refusal gate
    refusal_state = agent.run(
        "What fermentation temperature produces highest yield in sourdough rye bakeries?"
    )
    assert refusal_state["low_confidence_refusal"] is True
    assert refusal_state["citations"] == []
    assert "below the required safety threshold" in refusal_state["final_answer"]


def test_weekly_digest_lambda_with_ses_and_dynamodb() -> None:
    with mock_aws():
        region = "us-east-1"
        ddb_client = boto3.client("dynamodb", region_name=region)
        ddb_client.create_table(
            TableName="LitLensStore",
            BillingMode="PROVISIONED",
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
        )
        ses_client = boto3.client("ses", region_name=region)
        ses_client.verify_email_identity(EmailAddress="digest@example.com")
        ses_client.verify_email_identity(EmailAddress="researcher@example.com")

        store = DynamoDBStore(table_name="LitLensStore", region=region)
        store.aws_enabled = True
        store.subscribe_topic("researcher@example.com", "Retrieval Augmented Generation")

        resp = digest_lambda_handler(
            {
                "mock_papers": [
                    {
                        "arxiv_id": "2401.10001",
                        "title": "Hybrid RRF Search",
                        "published": "2024-01-15",
                        "summary": "Dense + BM25 RRF fusion.",
                        "url": "https://arxiv.org/abs/2401.10001",
                    }
                ]
            },
            None,
        )
        assert resp["statusCode"] == 200
        assert resp["emails_sent"] == 1


def test_fastapi_endpoints_and_eval_harness() -> None:
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["indexed_papers"] >= 6

    chat_resp = client.post(
        "/api/chat",
        json={"query": "By what percentage does hybrid RRF improve top-5 recall on SciFact?"},
    )
    assert chat_resp.status_code == 200
    data: dict[str, Any] = chat_resp.json()
    assert data["low_confidence_refusal"] is False
    assert any(c.startswith("2401.10001#") for c in data["citations"])

    eval_summary = run_evaluation(chunk_size_tokens=512, model_size="1.5B", persist_to_dynamodb=False)
    assert eval_summary["num_questions"] == 30
    assert eval_summary["retrieval_hit_rate"] >= 0.85
    assert eval_summary["citation_precision"] >= 0.85
    assert eval_summary["abstention_accuracy"] >= 0.85


def test_pdf_upload_and_citation_chat_endpoint() -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Quantum Sparse Attention Paper\n\nAbstract\nWe introduce Quantum Sparse Attention (QSA) which reduces transformer KV-cache memory by 73.4 percent on LongBench.",
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    client = TestClient(app)
    upload_resp = client.post(
        "/api/upload_pdf",
        files={"file": ("quantum_sparse_attention.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 200
    up_data = upload_resp.json()
    uploaded_id = up_data["arxiv_id"]
    assert uploaded_id.startswith("upload-quantum-sparse")
    assert up_data["chunks_total"] >= 1

    chat_resp = client.post(
        "/api/chat",
        json={
            "query": "By what percent does Quantum Sparse Attention reduce transformer KV-cache memory on LongBench?",
            "arxiv_ids": [uploaded_id],
        },
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["low_confidence_refusal"] is False
    assert any(c.startswith(f"{uploaded_id}#c") for c in chat_data["citations"])
    assert "73.4" in chat_data["answer"]
