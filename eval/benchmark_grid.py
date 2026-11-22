"""Benchmark grid comparing chunk sizes (256, 512, 1024) x model sizes (1.5B, 3B) and outputting a Markdown table."""

from __future__ import annotations

from typing import Any

from eval.runner import run_evaluation


def run_benchmark_grid(
    chunk_sizes: tuple[int, ...] = (256, 512, 1024),
    model_sizes: tuple[str, ...] = ("1.5B", "3B"),
    force_hash_embeddings: bool = True,
) -> dict[str, Any]:
    """Evaluate all (chunk_size, model_size) combinations on the 30-question Golden Set."""
    rows: list[dict[str, Any]] = []

    for model_size in model_sizes:
        for chunk_size in chunk_sizes:
            res = run_evaluation(
                chunk_size_tokens=chunk_size,
                model_size=model_size,
                force_hash_embeddings=force_hash_embeddings,
                persist_to_dynamodb=False,
            )
            rows.append(
                {
                    "model_size": f"Qwen2.5-{model_size}-Instruct (Q4_K_M)",
                    "chunk_size": chunk_size,
                    "retrieval_hit_rate": res["retrieval_hit_rate"],
                    "mrr": res["mrr"],
                    "citation_precision": res["citation_precision"],
                    "citation_recall": res["citation_recall"],
                    "faithfulness": res["faithfulness"],
                    "abstention_accuracy": res["abstention_accuracy"],
                    "avg_latency_ms": res["avg_latency_ms"],
                }
            )

    markdown = format_grid_markdown(rows)
    return {"rows": rows, "markdown": markdown}


def format_grid_markdown(rows: list[dict[str, Any]]) -> str:
    """Format benchmark comparison rows into a GitHub-Flavored Markdown table."""
    header = (
        "| Model Size | Chunk Size (Tokens) | Retrieval Hit@5 | MRR | Citation Precision | "
        "Citation Recall | Faithfulness | Abstention Acc. | Avg Latency (ms) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['model_size']} | {r['chunk_size']} | {r['retrieval_hit_rate']:.2%} | "
            f"{r['mrr']:.3f} | {r['citation_precision']:.2%} | {r['citation_recall']:.2%} | "
            f"{r['faithfulness']:.2%} | {r['abstention_accuracy']:.2%} | {r['avg_latency_ms']:.1f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    grid = run_benchmark_grid()
    print(grid["markdown"])
