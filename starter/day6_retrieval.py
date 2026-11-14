"""Compare Day 6 dense retrieval, Day 7 TF-IDF, and RRF hybrid on dev/test.

Run with the Python environment from Day 6, which has psycopg, pgvector,
sentence-transformers, and python-dotenv installed. Gold is used only after
retrieval to score the returned chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from baseline import TfidfIndex, hit, pack_context, validate


DAY7_ROOT = Path(__file__).resolve().parents[1]
DAY6_ROOT = Path(os.getenv("DAY6_ROOT", str(DAY7_ROOT.parent / "Lab Day 6")))
DAY6_STARTER = DAY6_ROOT / "starter"
sys.path.insert(0, str(DAY6_STARTER))


def rrf(dense: list[dict[str, Any]], lexical: list[dict[str, Any]], *, k: int = 60) -> list[dict[str, Any]]:
    """Fuse two ranked lists with Reciprocal Rank Fusion."""
    by_id: dict[str, dict[str, Any]] = {}
    fused: dict[str, float] = {}
    for rows in (dense, lexical):
        for rank, row in enumerate(rows, start=1):
            cid = row["id"]
            by_id.setdefault(cid, row)
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))
    return [{**by_id[cid], "score": fused[cid], "strategy": "hybrid_rrf"} for cid in ordered]


def normalize_dense(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["chunk_id"],
            "source_id": row["document_id"],
            "source": row["source"],
            "page": row["page"],
            "text": row["text"],
            "metadata": row["metadata"],
            "score": row["score"],
            "strategy": "dense_day6_B",
        }
        for row in rows
    ]


def score_run(
    method: str,
    questions: list[dict[str, Any]],
    results_by_id: dict[str, list[dict[str, Any]]],
    elapsed_by_id: dict[str, float],
    budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = []
    for question in questions:
        rows = results_by_id[question["id"]]
        ranked = [{**row, "rank": rank, "hit": hit(row, question["gold"])} for rank, row in enumerate(rows[:3], 1)]
        context, context_ids = pack_context(rows[:3], budget=budget)
        context_hit = any(row["hit"] and row["id"] in context_ids for row in ranked)
        traces.append({
            "query_id": question["id"],
            "split": question["split"],
            "question": question["question"],
            "method": method,
            "top_k": 3,
            "budget_chars": budget,
            "results": ranked,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_by_id[question["id"]],
            "hit1": bool(ranked and ranked[0]["hit"]),
            "hit3": any(row["hit"] for row in ranked),
            "context_hit": context_hit,
        })
    n = len(traces)
    summary = {
        "method": method,
        "split": questions[0]["split"] if questions else None,
        "n": n,
        "hit1": sum(row["hit1"] for row in traces),
        "hit3": sum(row["hit3"] for row in traces),
        "context_hit": sum(row["context_hit"] for row in traces),
        "avg_context_chars": statistics.mean(row["context_chars"] for row in traces) if n else 0,
        "median_ms": statistics.median(row["elapsed_ms"] for row in traces) if n else 0,
        "service_cost_usd": 0,
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense Day 6, TF-IDF, and hybrid RRF")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--strategy", choices=("A", "B", "C"), default="B")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_k < 3 or args.budget < 1:
        raise ValueError("candidate-k must be >= 3 and budget must be positive")

    chunks_path = DAY7_ROOT / "data" / f"day6_chunks_{args.strategy}.json"
    questions_path = DAY7_ROOT / "data" / "benchmark-day7" / "questions.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = [row for row in all_questions if row.get("split") == args.split]
    validate(chunks, questions)

    from dotenv import load_dotenv

    load_dotenv(DAY6_STARTER / ".env")
    database_url = os.getenv("DATABASE_URL")
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if not database_url:
        raise ValueError(f"DATABASE_URL is missing from {DAY6_STARTER / '.env'}")

    from src.embeddings import MyEmbedder
    from src.store import PgVectorStore

    setup_started = time.perf_counter()
    embedder = MyEmbedder(model_name)
    store = PgVectorStore(database_url, dimensions=embedder.dimensions)
    if embedder.dimensions != 384:
        raise ValueError(f"Day 6 index expects 384 dimensions, got {embedder.dimensions}")
    setup_seconds = time.perf_counter() - setup_started

    lexical_index = TfidfIndex(chunks)
    dense_by_id: dict[str, list[dict[str, Any]]] = {}
    lexical_by_id: dict[str, list[dict[str, Any]]] = {}
    hybrid_by_id: dict[str, list[dict[str, Any]]] = {}
    dense_ms: dict[str, float] = {}
    lexical_ms: dict[str, float] = {}
    hybrid_ms: dict[str, float] = {}
    embedding_seconds = 0.0
    dense_seconds = 0.0

    for question in questions:
        qid = question["id"]
        filters = question.get("filter")
        pipeline_started = time.perf_counter()
        started = time.perf_counter()
        query_vector = embedder.embed_query(question["question"])
        embedding_seconds += time.perf_counter() - started
        started = time.perf_counter()
        dense_rows = normalize_dense(store.search(
            strategy=args.strategy,
            query_vector=query_vector,
            top_k=args.candidate_k,
            filters=filters,
        ))
        dense_seconds += time.perf_counter() - started
        dense_by_id[qid] = dense_rows
        dense_ms[qid] = (time.perf_counter() - pipeline_started) * 1000

        started = time.perf_counter()
        lexical_rows = lexical_index.search(question["question"], top_k=args.candidate_k)
        lexical_by_id[qid] = lexical_rows
        lexical_ms[qid] = (time.perf_counter() - started) * 1000

        hybrid_by_id[qid] = rrf(dense_rows, lexical_rows)
        hybrid_ms[qid] = (time.perf_counter() - pipeline_started) * 1000

    summaries = []
    traces = []
    for method, result_map, elapsed in (
        ("dense_day6", dense_by_id, dense_ms),
        ("tfidf", lexical_by_id, lexical_ms),
        ("hybrid_rrf", hybrid_by_id, hybrid_ms),
    ):
        summary, method_traces = score_run(method, questions, result_map, elapsed, args.budget)
        summaries.append(summary)
        traces.extend(method_traces)

    for summary in summaries:
        summary["candidate_k"] = args.candidate_k
        summary["day6_strategy"] = args.strategy
        summary["rrf_k"] = 60
    metadata = {
        "embedding_model": model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL + pgvector",
        "day6_strategy": args.strategy,
        "candidate_k": args.candidate_k,
        "rrf_k": 60,
        "context_budget_chars": args.budget,
        "setup_seconds_including_model_load": setup_seconds,
        "query_embedding_seconds_total": embedding_seconds,
        "dense_search_seconds_total": dense_seconds,
        "retrieval_methods": ["dense_day6", "tfidf", "hybrid_rrf"],
        "note": "Gold is used only after retrieval to score evidence; it is not passed to the retriever.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summaries, "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
