"""Export metadata and text for one Day 6 pgvector strategy as a JSON array.

Embeddings are intentionally omitted. This exports source text, so only share
the output when the corpus owner has authorized redistribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import dotenv_values


DAY7_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day6-root", type=Path, default=DAY7_ROOT.parent / "Lab Day 6")
    parser.add_argument("--strategy", choices=("A", "B", "C"), default="B")
    parser.add_argument("--output", type=Path, default=DAY7_ROOT / "data" / "day6_chunks_B.json")
    args = parser.parse_args()

    env = dotenv_values(args.day6_root / "starter" / ".env")
    database_url = env.get("DATABASE_URL")
    if not database_url:
        raise ValueError(f"DATABASE_URL missing in {args.day6_root / 'starter' / '.env'}")

    import psycopg

    with psycopg.connect(str(database_url)) as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, document_id, source, page, text, metadata
            FROM day6_chunks
            WHERE strategy = %s
            ORDER BY page, chunk_id
            """,
            (args.strategy,),
        ).fetchall()

    chunks = [
        {
            "id": row[0],
            "source_id": row[1],
            "source": row[2],
            "page": row[3],
            "text": row[4],
            "metadata": row[5] or {},
        }
        for row in rows
    ]
    if not chunks:
        raise ValueError(f"No chunks found for Day 6 strategy {args.strategy}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(chunks)} chunks for strategy {args.strategy} to {args.output}")
    print("The JSON contains source text; check corpus permissions before sharing it.")


if __name__ == "__main__":
    main()
