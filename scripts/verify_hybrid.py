"""Verify FTS + hybrid search on the real `lectures_sample` table (cdgl-csl).

Unlike scripts/lancedb_probe.py (seeded random vectors), this embeds queries
locally with Nomic v1.5 via lecture_search.embed.embed_query and runs against
the real table loaded by scripts/load_sample.py:

1. Verify the FTS index on `text` and run an FTS-only (BM25) query.
2. Hybrid search with explicit .vector().text() and
   MRRReranker(weight_vector, weight_fts) at weight_vector in {0.2, 0.5, 0.8}
   (weight_fts = 1 - weight_vector); show that rankings differ by weight.
3. Print everything needed to rewrite docs/lancedb-findings.md.

Never prints secrets. Read-only against the table (no writes, no reindexing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import lancedb
from lancedb.rerankers import MRRReranker

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # lecture_search is not an installed package

from lecture_search.embed import embed_query  # noqa: E402

LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
TABLE_NAME = "lectures_sample"
TOP_K = 5

QUERIES = [
    "how does dbt transform data in the warehouse",
    "distributed SQL database replication",
    "YugabyteDB",
]
WEIGHTS_VECTOR = [0.2, 0.5, 0.8]


def load_api_key() -> str:
    env_file = ROOT_DIR / ".env.local"
    if not env_file.exists():
        msg = ".env.local not found"
        raise FileNotFoundError(msg)
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("LANCEDB_API_KEY="):
            return line.split("=", 1)[1].strip('"\'')
    msg = "LANCEDB_API_KEY not found in .env.local"
    raise ValueError(msg)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "id": row["id"],
            "title": row["title"],
            "text_preview": row["text"][:80].replace("\n", " "),
            **{
                key: round(float(row[key]), 4)
                for key in ("_score", "_distance", "_relevance_score")
                if key in row and row[key] is not None
            },
        }
        for rank, row in enumerate(rows, start=1)
    ]


def main() -> int:
    db = lancedb.connect(LANCEDB_URI, region=LANCEDB_REGION, api_key=load_api_key())
    table = db.open_table(TABLE_NAME)
    row_count = table.count_rows()
    print(f"Table {TABLE_NAME}: {row_count} rows")

    # --- Step 1: FTS index on text -------------------------------------
    indices = table.list_indices()
    fts_indices = [
        idx for idx in indices
        if idx.index_type == "FTS" and "text" in (idx.columns or [])
    ]
    if not fts_indices:
        print("FAIL: no FTS index on column 'text'", file=sys.stderr)
        return 1
    idx = fts_indices[0]
    print(f"FTS index verified: {idx.name} on {idx.columns} "
          f"({idx.num_indexed_rows} indexed rows)")

    report: dict[str, Any] = {"table": TABLE_NAME, "rows": row_count,
                    "fts_index": {"name": idx.name, "columns": idx.columns},
                    "queries": {}}

    for query in QUERIES:
        print(f"\n=== Query: {query!r} ===")
        q_vec = embed_query(query)
        assert len(q_vec) == 768

        entry: dict[str, Any] = {}

        # FTS-only (BM25)
        fts_rows = (
            table.search(query, query_type="fts")
            .limit(TOP_K)
            .to_list()
        )
        entry["fts_only"] = summarize(fts_rows)
        print(f"FTS-only top {TOP_K}: {[r['id'] for r in entry['fts_only']]}")

        # Vector-only
        vec_rows = (
            table.search(q_vec, query_type="vector")
            .limit(TOP_K)
            .to_list()
        )
        entry["vector_only"] = summarize(vec_rows)
        print(f"Vector-only top {TOP_K}: {[r['id'] for r in entry['vector_only']]}")

        # Hybrid with explicit .vector().text() + MRRReranker at 3 weights
        hybrid: dict[str, list[dict[str, Any]]] = {}
        for w_vec in WEIGHTS_VECTOR:
            w_fts = round(1.0 - w_vec, 1)
            rows = (
                table.search(query_type="hybrid")
                .vector(q_vec)
                .text(query)
                .rerank(MRRReranker(weight_vector=w_vec, weight_fts=w_fts))
                .limit(TOP_K)
                .to_list()
            )
            hybrid[f"vector={w_vec}/fts={w_fts}"] = summarize(rows)
            print(f"Hybrid w_vec={w_vec} w_fts={w_fts}: "
                  f"{[r['id'] for r in hybrid[f'vector={w_vec}/fts={w_fts}']]}")
        entry["hybrid_mrr"] = hybrid

        # Rankings must differ across weights for at least one query overall
        rank_lists = [tuple(r["id"] for r in rows) for rows in hybrid.values()]
        entry["rankings_differ_across_weights"] = len(set(rank_lists)) > 1

        report["queries"][query] = entry

    out = ROOT_DIR / "data" / "hybrid_verification.json"
    _ = out.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
