"""Verify FTS + hybrid search on the real `lectures_sample` table (cdgl-csl).

Unlike scripts/lancedb_probe.py (seeded random vectors), this embeds queries
locally with Nomic v1.5 via lecture_search.embed.embed_query and runs against
the real table loaded by scripts/load_sample.py:

1. Verify the FTS index on `text` and run an FTS-only (BM25) query.
2. Hybrid search with explicit .vector().text() and
   MRRReranker(weight_vector, weight_fts) at weight_vector in {0.2, 0.5, 0.8}
   (weight_fts = 1 - weight_vector); show that rankings differ by weight.
3. Write everything needed to rewrite docs/lancedb-findings.md.

Never prints secrets. Read-only against the table (no writes, no reindexing).
"""  # noqa: INP001  # standalone script, not a package

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

import lancedb
from lancedb.rerankers import MRRReranker

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # lecture_search is not an installed package

from lecture_search.embed import EMBEDDING_DIMS, embed_query  # noqa: E402
from scripts.lancedb_typed import (  # noqa: E402
    ScoredSearchRow,
    fts_search_rows,
    hybrid_search_rows,
    vector_search_rows,
)

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


class SummaryRow(TypedDict):
    """One ranked hit in a query-mode summary."""

    rank: int
    id: str
    title: str
    text_preview: str


class ScoredSummaryRow(SummaryRow, total=False):
    """SummaryRow plus whichever score column the query mode returned."""

    _score: float
    _distance: float
    _relevance_score: float


class QueryEntry(TypedDict, total=False):
    """Per-query report entry across retrieval modes."""

    fts_only: list[ScoredSummaryRow]
    vector_only: list[ScoredSummaryRow]
    hybrid_mrr: dict[str, list[ScoredSummaryRow]]
    rankings_differ_across_weights: bool


class FtsIndexInfo(TypedDict):
    """FTS index details recorded in the report."""

    name: str
    columns: list[str] | None


class Report(TypedDict):
    """Full verification report written to data/hybrid_verification.json."""

    table: str
    rows: int
    fts_index: FtsIndexInfo
    queries: dict[str, QueryEntry]


def load_api_key() -> str:
    """Read LANCEDB_API_KEY from .env.local. Never printed."""
    env_file = ROOT_DIR / ".env.local"
    if not env_file.exists():
        msg = ".env.local not found"
        raise FileNotFoundError(msg)
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("LANCEDB_API_KEY="):
            return line.split("=", 1)[1].strip("\"'")
    msg = "LANCEDB_API_KEY not found in .env.local"
    raise ValueError(msg)


def summarize(rows: list[ScoredSearchRow]) -> list[ScoredSummaryRow]:
    """Trim raw LanceDB rows to rank/id/title/preview plus score columns."""
    summarized: list[ScoredSummaryRow] = []
    for rank, row in enumerate(rows, start=1):
        entry = ScoredSummaryRow(
            rank=rank,
            id=row["id"],
            title=row["title"],
            text_preview=row["text"][:80].replace("\n", " "),
        )
        if (score := row.get("_score")) is not None:
            entry["_score"] = round(score, 4)
        if (distance := row.get("_distance")) is not None:
            entry["_distance"] = round(distance, 4)
        if (relevance := row.get("_relevance_score")) is not None:
            entry["_relevance_score"] = round(relevance, 4)
        summarized.append(entry)
    return summarized


def run_query(table: lancedb.table.Table, query: str) -> QueryEntry:
    """Run one query in FTS-only, vector-only, and weighted-hybrid modes."""
    q_vec = embed_query(query)
    if len(q_vec) != EMBEDDING_DIMS:
        msg = f"expected {EMBEDDING_DIMS}-dim query vector, got {len(q_vec)}"
        raise ValueError(msg)

    entry = QueryEntry(
        fts_only=summarize(fts_search_rows(table, query, limit=TOP_K)),
        vector_only=summarize(vector_search_rows(table, q_vec, limit=TOP_K)),
    )

    # Hybrid with explicit vector + text and MRRReranker at 3 weights.
    hybrid: dict[str, list[ScoredSummaryRow]] = {}
    for w_vec in WEIGHTS_VECTOR:
        w_fts = round(1.0 - w_vec, 1)
        hybrid[f"vector={w_vec}/fts={w_fts}"] = summarize(
            hybrid_search_rows(
                table,
                query,
                q_vec,
                reranker=MRRReranker(weight_vector=w_vec, weight_fts=w_fts),
                limit=TOP_K,
            ),
        )
    entry["hybrid_mrr"] = hybrid

    # Rankings must differ across weights for at least one query overall
    rank_lists = [tuple(r["id"] for r in rows) for rows in hybrid.values()]
    entry["rankings_differ_across_weights"] = len(set(rank_lists)) > 1
    return entry


def main() -> int:
    """Verify the FTS index, run all queries, and write the JSON report."""
    db = lancedb.connect(LANCEDB_URI, region=LANCEDB_REGION, api_key=load_api_key())
    table = db.open_table(TABLE_NAME)
    row_count = table.count_rows()

    # --- Step 1: FTS index on text -------------------------------------
    indices = table.list_indices()
    fts_indices = [
        idx
        for idx in indices
        if idx.index_type == "FTS" and "text" in (idx.columns or [])
    ]
    if not fts_indices:
        return 1
    idx = fts_indices[0]

    report = Report(
        table=TABLE_NAME,
        rows=row_count,
        fts_index={"name": idx.name, "columns": idx.columns},
        queries={query: run_query(table, query) for query in QUERIES},
    )

    out = ROOT_DIR / "data" / "hybrid_verification.json"
    _ = out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
