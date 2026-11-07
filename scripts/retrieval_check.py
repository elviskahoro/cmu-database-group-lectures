"""Retrieval check over the full `lectures` table (cdgl-ea1).

Sample-table predecessor: cdgl-x8u.

Runs all 15 retrieval queries: the original 10 from the bead notes
(queries 4, 5, 7 were deferred until the full corpus load) plus 5 new
realistic queries (11-15). For each query: vector-only top-10 and hybrid
top-10 via MRRReranker (same reranker as the Modal search API and
notebooks/search.py) at fts_weight in {0.2, 0.5, 0.8}. Records whether the
expected lecture (video_id) appears in the top 10 for every mode.

Writes the full report to data/retrieval_check.json and exits nonzero when
any expected lecture misses the top 10. Never prints secrets. Read-only
against the table.
"""  # noqa: INP001  # standalone script, not a package

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

import lancedb
from lancedb.rerankers import MRRReranker

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # lecture_search is not an installed package

from lecture_search.embed import EMBEDDING_DIMS, embed_query  # noqa: E402
from scripts.lancedb_typed import (  # noqa: E402
    ScoredSearchRow,
    hybrid_search_rows,
    vector_search_rows,
)

LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
TABLE_NAME = "lectures"
TOP_K = 10
FTS_WEIGHTS = [0.2, 0.5, 0.8]
REPORT_PATH = ROOT_DIR / "data" / "retrieval_check.json"


@dataclass(frozen=True)
class RetrievalQuery:
    """One retrieval probe query and its human-judged expected lecture."""

    number: int
    text: str
    expected_video_id: str | None  # None = deferred to the full corpus


class RankedRow(TypedDict):
    """Minimal ranked-hit shape expected_rank works against."""

    rank: int
    video_id: str


class SummaryRow(RankedRow, total=False):
    """One ranked hit in a query-mode summary (plus any score column)."""

    id: str
    title: str
    start_seconds: float
    _score: float
    _distance: float
    _relevance_score: float


class QueryResult(TypedDict, total=False):
    """Per-query report entry across all retrieval modes."""

    number: int
    query: str
    expected_video_id: str | None
    modes: dict[str, list[SummaryRow]]
    expected_rank_by_mode: dict[str, int | None]


# Queries 1-10 are from the bead notes; 11-15 are new realistic queries
# added for the full-corpus run (cdgl-ea1). expected_video_id is the lecture
# a human would judge the right answer (verified against lecture titles);
# all expected lectures are loaded in `lectures`.
#
# For the full-corpus run the expected lectures for queries 3, 8, 9, 10 were
# re-pointed from the sample-table choices: the full corpus contains a more
# canonical lecture for each (e.g. the RocksDB Databaseology lecture for
# LSM compaction/write amplification — Mark Callaghan's home turf). See
# docs/retrieval-check.md.
QUERIES = [
    RetrievalQuery(1, "how does MVCC avoid read locks", "tUFha9-DuSk"),
    RetrievalQuery(2, "why use a write-ahead log", "CedEy54pe3g"),
    RetrievalQuery(3, "LSM tree compaction write amplification", "jGCv4r8CJEI"),
    RetrievalQuery(4, "B+ tree vs skip list for indexes", "pCoox0XbU8Y"),
    RetrievalQuery(5, "how does Raft elect a leader", "JLlIAWjvHxM"),
    RetrievalQuery(
        6,
        "query optimizer cost model cardinality estimation",
        "Kf9Nj3zI9WI",
    ),
    RetrievalQuery(7, "columnar storage and vectorized execution", "fGG9dApIhDU"),
    RetrievalQuery(8, "how dbt models become tables in the warehouse", "gPhreHDIenE"),
    RetrievalQuery(9, "distributed SQL replication and consensus", "yu2TZF7S1Mg"),
    RetrievalQuery(10, "snapshot isolation vs serializable", "-7VjPVmw4vM"),
    RetrievalQuery(
        11,
        "how does the ARIES recovery algorithm restore a database after a crash",
        "4VGkRXVM5fk",
    ),
    RetrievalQuery(12, "how does DuckDB process analytical queries", "bZOvAKGkzpQ"),
    RetrievalQuery(
        13,
        "how does Snowflake separate storage from compute",
        "NhWp1bTG0Cw",
    ),
    RetrievalQuery(14, "vector similarity search with HNSW indexes", "YU66YlZW6dA"),
    RetrievalQuery(
        15,
        "how does the buffer pool decide which pages to evict",
        "Moz2AgC9hG4",
    ),
]


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


def summarize(rows: list[ScoredSearchRow]) -> list[SummaryRow]:
    """Trim raw LanceDB rows to rank/id/video/title plus any score columns."""
    summarized: list[SummaryRow] = []
    for rank, row in enumerate(rows, start=1):
        entry = SummaryRow(
            rank=rank,
            id=row["id"],
            video_id=row["video_id"],
            title=row["title"],
            start_seconds=row["start_seconds"],
        )
        if (score := row.get("_score")) is not None:
            entry["_score"] = round(score, 4)
        if (distance := row.get("_distance")) is not None:
            entry["_distance"] = round(distance, 4)
        if (relevance := row.get("_relevance_score")) is not None:
            entry["_relevance_score"] = round(relevance, 4)
        summarized.append(entry)
    return summarized


def expected_rank(
    rows: Sequence[RankedRow],
    expected_video_id: str,
) -> int | None:
    """Rank of the expected lecture in a summary, or None when absent."""
    for row in rows:
        if row["video_id"] == expected_video_id:
            return row["rank"]
    return None


def run_query(table: lancedb.table.Table, query: RetrievalQuery) -> QueryResult:
    """Run one query in vector-only and hybrid (per fts_weight) modes."""
    query_vector = embed_query(query.text)
    if len(query_vector) != EMBEDDING_DIMS:
        msg = f"expected {EMBEDDING_DIMS}-dim query vector, got {len(query_vector)}"
        raise ValueError(msg)

    modes: dict[str, list[SummaryRow]] = {
        "vector_only": summarize(
            vector_search_rows(table, query_vector, limit=TOP_K),
        ),
    }

    for fts_weight in FTS_WEIGHTS:
        modes[f"hybrid_fts_weight={fts_weight}"] = summarize(
            hybrid_search_rows(
                table,
                query.text,
                query_vector,
                reranker=MRRReranker(
                    weight_vector=1.0 - fts_weight,
                    weight_fts=fts_weight,
                ),
                limit=TOP_K,
            ),
        )

    result = QueryResult(
        number=query.number,
        query=query.text,
        expected_video_id=query.expected_video_id,
        modes=modes,
    )
    if query.expected_video_id is not None:
        result["expected_rank_by_mode"] = {
            mode: expected_rank(rows, query.expected_video_id)
            for mode, rows in modes.items()
        }
    return result


def main() -> int:
    """Run every runnable query and write data/retrieval_check.json."""
    db = lancedb.connect(LANCEDB_URI, region=LANCEDB_REGION, api_key=load_api_key())
    table = db.open_table(TABLE_NAME)
    row_count = table.count_rows()

    runnable = [query for query in QUERIES if query.expected_video_id is not None]
    deferred = [query for query in QUERIES if query.expected_video_id is None]

    results: list[QueryResult] = []
    failures: list[str] = []
    for query in runnable:
        entry = run_query(table, query)
        results.append(entry)
        # runnable queries always carry expected_rank_by_mode (see run_query)
        for mode, rank in entry.get("expected_rank_by_mode", {}).items():
            if rank is None:
                failures.append(f"query {query.number} ({mode})")

    report = {
        "table": TABLE_NAME,
        "rows": row_count,
        "top_k": TOP_K,
        "fts_weights": FTS_WEIGHTS,
        "reranker": "MRRReranker",
        "deferred_queries": [
            {"number": query.number, "query": query.text} for query in deferred
        ],
        "results": results,
    }
    _ = REPORT_PATH.write_text(json.dumps(report, indent=2))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
