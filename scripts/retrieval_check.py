# Retrieval check over the full `lectures` table (cdgl-ea1; sample-table
# predecessor: cdgl-x8u).
#
# Runs all 15 retrieval queries: the original 10 from the bead notes
# (queries 4, 5, 7 were deferred until the full corpus load) plus 5 new
# realistic queries (11-15). For each query: vector-only top-10 and hybrid
# top-10 via MRRReranker (same reranker as the Modal search API and
# notebooks/search.py) at fts_weight in {0.2, 0.5, 0.8}. Records whether the
# expected lecture (video_id) appears in the top 10 for every mode.
#
# Writes the full report to data/retrieval_check.json and prints a summary.
# Never prints secrets. Read-only against the table.

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
from lancedb.rerankers import MRRReranker

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # lecture_search is not an installed package

from lecture_search.embed import embed_query  # noqa: E402

LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
TABLE_NAME = "lectures"
TOP_K = 10
FTS_WEIGHTS = [0.2, 0.5, 0.8]
REPORT_PATH = ROOT_DIR / "data" / "retrieval_check.json"


@dataclass(frozen=True)
class RetrievalQuery:
    number: int
    text: str
    expected_video_id: str | None  # None = deferred to the full corpus


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
    env_file = ROOT_DIR / ".env.local"
    if not env_file.exists():
        msg = ".env.local not found"
        raise FileNotFoundError(msg)
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("LANCEDB_API_KEY="):
            return line.split("=", 1)[1].strip("\"'")
    msg = "LANCEDB_API_KEY not found in .env.local"
    raise ValueError(msg)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "id": row["id"],
            "video_id": row["video_id"],
            "title": row["title"],
            "start_seconds": row["start_seconds"],
            **{
                key: round(float(row[key]), 4)
                for key in ("_score", "_distance", "_relevance_score")
                if key in row and row[key] is not None
            },
        }
        for rank, row in enumerate(rows, start=1)
    ]


def expected_rank(rows: list[dict[str, Any]], expected_video_id: str) -> int | None:
    for row in rows:
        if row["video_id"] == expected_video_id:
            return int(row["rank"])
    return None


def run_query(table: Any, query: RetrievalQuery) -> dict[str, Any]:
    query_vector = embed_query(query.text)
    assert len(query_vector) == 768

    vector_rows = summarize(
        table.search(query_vector, query_type="vector").limit(TOP_K).to_list(),
    )
    modes: dict[str, Any] = {"vector_only": vector_rows}

    for fts_weight in FTS_WEIGHTS:
        rows = summarize(
            table.search(query_type="hybrid")
            .vector(query_vector)
            .text(query.text)
            .rerank(MRRReranker(weight_vector=1.0 - fts_weight, weight_fts=fts_weight))
            .limit(TOP_K)
            .to_list(),
        )
        modes[f"hybrid_fts_weight={fts_weight}"] = rows

    result: dict[str, Any] = {
        "number": query.number,
        "query": query.text,
        "expected_video_id": query.expected_video_id,
        "modes": modes,
    }
    if query.expected_video_id is not None:
        result["expected_rank_by_mode"] = {
            mode: expected_rank(rows, query.expected_video_id)
            for mode, rows in modes.items()
        }
    return result


def main() -> int:
    db = lancedb.connect(LANCEDB_URI, region=LANCEDB_REGION, api_key=load_api_key())
    table = db.open_table(TABLE_NAME)
    row_count = table.count_rows()

    runnable = [query for query in QUERIES if query.expected_video_id is not None]
    deferred = [query for query in QUERIES if query.expected_video_id is None]

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for query in runnable:
        entry = run_query(table, query)
        results.append(entry)
        for mode, rank in entry["expected_rank_by_mode"].items():
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

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_expected_rank_finds_first_match() -> None:
    rows = [
        {"rank": 1, "video_id": "aaa"},
        {"rank": 2, "video_id": "bbb"},
        {"rank": 3, "video_id": "aaa"},
    ]
    assert expected_rank(rows, "aaa") == 1
    assert expected_rank(rows, "bbb") == 2


def test_expected_rank_returns_none_on_miss() -> None:
    rows = [{"rank": 1, "video_id": "aaa"}]
    assert expected_rank(rows, "zzz") is None


def test_all_fifteen_queries_have_expected_lectures() -> None:
    assert len(QUERIES) == 15
    assert [query.number for query in QUERIES] == list(range(1, 16))
    assert all(query.expected_video_id is not None for query in QUERIES)
