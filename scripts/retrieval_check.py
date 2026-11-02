# Retrieval smoke check over the real `lectures_sample` table (cdgl-x8u).
#
# Runs the sample-appropriate subset of the 10 retrieval queries from the
# bead notes (queries 1, 2, 3, 6, 8, 9, 10 map to loaded lectures; 4, 5, 7
# are deferred to the full corpus). For each query: vector-only top-10 and
# hybrid top-10 via LinearCombinationReranker (same reranker as
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
from lancedb.rerankers import LinearCombinationReranker

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # lecture_search is not an installed package

from lecture_search.embed import embed_query  # noqa: E402

LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
TABLE_NAME = "lectures_sample"
TOP_K = 10
FTS_WEIGHTS = [0.2, 0.5, 0.8]
REPORT_PATH = ROOT_DIR / "data" / "retrieval_check.json"


@dataclass(frozen=True)
class RetrievalQuery:
    number: int
    text: str
    expected_video_id: str | None  # None = deferred to the full corpus


# The 10 queries from the bead notes. expected_video_id is the lecture in
# the sample table that should appear in the top 10; None means no matching
# lecture is loaded yet (run on the full corpus instead).
QUERIES = [
    RetrievalQuery(1, "how does MVCC avoid read locks", "tUFha9-DuSk"),
    RetrievalQuery(2, "why use a write-ahead log", "CedEy54pe3g"),
    RetrievalQuery(3, "LSM tree compaction write amplification", "2_sTdS4h-bY"),
    RetrievalQuery(4, "B+ tree vs skip list for indexes", None),
    RetrievalQuery(5, "how does Raft elect a leader", None),
    RetrievalQuery(
        6, "query optimizer cost model cardinality estimation", "Kf9Nj3zI9WI"
    ),
    RetrievalQuery(7, "columnar storage and vectorized execution", None),
    RetrievalQuery(8, "how dbt models become tables in the warehouse", "QnGBxn_XP5U"),
    RetrievalQuery(9, "distributed SQL replication and consensus", "9NdAOHyBq44"),
    RetrievalQuery(10, "snapshot isolation vs serializable", "tUFha9-DuSk"),
]


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
        table.search(query_vector, query_type="vector").limit(TOP_K).to_list()
    )
    modes: dict[str, Any] = {"vector_only": vector_rows}

    for fts_weight in FTS_WEIGHTS:
        rows = summarize(
            table.search(query_type="hybrid")
            .vector(query_vector)
            .text(query.text)
            .rerank(LinearCombinationReranker(weight=fts_weight))
            .limit(TOP_K)
            .to_list()
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
    print(f"Table {TABLE_NAME}: {row_count} rows, top_k={TOP_K}")

    runnable = [query for query in QUERIES if query.expected_video_id is not None]
    deferred = [query for query in QUERIES if query.expected_video_id is None]
    print(
        f"Running {len(runnable)} queries "
        f"({len(deferred)} deferred to full corpus: "
        f"{[query.number for query in deferred]})"
    )

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for query in runnable:
        print(f"\n=== Query {query.number}: {query.text!r} "
              f"(expect {query.expected_video_id}) ===")
        entry = run_query(table, query)
        results.append(entry)
        for mode, rank in entry["expected_rank_by_mode"].items():
            status = f"rank {rank}" if rank is not None else "MISS"
            print(f"  {mode}: {status}")
            if rank is None:
                failures.append(f"query {query.number} ({mode})")

    report = {
        "table": TABLE_NAME,
        "rows": row_count,
        "top_k": TOP_K,
        "fts_weights": FTS_WEIGHTS,
        "reranker": "LinearCombinationReranker",
        "deferred_queries": [
            {"number": query.number, "query": query.text} for query in deferred
        ],
        "results": results,
    }
    _ = REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {REPORT_PATH}")

    if failures:
        print(f"FAIL: expected lecture missing from top {TOP_K} for: {failures}",
              file=sys.stderr)
        return 1
    print(f"PASS: expected lecture in top {TOP_K} for every query and mode")
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


def test_sample_queries_cover_seven_loaded_lectures() -> None:
    runnable = [query for query in QUERIES if query.expected_video_id is not None]
    assert len(QUERIES) == 10
    assert len(runnable) == 7
    assert {query.number for query in runnable} == {1, 2, 3, 6, 8, 9, 10}
