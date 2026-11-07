"""Tests for the pure helpers in scripts/retrieval_check.py."""

from __future__ import annotations

from scripts.retrieval_check import QUERIES, expected_rank

EXPECTED_QUERY_COUNT = 15


def test_expected_rank_finds_first_match() -> None:
    rows = [
        {"rank": 1, "video_id": "aaa"},
        {"rank": 2, "video_id": "bbb"},
        {"rank": 3, "video_id": "aaa"},
    ]
    assert expected_rank(rows, "aaa") == rows[0]["rank"]
    assert expected_rank(rows, "bbb") == rows[1]["rank"]


def test_expected_rank_returns_none_on_miss() -> None:
    rows = [{"rank": 1, "video_id": "aaa"}]
    assert expected_rank(rows, "zzz") is None


def test_all_fifteen_queries_have_expected_lectures() -> None:
    assert len(QUERIES) == EXPECTED_QUERY_COUNT
    assert [query.number for query in QUERIES] == list(
        range(1, EXPECTED_QUERY_COUNT + 1),
    )
    assert all(query.expected_video_id is not None for query in QUERIES)
