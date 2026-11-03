"""Guards for the Modal search API's pure logic (lecture_search/search_api.py).

The endpoint is public and unauthenticated, so the tests that matter most
are the rejection cases: oversized or wrongly typed inputs, SQL-injection
shaped video IDs, and proof that nothing user-controlled is ever
interpolated into a WHERE clause, and no vectors/keys leave in a response.
Runs fully offline with fake table/embedder doubles — no Modal, no network.
"""

from __future__ import annotations

from datetime import date

import pytest
from lancedb.rerankers import MRRReranker

from lecture_search.embed import EMBEDDING_DIMS, QUERY_PREFIX
from lecture_search.search_api import (
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    MAX_VIDEO_IDS,
    RequestValidationError,
    SearchParams,
    build_where,
    embed_query_once,
    parse_search_request,
    run_hybrid_search,
    serialize_row,
    summarize_lectures,
)

VALID_VIDEO_ID = "QnGBxn_XP5U"
WRONG_DIMS = 8
CUSTOM_TOP_K = 7


def _params(**overrides: object) -> SearchParams:
    base: dict[str, object] = {
        "query": "what is a B-tree",
        "top_k": 5,
        "fts_weight": 0.5,
        "date_from": None,
        "date_to": None,
        "video_ids": (),
    }
    base.update(overrides)
    return SearchParams(**base)  # type: ignore[arg-type]


# --- request validation -------------------------------------------------


def test_valid_payload_parses_with_defaults() -> None:
    params = parse_search_request({"query": "mvcc"})

    assert params.query == "mvcc"
    assert 1 <= params.top_k <= MAX_TOP_K
    assert 0.0 <= params.fts_weight <= 1.0
    assert params.date_from is None
    assert params.date_to is None
    assert params.video_ids == ()


def test_valid_payload_with_all_filters() -> None:
    params = parse_search_request(
        {
            "query": "write-ahead log",
            "top_k": 20,
            "fts_weight": 0.8,
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "video_ids": [VALID_VIDEO_ID],
        },
    )

    assert params.date_from == date(2024, 1, 1)
    assert params.date_to == date(2024, 12, 31)
    assert params.video_ids == (VALID_VIDEO_ID,)


@pytest.mark.parametrize(
    "payload",
    [
        "not an object",
        ["query"],
        {},
        {"query": ""},
        {"query": "   "},
        {"query": 42},
        {"query": "x" * (MAX_QUERY_CHARS + 1)},
        {"query": "q", "top_k": 0},
        {"query": "q", "top_k": MAX_TOP_K + 1},
        {"query": "q", "top_k": 1000},
        {"query": "q", "top_k": True},
        {"query": "q", "top_k": "5"},
        {"query": "q", "top_k": 5.0},
        {"query": "q", "fts_weight": -0.1},
        {"query": "q", "fts_weight": 1.01},
        {"query": "q", "fts_weight": True},
        {"query": "q", "fts_weight": "0.5"},
        {"query": "q", "date_from": "yesterday"},
        {"query": "q", "date_from": "2024-13-01"},
        {"query": "q", "date_from": "2024-01-01' OR 1=1 --"},
        {"query": "q", "date_to": 20240101},
        {"query": "q", "date_from": "2024-12-31", "date_to": "2024-01-01"},
        {"query": "q", "video_ids": "not-a-list"},
        {"query": "q", "video_ids": [VALID_VIDEO_ID] * (MAX_VIDEO_IDS + 1)},
        {"query": "q", "video_ids": ["x' OR 1=1"]},
        {"query": "q", "video_ids": ["'; DROP TABLE lectures; --"]},
        {"query": "q", "video_ids": ["short"]},
        {"query": "q", "video_ids": [VALID_VIDEO_ID + "x"]},
        {"query": "q", "video_ids": ["QnGBxn_XP5!"]},
        {"query": "q", "video_ids": [12345678901]},
    ],
    ids=repr,
)
def test_invalid_payloads_rejected(payload: object) -> None:
    with pytest.raises(RequestValidationError):
        parse_search_request(payload)


def test_validation_messages_do_not_echo_input() -> None:
    malicious = "'; DROP TABLE lectures; --"
    with pytest.raises(RequestValidationError) as excinfo:
        parse_search_request({"query": "q", "video_ids": [malicious]})

    assert malicious not in str(excinfo.value)


# --- WHERE clause construction ------------------------------------------


def test_build_where_none_without_filters() -> None:
    assert build_where(_params()) is None


def test_build_where_combines_dates_and_video_ids() -> None:
    params = _params(
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
        video_ids=(VALID_VIDEO_ID, "9NdAOHyBq44"),
    )

    assert build_where(params) == (
        "date >= '2024-01-01' AND date <= '2024-12-31' "
        "AND video_id IN ('QnGBxn_XP5U', '9NdAOHyBq44')"
    )


# --- query embedding ----------------------------------------------------


class _RecordingEmbedder:
    """Records inputs; returns fixed-dim zero vectors."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.sent.extend(texts)
        return [[0.0] * EMBEDDING_DIMS for _ in texts]


def test_query_prefix_applied_exactly_once() -> None:
    embedder = _RecordingEmbedder()
    vector = embed_query_once(embedder, "what is a b-tree")

    assert embedder.sent == [QUERY_PREFIX + "what is a b-tree"]
    assert embedder.sent[0].count(QUERY_PREFIX) == 1
    assert len(vector) == EMBEDDING_DIMS


def test_embedding_dimension_mismatch_raises() -> None:
    class _WrongDims:
        def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * WRONG_DIMS for _ in texts]

    with pytest.raises(RuntimeError, match="dimension"):
        embed_query_once(_WrongDims(), "q")


# --- hybrid search plumbing ---------------------------------------------


def _chunk_row() -> dict[str, object]:
    return {
        "id": "QnGBxn_XP5U:0",
        "video_id": VALID_VIDEO_ID,
        "title": "Introduction to dbt",
        "date": "2024-03-01",
        "chunk_index": 0,
        "start_seconds": 12.0,
        "end_seconds": 48.5,
        "text": "dbt builds tables in the warehouse",
        "deep_link": "https://youtu.be/QnGBxn_XP5U?t=12",
        "vector": [0.0] * EMBEDDING_DIMS,
        "_relevance_score": 0.42,
    }


class _FakeBuilder:
    """Records the chained query-builder calls; returns canned rows."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.vector_value: list[float] | None = None
        self.text_value: str | None = None
        self.reranker: MRRReranker | None = None
        self.limit_value: int | None = None
        self.where_clause: str | None = None

    def vector(self, vector: list[float]) -> _FakeBuilder:
        self.vector_value = vector
        return self

    def text(self, text: str) -> _FakeBuilder:
        self.text_value = text
        return self

    def rerank(self, reranker: MRRReranker) -> _FakeBuilder:
        self.reranker = reranker
        return self

    def limit(self, limit: int) -> _FakeBuilder:
        self.limit_value = limit
        return self

    def where(self, where: str) -> _FakeBuilder:
        self.where_clause = where
        return self

    def to_list(self) -> list[dict[str, object]]:
        return self.rows


class _FakeTable:
    """Records query_type; hands out the canned builder."""

    def __init__(self, builder: _FakeBuilder) -> None:
        self.builder = builder
        self.query_type: str | None = None

    def search(self, query_type: str = "auto") -> _FakeBuilder:
        self.query_type = query_type
        return self.builder


def test_run_hybrid_search_wires_vector_text_rerank_limit_where() -> None:
    builder = _FakeBuilder([_chunk_row()])
    table = _FakeTable(builder)
    embedder = _RecordingEmbedder()
    params = _params(
        query="warehouse transformations",
        top_k=CUSTOM_TOP_K,
        fts_weight=0.25,
        video_ids=(VALID_VIDEO_ID,),
    )

    results = run_hybrid_search(table, embedder, params)

    assert table.query_type == "hybrid"
    assert embedder.sent == [QUERY_PREFIX + "warehouse transformations"]
    assert builder.vector_value is not None
    assert len(builder.vector_value) == EMBEDDING_DIMS
    assert builder.text_value == "warehouse transformations"
    assert isinstance(builder.reranker, MRRReranker)
    assert builder.reranker.weight_vector == pytest.approx(0.75)
    assert builder.reranker.weight_fts == pytest.approx(0.25)
    assert builder.limit_value == CUSTOM_TOP_K
    assert builder.where_clause == f"video_id IN ('{VALID_VIDEO_ID}')"
    assert len(results) == 1


def test_run_hybrid_search_skips_where_without_filters() -> None:
    builder = _FakeBuilder([_chunk_row()])
    run_hybrid_search(_FakeTable(builder), _RecordingEmbedder(), _params())

    assert builder.where_clause is None


# --- response serialization ---------------------------------------------


def test_serialize_row_whitelists_fields_and_maps_score() -> None:
    serialized = serialize_row(_chunk_row())

    assert set(serialized) == {
        "video_id",
        "title",
        "date",
        "start_seconds",
        "end_seconds",
        "deep_link",
        "text",
        "score",
    }
    assert serialized["score"] == pytest.approx(0.42)
    assert serialized["video_id"] == VALID_VIDEO_ID


def test_serialize_row_drops_vectors_and_internal_columns() -> None:
    serialized = serialize_row(_chunk_row())

    for forbidden in ("vector", "id", "chunk_index", "_relevance_score"):
        assert forbidden not in serialized
    assert not any(isinstance(value, list) for value in serialized.values())


def test_serialize_row_missing_score_is_none() -> None:
    row = _chunk_row()
    del row["_relevance_score"]

    assert serialize_row(row)["score"] is None


# --- lecture listing -----------------------------------------------------


def test_summarize_lectures_dedupes_and_sorts_date_desc() -> None:
    rows = [
        {"video_id": "a" * 11, "title": "Older", "date": "2023-01-01"},
        {"video_id": VALID_VIDEO_ID, "title": "Newer", "date": "2024-03-01"},
        {"video_id": "a" * 11, "title": "Older", "date": "2023-01-01"},
    ]

    lectures = summarize_lectures(rows)

    assert lectures == [
        {"video_id": VALID_VIDEO_ID, "title": "Newer", "date": "2024-03-01"},
        {"video_id": "a" * 11, "title": "Older", "date": "2023-01-01"},
    ]
