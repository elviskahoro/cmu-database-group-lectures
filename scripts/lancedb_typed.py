"""Typed wrappers over LanceDB's partially-Unknown client API.

lancedb 0.40 ships annotations, but its ``VEC``/``DATA``/``QueryType``
aliases and ``LanceQueryBuilder.to_list()`` leave bare ``list``/``dict``/
``Tuple`` in signatures, so pyright strict reports partially-Unknown member
types at every direct call site. The scripts in this directory funnel all
LanceDB I/O through the wrappers below; each wrapper carries one narrow
per-line pyright ignore for exactly that third-party stub gap.
"""  # noqa: INP001  # standalone script helper, not a package

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import lancedb
    import lancedb.table
    from lancedb.rerankers import MRRReranker


class InsertRow(TypedDict):
    """One lecture-chunk row written to LanceDB (mirrors LectureChunk)."""

    id: str
    video_id: str
    date: str
    title: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str
    deep_link: str
    vector: list[float]


class SearchRow(TypedDict):
    """Columns present on every lecture-table row LanceDB returns."""

    id: str
    video_id: str
    date: str
    title: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str
    deep_link: str
    vector: list[float]


class ScoredSearchRow(SearchRow, total=False):
    """SearchRow plus whichever score column the query mode adds."""

    _distance: float
    _score: float
    _relevance_score: float


def create_table_from_rows(
    db: lancedb.DBConnection,
    name: str,
    rows: Sequence[Mapping[str, object]],
) -> lancedb.table.Table:
    """Overwrite-create a table from row dicts."""
    return db.create_table(name, data=rows, mode="overwrite")  # pyright: ignore[reportUnknownMemberType]  # lancedb's DATA alias is partially Unknown


def add_rows(
    table: lancedb.table.Table,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Bulk-append rows to an existing table."""
    _ = table.add(rows)  # pyright: ignore[reportUnknownMemberType]  # lancedb's DATA alias is partially Unknown


def merge_insert_rows(
    table: lancedb.table.Table,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Idempotent upsert keyed on `id` (update matches, insert misses)."""
    _ = (  # pyright: ignore[reportUnknownVariableType]  # execute() return type is Unknown
        table.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(rows)  # pyright: ignore[reportUnknownMemberType]  # lancedb's DATA alias is partially Unknown
    )


def vector_search_rows(
    table: lancedb.table.Table,
    vector: list[float],
    *,
    limit: int,
) -> list[ScoredSearchRow]:
    """ANN search over `vector`, returning the top `limit` rows."""
    return cast(
        "list[ScoredSearchRow]",
        table.search(vector, query_type="vector").limit(limit).to_list(),  # pyright: ignore[reportUnknownMemberType]  # lancedb's VEC alias and to_list() are partially Unknown
    )


def fts_search_rows(
    table: lancedb.table.Table,
    text: str,
    *,
    limit: int,
) -> list[ScoredSearchRow]:
    """Full-text (BM25) search over `text`, returning the top `limit` rows."""
    return cast(
        "list[ScoredSearchRow]",
        table.search(text, query_type="fts").limit(limit).to_list(),  # pyright: ignore[reportUnknownMemberType]  # lancedb's QueryType alias and to_list() are partially Unknown
    )


def filtered_vector_search_rows(
    table: lancedb.table.Table,
    vector: list[float],
    *,
    where: str,
    limit: int,
) -> list[ScoredSearchRow]:
    """Vector search with a SQL prefilter (LanceDB's prefilter 'hybrid')."""
    return cast(
        "list[ScoredSearchRow]",
        table.search(vector).select(["*"]).where(where).limit(limit).to_list(),  # pyright: ignore[reportUnknownMemberType]  # lancedb's VEC alias and to_list() are partially Unknown
    )


def hybrid_search_rows(
    table: lancedb.table.Table,
    text: str,
    vector: list[float],
    *,
    reranker: MRRReranker,
    limit: int,
) -> list[ScoredSearchRow]:
    """True hybrid search: FTS + vector, fused by the given reranker."""
    builder = table.search(query_type="hybrid")  # pyright: ignore[reportUnknownMemberType]  # lancedb's QueryType alias is partially Unknown
    fused = builder.vector(vector).text(text).rerank(reranker)  # pyright: ignore[reportUnknownMemberType]  # lancedb's VEC alias is partially Unknown
    return cast(
        "list[ScoredSearchRow]",
        fused.limit(limit).to_list(),  # pyright: ignore[reportUnknownMemberType]  # to_list() dicts are partially Unknown
    )
