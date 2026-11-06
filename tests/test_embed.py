"""Guards for lecture_search.embed: prefix application and table schema.

The stub embedder records exactly what it is handed, proving the Nomic task
prefixes are prepended exactly once; the schema guard pins the vector column
to a fixed-size 768-d float32 list (the Nomic v1.5 embedding shape).
"""

from __future__ import annotations

from lecture_search.embed import (
    EMBEDDING_DIMS,
    LectureChunk,
    embed_documents,
    embed_query,
)


class _StubEmbedder:
    """Verbatim embedder double: records inputs, returns index-filled vectors."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(i)] * EMBEDDING_DIMS for i, _ in enumerate(texts)]


def test_document_prefix_applied() -> None:
    texts = ["alpha", "beta"]
    stub = _StubEmbedder()
    vectors = embed_documents(texts, embedder=stub)

    assert stub.seen == ["search_document: alpha", "search_document: beta"]
    assert len(vectors) == len(texts)
    assert all(len(vector) == EMBEDDING_DIMS for vector in vectors)


def test_query_prefix_applied() -> None:
    stub = _StubEmbedder()
    vector = embed_query("what is a b-tree", embedder=stub)

    assert stub.seen == ["search_query: what is a b-tree"]
    assert len(vector) == EMBEDDING_DIMS


def test_schema_vector_is_fixed_size_float32_768() -> None:
    schema = LectureChunk.to_arrow_schema()
    vector_field = schema.field("vector")

    assert vector_field.type.list_size == EMBEDDING_DIMS
    assert str(vector_field.type.value_field.type) == "float"
    expected_columns = {
        "id",
        "video_id",
        "date",
        "title",
        "chunk_index",
        "start_seconds",
        "end_seconds",
        "text",
        "deep_link",
        "vector",
    }
    assert set(schema.names) == expected_columns
