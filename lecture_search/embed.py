from __future__ import annotations

from typing import Protocol

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMS = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Registry embedder instantiated ONCE at module import. Instantiating is
# cheap (the underlying SentenceTransformer model is lazy-loaded and cached
# per process); never construct it inside a row-add loop. Nomic prefixes are
# NOT auto-applied by the registry, so embed_documents/embed_query prepend
# them manually.
_EMBEDDER = (
    get_registry()
    .get("sentence-transformers")
    .create(
        name=NOMIC_MODEL,
        device="cpu",
        trust_remote_code=True,
    )
)


class EmbeddingFunction(Protocol):
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]: ...


class LectureChunk(LanceModel):
    id: str
    video_id: str
    date: str
    title: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str
    deep_link: str
    vector: Vector(EMBEDDING_DIMS)


def _embed_with_prefix(
    texts: list[str],
    prefix: str,
    embedder: EmbeddingFunction,
) -> list[list[float]]:
    prefixed = [prefix + text for text in texts]
    return [
        [float(value) for value in vector]
        for vector in embedder.generate_embeddings(prefixed)
    ]


def embed_documents(
    texts: list[str],
    embedder: EmbeddingFunction = _EMBEDDER,
) -> list[list[float]]:
    return _embed_with_prefix(texts, DOCUMENT_PREFIX, embedder)


def embed_query(
    text: str,
    embedder: EmbeddingFunction = _EMBEDDER,
) -> list[float]:
    return _embed_with_prefix([text], QUERY_PREFIX, embedder)[0]


def create_lectures_table(db: lancedb.DBConnection, name: str) -> lancedb.table.Table:
    return db.create_table(name, schema=LectureChunk, mode="overwrite")


class _StubEmbedder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(i)] * EMBEDDING_DIMS for i, _ in enumerate(texts)]


def test_document_prefix_applied() -> None:
    stub = _StubEmbedder()
    vectors = embed_documents(["alpha", "beta"], embedder=stub)

    assert stub.seen == ["search_document: alpha", "search_document: beta"]
    assert len(vectors) == 2
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


if __name__ == "__main__":
    test_document_prefix_applied()
    test_query_prefix_applied()
    test_schema_vector_is_fixed_size_float32_768()
