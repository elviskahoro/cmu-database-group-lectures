"""Client-side Nomic v1.5 embeddings and the LanceDB lectures table schema.

Embedding is client-side by design: callers embed with
``embed_documents``/``embed_query`` (which prepend the Nomic task prefixes
exactly once) and send vectors to the search API — the Modal container
never runs a model. ``LectureChunk`` is the LanceDB table schema; the
vector column is a fixed-size 768-d float32 list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from lancedb.embeddings import (
    TextEmbeddingFunction as _TextEmbeddingFunction,
    get_registry,
)

# Vector(dim) builds a pyarrow-backed type dynamically — the canonical
# lancedb schema idiom; its value_type parameter is untyped upstream.
from lancedb.pydantic import (
    LanceModel,
    Vector,  # pyright: ignore[reportUnknownVariableType]
)

if TYPE_CHECKING:
    import lancedb

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMS = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingFunction(Protocol):
    """The slice of an embedding backend this module depends on."""

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed each text into one fixed-dimension vector."""
        ...


class _RegistryEmbedder:
    """Adapt a lancedb registry embedder to the local EmbeddingFunction.

    lancedb's ``EmbeddingFunction.generate_embeddings`` is typed as
    returning ``list[np.ndarray | None]``; this adapter normalizes to plain
    float lists so embed_documents/embed_query stay backend-agnostic.
    """

    def __init__(self, inner: _TextEmbeddingFunction) -> None:
        self._inner = inner

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the wrapped lancedb embedder."""
        embeddings: list[list[float]] = []
        # lancedb annotates the return with `np.array` (a function, not a
        # type), so pyright sees the elements as unknown; they are really
        # float arrays (or None for a failed embedding).
        raw = cast(
            "list[list[float] | None]",
            self._inner.generate_embeddings(  # pyright: ignore[reportUnknownMemberType]
                texts,
            ),
        )
        for vector in raw:
            if vector is None:
                msg = "embedder returned no vector for a text"
                raise RuntimeError(msg)
            embeddings.append([float(value) for value in vector])
        return embeddings


def _create_registry_embedder() -> EmbeddingFunction:
    """Instantiate the Nomic v1.5 sentence-transformers embedder.

    Instantiating is cheap (the underlying SentenceTransformer model is
    lazy-loaded and cached per process); never construct it inside a
    row-add loop.
    """
    embedder_cls = get_registry().get("sentence-transformers")
    # lancedb's EmbeddingFunction.create(**kwargs) is untyped upstream.
    inner = embedder_cls.create(  # pyright: ignore[reportUnknownMemberType]
        name=NOMIC_MODEL,
        device="cpu",
        trust_remote_code=True,
    )
    if not isinstance(inner, _TextEmbeddingFunction):
        msg = "sentence-transformers embedder is not a TextEmbeddingFunction"
        raise TypeError(msg)
    return _RegistryEmbedder(inner)


# Registry embedder instantiated ONCE at module import (see
# _create_registry_embedder). Nomic prefixes are NOT auto-applied by the
# registry, so embed_documents/embed_query prepend them manually.
_EMBEDDER: EmbeddingFunction = _create_registry_embedder()


class LectureChunk(LanceModel):
    """LanceDB schema for one chunk row (vector is 768-d float32)."""

    id: str
    video_id: str
    date: str
    title: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str
    deep_link: str
    # Vector(dim) constructs the column type at call time; there is no
    # static type form for it (dynamic lancedb/pyarrow API).
    # pyrefly: ignore[invalid-annotation]
    vector: Vector(EMBEDDING_DIMS)  # pyright: ignore[reportInvalidTypeForm]


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
    """Embed corpus documents with the Nomic ``search_document: `` prefix."""
    return _embed_with_prefix(texts, DOCUMENT_PREFIX, embedder)


def embed_query(
    text: str,
    embedder: EmbeddingFunction = _EMBEDDER,
) -> list[float]:
    """Embed one query with the Nomic ``search_query: `` prefix."""
    return _embed_with_prefix([text], QUERY_PREFIX, embedder)[0]


def create_lectures_table(db: lancedb.DBConnection, name: str) -> lancedb.table.Table:
    """Create (overwriting) the lectures table with the LectureChunk schema."""
    # lancedb's create_table(data=...) parameter is untyped upstream; this
    # call passes only schema=.
    return db.create_table(  # pyright: ignore[reportUnknownMemberType]
        name,
        schema=LectureChunk,
        mode="overwrite",
    )
