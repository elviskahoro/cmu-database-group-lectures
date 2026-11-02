"""Guard: the Nomic task prefix is added client-side, exactly once.

Fireworks embeds input verbatim (see docs/parity-gate.md), so any embedder
we swap in behaves like this verbatim stub: whatever embed_documents /
embed_query hand it is what gets embedded. Omitting the prefix drops cosine
to the stored vectors to ~0.93; doubling it is also wrong.
"""

from __future__ import annotations

from lecture_search.embed import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMS,
    QUERY_PREFIX,
    embed_documents,
    embed_query,
)


class _VerbatimEmbedder:
    """Records exactly what would be sent to a no-prefix endpoint like Fireworks."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.sent.extend(texts)
        return [[0.0] * EMBEDDING_DIMS for _ in texts]


def test_documents_are_prefixed_exactly_once():
    embedder = _VerbatimEmbedder()
    embed_documents(["hash joins", "MVCC"], embedder=embedder)

    assert len(embedder.sent) == 2
    for sent, raw in zip(embedder.sent, ["hash joins", "MVCC"], strict=True):
        assert sent == DOCUMENT_PREFIX + raw
        assert sent.count(DOCUMENT_PREFIX) == 1


def test_query_is_prefixed_exactly_once():
    embedder = _VerbatimEmbedder()
    embed_query("what is a b-tree", embedder=embedder)

    assert embedder.sent == [QUERY_PREFIX + "what is a b-tree"]
    assert embedder.sent[0].count(QUERY_PREFIX) == 1


def test_document_and_query_prefixes_differ():
    assert DOCUMENT_PREFIX != QUERY_PREFIX
