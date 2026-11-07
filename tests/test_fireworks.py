"""Guards for lecture_search.fireworks: cache, batching, retry/backoff.

Runs fully offline against an httpx MockTransport. The retry tests also pin
the security posture: the API key must never appear in raised errors.
"""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lecture_search.fireworks import EmbeddingCache, FireworksEmbedder, cache_key

MOCK_EMBEDDING_DIMS = 3
TOKENS_PER_INPUT_TEXT = 7


def _mock_transport(
    status_sequence: list[int],
) -> tuple[httpx.MockTransport, list[list[str]]]:
    """Transport returning the given statuses; final entry is a 200 payload."""
    seen_inputs: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_inputs.append(body["input"])
        status = status_sequence[min(len(seen_inputs) - 1, len(status_sequence) - 1)]
        if status != HTTPStatus.OK:
            return httpx.Response(status)
        vectors = [
            {"index": i, "embedding": [float(i)] * MOCK_EMBEDDING_DIMS}
            for i, _ in enumerate(body["input"])
        ]
        return httpx.Response(
            HTTPStatus.OK,
            json={
                "data": vectors,
                "usage": {"total_tokens": TOKENS_PER_INPUT_TEXT * len(body["input"])},
            },
        )

    return httpx.MockTransport(handler), seen_inputs


def _embedder(transport: httpx.MockTransport, **overrides: object) -> FireworksEmbedder:
    kwargs: dict[str, object] = {
        "batch_size": 64,
        "client": httpx.Client(transport=transport),
    }
    kwargs.update(overrides)
    return FireworksEmbedder("key", **kwargs)  # type: ignore[arg-type]


def test_cache_key_tracks_text() -> None:
    assert cache_key("vid:0", "alpha") == cache_key("vid:0", "alpha")
    assert cache_key("vid:0", "alpha") != cache_key("vid:0", "beta")
    assert cache_key("vid:0", "alpha") != cache_key("vid:1", "alpha")
    assert cache_key("vid:0", "alpha").startswith("vid:0:")


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.json")
    assert cache.get("k") is None
    cache.set("k", [1.0, 2.0])
    cache.save()

    reloaded = EmbeddingCache(tmp_path / "cache.json")
    assert len(reloaded) == 1
    assert reloaded.get("k") == [1.0, 2.0]


def test_batches_and_token_accounting() -> None:
    transport, seen = _mock_transport([HTTPStatus.OK])
    texts = ["a", "b", "c", "d", "e"]
    embedder = FireworksEmbedder(
        "key",
        batch_size=2,
        client=httpx.Client(transport=transport),
    )
    vectors = embedder.generate_embeddings(texts)

    assert seen == [["a", "b"], ["c", "d"], ["e"]]
    assert len(vectors) == len(texts)
    assert embedder.requests_made == len(seen)
    assert embedder.tokens_used == TOKENS_PER_INPUT_TEXT * len(texts)


def test_retry_on_429_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = [HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.OK]
    transport, seen = _mock_transport(statuses)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    embedder = _embedder(transport)
    vectors = embedder.generate_embeddings(["x"])

    assert len(seen) == len(statuses)  # one 429, one 200
    assert len(vectors) == 1
    assert embedder.requests_made == 1


def test_retry_on_5xx_exhaustion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, _ = _mock_transport([HTTPStatus.INTERNAL_SERVER_ERROR])
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    embedder = _embedder(transport, max_retries=2)

    with pytest.raises(RuntimeError, match="HTTP 500") as exc_info:
        embedder.generate_embeddings(["x"])
    assert "key" not in str(exc_info.value)  # never leak the API key


def test_non_retryable_400_raises_immediately() -> None:
    transport, seen = _mock_transport([HTTPStatus.BAD_REQUEST])
    embedder = _embedder(transport)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        embedder.generate_embeddings(["x"])
    assert len(seen) == 1  # no retry on 4xx other than 408/409/425/429
