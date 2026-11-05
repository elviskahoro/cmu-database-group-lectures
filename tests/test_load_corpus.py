"""Guards for the full-corpus loader (scripts/load_corpus.py, cdgl-tds).

Covers the pure/testable seams without touching the network or LanceDB:
- titles come from metadata/titles_all.json, not a per-directory titles.json
- the Fireworks embedder batches, retries, and never leaks the API key
- 'search_document: ' is prepended exactly once before hitting Fireworks
- the disk cache makes a rerun cost zero API calls
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from lecture_search import ChunkRecord, load_titles_all, process_corpus_transcripts
from lecture_search.embed import DOCUMENT_PREFIX, EMBEDDING_DIMS, embed_documents
from lecture_search.fireworks import EmbeddingCache, FireworksEmbedder, cache_key

TOTAL_LECTURES = 595  # full corpus size; metadata/titles_all.json is the source of truth


def _write_transcript(
    directory: Path,
    date: str,
    video_id: str,
    segments: int = 5,
) -> None:
    transcript = {
        "file_path": f"out/{video_id}.mp3",
        "language": "en",
        "text": " ".join(f"segment_{i}" for i in range(segments)),
        "segments": [
            {"start": float(i * 10), "end": float((i + 1) * 10), "text": f"segment_{i}"}
            for i in range(segments)
        ],
    }
    (directory / f"{date}·{video_id}·transcription.json").write_text(
        json.dumps(transcript),
    )


def test_titles_come_from_titles_all_not_per_dir_titles_json(tmp_path: Path) -> None:
    _write_transcript(tmp_path, "2016-05-16", "vid1")
    # A stale per-directory titles.json must be ignored by the corpus path.
    (tmp_path / "titles.json").write_text(
        json.dumps({"vid1": {"title": "STALE", "date": "1999-01-01"}}),
    )
    titles = {"vid1": {"date": "2016-05-16", "title": "Real Title", "duration": 50}}

    results = process_corpus_transcripts(tmp_path, titles)

    assert set(results) == {"vid1"}
    assert all(chunk.title == "Real Title" for chunk in results["vid1"])


def test_load_titles_all_full_metadata() -> None:
    titles = load_titles_all(Path("metadata/titles_all.json"))
    assert len(titles) == TOTAL_LECTURES
    entry = next(iter(titles.values()))
    assert {"date", "title", "duration"} <= set(entry)


def _recording_embedder(
    dims: int = EMBEDDING_DIMS,
) -> tuple[FireworksEmbedder, list[list[str]]]:
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-key"
        body = json.loads(request.content)
        assert body["model"] == "nomic-ai/nomic-embed-text-v1.5"
        sent.append(body["input"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [0.0] * dims}
                    for i, _ in enumerate(body["input"])
                ],
                "usage": {"total_tokens": sum(len(t.split()) for t in body["input"])},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FireworksEmbedder("secret-key", client=client), sent


def _chunk(video_id: str, index: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=f"{video_id}:{index}",
        video_id=video_id,
        date="2016-05-16",
        title="T",
        chunk_index=index,
        start_seconds=0.0,
        end_seconds=10.0,
        text=text,
        deep_link=f"https://youtu.be/{video_id}?t=0",
    )


def test_prefix_added_exactly_once_before_fireworks() -> None:
    embedder, sent = _recording_embedder()
    embed_documents(["b-trees", "write ahead log"], embedder=embedder)

    flat = [text for batch in sent for text in batch]
    assert flat == [DOCUMENT_PREFIX + "b-trees", DOCUMENT_PREFIX + "write ahead log"]
    for text in flat:
        assert text.count(DOCUMENT_PREFIX) == 1


def test_retries_on_429_and_never_leaks_key(monkeypatch) -> None:
    statuses = iter([429, 200])
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if next(statuses) == 429:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [0.0] * EMBEDDING_DIMS}],
                "usage": {"total_tokens": 3},
            },
        )

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = FireworksEmbedder("secret-key", client=client)

    vectors = embedder.generate_embeddings(["alpha"])

    assert attempts == 2
    assert len(vectors) == 1
    assert embedder.requests_made == 1
    assert embedder.tokens_used == 3


def test_cached_rerun_makes_zero_api_calls(tmp_path: Path) -> None:
    embedder, sent = _recording_embedder()
    cache = EmbeddingCache(tmp_path / "cache.json")
    chunks = [_chunk("vid", i, f"chunk text {i}") for i in range(3)]

    def embed_with_cache(
        chunks: list[ChunkRecord],
        embedder: FireworksEmbedder,
        cache: EmbeddingCache,
    ) -> list[list[float]]:
        keys = [cache_key(chunk.id, chunk.text) for chunk in chunks]
        misses = [i for i, key in enumerate(keys) if cache.get(key) is None]
        if misses:
            vectors = embed_documents(
                [chunks[i].text for i in misses],
                embedder=embedder,
            )
            for i, vector in zip(misses, vectors, strict=True):
                cache.set(keys[i], vector)
            cache.save()
        return [cache.get(key) for key in keys if cache.get(key) is not None]

    first = embed_with_cache(chunks, embedder, cache)
    assert len(first) == 3 and embedder.requests_made > 0

    reloaded = EmbeddingCache(tmp_path / "cache.json")
    fresh_embedder, fresh_sent = _recording_embedder()
    second = embed_with_cache(chunks, fresh_embedder, reloaded)

    assert len(second) == 3
    assert fresh_embedder.requests_made == 0  # rerun costs zero API calls
    assert fresh_sent == []
