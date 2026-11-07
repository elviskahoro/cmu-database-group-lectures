"""Fireworks hosted embeddings client: batching, retry with backoff, disk cache.

Swap-in for the local sentence-transformers embedder behind the same
EmbeddingFunction protocol as lecture_search.embed. The Nomic task prefixes
('search_document: ' / 'search_query: ') are NOT applied here — Fireworks
embeds input verbatim (docs/parity-gate.md), so callers must go through
lecture_search.embed.embed_documents / embed_query, which prepend the prefix
exactly once.

The disk cache is keyed by chunk id + text hash so a rerun over an unchanged
corpus makes zero API calls (cdgl-tds).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from http import HTTPStatus
from pathlib import Path
from typing import NotRequired, TypedDict

import httpx

FIREWORKS_EMBEDDINGS_URL = "https://api.fireworks.ai/inference/v1/embeddings"
FIREWORKS_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBED_BATCH_SIZE = 64
MAX_RETRIES = 6
INITIAL_BACKOFF_SECONDS = 1.0
# Serverless list price for nomic-embed-text-v1.5 on Fireworks
# (https://fireworks.ai/pricing, serverless text-embedding tier).
COST_PER_MILLION_TOKENS_USD = 0.008

RETRYABLE_STATUS_CODES = {408, 409, 425, 429}


def load_fireworks_api_key(env_file: Path) -> str:
    """Read FIREWORKS_API_KEY from a .env.local-style file. Never printed."""
    if not env_file.exists():
        msg = f"{env_file} not found"
        raise FileNotFoundError(msg)
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("FIREWORKS_API_KEY="):
            return line.split("=", 1)[1].strip("\"'")
    msg = f"FIREWORKS_API_KEY not found in {env_file}"
    raise ValueError(msg)


def cache_key(chunk_id: str, text: str) -> str:
    """Cache key for a chunk: stable id plus a digest of the exact text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{chunk_id}:{digest}"


class EmbeddingCache:
    """JSON-file cache: key -> 768-d float vector. Rewrite-on-save, atomic."""

    def __init__(self, path: Path) -> None:
        """Load (or lazily create) the cache file at `path`."""
        self.path = Path(path)
        self._vectors: dict[str, list[float]] = {}
        if self.path.exists():
            self._vectors = json.loads(self.path.read_text())

    def __len__(self) -> int:
        """Number of cached vectors."""
        return len(self._vectors)

    def get(self, key: str) -> list[float] | None:
        """Return the cached vector for `key`, or None on a miss."""
        return self._vectors.get(key)

    def set(self, key: str, vector: list[float]) -> None:
        """Store `vector` under `key` (in memory until save())."""
        self._vectors[key] = vector

    def save(self) -> None:
        """Persist atomically: write a temp file, then replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=self.path.name,
            suffix=".tmp",
        )
        with os.fdopen(fd, "w") as tmp:
            json.dump(self._vectors, tmp)
        _ = Path(tmp_name).replace(self.path)


class _EmbeddingItem(TypedDict):
    """One entry of the Fireworks embeddings response ``data`` array."""

    index: int
    embedding: list[float]


class _Usage(TypedDict, total=False):
    """Token accounting block of a Fireworks embeddings response."""

    total_tokens: int


class _EmbeddingsResponse(TypedDict):
    """Parsed Fireworks embeddings response payload."""

    data: list[_EmbeddingItem]
    usage: NotRequired[_Usage]


class FireworksEmbedder:
    """Batched Fireworks embeddings with retry/backoff on 429/5xx.

    Counters `requests_made` and `tokens_used` let the loader report measured
    cost and prove idempotent reruns make zero API calls.
    """

    def __init__(
        self,
        api_key: str,
        *,
        batch_size: int = EMBED_BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Create an embedder; pass `client` to inject a mock transport."""
        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValueError(msg)
        self._api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self.requests_made = 0
        self.tokens_used = 0

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` verbatim (no prefix), batching and retrying."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = self._post_with_retry(batch)
            by_index = sorted(payload["data"], key=lambda item: item["index"])
            vectors.extend([float(v) for v in item["embedding"]] for item in by_index)
        return vectors

    def _post_with_retry(self, batch: list[str]) -> _EmbeddingsResponse:
        delay = INITIAL_BACKOFF_SECONDS
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(
                    FIREWORKS_EMBEDDINGS_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": FIREWORKS_MODEL, "input": batch},
                )
            except httpx.TransportError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code < HTTPStatus.BAD_REQUEST:
                payload: _EmbeddingsResponse = response.json()
                self.requests_made += 1
                usage = payload.get("usage") or {}
                self.tokens_used += int(usage.get("total_tokens", 0))
                return payload

            retryable = (
                response.status_code in RETRYABLE_STATUS_CODES
                or response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
            )
            if not retryable or attempt == self.max_retries - 1:
                # Never include the request headers (they carry the API key).
                msg = (
                    f"Fireworks embeddings failed: HTTP {response.status_code} "
                    f"after {attempt + 1} attempt(s)"
                )
                raise RuntimeError(msg)
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            time.sleep(wait)
            delay *= 2

        msg = "unreachable"  # pragma: no cover - loop always returns or raises
        raise RuntimeError(msg)
