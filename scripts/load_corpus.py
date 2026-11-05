"""Full-corpus loader: transcripts -> Fireworks embeddings -> LanceDB (cdgl-tds).

Chunks every transcript in --transcripts-dir with lecture_search.chunking
(titles/dates from metadata/titles_all.json), drops degenerate Whisper
chunks (repetition loops, 'you you you' tails, near-empty stretches; see
lecture_search.quality and docs/corpus-quality.md), embeds with the hosted
Fireworks nomic-ai/nomic-embed-text-v1.5 endpoint ('search_document: '
prepended client-side exactly once by lecture_search.embed.embed_documents),
caches vectors on disk keyed by chunk id + text hash so reruns make zero API
calls, and upserts into LanceDB Cloud with batched, idempotent merge_insert
on `id`.

Usage:
    uv run python scripts/load_corpus.py --transcripts-dir data/corpus/transcription \
        --table lectures_dev --limit 100
    uv run python scripts/load_corpus.py --transcripts-dir data/corpus/transcription \
        --dry-run

Never prints secrets. Vector index is created only above the sequential-scan
threshold (~50k rows, see .agents/skills/sdk-lancedb INDEXING_AND_TUNING);
FTS index on `text` is always (re)built.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import lancedb
from lancedb.index import FTS

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # lecture_search is not an installed package

from lecture_search import (  # noqa: E402
    ChunkRecord,
    load_titles_all,
    process_corpus_transcripts,
)
from lecture_search.embed import (  # noqa: E402
    EMBEDDING_DIMS,
    LectureChunk,
    embed_documents,
)
from lecture_search.fireworks import (  # noqa: E402
    COST_PER_MILLION_TOKENS_USD,
    EmbeddingCache,
    FireworksEmbedder,
    cache_key,
    load_fireworks_api_key,
)
from lecture_search.quality import (  # noqa: E402
    drop_report_lines,
    partition_chunks,
)

LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
DEFAULT_TITLES_PATH = ROOT_DIR / "metadata" / "titles_all.json"
DEFAULT_CACHE_PATH = ROOT_DIR / "data" / "corpus" / "embedding_cache.json"
ENV_FILE = ROOT_DIR / ".env.local"
UPSERT_BATCH_SIZE = 1024
# Below this many rows a sequential scan is fast and exact; skip IVF_PQ.
VECTOR_INDEX_MIN_ROWS = 50_000
WORDS_PER_TOKEN = 0.75  # rough prose ratio for dry-run token estimates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (see module docstring for usage)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        required=True,
        help="Directory of 'YYYY-MM-DD·video_id·transcription.json' files",
    )
    parser.add_argument("--table", default="lectures", help="LanceDB table name")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N transcripts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk and estimate tokens only; no API calls, no DB writes",
    )
    parser.add_argument("--titles-path", type=Path, default=DEFAULT_TITLES_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=UPSERT_BATCH_SIZE)
    return parser.parse_args(argv)


def load_lancedb_api_key() -> str:
    """Read LANCEDB_API_KEY from .env.local. Never printed."""
    if not ENV_FILE.exists():
        msg = f"{ENV_FILE} not found"
        raise FileNotFoundError(msg)
    for raw_line in ENV_FILE.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("LANCEDB_API_KEY="):
            return line.split("=", 1)[1].strip("\"'")
    msg = "LANCEDB_API_KEY not found in .env.local"
    raise ValueError(msg)


def chunk_to_row(chunk: ChunkRecord, vector: list[float]) -> dict:
    """Flatten a ChunkRecord + embedding into a LanceDB row."""
    return {
        "id": chunk.id,
        "video_id": chunk.video_id,
        "date": chunk.date,
        "title": chunk.title,
        "chunk_index": chunk.chunk_index,
        "start_seconds": chunk.start_seconds,
        "end_seconds": chunk.end_seconds,
        "text": chunk.text,
        "deep_link": chunk.deep_link,
        "vector": vector,
    }


def embed_chunks_with_cache(
    chunks: list[ChunkRecord],
    embedder: FireworksEmbedder,
    cache: EmbeddingCache,
) -> list[list[float]]:
    """Return one vector per chunk, embedding only cache misses.

    embed_documents prepends 'search_document: ' exactly once; the cache key
    covers chunk id + raw text, so a rerun over unchanged chunks costs zero
    API calls.
    """
    keys = [cache_key(chunk.id, chunk.text) for chunk in chunks]
    vectors: list[list[float] | None] = [cache.get(key) for key in keys]

    miss_positions = [i for i, vector in enumerate(vectors) if vector is None]
    if miss_positions:
        miss_texts = [chunks[i].text for i in miss_positions]
        print(
            f"Embedding {len(miss_texts)} uncached chunks "
            f"({len(chunks) - len(miss_texts)} cache hits)...",
        )
        embedded = embed_documents(miss_texts, embedder=embedder)
        for position, vector in zip(miss_positions, embedded, strict=True):
            if len(vector) != EMBEDDING_DIMS:
                msg = f"expected {EMBEDDING_DIMS}-dim vector, got {len(vector)}"
                raise ValueError(msg)
            vectors[position] = vector
            cache.set(keys[position], vector)
        cache.save()
    else:
        print(f"All {len(chunks)} chunks served from embedding cache (0 API calls)")

    return [vector for vector in vectors if vector is not None]


def upsert_rows(
    db: lancedb.DBConnection,
    table_name: str,
    rows: list[dict],
    batch_size: int,
) -> lancedb.table.Table:
    """Idempotent batched merge_insert on `id`; creates the table if absent."""
    if table_name in db.list_tables().tables:
        table = db.open_table(table_name)
    else:
        table = db.create_table(table_name, schema=LectureChunk)

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(batch)
        )
        print(f"Upserted {min(start + batch_size, len(rows))}/{len(rows)} rows")
    return table


def maybe_create_vector_index(table: lancedb.table.Table, row_count: int) -> str:
    """Build IVF_PQ only above the sequential-scan threshold; return a note."""
    if row_count < VECTOR_INDEX_MIN_ROWS:
        return (
            f"skipped vector index: {row_count} rows < {VECTOR_INDEX_MIN_ROWS} "
            "(sequential scan is exact at this size)"
        )
    table.create_index(
        metric="cosine",
        num_partitions=int(math.sqrt(row_count)),
        num_sub_vectors=EMBEDDING_DIMS // 8,
        index_type="IVF_PQ",
        replace=True,
    )
    return f"created IVF_PQ vector index for {row_count} rows"


def main(argv: list[str] | None = None) -> int:
    """Run the chunk -> embed -> upsert -> index pipeline."""
    args = parse_args(argv)
    started = time.monotonic()

    if not args.transcripts_dir.exists():
        print(f"Transcripts dir not found: {args.transcripts_dir}", file=sys.stderr)
        return 1

    titles = load_titles_all(args.titles_path)
    per_video = process_corpus_transcripts(args.transcripts_dir, titles, args.limit)
    all_chunks = [
        chunk for video_chunks in per_video.values() for chunk in video_chunks
    ]
    if not all_chunks:
        print("No chunks produced", file=sys.stderr)
        return 1

    chunks, dropped = partition_chunks(all_chunks)
    for line in drop_report_lines(all_chunks, dropped):
        print(line)
    if not chunks:
        print("No chunks survived the quality filter", file=sys.stderr)
        return 1

    total_words = sum(len(chunk.text.split()) for chunk in chunks)
    estimated_tokens = int(total_words / WORDS_PER_TOKEN)
    print(
        f"Chunked {len(per_video)} transcripts -> {len(chunks)} kept chunks "
        f"(~{estimated_tokens:,} tokens estimated)",
    )

    if args.dry_run:
        print("Dry run: no embedding calls, no DB writes.")
        return 0

    cache = EmbeddingCache(args.cache_path)
    embedder = FireworksEmbedder(
        load_fireworks_api_key(ENV_FILE),
        batch_size=args.embed_batch_size,
    )
    vectors = embed_chunks_with_cache(chunks, embedder, cache)
    rows = [
        chunk_to_row(chunk, vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    db = lancedb.connect(
        LANCEDB_URI,
        region=LANCEDB_REGION,
        api_key=load_lancedb_api_key(),
    )
    table = upsert_rows(db, args.table, rows, args.upsert_batch_size)

    print("Creating FTS index on text...")
    table.create_index("text", config=FTS(with_position=True), replace=True)

    row_count = table.count_rows()
    print(maybe_create_vector_index(table, row_count))

    elapsed = time.monotonic() - started
    cost = embedder.tokens_used / 1_000_000 * COST_PER_MILLION_TOKENS_USD
    print(
        f"Done: {row_count} rows in {args.table}. "
        f"Embedding API: {embedder.requests_made} requests, "
        f"{embedder.tokens_used:,} tokens measured "
        f"(~${cost:.4f} at ${COST_PER_MILLION_TOKENS_USD}/1M), "
        f"{elapsed:.1f}s wall clock.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
