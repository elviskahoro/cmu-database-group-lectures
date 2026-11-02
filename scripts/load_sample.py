from __future__ import annotations

import sys
from pathlib import Path

import lancedb

from lecture_search import ChunkRecord, process_sample_transcripts
from lecture_search.embed import create_lectures_table, embed_documents

PILOT_TRANSCRIPTS = Path(
    "/Users/elvis/Documents/hq/repos/epic-sermons"
    "/out/pilot2_noprompt/transcription"
)
LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
TABLE_NAME = "lectures_sample"
ROOT_DIR = Path(__file__).resolve().parent.parent
EMBED_BATCH_SIZE = 16


def load_api_key() -> str:
    env_file = ROOT_DIR / ".env.local"
    if not env_file.exists():
        msg = ".env.local not found"
        raise FileNotFoundError(msg)

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("LANCEDB_API_KEY="):
            return line.split("=", 1)[1].strip('"\'')
    msg = "LANCEDB_API_KEY not found in .env.local"
    raise ValueError(msg)


def chunk_to_row(chunk: ChunkRecord, vector: list[float]) -> dict:
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


def embed_chunks(chunks: list[ChunkRecord]) -> list[dict]:
    rows: list[dict] = []
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        vectors = embed_documents([chunk.text for chunk in batch])
        rows.extend(chunk_to_row(chunk, vector) for chunk, vector in zip(batch, vectors))
        print(f"Embedded {len(rows)}/{len(chunks)} chunks")
    return rows


def main() -> int:
    if not PILOT_TRANSCRIPTS.exists():
        print(f"Sample transcripts not found at {PILOT_TRANSCRIPTS}", file=sys.stderr)
        return 1

    api_key = load_api_key()

    print("Chunking sample transcripts...")
    per_video = process_sample_transcripts(PILOT_TRANSCRIPTS)
    chunks = [chunk for video_chunks in per_video.values() for chunk in video_chunks]
    if not chunks:
        print("No chunks produced from sample transcripts", file=sys.stderr)
        return 1
    print(f"Produced {len(chunks)} chunks from {len(per_video)} transcripts")

    print("Embedding chunks locally with Nomic v1.5 (CPU)...")
    rows = embed_chunks(chunks)

    print("Connecting to LanceDB Cloud...")
    db = lancedb.connect(LANCEDB_URI, region=LANCEDB_REGION, api_key=api_key)

    print(f"Creating table {TABLE_NAME}...")
    table = create_lectures_table(db, TABLE_NAME)
    table.add(rows)

    print("Creating FTS index on text...")
    table.create_fts_index("text", replace=True)

    print(f"Done: {table.count_rows()} rows in {TABLE_NAME} (FTS indexed, no vector index at sample size)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
