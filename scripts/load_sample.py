from __future__ import annotations

import sys
from pathlib import Path

import lancedb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # lecture_search is not an installed package

from lecture_search import ChunkRecord, process_sample_transcripts  # noqa: E402
from lecture_search.embed import create_lectures_table, embed_documents  # noqa: E402

LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
TABLE_NAME = "lectures_sample"
SAMPLE_DIR = ROOT_DIR / "data" / "sample"
EMBED_BATCH_SIZE = 16


def load_api_key() -> str:
    env_file = ROOT_DIR / ".env.local"
    if not env_file.exists():
        msg = ".env.local not found"
        raise FileNotFoundError(msg)

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("LANCEDB_API_KEY="):
            return line.split("=", 1)[1].strip("\"'")
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
        rows.extend(
            chunk_to_row(chunk, vector)
            for chunk, vector in zip(batch, vectors, strict=False)
        )
    return rows


def main() -> int:
    if not SAMPLE_DIR.exists():
        return 1

    api_key = load_api_key()

    per_video = process_sample_transcripts(SAMPLE_DIR)
    chunks = [chunk for video_chunks in per_video.values() for chunk in video_chunks]
    if not chunks:
        return 1

    rows = embed_chunks(chunks)

    db = lancedb.connect(LANCEDB_URI, region=LANCEDB_REGION, api_key=api_key)

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
    else:
        table = create_lectures_table(db, TABLE_NAME)
        table.add(rows)

    table.create_fts_index("text", replace=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
