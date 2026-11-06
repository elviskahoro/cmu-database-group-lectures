"""Chunking, embedding, and search plumbing for the lecture corpus."""

from lecture_search.chunking import (
    ChunkContext,
    ChunkRecord,
    Segment,
    chunk_transcript,
    load_titles_all,
    process_corpus_transcripts,
    process_sample_transcripts,
)

__all__ = [
    "ChunkContext",
    "ChunkRecord",
    "Segment",
    "chunk_transcript",
    "load_titles_all",
    "process_corpus_transcripts",
    "process_sample_transcripts",
]
