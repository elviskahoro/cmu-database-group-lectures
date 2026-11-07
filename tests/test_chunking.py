"""Guards for lecture_search.chunking: windowing, overlap, IDs, deep links.

Also covers the transcript-directory processors: titles.json must not be
treated as a transcript, filename dates are the fallback when a video_id is
missing from the metadata, and `limit` caps partial corpus runs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from lecture_search import (
    ChunkContext,
    ChunkRecord,
    Segment,
    chunk_transcript,
    process_corpus_transcripts,
    process_sample_transcripts,
)

TEST_VIDEO_ID = "test_video"
TEST_DATE = "2026-01-01"
TEST_TITLE = "Test Title"


def _segments(count: int) -> list[Segment]:
    return [
        Segment(start=float(i * 10), end=float((i + 1) * 10), text=f"segment_{i}")
        for i in range(count)
    ]


def _chunk(
    segments: list[Segment],
    video_id: str = TEST_VIDEO_ID,
) -> list[ChunkRecord]:
    return chunk_transcript(
        segments=segments,
        context=ChunkContext(video_id=video_id, date=TEST_DATE, title=TEST_TITLE),
        window_duration=75.0,
        overlap_duration=15.0,
    )


def test_window_bounds() -> None:
    chunks = _chunk(_segments(10))

    assert len(chunks) > 0, "Should produce chunks"
    for chunk in chunks:
        assert chunk.start_seconds < chunk.end_seconds, (
            f"Start should be before end: {chunk}"
        )
        assert chunk.text, "Chunk should have text"
        assert chunk.video_id == TEST_VIDEO_ID
        assert chunk.date == TEST_DATE
        assert chunk.title == TEST_TITLE


def test_overlap() -> None:
    chunks = _chunk(_segments(15))

    assert len(chunks) > 1, "Should produce multiple chunks"

    for i in range(len(chunks) - 1):
        current_end = chunks[i].end_seconds
        next_start = chunks[i + 1].start_seconds
        overlap = current_end - next_start
        assert overlap >= 0, f"Chunks should overlap or be adjacent ({overlap=})"


def test_chunk_ids() -> None:
    chunks = _chunk(_segments(20), video_id="video123")

    for i, chunk in enumerate(chunks):
        assert chunk.id == f"video123:{i}", "ID should be video_id:chunk_index"
        assert chunk.chunk_index == i


def test_deep_link() -> None:
    chunks = _chunk(_segments(8), video_id="abc123")

    for chunk in chunks:
        assert f"youtu.be/{chunk.video_id}" in chunk.deep_link
        assert "?t=" in chunk.deep_link


def test_process_sample_transcripts_reads_titles_json(tmp_path: Path) -> None:
    transcript = {"segments": [vars(segment) for segment in _segments(10)]}
    (tmp_path / "2026-01-01·vid123·transcription.json").write_text(
        json.dumps(transcript),
        encoding="utf-8",
    )
    titles_json = json.dumps(
        {"vid123": {"date": "2026-01-01", "title": "My Lecture", "duration": 100}},
    )
    (tmp_path / "titles.json").write_text(titles_json, encoding="utf-8")

    results = process_sample_transcripts(tmp_path)

    assert set(results) == {"vid123"}, "titles.json must not be treated as a transcript"
    assert all(chunk.title == "My Lecture" for chunk in results["vid123"])


def test_process_sample_transcripts_missing_titles_json(tmp_path: Path) -> None:
    transcript = {"segments": [{"start": 0.0, "end": 10.0, "text": "only"}]}
    (tmp_path / "2026-01-01·vid999·transcription.json").write_text(
        json.dumps(transcript),
        encoding="utf-8",
    )

    results = process_sample_transcripts(tmp_path)

    assert set(results) == {"vid999"}
    assert all(chunk.title == "" for chunk in results["vid999"])


def test_process_corpus_transcripts_reads_titles_all(tmp_path: Path) -> None:
    transcript = {"segments": [vars(segment) for segment in _segments(10)]}
    (tmp_path / "2016-05-16·vid123·transcription.json").write_text(
        json.dumps(transcript),
        encoding="utf-8",
    )
    (tmp_path / "2016-05-17·vid456·transcription.json").write_text(
        json.dumps(transcript),
        encoding="utf-8",
    )
    titles = {
        "vid123": {
            "date": "2016-05-16",
            "title": "Full Corpus Lecture",
            "duration": 100,
        },
    }

    results = process_corpus_transcripts(tmp_path, titles)

    assert set(results) == {"vid123", "vid456"}
    assert all(chunk.title == "Full Corpus Lecture" for chunk in results["vid123"])
    assert all(chunk.date == "2016-05-16" for chunk in results["vid123"])
    # Missing from metadata: empty title, filename date as fallback.
    assert all(chunk.title == "" for chunk in results["vid456"])
    assert all(chunk.date == "2016-05-17" for chunk in results["vid456"])


def test_process_corpus_transcripts_limit(tmp_path: Path) -> None:
    transcript = {"segments": [{"start": 0.0, "end": 10.0, "text": "only"}]}
    for i in range(3):
        (tmp_path / f"2026-01-0{i + 1}·vid{i}·transcription.json").write_text(
            json.dumps(transcript),
            encoding="utf-8",
        )

    results = process_corpus_transcripts(tmp_path, {}, limit=2)

    assert set(results) == {"vid0", "vid1"}
