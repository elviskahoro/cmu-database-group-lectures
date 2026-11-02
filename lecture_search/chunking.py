from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from datetime import datetime

import json
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class ChunkRecord:
    id: str
    video_id: str
    date: str
    title: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str
    deep_link: str


def chunk_transcript(
    segments: list[Segment],
    video_id: str,
    date: str,
    title: str,
    window_duration: float = 75.0,
    overlap_duration: float = 15.0,
) -> list[ChunkRecord]:
    if not segments:
        return []

    chunk_index = 0
    chunks: list[ChunkRecord] = []
    current_start = segments[0].start
    current_text_parts: list[str] = []
    current_start_time = segments[0].start

    for segment in segments:
        if segment.end - current_start > window_duration:
            if current_text_parts:
                chunk_text = " ".join(current_text_parts)
                chunk_end = segment.start
                chunks.append(
                    ChunkRecord(
                        id=f"{video_id}:{chunk_index}",
                        video_id=video_id,
                        date=date,
                        title=title,
                        chunk_index=chunk_index,
                        start_seconds=current_start_time,
                        end_seconds=chunk_end,
                        text=chunk_text,
                        deep_link=f"https://youtu.be/{video_id}?t={int(current_start_time)}",
                    )
                )
                chunk_index += 1

            current_start_time = max(current_start - overlap_duration, segments[0].start)
            current_text_parts = []
            current_start = segment.start

        current_text_parts.append(segment.text)

    if current_text_parts:
        chunk_text = " ".join(current_text_parts)
        chunk_end = segments[-1].end
        chunks.append(
            ChunkRecord(
                id=f"{video_id}:{chunk_index}",
                video_id=video_id,
                date=date,
                title=title,
                chunk_index=chunk_index,
                start_seconds=current_start_time,
                end_seconds=chunk_end,
                text=chunk_text,
                deep_link=f"https://youtu.be/{video_id}?t={int(current_start_time)}",
            )
        )

    return chunks


def load_transcript(transcript_path: Path) -> dict[str, Any]:
    with open(transcript_path, "r") as f:
        return json.load(f)


def load_titles(sample_dir: Path) -> dict[str, dict[str, Any]]:
    """Load video_id -> {date, title, duration} metadata from titles.json."""
    titles_path = sample_dir / "titles.json"
    if not titles_path.exists():
        return {}
    return json.loads(titles_path.read_text())


def process_sample_transcripts(
    sample_dir: Path,
) -> dict[str, list[ChunkRecord]]:
    results = {}
    titles = load_titles(sample_dir)

    for transcript_file in sorted(sample_dir.glob("*·transcription.json")):
        filename = transcript_file.name
        date, video_id, _ = filename.split("·", 2)

        data = load_transcript(transcript_file)

        title = titles.get(video_id, {}).get("title", "")

        segments = [
            Segment(start=seg["start"], end=seg["end"], text=seg["text"])
            for seg in data.get("segments", [])
        ]

        chunks = chunk_transcript(
            segments=segments,
            video_id=video_id,
            date=date,
            title=title,
        )
        results[video_id] = chunks

    return results


def test_window_bounds():
    segments = [
        Segment(start=0.0, end=10.0, text="first"),
        Segment(start=10.0, end=20.0, text="second"),
        Segment(start=20.0, end=30.0, text="third"),
        Segment(start=30.0, end=40.0, text="fourth"),
        Segment(start=40.0, end=50.0, text="fifth"),
        Segment(start=50.0, end=60.0, text="sixth"),
        Segment(start=60.0, end=70.0, text="seventh"),
        Segment(start=70.0, end=80.0, text="eighth"),
        Segment(start=80.0, end=90.0, text="ninth"),
        Segment(start=90.0, end=100.0, text="tenth"),
    ]

    chunks = chunk_transcript(
        segments=segments,
        video_id="test_video",
        date="2026-01-01",
        title="Test Title",
        window_duration=75.0,
        overlap_duration=15.0,
    )

    assert len(chunks) > 0, "Should produce chunks"
    for chunk in chunks:
        assert chunk.start_seconds < chunk.end_seconds, f"Start should be before end: {chunk}"
        assert chunk.text, "Chunk should have text"
        assert chunk.video_id == "test_video"
        assert chunk.date == "2026-01-01"
        assert chunk.title == "Test Title"


def test_overlap():
    segments = [
        Segment(start=float(i * 10), end=float((i + 1) * 10), text=f"segment_{i}")
        for i in range(15)
    ]

    chunks = chunk_transcript(
        segments=segments,
        video_id="test_video",
        date="2026-01-01",
        title="Test Title",
        window_duration=75.0,
        overlap_duration=15.0,
    )

    assert len(chunks) > 1, "Should produce multiple chunks"

    for i in range(len(chunks) - 1):
        current_end = chunks[i].end_seconds
        next_start = chunks[i + 1].start_seconds
        overlap = current_end - next_start
        assert overlap >= 0, f"Chunks should overlap or be adjacent (overlap: {overlap})"


def test_chunk_ids():
    segments = [
        Segment(start=float(i * 10), end=float((i + 1) * 10), text=f"segment_{i}")
        for i in range(20)
    ]

    chunks = chunk_transcript(
        segments=segments,
        video_id="video123",
        date="2026-01-01",
        title="Test Title",
        window_duration=75.0,
        overlap_duration=15.0,
    )

    for i, chunk in enumerate(chunks):
        assert chunk.id == f"video123:{i}", f"ID should be video_id:chunk_index"
        assert chunk.chunk_index == i


def test_deep_link():
    segments = [
        Segment(start=0.0, end=10.0, text="first"),
        Segment(start=10.0, end=20.0, text="second"),
        Segment(start=20.0, end=30.0, text="third"),
        Segment(start=30.0, end=40.0, text="fourth"),
        Segment(start=40.0, end=50.0, text="fifth"),
        Segment(start=50.0, end=60.0, text="sixth"),
        Segment(start=60.0, end=70.0, text="seventh"),
        Segment(start=70.0, end=80.0, text="eighth"),
    ]

    chunks = chunk_transcript(
        segments=segments,
        video_id="abc123",
        date="2026-01-01",
        title="Test Title",
        window_duration=75.0,
        overlap_duration=15.0,
    )

    for chunk in chunks:
        assert f"youtu.be/{chunk.video_id}" in chunk.deep_link
        assert "?t=" in chunk.deep_link


def test_process_sample_transcripts_reads_titles_json(tmp_path):
    transcript = {
        "segments": [
            {"start": float(i * 10), "end": float((i + 1) * 10), "text": f"segment_{i}"}
            for i in range(10)
        ]
    }
    (tmp_path / "2026-01-01·vid123·transcription.json").write_text(json.dumps(transcript))
    (tmp_path / "titles.json").write_text(
        json.dumps({"vid123": {"date": "2026-01-01", "title": "My Lecture", "duration": 100}})
    )

    results = process_sample_transcripts(tmp_path)

    assert set(results) == {"vid123"}, "titles.json must not be treated as a transcript"
    assert all(chunk.title == "My Lecture" for chunk in results["vid123"])


def test_process_sample_transcripts_missing_titles_json(tmp_path):
    transcript = {"segments": [{"start": 0.0, "end": 10.0, "text": "only"}]}
    (tmp_path / "2026-01-01·vid999·transcription.json").write_text(json.dumps(transcript))

    results = process_sample_transcripts(tmp_path)

    assert set(results) == {"vid999"}
    assert all(chunk.title == "" for chunk in results["vid999"])


if __name__ == "__main__":
    test_window_bounds()
    test_overlap()
    test_chunk_ids()
    test_deep_link()
    print("All tests passed!")

    sample_dir = Path(__file__).parent.parent / "data" / "sample"
    results = process_sample_transcripts(sample_dir)
    assert "titles" not in results, "titles.json must not be parsed as a transcript"
    for video_id, chunks in results.items():
        print(f"\n{video_id}: {len(chunks)} chunks")
        for chunk in chunks[:2]:
            print(
                f"  [{chunk.chunk_index}] {chunk.start_seconds:.1f}s-{chunk.end_seconds:.1f}s: {chunk.text[:60]}..."
            )
