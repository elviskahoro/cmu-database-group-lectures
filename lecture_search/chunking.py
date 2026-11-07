"""Chunk lecture transcripts into overlapping, deep-linkable windows.

Transcripts are YouTube auto-captions stored as
``<date>·<video_id>·transcription.json`` files. Each lecture becomes a
sequence of ``ChunkRecord`` windows (default 75 s with 15 s overlap) whose
metadata — video_id, date, title, deep link — comes from a per-directory
``titles.json`` (sample corpus) or the full-corpus ``metadata/titles_all.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


@dataclass
class Segment:
    """One transcript segment: a timed span of caption text."""

    start: float
    end: float
    text: str


@dataclass
class ChunkRecord:
    """One chunk window over a lecture, with deep link back to the video."""

    id: str
    video_id: str
    date: str
    title: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    text: str
    deep_link: str


@dataclass(frozen=True)
class ChunkContext:
    """Lecture identity stamped onto every chunk the transcript yields."""

    video_id: str
    date: str
    title: str


class TranscriptSegment(TypedDict):
    """One segment entry in a ``*·transcription.json`` file."""

    start: float
    end: float
    text: str


class Transcript(TypedDict, total=False):
    """Parsed ``*·transcription.json`` payload."""

    segments: list[TranscriptSegment]


class TitleMeta(TypedDict, total=False):
    """One entry of a titles mapping: per-lecture metadata."""

    date: str
    title: str
    duration: float


def chunk_transcript(
    segments: list[Segment],
    context: ChunkContext,
    window_duration: float = 75.0,
    overlap_duration: float = 15.0,
) -> list[ChunkRecord]:
    """Split one lecture's segments into overlapping chunk windows.

    A new chunk starts whenever the accumulated window would exceed
    ``window_duration`` seconds; the next window rewinds by
    ``overlap_duration`` seconds (bounded by the lecture start) so
    sentences spanning a boundary appear in both chunks.
    """
    if not segments:
        return []

    video_id = context.video_id
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
                        date=context.date,
                        title=context.title,
                        chunk_index=chunk_index,
                        start_seconds=current_start_time,
                        end_seconds=chunk_end,
                        text=chunk_text,
                        deep_link=f"https://youtu.be/{video_id}?t={int(current_start_time)}",
                    ),
                )
                chunk_index += 1

            current_start_time = max(
                current_start - overlap_duration,
                segments[0].start,
            )
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
                date=context.date,
                title=context.title,
                chunk_index=chunk_index,
                start_seconds=current_start_time,
                end_seconds=chunk_end,
                text=chunk_text,
                deep_link=f"https://youtu.be/{video_id}?t={int(current_start_time)}",
            ),
        )

    return chunks


def load_transcript(transcript_path: Path) -> Transcript:
    """Parse one ``*·transcription.json`` file."""
    return cast("Transcript", json.loads(transcript_path.read_text(encoding="utf-8")))


def load_titles(sample_dir: Path) -> dict[str, TitleMeta]:
    """Load video_id -> {date, title, duration} metadata from titles.json."""
    titles_path = sample_dir / "titles.json"
    if not titles_path.exists():
        return {}
    return cast(
        "dict[str, TitleMeta]",
        json.loads(titles_path.read_text(encoding="utf-8")),
    )


def load_titles_all(metadata_path: Path) -> dict[str, TitleMeta]:
    """Load video_id -> {date, title, duration} for the full 595-lecture corpus."""
    return cast(
        "dict[str, TitleMeta]",
        json.loads(Path(metadata_path).read_text(encoding="utf-8")),
    )


def process_corpus_transcripts(
    transcripts_dir: Path,
    titles: dict[str, TitleMeta],
    limit: int | None = None,
) -> dict[str, list[ChunkRecord]]:
    """Chunk a directory of full-corpus transcripts.

    Like process_sample_transcripts, but titles/dates come from the
    metadata/titles_all.json mapping (passed in) instead of a per-directory
    titles.json. Filename date is the fallback when a video_id is missing
    from the metadata. `limit` caps the number of transcripts processed
    (sorted order) for partial runs.
    """
    results: dict[str, list[ChunkRecord]] = {}
    transcript_files = sorted(transcripts_dir.glob("*·transcription.json"))
    if limit is not None:
        transcript_files = transcript_files[:limit]

    for transcript_file in transcript_files:
        date_from_name, video_id, _ = transcript_file.name.split("·", 2)
        data = load_transcript(transcript_file)

        meta = titles.get(video_id, {})
        title = meta.get("title", "")
        date = meta.get("date", date_from_name)

        segments = [
            Segment(start=seg["start"], end=seg["end"], text=seg["text"])
            for seg in data.get("segments", [])
        ]

        chunks = chunk_transcript(
            segments=segments,
            context=ChunkContext(video_id=video_id, date=date, title=title),
        )
        results[video_id] = chunks

    return results


def process_sample_transcripts(
    sample_dir: Path,
) -> dict[str, list[ChunkRecord]]:
    """Chunk a sample directory, reading titles from its titles.json."""
    results: dict[str, list[ChunkRecord]] = {}
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
            context=ChunkContext(video_id=video_id, date=date, title=title),
        )
        results[video_id] = chunks

    return results
