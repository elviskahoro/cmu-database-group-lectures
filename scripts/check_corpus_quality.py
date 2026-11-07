"""Quality checks over the pulled transcript corpus.

Checks (per bead cdgl-575):
- every video_id in metadata/titles_all.json has a transcript
- no empty transcripts
- no repetition loops (same segment text repeated >30x)
- no >120s gaps between consecutive segments
- wpm sanity (120-230)
- last segment end within 5% of the lecture duration
"""  # noqa: INP001  # standalone script, not a package

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

CORPUS = Path("data/corpus/transcription")
TITLES = Path("metadata/titles_all.json")
ISSUE_KINDS = ("empty", "repetition", "gap", "wpm", "tail")
REPETITION_LIMIT = 30
MAX_GAP_SECONDS = 120
WPM_MIN = 120
WPM_MAX = 230
TAIL_TOLERANCE_RATIO = 0.05
SECONDS_PER_MINUTE = 60

IssueMap = dict[str, list[str]]


class TitleMeta(TypedDict, total=False):
    """Per-video metadata from metadata/titles_all.json."""

    date: str
    title: str
    duration: float


class Segment(TypedDict):
    """One Whisper transcript segment."""

    text: str
    start: float
    end: float


class Transcription(TypedDict, total=False):
    """Transcription JSON payload for one video."""

    text: str
    segments: list[Segment]


def repetition_issue(label: str, segs: list[Segment]) -> str | None:
    """Flag transcripts whose most common segment text loops suspiciously."""
    counts = Counter(re.sub(r"\s+", " ", s["text"].strip()) for s in segs)
    worst, n = counts.most_common(1)[0]
    if n > REPETITION_LIMIT:
        return f"{label} | x{n}: {worst[:80]!r}"
    return None


def gap_issue(label: str, segs: list[Segment]) -> str | None:
    """Flag long silences between consecutive segments."""
    gaps: list[tuple[float, float]] = []
    prev_end = segs[0]["start"]
    for s in segs:
        gap = s["start"] - prev_end
        if gap > MAX_GAP_SECONDS:
            gaps.append((prev_end, s["start"]))
        prev_end = max(prev_end, s["end"])
    if gaps:
        gtxt = ", ".join(f"{a:.0f}s-{b:.0f}s" for a, b in gaps[:3])
        return f"{label} | {len(gaps)} gap(s): {gtxt}"
    return None


def pacing_issues(
    label: str,
    segs: list[Segment],
    words: int,
    duration: float | None,
) -> list[tuple[str, str]]:
    """Wpm sanity and transcript-vs-duration tail checks."""
    if not duration:
        return []
    found: list[tuple[str, str]] = []
    minutes = segs[-1]["end"] / SECONDS_PER_MINUTE
    wpm = words / minutes if minutes > 0 else 0
    if not (WPM_MIN <= wpm <= WPM_MAX):
        found.append(("wpm", f"{label} | wpm={wpm:.0f} ({words}w/{minutes:.1f}m)"))
    tail_gap = duration - segs[-1]["end"]
    if abs(tail_gap) > TAIL_TOLERANCE_RATIO * duration:
        found.append(
            (
                "tail",
                (
                    f"{label} | ends {segs[-1]['end']:.0f}s of {duration}s "
                    f"(off {tail_gap:+.0f}s)"
                ),
            ),
        )
    return found


def check_video(vid: str, path: Path, meta: TitleMeta) -> IssueMap:
    """Run all per-video checks; return issue kind -> messages."""
    data: Transcription = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    label = f"{vid} {meta.get('date', '?')} {meta.get('title', '')[:60]}"
    words = len(data.get("text", "").split())

    found: IssueMap = {kind: [] for kind in ISSUE_KINDS}
    if not segs or words == 0:
        found["empty"].append(label)
        return found
    if (msg := repetition_issue(label, segs)) is not None:
        found["repetition"].append(msg)
    if (msg := gap_issue(label, segs)) is not None:
        found["gap"].append(msg)
    for kind, msg in pacing_issues(label, segs, words, meta.get("duration")):
        found[kind].append(msg)
    return found


def main() -> None:
    """Check every transcript against the metadata and print a report."""
    titles: dict[str, TitleMeta] = json.loads(TITLES.read_text(encoding="utf-8"))
    files = sorted(CORPUS.glob("*.json"))
    by_vid: dict[str, Path] = {}
    for path in files:
        # filename: <date>·<video_id>·transcription.json
        by_vid[path.name.split("·")[1]] = path

    missing = sorted(set(titles) - set(by_vid))
    extra = sorted(set(by_vid) - set(titles))
    print(f"transcripts: {len(files)}, titles: {len(titles)}")  # noqa: T201  # CLI report output
    print(f"missing transcripts for video_ids: {len(missing)}")  # noqa: T201  # CLI report output
    for v in missing:
        print(f"  MISSING {v} {titles[v].get('title', '?')[:70]}")  # noqa: T201  # CLI report output
    if extra:
        print(f"extra transcripts not in titles: {len(extra)}")  # noqa: T201  # CLI report output
        for v in extra:
            print(f"  EXTRA {v}")  # noqa: T201  # CLI report output

    issues: IssueMap = {kind: [] for kind in ISSUE_KINDS}
    for vid, path in sorted(by_vid.items()):
        for kind, messages in check_video(vid, path, titles.get(vid, {})).items():
            issues[kind].extend(messages)

    print()  # noqa: T201  # CLI report output
    for kind, rows in issues.items():
        print(f"=== {kind}: {len(rows)} ===")  # noqa: T201  # CLI report output
        for row in rows:
            print(f"  {row}")  # noqa: T201  # CLI report output


if __name__ == "__main__":
    main()
