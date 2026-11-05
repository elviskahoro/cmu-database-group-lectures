"""Chunk-quality filter: drop degenerate Whisper chunks before embedding.

docs/corpus-quality.md found deterministic Whisper failures (repetition
loops, 'you you you' tails, near-empty stretches) in tEa5sAh-kVk (LMDB),
6H0yuYPHw0Q, YeNF3pJwLKw, and OIT9Y1IBZUU. Re-transcription does not help
(the source audio is silent or ~-38 dB after minute 20), so
scripts/load_corpus.py routes chunks through `chunk_drop_reasons` before
spending embedding calls on them.

A chunk is dropped when any rule fires:

- too short: fewer than MIN_WORDS words (empty / near-empty stretches)
- low variety: unique-word ratio below MIN_UNIQUE_RATIO
- dominant word: one token over MAX_DOMINANT_WORD_SHARE of the chunk
  ('you you you', 'the, the, the')
- dominant bigram: one bigram over MAX_DOMINANT_BIGRAM_SHARE ('Hold on.' x59)
- duplicate sentences: over MAX_DUPLICATE_SENTENCE_SHARE of the >=3-word
  sentences are verbatim repeats (hallucinated loops)
- repeated trigrams: over MAX_REPEATED_TRIGRAM_COVERAGE of trigram slots
  are repeats of an earlier trigram (catches paraphrased loops that
  verbatim-sentence checks miss)

Thresholds were tuned on the full 595-transcript corpus (2026-11-05):
all 58 chunks of tEa5sAh-kVk (degenerate across the whole file) and the
degenerate tail chunks of the other three bad lectures are dropped, while
114/33,440 chunks (0.34%) from healthy lectures are dropped — under the
0.5% budget from bead cdgl-819. Healthy-side drops are themselves mostly
stutter loops ('Thank you. Thank you. Thank you.'), so the filter errs on
the side of removing non-retrievable text.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

MIN_WORDS = 10
MIN_UNIQUE_RATIO = 0.30
MAX_DOMINANT_WORD_SHARE = 0.35
MAX_DOMINANT_BIGRAM_SHARE = 0.30
MAX_DUPLICATE_SENTENCE_SHARE = 0.25
MAX_REPEATED_TRIGRAM_COVERAGE = 0.25

# Sentences shorter than this are normal back-and-forth ('Yeah.', 'Thank
# you.') and must not count toward the duplicate-sentence rule.
MIN_SENTENCE_WORDS = 3
# Need at least this many qualifying sentences before the
# duplicate-sentence share is meaningful.
MIN_SENTENCE_SAMPLE = 4

WORD_RE = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


class ChunkLike(Protocol):
    """Structural type for anything partition_chunks can filter."""

    video_id: str
    title: str
    chunk_index: int
    text: str


C = TypeVar("C", bound=ChunkLike)


def chunk_drop_reasons(text: str) -> list[str]:
    """Return why a chunk is degenerate; an empty list means keep it."""
    words = WORD_RE.findall(text.lower())
    word_count = len(words)
    if word_count < MIN_WORDS:
        return ["too-short"]

    reasons: list[str] = []
    word_counts = Counter(words)

    if len(word_counts) / word_count < MIN_UNIQUE_RATIO:
        reasons.append("low-variety")

    top_word_share = word_counts.most_common(1)[0][1] / word_count
    if top_word_share > MAX_DOMINANT_WORD_SHARE:
        reasons.append("dominant-word")

    bigram_counts = Counter(pairwise(words))
    if bigram_counts:
        top_bigram_share = bigram_counts.most_common(1)[0][1] / (word_count - 1)
        if top_bigram_share > MAX_DOMINANT_BIGRAM_SHARE:
            reasons.append("dominant-bigram")

    trigram_slots = word_count - 2
    if trigram_slots > 0:
        trigram_counts = Counter(zip(words, words[1:], words[2:], strict=False))
        repeated_slots = sum(
            count - 1 for count in trigram_counts.values() if count > 1
        )
        if repeated_slots / trigram_slots > MAX_REPEATED_TRIGRAM_COVERAGE:
            reasons.append("repeated-trigrams")

    sentences = [
        " ".join(WORD_RE.findall(sentence.lower()))
        for sentence in SENTENCE_SPLIT_RE.split(text)
    ]
    sentences = [
        sentence
        for sentence in sentences
        if len(sentence.split()) >= MIN_SENTENCE_WORDS
    ]
    if len(sentences) >= MIN_SENTENCE_SAMPLE:
        duplicate_share = 1 - len(set(sentences)) / len(sentences)
        if duplicate_share > MAX_DUPLICATE_SENTENCE_SHARE:
            reasons.append("duplicate-sentences")

    return reasons


def partition_chunks(chunks: Sequence[C]) -> tuple[list[C], list[tuple[C, list[str]]]]:
    """Split chunks into (kept, [(dropped chunk, drop reasons)])."""
    kept: list[C] = []
    dropped: list[tuple[C, list[str]]] = []
    for chunk in chunks:
        reasons = chunk_drop_reasons(chunk.text)
        if reasons:
            dropped.append((chunk, reasons))
        else:
            kept.append(chunk)
    return kept, dropped


def drop_report_lines(
    chunks: Sequence[C],
    dropped: Sequence[tuple[C, list[str]]],
    sample_limit: int = 10,
) -> list[str]:
    """Human-readable dry-run report: totals, top droppers, samples."""
    per_video_total: Counter[tuple[str, str]] = Counter()
    for chunk in chunks:
        per_video_total[(chunk.video_id, chunk.title)] += 1
    per_video_dropped: Counter[tuple[str, str]] = Counter()
    for chunk, _ in dropped:
        per_video_dropped[(chunk.video_id, chunk.title)] += 1

    total = len(chunks)
    kept = total - len(dropped)
    share = (len(dropped) / total * 100) if total else 0.0
    summary = (
        f"Chunk-quality filter: {total} chunks -> {kept} kept, "
        f"{len(dropped)} dropped ({share:.2f}%)"
    )
    lines = [summary]

    if per_video_dropped:
        lines.append("Top droppers (chunks dropped / total per lecture):")
        worst = per_video_dropped.most_common(10)
        for (video_id, title), count in worst:
            total_for_video = per_video_total[(video_id, title)]
            lines.append(
                f"  {video_id}  {count}/{total_for_video}  {title[:60]}",
            )

    if dropped:
        lines.append(f"Sample dropped chunks (up to {sample_limit}):")
        for chunk, reasons in list(dropped)[:sample_limit]:
            preview = " ".join(chunk.text.split())[:80]
            lines.append(
                (
                    f"  {chunk.video_id}:{chunk.chunk_index} [{', '.join(reasons)}]"
                    f' "{preview}"'
                ),
            )
    return lines
