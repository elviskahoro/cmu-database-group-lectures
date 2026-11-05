"""Guards for the chunk-quality filter (lecture_search/quality.py, cdgl-819).

Synthetic examples cover each drop rule (too short, low variety, dominant
word/bigram, duplicate sentences, repeated trigrams) plus healthy text that
must survive. A corpus-grounded test re-checks the tuned acceptance
thresholds against the real 595-transcript pull when data/corpus is present
(gitignored; skipped in CI): every degenerate-region chunk of the four
lectures named in docs/corpus-quality.md is dropped, and healthy-lecture
drops stay under 0.5%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from lecture_search import load_titles_all, process_corpus_transcripts
from lecture_search.quality import (
    MIN_WORDS,
    chunk_drop_reasons,
    drop_report_lines,
    partition_chunks,
)


@dataclass
class FakeChunk:
    """Minimal stand-in for ChunkRecord satisfying quality.ChunkLike."""

    video_id: str
    title: str
    chunk_index: int
    text: str


CORPUS_DIR = Path("data/corpus/transcription")
TITLES_PATH = Path("metadata/titles_all.json")

# Lectures with known Whisper failures (docs/corpus-quality.md). The whole
# tEa5sAh-kVk file is a repetition loop; the other three fail only in
# their tail regions (silent/quiet audio after ~minute 20), which the
# chunker lumps into one stretched final chunk.
DEGENERATE_WHOLE_FILE = "tEa5sAh-kVk"
DEGENERATE_TAILS = {"6H0yuYPHw0Q", "YeNF3pJwLKw", "OIT9Y1IBZUU"}
MAX_HEALTHY_DROP_SHARE = 0.005
MIN_HEALTHY_CHUNKS = 30_000  # sanity guard against an empty/partial corpus
EXPECTED_PARTITION_KEPT = 2
EXPECTED_PARTITION_DROPPED = 1
EXPECTED_REPORT_KEPT = 2

HEALTHY_PARAGRAPH = (
    "So the idea behind external merge sort is that we break the table "
    "into runs that fit in memory, sort each one, and then merge them "
    "back together a few pages at a time. The number of passes depends "
    "on how many buffer pages the database system gives us, and each "
    "pass reads and writes every page exactly once, which is why the "
    "IO cost dominates everything else in this algorithm."
)


def test_empty_and_punctuation_only_chunks_are_dropped() -> None:
    assert chunk_drop_reasons("") == ["too-short"]
    assert chunk_drop_reasons("   \n  ") == ["too-short"]
    assert chunk_drop_reasons(". . . . . . . . . . . .") == ["too-short"]


def test_very_short_chunk_is_dropped() -> None:
    assert chunk_drop_reasons("Thank you. Bye.") == ["too-short"]


def test_min_words_boundary_keeps_varied_text() -> None:
    text = " ".join(f"word{i}" for i in range(MIN_WORDS))
    assert chunk_drop_reasons(text) == []


def test_you_you_you_loop_is_dropped() -> None:
    text = " ".join(["you"] * 40)
    reasons = chunk_drop_reasons(text)
    assert "low-variety" in reasons
    assert "dominant-word" in reasons


def test_stutter_loop_is_dropped() -> None:
    text = "Like the, " * 30 + "the answer."
    reasons = chunk_drop_reasons(text)
    assert "dominant-word" in reasons
    assert "dominant-bigram" in reasons


def test_hold_on_bigram_loop_is_dropped() -> None:
    text = " ".join(["Hold on."] * 40)
    reasons = chunk_drop_reasons(text)
    assert "dominant-bigram" in reasons
    assert "duplicate-sentences" not in reasons  # too short to qualify


def test_repeated_verbatim_sentences_are_dropped() -> None:
    text = (
        "I'm not a computer, I'm not a computer. " * 8
        + "What is going on here today folks?"
    )
    assert "duplicate-sentences" in chunk_drop_reasons(text)


def test_short_sentence_repetition_is_not_counted() -> None:
    # Speaker tics ('Yeah.', 'Thank you.') are normal conversation.
    text = (
        HEALTHY_PARAGRAPH
        + " Yeah. Thank you. Yeah. Okay. Thank you. Right. Yeah. Okay."
    )
    assert chunk_drop_reasons(text) == []


def test_paraphrased_loop_is_caught_by_trigram_coverage() -> None:
    # Every sentence is distinct verbatim, but the trigram skeleton repeats.
    text = " ".join(
        f"I'm going to start the script variant {name}."
        for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    )
    reasons = chunk_drop_reasons(text)
    assert "repeated-trigrams" in reasons
    assert "duplicate-sentences" not in reasons


def test_healthy_paragraph_is_kept() -> None:
    assert chunk_drop_reasons(HEALTHY_PARAGRAPH) == []


def test_partition_chunks_splits_and_preserves_order() -> None:
    chunks = [
        FakeChunk("a", "A", 0, HEALTHY_PARAGRAPH),
        FakeChunk("a", "A", 1, "you " * 40),
        FakeChunk("b", "B", 0, HEALTHY_PARAGRAPH),
    ]
    kept, dropped = partition_chunks(chunks)
    assert [c.video_id for c in kept] == ["a", "b"]
    assert len(kept) == EXPECTED_PARTITION_KEPT
    assert len(dropped) == EXPECTED_PARTITION_DROPPED
    dropped_chunk, reasons = dropped[0]
    assert dropped_chunk.chunk_index == 1
    assert "dominant-word" in reasons


def test_drop_report_lines_shows_totals_top_droppers_and_samples() -> None:
    chunks = [
        FakeChunk("good", "Healthy", 0, HEALTHY_PARAGRAPH),
        *[FakeChunk("bad", "Degenerate", i, "you " * 40) for i in range(3)],
        FakeChunk("bad", "Degenerate", 3, HEALTHY_PARAGRAPH),
    ]
    kept, dropped = partition_chunks(chunks)
    lines = drop_report_lines(chunks, dropped)
    report = "\n".join(lines)
    assert "5 chunks -> 2 kept, 3 dropped" in report
    assert "bad  3/4  Degenerate" in report
    assert "bad:0 [low-variety, dominant-word" in report
    assert len(kept) == EXPECTED_REPORT_KEPT


@pytest.mark.skipif(
    not CORPUS_DIR.exists(),
    reason="full corpus is gitignored; pull with modal volume get",
)
def test_tuned_thresholds_on_real_corpus() -> None:
    titles = load_titles_all(TITLES_PATH)
    per_video = process_corpus_transcripts(CORPUS_DIR, titles)

    healthy_total = 0
    healthy_dropped = 0
    for video_id, chunks in per_video.items():
        kept, dropped = partition_chunks(chunks)
        dropped_ids = {c.id for c, _ in dropped}
        if video_id == DEGENERATE_WHOLE_FILE:
            assert not kept, f"{video_id}: expected every chunk dropped"
        elif video_id in DEGENERATE_TAILS:
            tail = chunks[-1]
            assert tail.id in dropped_ids, (
                f"{video_id}:{tail.chunk_index} degenerate tail chunk kept"
            )
        else:
            healthy_total += len(chunks)
            healthy_dropped += len(dropped)

    assert healthy_total > MIN_HEALTHY_CHUNKS
    share = healthy_dropped / healthy_total
    assert share < MAX_HEALTHY_DROP_SHARE, (
        f"healthy drop share {share:.4%} exceeds 0.5% budget"
    )
