# Corpus quality report — full 595-transcript pull

Date: 2026-11-05
Source: Modal volume `cmu-transcripts` (`/transcription`), pulled with
`MODAL_PROFILE=marcella modal volume get --force cmu-transcripts /transcription data/corpus`
(gitignored). Checker script: `scripts/check_corpus_quality.py`.

## Gate

- Volume held **595** files (`modal volume ls ... | wc -l` = 595). ✓
- `modal app list` showed no running `youtube-transcription-r2` app
  (only the deployed `cmu-lecture…` app, 0 tasks). ✓

## Automated checks (all 595 files)

| Check | Result |
|---|---|
| File count | 595/595 ✓ |
| Empty files / empty transcripts | 0 ✓ |
| Every `video_id` in `metadata/titles_all.json` present | 595/595, none missing, none extra ✓ |
| Gaps > 120 s between segments | 0 ✓ |
| Repetition loops (same segment text > 30×) | 5 flagged, 1 real (see below) |
| wpm sanity (120–230) | 9 flagged, 2 real failures (see below) |
| Last segment end within 5 % of duration | 1 flagged |

## Outliers

### Definite transcription failures — re-transcribe

1. **`tEa5sAh-kVk` — 2016-05-16 Howard Chu, LMDB (Databaseology)**
   Unique-word ratio 0.042 (next-lowest in corpus is 0.115; median 0.162),
   wpm 280, 44× repeated segment "I'm going to go ahead and start".
   Degenerate Whisper repetition loop across the whole file.
2. **`6H0yuYPHw0Q` — 2017-01-30 L04 Optimistic Concurrency Control**
   wpm 39 (3 051 words over 77 min). Nearly all text sits in the first
   ~20 min; segments after that are near-empty with broken timestamps and a
   "because because Thank you" hallucinated ending.
3. **`YeNF3pJwLKw` — 2021-09-09 03 Database Storage I**
   wpm 33 (2 236 words over 67 min). Same failure mode: content stops after
   ~20 min, ends in a "you you you you" loop.

### Borderline — recommend re-transcribe

4. **`OIT9Y1IBZUU` — 2017-10-05 11 Sorting & Joins (Fall 2017)**
   Last segment ends at 4 294 s of a 4 542 s lecture (248 s / 5.5 % early)
   and the final segment is a "you you you …" loop. Bulk of the lecture
   (699 words in the last 5 min of transcript) is fine, but the tail is
   truncated/degenerate.

### Flagged but benign (no action)

- Repetition flags in `0_m5gPfzEYQ` (75× "All right."), `nUwT7PEQ49o`
  (69× "Yes."), `2HtfGdsrwqA` (31× ".") are spread across the full lecture
  with normal unique-word ratios (0.129–0.169) — speaker tics, not loops.
- `uikbtpVZS2s` (2022-08-31) has 59× "Hold on." clustered within a 6-second
  window (~1 525 s) — a local glitch in an otherwise healthy transcript.
- Low-wpm industry talks (`Fmz2DUGJLUY` eXtremeDB 100, `HO9UKYomI7c`
  BerkeleyDB 117, `J_YtAoPL2c4` Apache Doris 115, `VEYdZL0bU-I` Odyssey 113,
  `VLAvZw0ZEwI` GreptimeDB 111) have no gaps and healthy text — just
  slower-paced speakers.
- `tNxvuZWW98E` (2018 SCS awards clip) is a genuine 5.4-min short video.

## Manual reading pass (jargon)

Read excerpts from 5 transcripts: 2016 (`-7VjPVmw4vM`, Concurrency Control
II), 2019 (`0_m5gPfzEYQ`, Distributed Databases), 2022 (`2HtfGdsrwqA`,
Database Storage 2), 2025 (`YWtH10gfcY0`, Intro lecture), and an industry
talk (`J_YtAoPL2c4`, Apache Doris). Technical content is coherent and
terminology (MVCC versions, log-structured vs page-oriented storage, tuple
layout, SQL parser pipeline, inverted index / variant type) is rendered
correctly.

One systematic artifact: **"sequel" in place of "SQL"** — 717 occurrences in
254 of 595 files (plus 171× "my sequel", 31× "no sequel"). Pervasive but
predictable; handle at embedding/retrieval time (normalize `sequel`→`SQL`)
rather than re-transcribing. No "postgrest"/"postgress" misrenderings found.

## Re-transcribe list

Delete these from the volume and re-run the resumable job:

```
tEa5sAh-kVk   2016-05-16  Howard Chu - LMDB
6H0yuYPHw0Q   2017-01-30  L04 - Optimistic Concurrency Control
YeNF3pJwLKw   2021-09-09  03 - Database Storage I
OIT9Y1IBZUU   2017-10-05  11 Sorting & Joins (borderline: tail truncated)
```

## Chunk-quality filter (cdgl-819, 2026-11-05)

Re-transcription does not fix the four failures above (deterministic
Whisper; `YeNF3pJwLKw` audio is digitally silent after minute 20,
`6H0yuYPHw0Q` is ~-38 dB), so `lecture_search/quality.py` drops degenerate
chunks before embedding in `scripts/load_corpus.py`. Rules: fewer than 10
words, unique-word ratio < 0.30, one word > 35% or one bigram > 30% of the
chunk, > 25% verbatim-duplicate >=3-word sentences, or > 25%
repeated-trigram coverage.

Dry-run over the full corpus
(`uv run python scripts/load_corpus.py --transcripts-dir data/corpus/transcription --dry-run`):

- 33,587 chunks -> 33,412 kept, **175 dropped (0.52%)**
- `tEa5sAh-kVk` 58/58 dropped (whole-file loop); the degenerate stretched
  tail chunks of `6H0yuYPHw0Q`, `YeNF3pJwLKw`, and `OIT9Y1IBZUU` dropped
- healthy lectures: 114/33,440 dropped (0.34%, under the 0.5% budget);
  spot-checks show these are real junk too (stutter loops, 'Thank you.
  Thank you.', 'We need to know exactly what's happening.' xN) — e.g.
  `VHSDhMO63ww` (07 Tree Indexes I, Fall 2018) has 14 hallucinated chunks
  this report's file-level checks did not flag
- guarded regression test: `tests/test_chunk_quality.py::test_tuned_thresholds_on_real_corpus`
