# Retrieval smoke check — lectures_sample (cdgl-x8u)

End-to-end retrieval verification over the sample table before scaling to the
full 595-lecture corpus. Reproduce with:

```bash
uv run python -m scripts.retrieval_check
```

The script is read-only against the table, embeds queries locally with Nomic
v1.5 (`search_query: ` prefix), and writes the full per-rank dump to
`data/retrieval_check.json`. It exits nonzero if any expected lecture falls
out of the top 10 in any mode.

## Setup

- Table: `lectures_sample` in LanceDB Cloud (`db://cmu-database-group-eqdwgi`,
  `us-east-1`) — **275 chunks across 6 videos** (4 full lectures: MVCC
  `tUFha9-DuSk`, log-structured storage `2_sTdS4h-bY`, WAL `CedEy54pe3g`,
  cardinality estimation `Kf9Nj3zI9WI`; plus dbt `QnGBxn_XP5U` and YugabyteDB
  `9NdAOHyBq44` clips).
- `top_k = 10` per query, per mode.
- Modes: **vector-only** (cosine on the Nomic embedding) and **hybrid**
  (vector + FTS over `text`, `MRRReranker(weight_vector=1-w, weight_fts=w)` — the same reranker
  as `notebooks/search.py`) at `fts_weight` ∈ {0.2, 0.5, 0.8}.
- Run date: 2026-11-02, on commit `4146b0a` (main) + this branch's script.

## Queries

10 queries from the bead notes; 7 map to lectures loaded in the sample and
are run here. Queries 4, 5, 7 have no matching lecture in the sample and are
**deferred to the full corpus**:

| # | query | expected lecture | status |
|---|---|---|---|
| 1 | how does MVCC avoid read locks | `tUFha9-DuSk` (MVCC) | run |
| 2 | why use a write-ahead log | `CedEy54pe3g` (WAL) | run |
| 3 | LSM tree compaction write amplification | `2_sTdS4h-bY` (log-structured) | run |
| 4 | B+ tree vs skip list for indexes | — (not in sample) | deferred |
| 5 | how does Raft elect a leader | — (not in sample) | deferred |
| 6 | query optimizer cost model cardinality estimation | `Kf9Nj3zI9WI` (cardinality) | run |
| 7 | columnar storage and vectorized execution | — (not in sample) | deferred |
| 8 | how dbt models become tables in the warehouse | `QnGBxn_XP5U` (dbt) | run |
| 9 | distributed SQL replication and consensus | `9NdAOHyBq44` (YugabyteDB) | run |
| 10 | snapshot isolation vs serializable | `tUFha9-DuSk` (MVCC) | run |

## Results: expected-lecture rank (top 10)

**PASS — the expected lecture is rank 1 in every query and every mode.**

| # | vector-only | hybrid fts=0.2 | hybrid fts=0.5 | hybrid fts=0.8 |
|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 |
| 2 | 1 | 1 | 1 | 1 |
| 3 | 1 | 1 | 1 | 1 |
| 6 | 1 | 1 | 1 | 1 |
| 8 | 1 | 1 | 1 | 1 |
| 9 | 1 | 1 | 1 | 1 |
| 10 | 1 | 1 | 1 | 1 |

Vector-only first-hit timestamps and cosine distances (`_distance`, lower is
closer):

| # | first-hit chunk start | distance |
|---|---|---|
| 1 | 7:04 | 0.507 |
| 2 | 45:14 | 0.349 |
| 3 | 1:00:56 | 0.744 |
| 6 | 2:12 | 0.541 |
| 8 | 0:57 | 0.533 |
| 9 | 0:00 | 0.664 |
| 10 | 20:27 | 0.560 |

## Observations

- At 275 chunks the corpus is tiny, so the top 10 is dominated by chunks of
  the one relevant lecture; vector-only and hybrid agree at rank 1 for all
  queries. Hybrid's effect shows up mid-list, exactly where it should:
  - Q3 ("LSM tree compaction write amplification"): hybrid at fts_weight 0.2
    pulls WAL chunks (`CedEy54pe3g`) up to rank 3 — a sensible lexical match
    on "write amplification" — while staying anchored on the log-structured
    lecture at ranks 1–2.
  - Q9 ("distributed SQL replication and consensus"): higher fts weights
    bring MVCC chunks (`tUFha9-DuSk`, consensus-adjacent vocabulary) into
    ranks 2–5, but the YugabyteDB clip holds rank 1 in all modes.
- Q6 is the only query whose top 10 is a single lecture in every mode
  (cardinality estimation vocabulary is distinctive in this sample).
- Nothing here distinguishes vector-only from hybrid at rank 1; the choice of
  `fts_weight` matters more on the full corpus where each topic has many
  competing chunks. Keep hybrid (default 0.5 in the notebook) and re-run this
  check — including deferred queries 4, 5, 7 — after the full load.

## Full-corpus follow-ups

- Re-run all 10 queries (plus any added since) after the 595-lecture load;
  queries 4, 5, 7 become meaningful then.
- Expect the single-lecture dominance to disappear; watch for rank-1 flips
  between vector-only and hybrid, and for near-duplicate chunks across
  re-recorded lectures crowding the top 10 (may need per-video
  diversification).
- Add an IVF_PQ vector index at full-corpus scale (see
  `docs/lectures-sample.md` — sequential scan stops being acceptable well
  before 595 lectures' worth of chunks).
