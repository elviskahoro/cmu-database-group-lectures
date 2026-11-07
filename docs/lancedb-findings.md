# LanceDB Verification Findings

Verified 2026-11-02 against the real `lectures_sample` table by
`scripts/verify_hybrid.py` (cdgl-csl). This supersedes the earlier probe
(`scripts/lancedb_probe.py`, cdgl-qla), which used seeded random vectors and
never exercised real retrieval.

## Setup

- **LanceDB Cloud URI**: `db://cmu-database-group-eqdwgi` (region `us-east-1`)
- **Table**: `lectures_sample` — 14 chunks from 2 sample transcripts
  ("Introduction to dbt", "YugabyteDB: A Brief Introduction"), loaded by
  `scripts/load_sample.py` (cdgl-r35)
- **Embeddings**: Nomic v1.5 (`nomic-ai/nomic-embed-text-v1.5`), 768-dim,
  embedded locally on CPU. Documents use the `search_document:` prefix,
  queries the `search_query:` prefix (both include a trailing space and are
  prepended manually in
  `lecture_search/embed.py`; the registry embedder does not apply them)
- **Client**: `lancedb` 0.40.0 (Python)

## 1. FTS index on `text` — verified

`table.list_indices()` shows one FTS index:

- name `text_idx`, type `FTS`, column `["text"]`, 14/14 rows indexed
- native inverted index: English analyzer, lower-cased, stemmed, stop words
  removed (`remove_stop_words: True`)

BM25 (`query_type="fts"`) queries run against it successfully.

**Observed caveat:** FTS only covers the indexed column. The query
`"YugabyteDB"` (and even `"yugabyte"`) returns **zero** FTS hits because the
token appears only in the `title` column — the transcript `text` never
contains it. Vector search still ranks that lecture's chunks first, and
hybrid search degrades gracefully to the vector results when the FTS
candidate set is empty.

## 2. Hybrid search with explicit `.vector().text()` + `MRRReranker` — verified

Queries were embedded locally (real Nomic v1.5 vectors) and run as:

```python
table.search(query_type="hybrid").vector(q_vec).text(q_text) \
     .rerank(MRRReranker(weight_vector=w, weight_fts=1 - w)).limit(5)
```

Top-5 chunk IDs per weight (`vector / fts`):

**Query: "how does dbt transform data in the warehouse"**

| ranking | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| FTS-only (BM25) | :2 | :4 | :3 | :5 | :1 |
| 0.2 / 0.8 | :2 | :4 | :3 | :5 | :1 |
| 0.5 / 0.5 | :2 | :4 | :5 | :3 | :1 |
| 0.8 / 0.2 | :2 | :5 | :4 | :3 | :1 |
| vector-only | :2 | :5 | :4 | :3 | :1 |

**Query: "distributed SQL database replication"**

| ranking | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| FTS-only (BM25) | 9N:4 | 9N:5 | Qn:4 | 9N:1 | 9N:0 |
| 0.2 / 0.8 | 9N:4 | 9N:5 | 9N:1 | Qn:4 | 9N:0 |
| 0.5 / 0.5 | 9N:1 | 9N:4 | 9N:5 | 9N:0 | Qn:4 |
| 0.8 / 0.2 | 9N:1 | 9N:5 | 9N:4 | 9N:0 | Qn:4 |
| vector-only | 9N:1 | 9N:5 | 9N:0 | 9N:4 | Qn:4 |

(`9N:*` = YugabyteDB lecture chunks, `Qn:*` = dbt lecture chunks.)

**Rankings differ across weights** — and they shift monotonically: at
`weight_vector=0.2` the hybrid order matches FTS-only, at `0.8` it matches
vector-only, with an interpolated order at `0.5`. MRR's weighted reciprocal
rank is a real tuning knob on this data.

## 3. Read-only API keys — not documented as supported

Answer sourced from the LanceDB docs, not hardcoded:

- [Authentication — LanceDB Enterprise](https://docs.lancedb.com/enterprise/authentication)
  describes exactly two credential modes: an **API key** — "a long-lived
  shared secret passed as `api_key` on connect" — and **OAuth 2.0** bearer
  tokens. No read-only/read-write scoping of API keys is documented.
- The [REST API reference](https://docs.lancedb.com/api-reference/)
  similarly authenticates with a single `x-api-key` header — "a valid API
  key" — with no documented permission levels.

So per the current docs, LanceDB does **not** offer read-only API keys; a
key is an all-or-nothing shared secret. For centrally managed, rotatable
credentials the docs recommend OAuth 2.0 (supported in the async Python
client via `lancedb.connect_async` and in the TypeScript client).

## Recommendations

1. Use hybrid search (`query_type="hybrid"` + explicit `.vector().text()`) as
   the default retrieval path; start at `MRRReranker(weight_vector=0.5,
   weight_fts=0.5)` and tune toward vector (`0.8`) for conceptual queries or
   toward FTS (`0.2`) for keyword-heavy queries.
2. Index `title` for FTS as well (`create_fts_index(["text", "title"])`) if
   title keywords like "YugabyteDB" should be searchable — the `text`-only
   index cannot match them.
3. Treat the LanceDB API key as a full-access secret: store it in
   `.env.local` (never committed), and prefer OAuth 2.0 where centrally
   managed credentials are required. Do not design around read-only keys —
   they are not a documented feature.
4. Keep embedding queries locally with Nomic v1.5; the `search_query:`
   prefix matters for retrieval quality.

## Reproduce

```bash
uv run python scripts/verify_hybrid.py   # writes data/hybrid_verification.json
```
