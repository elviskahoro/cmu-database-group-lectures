# Search API (Modal)

Public, read-only hybrid search over the lecture corpus, served by the Modal
app **`cmu-lecture-search`** (`lecture_search/search_api.py`, cdgl-bcy).

**Why a server at all:** LanceDB Cloud has no read-only API keys — a key is
an all-or-nothing shared secret (see `docs/lancedb-findings.md`, §3). The
public notebook therefore never touches LanceDB; this endpoint holds the key
as a Modal secret and answers search requests.

## Endpoint

Base URL (workspace `marcella`):

```text
https://marcella--cmu-lecture-search-searchservice-web.us-east.modal.run
```

### `POST /search`

```json
{
  "query": "how does dbt transform data in the warehouse",
  "top_k": 5,
  "fts_weight": 0.5,
  "date_from": "2025-01-01",   // optional, ISO date
  "date_to": "2025-12-31",     // optional, ISO date
  "video_ids": ["QnGBxn_XP5U"] // optional, <= 50 YouTube IDs
}
```

Returns `{"results": [...]}` where each row is exactly:

```text
video_id, title, date, start_seconds, end_seconds, deep_link, text, score
```

No vectors, no internal columns, no keys, no stack traces. The query is
embedded server-side (Nomic v1.5, `search_query:` prefix applied exactly once) and run as hybrid search: `.vector(qvec).text(query)` reranked with
`MRRReranker(weight_vector=1-fts_weight, weight_fts=fts_weight)` — the same
construction as `notebooks/search.py`. `fts_weight=0` is pure vector,
`1` is pure FTS.

### `GET /lectures`

Returns `{"lectures": [{video_id, title, date}]}`, one entry per lecture,
sorted by date descending. Cached per container for 1 hour.

## Security posture (public, unauthenticated, read-only)

Hard validation in `parse_search_request` (all violations → HTTP 422):

- `query`: non-empty string, ≤ 300 chars
- `top_k`: integer 1–50
- `fts_weight`: number 0–1
- `video_ids`: ≤ 50 items, each `^[A-Za-z0-9_-]{11}$`
- `date_from` / `date_to`: parsed as ISO dates

WHERE clauses are built **only** from values that survived that validation
(dates render via `date.isoformat()`, video IDs are regex-locked), so raw
user input is never interpolated into SQL. Error responses are generic JSON;
internal exceptions never leave the container. CORS is allow-all (read-only
data, no credentials). There are no write paths.

Container limits: `max_containers=5`, `@modal.concurrent(max_inputs=8)`,
`scaledown_window=300s`, CPU-only (1 vCPU / 2 GiB).

## Deploy

The LanceDB config lives in Modal secret **`lancedb-search`**
(`LANCEDB_URI`, `LANCEDB_API_KEY`, `LANCEDB_REGION`, `LANCEDB_TABLE`). It was
created from `.env.local` without printing values:

```bash
set -a; source .env.local; set +a
MODAL_PROFILE=marcella modal secret create lancedb-search \
  LANCEDB_URI=db://cmu-database-group-eqdwgi \
  LANCEDB_REGION=us-east-1 \
  LANCEDB_TABLE=lectures_sample \
  LANCEDB_API_KEY="$LANCEDB_API_KEY"
```

To switch to the full corpus, update the secret's `LANCEDB_TABLE` to
`lectures` (`modal secret create ...` creates a new version; redeploy to pick
it up).

Deploy:

```bash
MODAL_PROFILE=marcella modal deploy lecture_search/search_api.py
```

The image pins the same deps as `notebooks/search.py` (lancedb 0.40.0,
sentence-transformers 5.7.0, transformers 4.57.6, einops 0.8.2) and bakes the
Nomic v1.5 weights in at build time (`snapshot_download` via `run_commands`),
so cold starts do not hit the Hugging Face hub.

## Cost model

Modal bills per **container-second** of CPU and memory actually used (plus a
small per-request fee); there is no idle cost once containers scale down.
With `scaledown_window=300s`, a container lingers at most 5 minutes after its
last request; `max_containers=5` caps worst-case concurrency. At 1 vCPU /
2 GiB, a traffic pattern of a few searches per minute costs cents per day —
see https://modal.com/pricing for current rates.

## Measured latency (2026-11-02, `lectures_sample`, 275 chunks / 5 lectures)

- Cold start (first request after scale-to-zero, includes container boot +
  model load + LanceDB connect): **~32 s**
- Warm `POST /search`: **~1.3 s**
- Warm `GET /lectures` (cached): **~0.5 s**

## Verification

- `uv run pytest tests/test_search_api.py -q` — offline guards: validation
  rejections (injection-shaped IDs, oversized inputs), WHERE-clause
  construction from validated values only, prefix applied exactly once,
  response whitelist (no vectors/internal columns).
- `uv run tach check && uv run tach check-external` — module boundary +
  dependency declarations (`modal`/`fastapi`/`huggingface_hub` are excluded
  as deploy-only, image-baked deps).
- Live `curl` checks against the deployed endpoint: valid search (200),
  `/lectures` (200), long query / `top_k=1000` / `"x' OR 1=1"` video ID /
  out-of-range `fts_weight` / non-ISO date / missing query (all 422), and a
  leak grep for `vector|api_key|LANCEDB|db://|sk_` over responses (clean).
