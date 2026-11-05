# Search API (Modal)

Public, read-only hybrid search over the lecture corpus, served by the Modal
app **`cmu-lecture-search`** (`lecture_search/search_api.py`, cdgl-bcy,
reworked in cdgl-99e).

**Why a server at all:** LanceDB Cloud has no read-only API keys — a key is
an all-or-nothing shared secret (see `docs/lancedb-findings.md`, §3). The
public notebook therefore never touches LanceDB; this endpoint holds the key
as a Modal secret and answers search requests.

**Embedding is client-side:** the caller (the molab notebook) embeds the
query itself — Nomic v1.5, `search_query: ` prefix, on a molab GPU if
available else CPU — and sends the vector in the request. The Modal image
carries only `lancedb` + `fastapi` (no sentence-transformers / transformers
/ einops / torch, no baked model weights), so the image is small, cold
starts are fast, and no container time is billed for embedding.

## Endpoint

Base URL (workspace `marcella`):

```text
https://marcella--cmu-lecture-search-searchservice-web.us-east.modal.run
```

### `POST /search`

Generate the query vector client-side, then POST it with the raw query text
(the text drives the FTS half of the hybrid search):

```bash
# Embed the query locally (Nomic v1.5, 'search_query: ' prefix applied by
# embed_query). Requires the dev venv: uv sync --group dev
VECTOR=$(uv run python -c '
import json
from lecture_search.embed import embed_query
print(json.dumps(embed_query("how does dbt transform data in the warehouse")))
')

curl -sS -X POST \
  https://marcella--cmu-lecture-search-searchservice-web.us-east.modal.run/search \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg q 'how does dbt transform data in the warehouse' \
        --argjson v "$VECTOR" \
        '{query: $q, vector: $v, top_k: 5, fts_weight: 0.5}')"
```

Body schema:

```json
{
  "query": "how does dbt transform data in the warehouse",
  "vector": [0.0123, -0.0456, "..."],
  "top_k": 5,
  "fts_weight": 0.5,
  "date_from": "2025-01-01",   // optional, ISO date
  "date_to": "2025-12-31",     // optional, ISO date
  "video_ids": ["QnGBxn_XP5U"] // optional, <= 50 YouTube IDs
}
```

- `query`: non-empty string, ≤ 300 chars — used verbatim for the FTS half.
- `vector`: **exactly 768 finite floats** (Nomic v1.5 output of
  `embed_query(query)`); NaN/inf, wrong length, and non-numeric components
  are rejected.

Returns `{"results": [...]}` where each row is exactly:

```text
video_id, title, date, start_seconds, end_seconds, deep_link, text, score
```

No vectors, no internal columns, no keys, no stack traces. The search runs
as `.vector(vector).text(query)` reranked with
`MRRReranker(weight_vector=1-fts_weight, weight_fts=fts_weight)` — the same
construction as `notebooks/search.py`. `fts_weight=0` is pure vector,
`1` is pure FTS.

### `GET /lectures`

Returns `{"lectures": [{video_id, title, date}]}`, one entry per lecture,
sorted by date descending. Cached per container for 1 hour.

## Security posture (public, unauthenticated, read-only)

Hard validation in `parse_search_request` (all violations → HTTP 422):

- request body: ≤ **20 KB** (`check_body_size` runs before JSON parsing; a
  legitimate body is dominated by the 768-float vector at ~15 KB)
- `query`: non-empty string, ≤ 300 chars
- `vector`: list of exactly 768 finite numbers (no NaN/inf/strings/bools)
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
`scaledown_window=300s`, CPU-only (1 vCPU / 1 GiB).

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

The image installs only `lancedb==0.40.0` and `fastapi` — no embedding
model, no torch, no Hugging Face download at build time.

## Cost model

Modal bills per **container-second** of CPU and memory actually used (plus a
small per-request fee); there is no idle cost once containers scale down.
With `scaledown_window=300s`, a container lingers at most 5 minutes after its
last request; `max_containers=5` caps worst-case concurrency. At 1 vCPU /
1 GiB with no model to load, a traffic pattern of a few searches per minute
costs cents per day — see https://modal.com/pricing for current rates.

## Measured latency (2026-11-05, client-side embedding rework, `lectures_sample`,
275 chunks / 5 lectures)

- Image build at deploy: **~10 s** (was minutes with baked Nomic weights)
- Cold start (first request after deploy, includes container boot + LanceDB
  connect, no model load): **~6.7 s**
- Warm `POST /search`: **~0.57 s**
- Warm `GET /lectures` (cached): **~0.42 s**

Pre-rework numbers (server-side embedding, baked Nomic weights, 2026-11-02):
~32 s cold, ~1.3 s warm search.

## Verification

- `uv run pytest tests/test_search_api.py -q` — offline guards: validation
  rejections (wrong-length / NaN / string vectors, oversized bodies,
  injection-shaped IDs, oversized inputs), WHERE-clause construction from
  validated values only, vector passed through verbatim, response whitelist
  (no vectors/internal columns).
- `uv run tach check && uv run tach check-external` — module boundary +
  dependency declarations (`modal`/`fastapi` are excluded as deploy-only,
  image-baked deps).
- Live `curl` checks against the deployed endpoint: valid search (200),
  `/lectures` (200), wrong-length vector / NaN vector / >20 KB body / long
  query / `top_k=1000` / `"x' OR 1=1"` video ID / out-of-range `fts_weight`
  / non-ISO date / missing query (all 422), and a leak grep for
  `vector|api_key|LANCEDB|db://|sk_` over responses (clean).
