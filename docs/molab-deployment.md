# molab deployment (cdgl-elw)

`notebooks/search.py` is deployed to [molab](https://molab.marimo.io) as a
GitHub-mirrored notebook (GitHub `main` is the source of truth; no build or
publish step is required — molab mirrors the public repo directly).

**App URL (share this one, not a notebook-editor URL):**

```text
https://molab.marimo.io/github/elviskahoro/cmu-database-group-lectures/blob/main/notebooks/search.py?include-code=false
```

`include-code=false` hides all code in the read-only view (viewers can still
reveal it), which is the app-like presentation
([molab query parameters](https://docs.marimo.io/guides/molab/#query-parameters)).
The plain URL without the query parameter shows the notebook with code.

The README carries an "Open in molab" badge pointing at the mirror URL.

## Verified 2026-11-06 (all checks run unauthenticated, from a clean shell)

### Mirror is live

`GET` on the molab URL above returns 200 and the server-rendered page embeds
the full current notebook source (verified byte-identical to
`notebooks/search.py` on `origin/main`, commit `fc8832b`). No login needed to
view; viewers can run it on an ephemeral server or fork it
([docs](https://docs.marimo.io/guides/molab/#mirror-notebooks-from-github)).

Static previews render **with outputs** because the session snapshot
`notebooks/__marimo__/session/search.py.json` is committed (regenerated with
`marimo export session notebooks/search.py --sandbox`; the old blanket
`notebooks/__marimo__/` gitignore rule was narrowed for this).

### Secrets check (logged-out view)

The notebook needs **no secrets in molab** — the LanceDB key lives in the
Modal secret `lancedb-search`, and the notebook only talks to the public
Modal search API (see `docs/search-api.md`). Verified:

- `notebooks/search.py`: no `LANCEDB_*` usage, no keys; the only URL is the
  public Modal endpoint. (`git grep` across all history: only
  env-var *lookups* in superseded commits — no key value was ever committed.)
- molab page HTML (fetched with no cookies, i.e. the logged-out view): no
  keys, only the public API URL inside the visible source cells.
- `marimo export html` static export: 0 hits for `LANCEDB_API_KEY`,
  `api_key`, `Bearer`, `db://`; the 4 "secret" hits are the source comment
  "Not a secret.", a lecture title, and a marimo frontend JS asset name.
- Committed session snapshot (`search.py.json`): 0 hits for the same
  patterns; outputs contain only lecture titles/snippets.

(The `RuntimeError: dictionary changed size during iteration` printed by
`marimo export html` is a known benign teardown race in marimo 0.25.2; the
export completed and the HTML is intact.)

### Can molab reach modal.run? (outbound HTTPS)

Yes, with this evidence:

- molab containers must have outbound HTTPS to function at all for this
  notebook: the built-in package manager installs missing PEP 723
  dependencies from PyPI on import, and the Nomic model weights
  (~0.5 GB) download from Hugging Face on first run. Both are documented
  molab features ([package management](https://docs.marimo.io/guides/molab/#package-management),
  [remote storage to S3/GDrive/HF](https://docs.marimo.io/guides/molab/#storage)).
- The Modal endpoint is plain public HTTPS with allow-all CORS and no
  auth, so nothing on the Modal side can block molab's egress IPs.
- End-to-end proof from this machine: `marimo export session --sandbox`
  (isolated env, exactly the PEP 723 deps molab installs) ran the notebook
  against the live Modal API and produced real search outputs.

The one thing not directly observed is a packet leaving molab's own NAT —
that needs an interactive molab session (Clerk login, not available to
agents). Given the above, the risk is negligible.

### Viewers' GPU access

- A shared app/notebook runs on the **owner's** compute config; viewers
  cannot attach a GPU to someone else's notebook. GPU attach (RTX Pro 6000
  Blackwell, 96 GB) is a per-notebook toggle in the app header, available to
  the owner and to anyone who **forks** into their own workspace
  ([docs FAQ](https://docs.marimo.io/guides/molab/#faq)).
- Not needed anyway: the notebook embeds queries client-side with
  `torch.cuda.is_available()` fallback to CPU
  (`notebooks/search.py`, `get_embedder`). Nomic v1.5 embeds a single query
  on CPU in milliseconds, and all heavy lifting (LanceDB hybrid search) is
  server-side on Modal.

### Concurrency / memory test (simulating several tabs)

Load test against the live Modal endpoint (`POST /search`, valid 768-dim
vector, `top_k=5`, `fts_weight=0.5`, 5.6 KB body):

| Round | Concurrent | Result | Latency (min / p50 / max) | Wall |
| --- | --- | --- | --- | --- |
| a | 8 | 8× HTTP 200, all 5 results | 1.55 s / 3.85 s / 4.01 s | 4.18 s |
| b | 24 | 24× HTTP 200, all 5 results | 0.33 s / 1.05 s / 1.61 s | 1.80 s |

Round (a) paid one extra container's spin-up (~4 s tail); round (b) ran
fully warm. No 5xx, no OOM, no throttling. Several browser tabs each issue
at most one in-flight search, so ~10 concurrent is a realistic ceiling for
organic traffic — well inside capacity
(`max_containers=5` × `@modal.concurrent(max_inputs=8)` = 40 concurrent
inputs per `docs/search-api.md`). Each container is capped at 1 vCPU / 1 GiB,
so memory pressure is bounded regardless of tab count.

### Cold start after idle

Endpoint left idle > 5.5 min (`scaledown_window=300s`), then measured:

| Request | Total | TTFB |
| --- | --- | --- |
| `GET /lectures` (cold container) | 4.57 s | 4.43 s |
| `GET /lectures` (warm) | 0.41 s | 0.27 s |
| `POST /search` (warm) | 0.42 s | 0.42 s |

(~5 s cold start = container boot + LanceDB connect; a second, independent
idle window reproduced it at 4.94 s. The notebook's UI already warns that
first load can take ~10 s, and `REQUEST_TIMEOUT_SECONDS = 120` gives ample
headroom.)

### Abuse / cost review of the Modal endpoint under concurrent tabs

The endpoint is public and unauthenticated by design (read-only data, CORS
allow-all). Cost is hard-capped by the container config:

- Modal list prices (2026-11, `modal.com/pricing`): CPU \$0.0000131/s/core,
  memory \$0.00000222/s/GiB → one container (1 vCPU + 1 GiB) ≈
  **\$0.055/hour**.
- `max_containers=5` bounds the whole app at **≈ \$0.28/hour ≈ \$199/month
  even if saturated 24/7** by an attacker. Realistic organic traffic is
  orders of magnitude below that.
- One warm search (~1 s) costs ~\$0.000015 of compute; an isolated search
  that idles a container through its 300 s scaledown window costs
  ~\$0.005. A full page load (`GET /lectures` + one search) is sub-cent.
- Request validation bounds per-request work: body ≤ 20 KB, exactly 768
  finite floats, `top_k` ≤ 50, query ≤ 300 chars, `video_ids` ≤ 50
  (regex-locked). There are no write paths and no way to make a request
  materially more expensive than a bounded scan of the 33,412-row table.
- Residual risks (acceptable): LanceDB Cloud per-query costs scale with
  traffic (tiny scans), and a sustained flood could degrade latency for
  real users once all 5 containers are busy. Mitigations if ever needed:
  Modal proxy auth token, per-IP rate limiting, or Cloudflare in front.

## Remaining manual step (requires molab login)

Everything above works without an account. The one thing that needs a human
in the molab UI (Clerk auth; no agent credentials exist in this repo's
environment): open the app URL, start the notebook on its server, and use
**Share → "Run as app"** if a persistent owner-hosted app session is wanted
(e.g. to pre-attach a GPU — unnecessary here — or to keep a warm kernel).
The GitHub-mirror URL above is otherwise the canonical share link; pushing
`main` updates it automatically.
