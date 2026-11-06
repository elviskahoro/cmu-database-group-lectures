# Fireworks vs Local Nomic Parity Gate

Verified 2026-11-02 by `scripts/parity_gate.py` (cdgl-zqm). Compares hosted
Fireworks embeddings against local sentence-transformers embeddings for
`nomic-ai/nomic-embed-text-v1.5` (768-dim) and gates on cosine similarity
>= 0.99 per text. Result: **PASS**.

## Setup

- **Hosted**: Fireworks inference API
  (`https://api.fireworks.ai/inference/v1/embeddings`), model
  `nomic-ai/nomic-embed-text-v1.5` (accepted as-is; the older
  `accounts/fireworks/models/nomic-embed-text-v1p5` id returns "not
  available"). API key read from `FIREWORKS_API_KEY` in `.env.local`
  (never printed). Total spend ~$0.001 (22 texts x 2 requests).
- **Local**: `sentence-transformers` on CPU, same model id,
  `trust_remote_code=True` — the same configuration as
  `lecture_search/embed.py`.
- **Corpus**: the 14 chunks from the 2 sample transcripts in
  `data/sample/` (via `lecture_search.process_sample_transcripts`) plus 8
  synthetic database-jargon sentences (hash joins, MVCC, WAL, LSM, etc.) —
  22 texts total.

## How Fireworks treats the `search_document:` prefix

**Fireworks does NOT apply the Nomic task prefix server-side.** The
request has no `prompt_template` field that injects it; whatever string
you send is embedded verbatim. The prefix must be prepended client-side,
exactly once.

Evidence (mean cosine over all 22 texts):

| comparison | mean cosine |
| --- | --- |
| Fireworks(raw) vs local(raw) | 0.999985 |
| Fireworks(prefixed) vs local(prefixed) | 0.999974 |
| Fireworks(raw) vs local(prefixed) | 0.926138 |

Raw-to-raw agrees to ~5 decimal places while raw-to-prefixed collapses to
0.93, so Fireworks is embedding exactly what it is sent. This matches the
existing convention in `lecture_search/embed.py`, which prepends
`search_document:` / `search_query:` manually — that code is compatible
with swapping the local embedder for the Fireworks endpoint as long as the
caller keeps prepending the prefix. Double-prefixing (prefixing before
calling an endpoint that also prefixes) is not a risk here because
Fireworks never prefixes; the failure mode to avoid is *omitting* the
prefix when moving to Fireworks.

## Gate result

Like-for-like comparison: Fireworks(prefixed) vs local(prefixed), the
baseline used by the lecture-search pipeline.

- texts: 22
- **min cosine: 0.999954**
- **mean cosine: 0.999974**
- threshold: 0.99 per text
- **verdict: PASS** (22/22 texts above threshold)

The residual ~2e-5 gap is numeric noise: Fireworks returns bf16-rounded
components (visible in the response payloads) while the local model
produces fp32.

Per-text results are in `data/parity_gate.json` (regenerate with
`uv run python scripts/parity_gate.py`; exits non-zero if any text falls
below the threshold).
