# lectures_sample — local Nomic v1.5 embeddings in LanceDB Cloud

`scripts/load_sample.py` chunks the two pilot transcripts (via
`lecture_search.chunking`), embeds them **locally on CPU** with Nomic
`nomic-ai/nomic-embed-text-v1.5` (768d, sentence-transformers), and loads them
into LanceDB Cloud (`db://cmu-database-group-eqdwgi`, region `us-east-1`) as
table `lectures_sample`.

## Schema

Defined as `LectureChunk` in `lecture_search/embed.py`:

| column | type |
|---|---|
| `id` | utf8 (`video_id:chunk_index`) |
| `video_id` | utf8 |
| `date` | utf8 |
| `title` | utf8 |
| `chunk_index` | int64 |
| `start_seconds` | double |
| `end_seconds` | double |
| `text` | utf8 (FTS-indexed) |
| `deep_link` | utf8 (`https://youtu.be/<video_id>?t=<start>`) |
| `vector` | fixed_size_list float32[768] |

## Conventions

- **Prefixes are manual.** The LanceDB sentence-transformers registry
  embedder does not apply Nomic's task prefixes, so `embed_documents` prepends
  `search_document: ` and `embed_query` prepends `search_query: `. Any query
  path (notebook, web) must call `embed_query` — never embed a raw query
  string.
- **One embedder per process.** The registry embedder is instantiated once at
  module top in `lecture_search/embed.py`; the underlying model is lazy-loaded
  and cached. Do not construct it inside loops.
- **FTS index on `text`** is created with `replace=True` after load
  (`create_fts_index`; lancedb 0.40 deprecates it in favor of
  `create_index(config=FTS())` — both work, kept per project convention).
- **transformers is pinned to 4.x**: Nomic v1.5's trust_remote_code modeling
  file uses `get_extended_attention_mask`, removed in transformers 5.x.
  `einops` is required by the same remote code.
- **No vector index** at sample size (~hundreds of rows): LanceDB's
  sequential scan is exact and fast below ~50k rows. Add IVF_PQ for the full
  corpus.
- The API key is read from `LANCEDB_API_KEY` in `.env.local`; it is never
  echoed.

## Run

```bash
uv sync --group dev
uv run python -m scripts.load_sample
```

## Deferred: Fireworks parity gate

The spec's Fireworks hosted-embedding parity check (compare local Nomic v1.5
vectors against the Fireworks-hosted model before the full run) is **deferred
to the full corpus load** — this sample establishes the schema, chunking, and
load path with local embeddings only. No Modal/Fireworks spend is incurred by
this script.
