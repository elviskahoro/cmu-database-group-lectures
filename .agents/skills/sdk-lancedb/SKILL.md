---
name: sdk-lancedb
description: Use when building, querying, or operating LanceDB — an embedded/serverless vector database for Python and TypeScript. Covers connections, Pydantic schemas, the embedding registry, vector/FTS/hybrid search, rerankers (Cohere, ColBERT, CrossEncoder, RRF, MRR, LinearCombination), IVF_PQ index tuning, multimodal (CLIP / OpenCLIP / Jina), versioning/time-travel, merge_insert upserts, and integrations (LangChain, LlamaIndex, Pandas/Arrow, LanceDB Cloud). Trigger on imports of `lancedb`, `lancedb.pydantic`, `lancedb.embeddings`, or `lancedb.rerankers`; on phrases like "vector database", "RAG store", "hybrid search", "rerank with Cohere/ColBERT", "FTS index", or "store embeddings"; and on requests to set up RAG / semantic search / multimodal retrieval.
license: Apache-2.0
compatibility: Requires Python 3.9+ and `lancedb` (pure-Python wheel — no separate server). For embedded use install just `lancedb`. For LanceDB Cloud use the same package plus an API key. Optional extras `lancedb[embeddings]`, `lancedb[fts]` (FTS native), `tantivy` (alternative FTS). Examples reference `pylance`, `pyarrow`, `pandas`, `pydantic`, `sentence-transformers`, `transformers`, `open_clip_torch`, `cohere`, `FlagEmbedding` depending on the recipe.
metadata:
  author: elviskahoro
  version: "1.0"
  tags: [lancedb, vector-database, rag, embeddings, hybrid-search, reranking, multimodal, semantic-search]
---

# LanceDB

LanceDB is an embedded vector database that runs in-process (no server to host), persists tables as columnar Lance files on local disk or object storage, and exposes one fluent Python API for vector search, full-text search (FTS / BM25), hybrid search, and SQL-style filters. The same connection works against a local directory, S3/GCS/Azure URIs, or LanceDB Cloud.

**Core mental model:** A `Database` is a directory. Each `Table` is a folder inside it backed by Lance files. You define a schema (often via a Pydantic `LanceModel` whose `Vector` field is wired to an embedding function in the registry), `add()` rows, and call `table.search(...)` to get back a `QueryBuilder` you compose with `.where()`, `.limit()`, `.rerank()`, `.select()`, then materialize with `.to_pandas()` / `.to_list()` / `.to_pydantic()`.

## When to Use

- Building a RAG pipeline that needs a persistent vector store with no separate server
- Combining dense vector search with BM25 full-text search (hybrid) and a reranker
- Storing multimodal data (CLIP / OpenCLIP / Jina) with multiple vector columns per row
- Migrating from Pinecone/Weaviate/Chroma to a file-backed, embedded store
- Running vector search inside a notebook, CLI, Modal job, or serverless function
- Time-travel queries / A-B testing embedding models against the same data
- Upserting via `merge_insert` instead of delete-then-insert
- Driving a vector store from LangChain or LlamaIndex with a uniform backend
- Reading/writing Lance datasets and querying them with DuckDB

## Don't Use For

- Managed multi-tenant search clusters with per-tenant access control — LanceDB Cloud handles this; embedded mode does not
- Transactional OLTP workloads — Lance is columnar, optimized for analytical scans
- Sub-millisecond replicated writes across regions — use a managed service

## Execution Steps for Agents

### Step 1: Install and connect

```python
# uv add lancedb pyarrow pandas pydantic
import lancedb

db = lancedb.connect("./data/lancedb")          # local directory
# db = lancedb.connect("s3://bucket/lancedb/")  # object storage
# db = lancedb.connect("az://container/lancedb/")
# db = lancedb.connect("db://my-project", api_key="sk_...", region="us-east-1")  # Cloud

# Async variant — only when you need it (async iteration, FastAPI, etc.)
# db = await lancedb.connect_async("./data/lancedb")
```

A connection is cheap; you can recreate it per request. Don't pre-warm anything.

### Step 2: Define schema with the embedding registry

Prefer the registry pattern — it stores the embedding function with the table, so `add()` and `search()` accept raw text and embed it for you. No bookkeeping.

```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

embed = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

class Doc(LanceModel):
    text: str = embed.SourceField()                # field that gets embedded
    vector: Vector(embed.ndims()) = embed.VectorField()  # output vector column
    source: str                                    # plain metadata column

table = db.create_table("docs", schema=Doc, mode="overwrite")
```

Available registry providers used in the example repo: `openai`, `sentence-transformers`, `colbert`, `cohere`, `instructor`, `open-clip`, plus any custom function registered with `@register("name")`. See [references/EMBEDDINGS_AND_SCHEMA.md](references/EMBEDDINGS_AND_SCHEMA.md).

When you can't (or shouldn't) use the registry — e.g. precomputed vectors from another service — pass an explicit PyArrow schema or a list of dicts that already contains `vector`:

```python
table = db.create_table(
    "docs",
    data=[{"vector": embed_query("Hello"), "text": "Hello", "id": "1"}],
    mode="overwrite",
)
```

### Step 3: Insert / upsert data

```python
# Add — registry-backed table embeds `text` automatically
table.add([{"text": "first chunk", "source": "a.pdf"},
           {"text": "second chunk", "source": "b.pdf"}])

# Add from pandas
import pandas as pd
table.add(pd.DataFrame({"text": chunks, "source": sources}))

# Upsert with merge_insert (preferred over delete-then-insert)
(table.merge_insert("id")
      .when_matched_update_all()
      .when_not_matched_insert_all()
      .execute(updates_df))
```

`mode="overwrite"` on `create_table` is the easiest way to wipe a dev table; in prod use `db.drop_table("name")` or `merge_insert`.

### Step 4: Search

```python
# Vector search — pass text (registry embeds) or a precomputed vector
hits = table.search("what is contextual retrieval?").limit(5).to_pandas()

# With metadata filter (prefilter is usually what you want)
hits = (table.search(query)
             .where("source = 'a.pdf'", prefilter=True)
             .limit(5)
             .to_list())

# Specific columns, ANN tuning
hits = (table.search(query)
             .select(["text", "source"])
             .nprobes(20)            # search more IVF partitions → recall ↑, latency ↑
             .refine_factor(10)      # rerank top-k*10 with exact distance → recall ↑
             .limit(10)
             .to_pandas())
```

`.search()` returns a `QueryBuilder`; chain composably and materialize last. Output options: `.to_list()`, `.to_pandas()`, `.to_arrow()`, `.to_pydantic(Model)`.

### Step 5: Add full-text search and go hybrid

```python
table.create_fts_index("text", replace=True)        # one-time, indexes the SourceField

# FTS-only (BM25)
table.search("contextual retrieval", query_type="fts").limit(10).to_pandas()

# Hybrid: dense + BM25 fused with a reranker
from lancedb.rerankers import RRFReranker
hits = (table.search("contextual retrieval", query_type="hybrid")
             .rerank(reranker=RRFReranker())
             .limit(10)
             .to_pandas())
```

For hybrid with precomputed vectors, pass a tuple `(vector, text)`:
```python
table.search((q_vec, q_text), query_type="hybrid").limit(10)
```

See [references/SEARCH_AND_RERANKING.md](references/SEARCH_AND_RERANKING.md) for all reranker types (`RRFReranker`, `LinearCombinationReranker`, `CohereReranker`, `ColbertReranker`, `CrossEncoderReranker`, `MRRReranker`, `JinaReranker`, `FlagEmbeddingReranker`) and when to pick each.

### Step 6: Build a vector index when the table grows

Sequential scan is fine up to ~50k rows. Past that, build IVF_PQ:

```python
table.create_index(
    metric="cosine",          # "l2" | "cosine" | "dot"
    num_partitions=256,       # ≈ sqrt(num_rows); more partitions = finer-grained, slower build
    num_sub_vectors=16,       # PQ M parameter; must divide vector dim
    index_type="IVF_PQ",
)
```

Tune recall vs latency at query time with `.nprobes()` and `.refine_factor()` — never rebuild the index for that. For multivector tables (ColBERT, ColPali), only `metric="cosine"` is supported. See [references/INDEXING_AND_TUNING.md](references/INDEXING_AND_TUNING.md).

### Step 7: Open existing tables and update them

```python
# Idempotent open
table = db.open_table("docs") if "docs" in db.table_names() else db.create_table("docs", schema=Doc)

# Delete by SQL predicate (no vector required)
table.delete("source = 'old.pdf'")

# Update by SQL predicate
table.update(where="source = 'a.pdf'", values={"source": "archive/a.pdf"})

# Versioning / time travel
table.list_versions()
table.checkout(3)             # read an older version
table.restore(version=3)      # make it current
```

`db.table_names()`, `db.drop_table(name)`, and `tbl.count_rows()` are the other operations you'll reach for.

### Step 8: Materialize how you'll consume it

| Output | Method |
|--------|--------|
| List of dicts | `.to_list()` |
| Pandas DataFrame | `.to_pandas()` (alias `.to_df()`) |
| PyArrow Table | `.to_arrow()` |
| Pydantic models (table must be schema'd with `LanceModel`) | `.to_pydantic(Doc)` |
| Lance dataset (for DuckDB / Polars) | `tbl.to_lance()` |

For DuckDB SQL over a LanceDB table:
```python
import duckdb
ds = table.to_lance()
duckdb.sql("SELECT source, COUNT(*) FROM ds GROUP BY source").to_df()
```

## Quick Reference

### Imports

| Import | Purpose |
|--------|---------|
| `import lancedb` | Top-level package; `lancedb.connect`, `lancedb.connect_async` |
| `from lancedb.pydantic import LanceModel, Vector` | Schema definition |
| `from lancedb.pydantic import pydantic_to_schema, vector` | Lower-level helpers |
| `from lancedb.embeddings import get_registry` | Registry-driven embedding fns |
| `from lancedb.embeddings import register, TextEmbeddingFunction` | Custom embedders |
| `from lancedb.embeddings import EmbeddingFunctionRegistry` | Direct registry handle |
| `from lancedb.embeddings import with_embeddings` | Embed a list/df once (no registry) |
| `from lancedb.embeddings import InstructorEmbeddingFunction` | Instructor models |
| `from lancedb.rerankers import RRFReranker, MRRReranker` | Score-fusion rerankers |
| `from lancedb.rerankers import LinearCombinationReranker` | Weighted mix of dense + FTS |
| `from lancedb.rerankers import CohereReranker, ColbertReranker, CrossEncoderReranker` | Model-based rerankers |
| `from lancedb.context import contextualize` | Window-based context expansion |

### Operations cheatsheet

| Operation | Call |
|-----------|------|
| Connect (local) | `lancedb.connect("./data/lancedb")` |
| Connect (cloud) | `lancedb.connect("db://slug", api_key=..., region=...)` |
| Connect (object store) | `lancedb.connect("s3://b/path")` |
| List tables | `db.table_names()` |
| Create table | `db.create_table(name, schema=Doc, mode="overwrite")` |
| Open table | `db.open_table(name)` |
| Drop table | `db.drop_table(name)` |
| Add rows | `tbl.add(data)` |
| Upsert | `tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(df)` |
| Delete rows | `tbl.delete("source = 'x.pdf'")` |
| Update rows | `tbl.update(where=..., values=...)` |
| Count | `tbl.count_rows()` |
| Vector index | `tbl.create_index(metric="cosine", num_partitions=256, num_sub_vectors=16)` |
| FTS index | `tbl.create_fts_index("text", replace=True)` |
| Scalar index | `tbl.create_scalar_index("source")` |
| Vector search | `tbl.search(query).limit(k).to_pandas()` |
| FTS search | `tbl.search(query, query_type="fts").limit(k)` |
| Hybrid search | `tbl.search(query, query_type="hybrid").rerank(reranker=RRFReranker()).limit(k)` |
| Filter | `.where("expr", prefilter=True)` |
| Project | `.select(["a", "b"])` |
| Tune ANN | `.nprobes(20).refine_factor(10)` |
| Multi-vector column | `tbl.search(vec, vector_column_name="image_embedding")` |
| Versions | `tbl.list_versions()`, `tbl.checkout(n)`, `tbl.restore(n)` |
| To Lance dataset | `tbl.to_lance()` |

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Creating the registry embedder inside the row-add loop | Re-instantiates the model per row, leaks GPU/CPU | Build once at module top; `embed = get_registry().get(...).create(...)` |
| Calling `tbl.search(text)` without an FTS index in `query_type="fts"` mode | `LanceError: no FTS index` | Run `tbl.create_fts_index("text", replace=True)` once before FTS / hybrid queries |
| Using `query_type="hybrid"` without a reranker | Returns a fused but unsorted list — quality is poor | Always chain `.rerank(reranker=RRFReranker())` (or another reranker) |
| `where(expr, prefilter=False)` (the default in some versions) on a small filtered set | Postfilter throws away most of the top-k → empty results | Pass `prefilter=True` whenever you have a selective predicate |
| Building a vector index on a tiny table | Slower than scan; index quality is poor | Skip until ~50k rows, or rely on `refine_factor` for accuracy |
| Setting `num_sub_vectors` that doesn't divide the vector dim | `create_index` errors out | Use a divisor of the dim (e.g. 16 or 32 for a 768-dim model) |
| Mixing precomputed vectors and registry-embedded text in one column | Schema conflict / dimension mismatch | Pick one. If you have precomputed vectors, drop the `SourceField` and pass `vector` directly |
| Calling `.to_pydantic()` on a table created without a `LanceModel` schema | Returns dicts, not models | Pass `schema=MyModel` to `create_table` so the type is preserved |
| Forgetting `mode="overwrite"` in dev | `RuntimeError: Table already exists` on rerun | Use `mode="overwrite"` while iterating; switch to `open_table` in prod |
| `tbl.delete()` to clear a table (no predicate) | `delete()` requires a SQL predicate | Use `db.drop_table(name)` then recreate, or `tbl.delete("true")` |
| Searching with `vector_column_name` left at default in a multi-vector schema | Hits the wrong column → poor results | Always pass `vector_column_name="image_embedding"` (or `"text_embedding"`) on multimodal tables |
| Loading the OpenAI embedder from the registry without `OPENAI_API_KEY` | Silent fallback or 401 at first `add()` | Set the env var (or use Doppler) before `get_registry().get("openai").create()` |
| Storing image bytes in a `str` column | `pyarrow` will fail to cast | Use a `bytes` field (`image: bytes` in `LanceModel`) or store the path |
| Using `tbl.checkout(n)` then writing | Writes are blocked on a checkout | Call `tbl.restore(n)` to make the version current, or open a fresh handle |
| Treating LanceDB like a queue (high write rate, small batches) | Excess small Lance files; fragmentation hurts read latency | Batch adds (≥1k rows), then `tbl.compact_files()` / `tbl.cleanup_old_versions()` periodically |

## References

- [Embeddings and Schema](references/EMBEDDINGS_AND_SCHEMA.md) — registry providers, `LanceModel`, custom `@register` embedders, multi-vector schemas
- [Search and Reranking](references/SEARCH_AND_RERANKING.md) — vector / FTS / hybrid query patterns, every reranker class with code
- [Indexing and Tuning](references/INDEXING_AND_TUNING.md) — IVF_PQ parameters, scalar indexes, `nprobes` / `refine_factor`, recall benchmarks
- [Multimodal](references/MULTIMODAL.md) — CLIP / OpenCLIP / Jina, multiple vector columns, image storage strategies
- [Integrations](references/INTEGRATIONS.md) — LangChain `LanceDB`, LlamaIndex `LanceDBVectorStore`, DuckDB, dlt, Pandas/Arrow interop, LanceDB Cloud
- [Examples](references/EXAMPLES.md) — end-to-end recipes (basic RAG, hybrid + Cohere reranker, multimodal CLIP, time-travel A/B test, custom embedder)
