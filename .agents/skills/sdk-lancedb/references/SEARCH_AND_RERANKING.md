# Search and Reranking

Every read in LanceDB starts with `tbl.search(...)`, which returns a `QueryBuilder`. Compose modifiers (`.where`, `.limit`, `.select`, `.rerank`, `.nprobes`, `.refine_factor`), then materialize. The same builder handles vector, FTS, and hybrid queries — `query_type` switches the mode.

## The QueryBuilder

```python
qb = tbl.search(query, query_type="vector", vector_column_name="vector")
qb = qb.where("source = 'a.pdf'", prefilter=True)
qb = qb.select(["text", "source"])
qb = qb.limit(10)
qb = qb.nprobes(20).refine_factor(10)        # ANN tuning
qb = qb.rerank(reranker=RRFReranker())       # only meaningful for hybrid / when you set query_string
hits = qb.to_pandas()                         # or .to_list() / .to_arrow() / .to_pydantic(Doc)
```

The chain is lazy. Nothing runs until you call a `to_*` method.

## Vector search

```python
# Pass text → registry-backed table embeds it
hits = tbl.search("contextual retrieval").limit(5).to_pandas()

# Pass a precomputed vector
hits = tbl.search(my_vec).limit(5).to_pandas()
```

Results carry `_distance` (vector mode). Lower is better for `l2`/`cosine`; higher for `dot`. A `_score` column appears in FTS / hybrid results.

### Filtering: prefilter vs postfilter

```python
hits = tbl.search(q).where("source = 'a.pdf'", prefilter=True).limit(5)
```

- `prefilter=True` (recommended for selective predicates) — apply the SQL filter first, then run vector search on the remaining rows. Recall is preserved even when the filter is selective.
- `prefilter=False` (the historical default) — vector search first, then drop hits that fail the filter. Cheap when the filter is broad; can return < `limit` rows when the filter is narrow.

Use `prefilter=True` whenever you have a selective metadata filter — date ranges, tenant scoping, source/category. The performance difference disappears once you `create_scalar_index(column)` on the filtered field.

### ANN tuning at query time

```python
hits = tbl.search(q).nprobes(20).refine_factor(10).limit(10).to_list()
```

- `nprobes` — how many IVF partitions to scan. Default 1; higher means more recall, more latency. 10–20 is a typical starting point on a 256-partition index.
- `refine_factor` — fetch `limit * refine_factor` candidates with the lossy PQ index, then rerank them with full-precision vectors. Default 1 (no refinement); 5–10 buys substantial recall at low cost.

These are the two knobs that matter most for ANN quality. Don't rebuild the index for these — change them at query time.

### Multi-vector tables

```python
tbl.search(q_vec, vector_column_name="text_embedding").limit(5)
tbl.search(q_img, vector_column_name="image_embedding").limit(5)
```

Always pass `vector_column_name` explicitly when the schema has more than one vector column.

## Full-text search (FTS / BM25)

LanceDB ships two FTS backends:

- Native (default in newer versions) — built into the Lance file format, supports incremental updates, scalar prefilter pushdown.
- Tantivy (legacy) — Rust BM25 implementation; requires `pip install tantivy`. Slightly different scoring.

```python
# One-time index — pass the SourceField column name
tbl.create_fts_index("text", replace=True)

# FTS-only query
hits = tbl.search("contextual retrieval", query_type="fts").limit(10).to_pandas()
```

Full-text supports BM25 keyword queries, phrase queries (`"exact phrase"`), boolean operators (`+required -excluded`), and wildcards. Add `replace=True` so reruns don't error.

You can index multiple columns: `tbl.create_fts_index(["text", "title"])`. Specify columns at query time with `.fts_columns(["title"])` or by configuring the index.

## Hybrid search

Hybrid runs both engines, fuses the candidate sets, and lets a reranker order the final list. Always pair with a reranker — the raw fused list is order-arbitrary.

```python
from lancedb.rerankers import RRFReranker

hits = (tbl.search("contextual retrieval", query_type="hybrid")
           .rerank(reranker=RRFReranker())
           .limit(10)
           .to_pandas())
```

Pass a tuple `(vector, text)` when you've precomputed the embedding:
```python
tbl.search((q_vec, "contextual retrieval"), query_type="hybrid").rerank(...).limit(10)
```

Hybrid results contain both `_distance` (vector) and `_score` (FTS) plus a `_relevance_score` injected by the reranker.

## Rerankers

All rerankers live in `lancedb.rerankers`. They take a candidate set (or two — vector and FTS — for hybrid) and produce a final ordering. Wrap any reranker in `.rerank(reranker=R)`.

### Score-fusion rerankers (no model)

These mix existing scores arithmetically — fast, no extra model calls.

```python
from lancedb.rerankers import RRFReranker, MRRReranker, LinearCombinationReranker

RRFReranker()                                                        # reciprocal rank fusion (default for hybrid)
MRRReranker(weight_vector=0.65, weight_fts=0.35)                     # weighted reciprocal-rank
LinearCombinationReranker(weight=0.7)                                # 0 = pure vector, 1 = pure FTS
```

`RRFReranker` is the safe default — bounded, model-free, robust to score scale differences. `LinearCombinationReranker` is intuitive when you want a knob to tune.

### Cross-encoder reranker

Run a sequence-classification cross-encoder over (query, doc) pairs. High quality, much slower than vector/FTS scoring — overfetch and rerank a small top-k.

```python
from lancedb.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-6")
# 'cross-encoder/ms-marco-MiniLM-L-12-v2' is a stronger / slower alternative

hits = (tbl.search(q, query_type="hybrid")
           .rerank(reranker=reranker)
           .limit(10))
```

### Cohere reranker

Hosted reranking via Cohere's API. Strong out-of-the-box quality on multilingual workloads.

```python
from lancedb.rerankers import CohereReranker

reranker = CohereReranker()                                       # default model
# reranker = CohereReranker(model_name="rerank-english-v3.0")
# reranker = CohereReranker(model_name="rerank-multilingual-v3.0")

hits = (tbl.search((q_vec, q_text), query_type="hybrid")
           .rerank(reranker=reranker)
           .limit(10))
```

Set `COHERE_API_KEY` (or pass `api_key=...`).

### ColBERT reranker

Late-interaction reranker — embeds the query and each candidate as a list of token vectors, scores by max-over-tokens. Local model, no API key.

```python
from lancedb.rerankers import ColbertReranker
reranker = ColbertReranker()
hits = tbl.search(q, query_type="hybrid").rerank(reranker=reranker).limit(10)
```

### Other rerankers in the wild

- `JinaReranker(api_key=...)` — Jina's hosted reranker.
- `OpenaiReranker()` — uses an OpenAI chat model as a judge (slow, expensive, but flexible).
- `FlagEmbeddingReranker(model="BAAI/bge-reranker-base", use_fp16=True)` — popular HuggingFace reranker; the example repo wires it in via `FlagEmbedding.FlagReranker` then sorts by score.

### Custom rerankers

Subclass `lancedb.rerankers.Reranker`. The example repo defines `MofidifiedLinearReranker(LinearCombinationReranker)` to add a hard-filter step before the linear combo:

```python
class TaggedLinear(LinearCombinationReranker):
    def __init__(self, weight: float, filters: list[str]):
        super().__init__(weight=weight)
        self.filters = filters

    def rerank_hybrid(self, query, vector_results, fts_results):
        merged = super().rerank_hybrid(query, vector_results, fts_results)
        keep = merged["text"].apply(lambda t: any(tag in t for tag in self.filters))
        return merged[keep]
```

### Reranker decision table

| Need | Use |
|------|-----|
| Default hybrid that "just works" | `RRFReranker()` |
| Tunable dense ↔ sparse mix | `LinearCombinationReranker(weight=0.7)` |
| Highest local quality | `CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")` |
| Highest hosted quality, English | `CohereReranker(model_name="rerank-english-v3.0")` |
| Multilingual or low-resource | `CohereReranker(model_name="rerank-multilingual-v3.0")` |
| Local late-interaction | `ColbertReranker()` |
| LLM-as-judge | `OpenaiReranker()` |
| Custom business rules | subclass `Reranker` (or `LinearCombinationReranker`) |

### Vector search + text reranking (without going hybrid)

When you only have a vector but want a text-aware reranker to do the final ordering:

```python
hits = (tbl.search(q_vec)
           .rerank(reranker=CohereReranker(), query_string=q_text)
           .limit(top_k * 2)        # overfetch — reranker keeps the best `limit`
           .to_pandas())
```

Pass `query_string` so the reranker has the natural-language query (vector search alone doesn't give it that).

## Output formats

| Method | Returns | When to use |
|--------|---------|-------------|
| `.to_list()` | `list[dict]` | Plain Python; small result sets |
| `.to_pandas()` / `.to_df()` | `pd.DataFrame` | Default in notebooks; use `.iloc`/`.loc` for downstream filtering |
| `.to_arrow()` | `pa.Table` | Hand off to DuckDB / Polars / Parquet |
| `.to_pydantic(Doc)` | `list[Doc]` | Type-safe pipelines (table must have a `LanceModel` schema) |

## Ranking signals attached to results

| Column | When present | Meaning |
|--------|--------------|---------|
| `_distance` | vector / hybrid | similarity distance (lower = closer for L2/cosine) |
| `_score` | FTS / hybrid | BM25 score (higher = more relevant) |
| `_relevance_score` | when `.rerank(...)` is used | reranker's final score (higher = better) |
| `_rowid` | optional, on every result | physical row id; useful for time-travel diffs |

When sorting in app code, prefer `_relevance_score` if you reranked, else `_score` for FTS, else negate `_distance`.
