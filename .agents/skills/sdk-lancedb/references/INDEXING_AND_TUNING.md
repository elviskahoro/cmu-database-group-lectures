# Indexing and Tuning

LanceDB does sequential scan by default. That is fast (millisecond range) up to roughly 50k rows, after which you want an ANN index. There are three index types you'll touch:

- **Vector index (IVF_PQ)** — accelerates `search(vector)`.
- **FTS index** — required for `query_type="fts"` and `query_type="hybrid"`.
- **Scalar index** — accelerates `where` predicates on a single column.

## When to build a vector index

- Up to ~50k rows: skip it. The scan is fast and recall is exact.
- 50k–10M rows: build IVF_PQ. Tune `nprobes` and `refine_factor` for recall.
- > 10M rows: same — IVF_PQ scales further by raising `num_partitions` (rule of thumb ≈ √(num_rows)).

## Building an IVF_PQ index

```python
import math

n = table.count_rows()

table.create_index(
    metric="cosine",                                 # "l2" | "cosine" | "dot"
    num_partitions=int(math.sqrt(n)),                # IVF coarse quantizer cells
    num_sub_vectors=16,                              # PQ M; must divide vector dim
    index_type="IVF_PQ",                             # default; explicit here for clarity
    accelerator="cuda",                              # optional, requires CUDA-enabled lance
    replace=True,
)
```

### Parameter cheat sheet

| Parameter | Default / heuristic | Effect |
|-----------|---------------------|--------|
| `metric` | `"l2"` | Distance metric. Must match the embedding model — most text encoders are normalized, use `"cosine"` |
| `num_partitions` | ≈ √(num_rows) | More partitions → finer-grained → higher recall at high `nprobes`, slower build |
| `num_sub_vectors` | 16 (or `dim/8`) | PQ chunks. Must divide the vector dim. Higher = more accurate but slower scan |
| `index_type` | `"IVF_PQ"` | Also `"IVF_HNSW_PQ"` and `"IVF_HNSW_SQ"` on newer versions |
| `accelerator` | None | `"cuda"` enables GPU index build |
| `replace` | `False` | Set `True` to rebuild |

### Multivector indexes

For ColBERT/ColPali tables (one row stores a list of vectors), only `metric="cosine"` is supported. The recommended `num_partitions` is the square root of the *total* number of embeddings, not number of rows:

```python
# from examples/multivector_xtr/main.ipynb
n_embeddings_per_doc = 1030          # measured on the model
num_rows = table.count_rows()

table.create_index(
    metric="cosine",                                  # mandatory for multivector
    num_partitions=int(math.sqrt(num_rows * n_embeddings_per_doc)),
    num_sub_vectors=32,
    index_type="IVF_PQ",
)
```

## Tuning recall at query time

Don't rebuild the index to chase recall — change `nprobes` and `refine_factor` per query.

```python
hits = (table.search(q)
              .nprobes(20)             # default 1; raise for recall
              .refine_factor(10)       # default 1; raise for recall
              .limit(10)
              .to_list())
```

- **`nprobes`** scans more IVF partitions. Cost grows roughly linearly. Typical sweet spots: 10 (fast), 20 (default for quality), 50 (high recall).
- **`refine_factor`** asks for `limit * refine_factor` candidates from the lossy PQ codes, then re-ranks them with full-precision vectors. Typical: 5–10. Costs negligible disk IO; mostly arithmetic.

If recall is still low after `nprobes=50, refine_factor=10`, the issue is upstream: wrong metric, low-quality embeddings, or too few partitions for the data size.

## FTS indexes

```python
table.create_fts_index("text", replace=True)               # one column
table.create_fts_index(["title", "body"], replace=True)    # multiple
```

There are two backends: native (default in newer versions, supports incremental updates) and tantivy (requires `pip install tantivy`). The native backend supports prefilter pushdown for `where` clauses on scalar-indexed columns; tantivy does not.

For high-write workloads, the native FTS picks up new rows incrementally — no manual reindex needed. Tantivy needs a full rebuild on update.

## Scalar indexes

Scalar indexes accelerate `where` clauses on a single column — especially useful when combined with `prefilter=True` on a vector or hybrid query. Without a scalar index, prefilter does a column scan.

```python
table.create_scalar_index("source")            # exact-match / range
table.create_scalar_index("created_at")
```

Build scalar indexes on:
- Tenant / partition keys (`tenant_id`, `org_id`)
- Frequent equality filters (`category`, `status`)
- Range filters (`created_at`, `price`)

Scalar indexes don't help free-text queries — that's what FTS is for.

## Maintenance

LanceDB stores writes as small append-only fragments. After heavy writes / merges, occasional compaction keeps reads fast:

```python
table.compact_files()                  # merge small fragments into larger ones
table.cleanup_old_versions()           # drop versions beyond the retention window
```

These are I/O-heavy — schedule off-peak. Both are safe to call on the same table that's serving reads.

## Choosing a metric

| Metric | Use when |
|--------|----------|
| `cosine` | Most text/image encoders. They normalize internally; cosine ignores magnitude. |
| `l2` | Embeddings are not normalized and magnitude is meaningful (rare). |
| `dot` | Inner-product retrieval (e.g. dual encoders trained on dot-product loss). On normalized vectors this is equivalent to cosine. |

Mismatching metric and embedding training is a silent quality killer. Read the model card before guessing.

## Verifying index quality

A pragmatic sanity check on a held-out query set:

```python
import numpy as np

def recall_at_k(true_ids, predicted_ids, k=10):
    return len(set(true_ids[:k]) & set(predicted_ids[:k])) / k

baseline = table.search(q).limit(100).to_pandas()["id"].tolist()       # exact
indexed  = table.search(q).nprobes(10).refine_factor(5).limit(100).to_pandas()["id"].tolist()
print(recall_at_k(baseline, indexed, k=10))
```

Run this for ~50 queries; iterate `nprobes` / `refine_factor` until `recall@10` is acceptable. The point of tuning is to find the knee of the latency/recall curve — usually around `nprobes=20, refine_factor=10` for IVF_PQ on a few million rows.
