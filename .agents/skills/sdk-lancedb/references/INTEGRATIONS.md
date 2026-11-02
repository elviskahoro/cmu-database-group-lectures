# Integrations

LanceDB is designed to slot under existing libraries. The most common backends to plug it into:

## Pandas / PyArrow

This is the most common interop — every `QueryBuilder` materializes to one of these natively.

```python
df = tbl.search(q).limit(10).to_pandas()
arrow_tbl = tbl.search(q).limit(10).to_arrow()

# Insert from a DataFrame
tbl.add(pd.DataFrame({"text": chunks, "source": sources}))

# Bulk-embed once with a callable, no registry
from lancedb.embeddings import with_embeddings
df = with_embeddings(my_embed_fn, df, column="text", batch_size=128)
tbl = db.create_table("docs", data=df)
```

## DuckDB

Convert a LanceDB table to a Lance dataset, then query with DuckDB. Useful for SQL aggregations and joins on top of vector search.

```python
import duckdb
ds = tbl.to_lance()                         # pyarrow-compatible dataset
duckdb.sql("SELECT source, COUNT(*) FROM ds GROUP BY source ORDER BY 2 DESC").to_df()

# Query metadata + embedding-derived stats together
duckdb.sql("""
    SELECT source, AVG(image_nsfw) AS avg_nsfw
    FROM ds
    WHERE image_nsfw >= 2
    GROUP BY source
""").to_df()
```

## LangChain

`langchain_community.vectorstores.LanceDB` wraps a `Table` so any LangChain retriever / chain can use it.

```python
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

embeddings = OpenAIEmbeddings()
db = lancedb.connect("./data/lancedb")

# Build the LangChain vector store from documents
docsearch = LanceDB.from_documents(documents, embeddings, connection=db)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
)
qa.run("What are the major features?")
```

To wire a reranker through LangChain, pass it to `LanceDB.from_documents`:

```python
from lancedb.rerankers import LinearCombinationReranker

reranker = LinearCombinationReranker(weight=0.3)         # 0.3 = mostly vector
docsearch = LanceDB.from_documents(documents, embeddings, reranker=reranker)
results = docsearch.similarity_search_with_relevance_scores("question text")
```

To delete by source after re-chunking the same document:
```python
small_chunk_table.delete('doc_id = "old"')
```

## LlamaIndex

`llama_index.vector_stores.lancedb.LanceDBVectorStore` — works for both local and Cloud.

```python
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

vector_store = LanceDBVectorStore(uri="./lancedb", table_name="docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Filter at the LanceDB level (raw where clause)
retriever = index.as_retriever(
    vector_store_kwargs={"where": "metadata.file_name = 'paul_graham_essay.txt'"}
)
retriever.retrieve("What did the author do growing up?")
```

LlamaIndex's `MetadataFilters` abstraction also pushes down to LanceDB:

```python
from llama_index.core.vector_stores import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="creation_date", operator=FilterOperator.EQ, value="2025-01-15"),
        MetadataFilter(key="file_size",     operator=FilterOperator.GT, value=75_040),
    ],
    condition=FilterCondition.AND,
)
query_engine = index.as_query_engine(filters=filters)
query_engine.query("How much did Viaweb charge per month?")
```

## LanceDB Cloud

Same Python package, just a different URI and an API key.

```python
import lancedb
db = lancedb.connect("db://my-project", api_key="sk_...", region="us-east-1")
tbl = db.create_table("docs", data=df)
hits = tbl.search("query").limit(10).to_pandas()
```

The hosted offering supports the same APIs: rerankers, hybrid search, FTS, scalar indexes. It does *not* support every embedded-only feature (e.g. `tbl.to_lance()` for a hosted table).

For LlamaIndex against Cloud:
```python
LanceDBVectorStore(uri="db://project-slug", api_key="sk_...", region="us-east-1")
```

## Object storage (S3 / GCS / Azure)

Embedded mode supports object-store URIs for the database path. Everything else (table names, schemas, search) works the same.

```python
db = lancedb.connect("s3://my-bucket/lancedb/")
db = lancedb.connect("gs://my-bucket/lancedb/")
db = lancedb.connect("az://my-container/lancedb/")

# Async variant for long-running pipelines
import lancedb
db = await lancedb.connect_async("az://my-container/lancedb/")
```

Credentials use the standard AWS / GCP / Azure environment variables. For private buckets, the storage layer is read at every query — there's no local cache, so put the LanceDB process in the same region.

## dlt (data load tool)

`dlt` has a LanceDB destination — declare a pipeline, point it at a LanceDB URI, and dlt handles batched writes plus schema evolution.

```python
import dlt
from dlt.destinations import lancedb as lancedb_dest

pipeline = dlt.pipeline(
    pipeline_name="docs",
    destination=lancedb_dest(uri="./data/lancedb"),
    dataset_name="raw",
)
pipeline.run([{"text": "...", "source": "..."}], table_name="docs")
```

This pairs well with the [dlt-pydantic-pii-validation](../../dlt-pydantic-pii-validation/SKILL.md) skill in this repo when ingestion needs PII gating.

## Async API

For FastAPI / async pipelines, use `connect_async` and `AsyncTable`:

```python
import lancedb

db = await lancedb.connect_async("./data/lancedb")
tbl = await db.open_table("docs")
async for batch in (await tbl.search(q).limit(100)).to_batches():
    process(batch)
```

The async client is feature-equivalent for the common path (search, add, create_index). Use it when you're already in an event loop; otherwise prefer the sync client for simplicity.

## Polars

```python
import polars as pl
df = pl.from_arrow(tbl.search(q).limit(10).to_arrow())
```

Or iterate row-by-row through `to_lance()` and let Polars stream the dataset:

```python
ds = tbl.to_lance()
df = pl.scan_pyarrow_dataset(ds).filter(pl.col("source") == "a.pdf").collect()
```

## Modal

LanceDB plays nicely with Modal because everything is in-process — no separate server to host. Two patterns:

```python
import modal
volume = modal.Volume.from_name("lancedb-data", create_if_missing=True)

@app.function(image=image, volumes={"/data": volume})
def query(q: str):
    import lancedb
    db = lancedb.connect("/data/lancedb")
    return db.open_table("docs").search(q).limit(5).to_list()
```

For higher-throughput serving, point Modal at S3 instead of a Volume so multiple containers share a single canonical store:
```python
db = lancedb.connect("s3://my-bucket/lancedb/")
```
Pair with `@modal.concurrent(max_inputs=...)` so each container handles many queries against its open `Table` handle.
