# Examples

End-to-end recipes lifted (and trimmed) from `tmp/lancedb-examples/`. Copy, adapt, run.

## 1. Basic RAG: registry-backed table, vector search

Minimal RAG store. Embeds with sentence-transformers, persists to disk, searches by string.

```python
import lancedb
import pandas as pd
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

embed = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

class Doc(LanceModel):
    text: str = embed.SourceField()
    vector: Vector(embed.ndims()) = embed.VectorField()
    source: str

db = lancedb.connect("./data/lancedb")
tbl = db.create_table("docs", schema=Doc, mode="overwrite")
tbl.add(pd.DataFrame({
    "text": ["Paris is the capital of France.",
             "Berlin is the capital of Germany.",
             "The Eiffel Tower is in Paris."],
    "source": ["wiki", "wiki", "wiki"],
}))

hits = tbl.search("Where is the Eiffel Tower?").limit(3).to_pandas()
print(hits[["text", "_distance"]])
```

## 2. Hybrid search with a Cohere reranker

Dense + BM25 with Cohere on top — the canonical hybrid recipe. Takes one FTS index and one `.rerank()` call.

```python
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CohereReranker

embed = get_registry().get("openai").create(name="text-embedding-3-small")

class Doc(LanceModel):
    text: str = embed.SourceField()
    vector: Vector(embed.ndims()) = embed.VectorField()
    source: str

db = lancedb.connect("./data/lancedb")
tbl = db.create_table("docs", schema=Doc, mode="overwrite")
tbl.add([{"text": "...", "source": "a.pdf"}, ...])

tbl.create_fts_index("text", replace=True)        # one-time

reranker = CohereReranker(model_name="rerank-english-v3.0")    # set COHERE_API_KEY

hits = (tbl.search("contextual retrieval", query_type="hybrid")
            .rerank(reranker=reranker)
            .limit(10)
            .to_pandas())
```

## 3. Multimodal CLIP search (text → image)

Build a single-vector image table and query it with text. The registry's `open-clip` provider handles both modalities through one `EmbeddingFunction`.

```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

clip = get_registry().get("open-clip").create()

class Image(LanceModel):
    image_uri: str = clip.SourceField()
    vector: Vector(clip.ndims()) = clip.VectorField()
    label: str

tbl = db.create_table("images", schema=Image, mode="overwrite")
tbl.add([
    {"image_uri": "/data/cat.jpg", "label": "cat"},
    {"image_uri": "/data/dog.jpg", "label": "dog"},
    {"image_uri": "/data/eiffel.jpg", "label": "landmark"},
])

# Text query into the same vector space
tbl.search("a cute kitten").limit(3).to_pandas()
# Image query into the same column
tbl.search("/data/query.jpg").limit(3).to_pandas()
```

## 4. Custom embedder via `@register`

A keyword-frequency "embedder" that's deterministic and demonstrates the registry contract. Useful for fixtures, tests, or domain-specific scoring.

```python
import math
from lancedb.embeddings import TextEmbeddingFunction, register, get_registry
from lancedb.pydantic import LanceModel, Vector

KEYWORDS = ("nvidia", "rag", "lancedb", "vector", "hybrid", "reranker")

@register("demo-keyword")
class KeywordCounter(TextEmbeddingFunction):
    keywords: tuple[str, ...] = KEYWORDS

    def ndims(self): return len(self.keywords)

    def generate_embeddings(self, texts):
        out = []
        for t in texts:
            counts = [float(t.lower().count(k)) for k in self.keywords]
            norm = math.sqrt(sum(c * c for c in counts)) or 1.0
            out.append([c / norm for c in counts])
        return out

embed = get_registry().get("demo-keyword").create()

class Chunk(LanceModel):
    text: str = embed.SourceField()
    vector: Vector(embed.ndims()) = embed.VectorField()

tbl = db.create_table("kw", schema=Chunk, mode="overwrite")
tbl.add([{"text": "RAG with LanceDB and a reranker"},
         {"text": "Hybrid search using vectors"}])
print(tbl.search("rag lancedb").limit(2).to_pandas())
```

## 5. Time-travel A/B test of two embedders

Index the same corpus with two embedding models and compare retrieval. LanceDB versioning makes the rollback trivial.

```python
import lancedb
db = lancedb.connect("./data/lancedb")

# Production embedder
prod_embed = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

class ProdDoc(LanceModel):
    text: str = prod_embed.SourceField()
    vector: Vector(prod_embed.ndims()) = prod_embed.VectorField()

prod = db.create_table("docs_prod", schema=ProdDoc, mode="overwrite")
prod.add(corpus_df)
prod_v = prod.version

# Experimental embedder (different model, same data)
exp_embed = get_registry().get("openai").create(name="text-embedding-3-small")

class ExpDoc(LanceModel):
    text: str = exp_embed.SourceField()
    vector: Vector(exp_embed.ndims()) = exp_embed.VectorField()

exp = db.create_table("docs_exp", schema=ExpDoc, mode="overwrite")
exp.add(corpus_df)

# Compare top-1 on the same query
query = "what is contextual retrieval?"
print("prod:", prod.search(query).limit(1).to_list()[0]["text"])
print("exp :", exp.search(query).limit(1).to_list()[0]["text"])

# Roll back if exp is worse
exp.restore(version=1)
```

For the same-table version-bump variant: `tbl.list_versions()`, `tbl.checkout(n)` to read past versions, `tbl.restore(n)` to make one current.

## 6. Upsert with `merge_insert`

Idempotent ingest that respects an `id` primary key. Faster and safer than delete-then-insert.

```python
(tbl.merge_insert("id")
    .when_matched_update_all()       # row with same id → overwrite
    .when_not_matched_insert_all()   # new id → insert
    .execute(updates_df))
```

`when_matched_update_all` and `when_not_matched_insert_all` are the most common combination. Other variants:
- `.when_matched_update_columns(["text", "vector"])` — partial update
- `.when_not_matched_by_source_delete()` — drop rows that disappeared from the source (full-sync semantics)

## 7. Filtered search with a scalar index

Tenant-scoped retrieval, optimized.

```python
tbl.create_scalar_index("tenant_id")     # one-time

hits = (tbl.search(query)
            .where("tenant_id = 'acme'", prefilter=True)
            .nprobes(20).refine_factor(10)
            .limit(10)
            .to_pandas())
```

Always `prefilter=True` plus a scalar index when the filter is selective. Without the index, prefilter scans the column.

## 8. Vector arithmetic

LanceDB doesn't have a special API for analogies — use vector math directly on numpy arrays, then `tbl.search(result_vec)`.

```python
import numpy as np

def vec(text):
    return np.array(tbl.search(text).limit(1).to_list()[0]["vector"])

# "king − man + woman"
target = vec("king") - vec("man") + vec("woman")
print(tbl.search(target).limit(5).to_pandas()[["text", "_distance"]])
```

The trick is that you go through the registry-embedded `vector` column to fetch each component vector, do arithmetic in numpy, then query. Works for any embedder that produces a normalized space.

## 9. Hybrid + cross-encoder for max quality

When you have a small candidate set and quality matters more than throughput.

```python
from lancedb.rerankers import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

hits = (tbl.search(query, query_type="hybrid")
            .rerank(reranker=reranker)
            .limit(10)
            .to_pandas())
```

Rule of thumb: candidate fan-out 50–200 candidates, then cross-encoder rerank to top 5–10. `MiniLM-L-12-v2` is the quality default; `TinyBERT-L-6` is ~5× faster with modest quality loss.

## 10. End-to-end RAG with LangChain

Tying it all together.

```python
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([raw_text])

# 2. Store
embeddings = OpenAIEmbeddings()
db = lancedb.connect("./data/lancedb")
docsearch = LanceDB.from_documents(docs, embeddings, connection=db)

# 3. Retrieve + answer
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 5}),
)
print(qa.run("What's the main point?"))
```

To bolt on a reranker, swap `LanceDB.from_documents(...)` for the `reranker=` overload (see [INTEGRATIONS.md](INTEGRATIONS.md)).
