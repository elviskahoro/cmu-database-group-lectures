# Embeddings and Schema

How to declare a LanceDB schema, wire it to an embedding function, and customize embedding behavior. The registry is the recommended path because it stores the embedding metadata with the table — anyone reopening the table later can `add()` and `search()` raw text without knowing which model produced the vectors.

## LanceModel basics

`LanceModel` is a thin Pydantic `BaseModel` subclass that maps fields to a Lance/Arrow column. `Vector(dim)` is a typed alias for a `list[float]` of fixed length.

```python
from lancedb.pydantic import LanceModel, Vector

class Doc(LanceModel):
    text: str
    vector: Vector(384)
    source: str
    page: int
```

Pass it as `schema=Doc` to `create_table`. Round-trip back into Pydantic models with `tbl.to_pydantic(Doc)`.

For tables with precomputed vectors, this is the whole picture — embed yourself before `add()`. For tables you want LanceDB to embed for you, use the registry.

## The embedding registry

`get_registry().get(NAME).create(**kwargs)` returns an `EmbeddingFunction`. Use it as `SourceField()` (the input column) and `VectorField()` (the output column) in a `LanceModel`.

```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

embed = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

class Doc(LanceModel):
    text: str = embed.SourceField()                     # gets embedded
    vector: Vector(embed.ndims()) = embed.VectorField() # produced from text

table = db.create_table("docs", schema=Doc, mode="overwrite")
table.add([{"text": "first chunk"}])    # vector populated automatically
table.search("query string").limit(5)   # query also embedded automatically
```

The function (provider name + kwargs) is serialized into the table's metadata. Reopening the table in a fresh process — even a different language — uses the same embedder.

### Providers used in the example repo

| Registry key | Class | Typical kwargs | Notes |
|--------------|-------|----------------|-------|
| `openai` | `OpenAIEmbeddings` | `name="text-embedding-3-small"` | needs `OPENAI_API_KEY` |
| `sentence-transformers` | `SentenceTransformerEmbeddings` | `name="all-MiniLM-L6-v2"`, `device="cuda"` | local; default in many examples |
| `cohere` | `CohereEmbeddings` | `name="embed-english-v3.0"` | needs `COHERE_API_KEY` |
| `colbert` | `ColbertEmbeddings` | `name="colbert-ir/colbertv2.0"` | multi-vector — produces a list of vectors per doc |
| `instructor` | `InstructorEmbeddingFunction` | `name="hkunlp/instructor-base"` | task-instructed |
| `open-clip` | `OpenClipEmbeddings` | `name="ViT-B-32"`, `pretrained="openai"` | text + image embedded into a shared space |
| `jina` | `JinaEmbeddings` | `name="jina-embeddings-v2-base-en"` | API or local |
| `bedrock-text` | `BedRockText` | model-specific kwargs | AWS Bedrock |

`registry.list()` returns the full list installed in your environment.

### Concrete examples

```python
# OpenAI
openai = get_registry().get("openai").create(name="text-embedding-ada-002")

class Schema(LanceModel):
    text: str = openai.SourceField()
    vector: Vector(1536) = openai.VectorField()
```

```python
# Cohere multilingual
cohere = get_registry().get("cohere").create(name="embed-multilingual-v3.0")

class Schema(LanceModel):
    text: str = cohere.SourceField()
    vector: Vector(cohere.ndims()) = cohere.VectorField()
```

```python
# OpenCLIP — text and image into the same vector space
clip = get_registry().get("open-clip").create()

class ImageDoc(LanceModel):
    image_uri: str = clip.SourceField()        # clip.embed_image(uri)
    vector: Vector(clip.ndims()) = clip.VectorField()
    caption: str
```

## Custom embedding functions

Subclass `TextEmbeddingFunction` (text in, vectors out) and tag with `@register("name")`. Once registered, it behaves like a built-in provider.

```python
import math
from lancedb.embeddings import TextEmbeddingFunction, register, get_registry
from lancedb.pydantic import LanceModel, Vector

KEYWORDS = ("nvidia", "rag", "lancedb", "vector", "hybrid", "reranker")

@register("demo-keyword")
class KeywordCounter(TextEmbeddingFunction):
    keywords: tuple[str, ...] = KEYWORDS

    def ndims(self) -> int:
        return len(self.keywords)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            counts = [float(t.lower().count(k)) for k in self.keywords]
            norm = math.sqrt(sum(c * c for c in counts)) or 1.0
            out.append([c / norm for c in counts])
        return out

embed = get_registry().get("demo-keyword").create()

class Chunk(LanceModel):
    text: str = embed.SourceField()
    vector: Vector(embed.ndims()) = embed.VectorField()
```

For image / multimodal embedders, subclass `ImageEmbeddingFunction` instead. Override `generate_image_embeddings(images)` and `generate_text_embeddings(texts)` so a single function handles both query types.

## Without the registry: `with_embeddings`

When you want to embed a Pandas DataFrame once at ingest time and skip the registry plumbing:

```python
from lancedb.embeddings import with_embeddings
import pandas as pd

def embed_fn(batch):
    return openai_client.embeddings.create(input=batch, model="...").data

df = pd.DataFrame({"text": chunks})
df = with_embeddings(embed_fn, df, column="text", batch_size=128)
table = db.create_table("docs", data=df)
```

`with_embeddings` adds a `vector` column in place. Use this when the schema and embedder come from outside your control (e.g. you're loading a precomputed Parquet).

## Multi-vector schemas (one table, several vector columns)

Two flavors are common:

**1. One row, multiple semantic vectors (e.g. text + image embedding for the same recipe).** Declare each vector column explicitly:

```python
class Recipe(LanceModel):
    id: str
    title: str
    text_embedding: Vector(384)
    image_embedding: Vector(512)
    image_binary: bytes
```

Search a specific column:
```python
tbl.search(text_vec, vector_column_name="text_embedding").limit(5)
tbl.search(image_vec, vector_column_name="image_embedding").limit(5)
```

**2. ColBERT / ColPali style — one row, one *list of vectors* per doc.** The `colbert` registry provider returns a multi-vector embedding; LanceDB stores it as a list and supports late-interaction scoring at query time. Build the index with `metric="cosine"` (the only metric currently supported for multivector):

```python
table.create_index(
    metric="cosine",
    num_partitions=int(math.sqrt(num_rows * 1030)),  # ~sqrt(total embeddings)
    num_sub_vectors=32,
    index_type="IVF_PQ",
)
```

## PyArrow schemas (when you need fine control)

Use `pydantic_to_schema` to get a PyArrow schema, edit it (e.g. adjust field nullability), then create the table with that schema:

```python
from lancedb.pydantic import pydantic_to_schema, vector
import pyarrow as pa
import pydantic

class ProductModel(pydantic.BaseModel):
    product_id: int
    product_name: str
    vector: vector(128)     # lowercase `vector` factory, takes dim

schema = pydantic_to_schema(ProductModel)
table = db.create_table("products", schema=schema, mode="overwrite")
```

`pydantic.BaseModel` (not `LanceModel`) plus the lowercase `vector(dim)` is the lower-level path — useful when you want a non-Pydantic Lance/Arrow schema downstream.

## Field types reference

| Python type | Lance/Arrow type |
|-------------|-------------------|
| `str` | `string` |
| `int` | `int64` |
| `float` | `float64` |
| `bool` | `bool` |
| `bytes` | `binary` |
| `datetime.datetime` | `timestamp[us]` |
| `list[float]` | `list<float>` (variable length) |
| `Vector(dim)` | `fixed_size_list<float, dim>` (use this for embeddings — required by ANN) |
| `dict` / nested Pydantic | `struct<...>` |
| `list[T]` | `list<T>` |
| `Optional[T]` | nullable |

`Vector(dim)` is **fixed-size** which is what the IVF_PQ index requires. Plain `list[float]` won't index — only use it when you intentionally want variable-length per row (multi-vector / ColBERT style).
