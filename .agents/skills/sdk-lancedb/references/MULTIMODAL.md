# Multimodal

Two patterns dominate multimodal LanceDB tables:

1. **Shared embedding space (CLIP family)** — one vector column, both text and image queries embed into it. Simpler, and what most retrieval-style apps want.
2. **Separate vector columns per modality** — text and image have their own encoders and their own column. Search one or the other depending on the query type.

Both compose with all the other LanceDB features (hybrid search, rerankers, filters).

## Shared embedding space with CLIP

The `open-clip` registry provider handles both modalities through one `EmbeddingFunction`.

```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

clip = get_registry().get("open-clip").create()    # default ViT-B-32, openai pretrained

class ImageDoc(LanceModel):
    image_uri: str = clip.SourceField()             # path or URL
    vector: Vector(clip.ndims()) = clip.VectorField()
    label: str

table = db.create_table("images", schema=ImageDoc, mode="overwrite")
table.add([
    {"image_uri": "/data/cat.jpg", "label": "cat"},
    {"image_uri": "/data/dog.jpg", "label": "dog"},
])

# Text query → CLIP embeds the prompt → ANN over image vectors
table.search("a photo of a cat").limit(5).to_pandas()

# Image query → CLIP embeds the image → ANN over the same column
table.search("/data/query.jpg").limit(5).to_pandas()
```

The registry's `SourceField` autodetects path/URL inputs and runs the image encoder; raw strings go through the text encoder. That gives you free image-to-image and text-to-image search out of one schema.

## Manual CLIP (no registry) — DiffusionDB

When you want explicit control over the encoder (custom preprocessing, batched embedding, GPU device placement), skip the registry and embed manually.

From `examples/multimodal_clip_diffusiondb/main.py`:

```python
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

MODEL_ID = "openai/clip-vit-base-patch32"
TOK = CLIPTokenizerFast.from_pretrained(MODEL_ID)
MODEL = CLIPModel.from_pretrained(MODEL_ID)

def embed_text(query: str):
    inputs = TOK([query], padding=True, return_tensors="pt")
    return MODEL.get_text_features(**inputs).detach().numpy()[0]

# Build table from a Lance dataset that already has image vectors
import lance
data = lance.dataset("path/to/diffusiondb.lance").to_table()
tbl = db.create_table("diffusiondb", data=data, mode="overwrite")
tbl.create_fts_index(["prompt"])

# Text → CLIP embedding → vector search
hits = tbl.search(embed_text("dragon over mountain")).limit(9).to_pandas()
```

## Separate vector columns per modality (recipes)

Use this when text and images have *different* semantics and a shared encoder would force unwanted compromises. From `applications/multimodal-recipe-agent/agent.py`:

```python
from lancedb.pydantic import LanceModel, Vector

class Recipe(LanceModel):
    id: str
    title: str
    ingredients: str
    text_embedding: Vector(384)        # sentence-transformers
    image_embedding: Vector(512)       # CLIP / Vision Transformer
    image_binary: bytes                # raw bytes; use `str` for paths

table = db.create_table("recipes", schema=Recipe, mode="overwrite")

# Always pass vector_column_name when searching a multi-vector table
text_hits = (table.search(text_vec, vector_column_name="text_embedding")
                  .limit(10).to_pandas())

img_hits = (table.search(image_vec, vector_column_name="image_embedding")
                 .limit(10).to_pandas())
```

To fetch a single recipe by id and stream its image:

```python
import base64
row = table.search().where(f"id = '{recipe_id}'").to_pandas().iloc[0]
b64 = base64.b64encode(row["image_binary"]).decode()
```

## Image storage strategies

| Strategy | When |
|----------|------|
| `image_path: str` (filesystem path) | Local-only pipelines; you control the disk. Lightest table. |
| `image_uri: str` (s3:// / https://) | Distributed pipelines; LanceDB stays free of bytes. CLIP autodetects URLs. |
| `image_binary: bytes` | Tables that must travel as a single artifact (Modal volumes, S3 sync). Avoid for >100KB images at scale; binary columns inflate Lance fragments. |
| Lance dataset of images side-by-side | Largest scale; uses `lance.dataset()` to read precomputed vectors. |

For paths/URIs, embed lazily at query time only when you re-embed; for `bytes`, embed at insert time so you don't repeatedly decode binary blobs.

## Jina CLIP v2 — multilingual + image

```python
clip = get_registry().get("jina").create(name="jina-clip-v2")

class Doc(LanceModel):
    text: str = clip.SourceField()
    vector: Vector(clip.ndims()) = clip.VectorField()
```

`jina-clip-v2` shares vector space across 80+ languages and images, so you can ingest English captions and query in Hindi, etc. Costs an API call per embed unless you set `local=True` and host the model locally.

## Voyage multimodal embeddings

```python
voyage = get_registry().get("voyageai").create(name="voyage-multimodal-3")
```

Voyage's `multimodal-3` series accepts interleaved text+image inputs and produces a single vector — useful when documents contain both (PDFs with figures, slides). Same `SourceField` / `VectorField` pattern.

## Hybrid + multimodal

Hybrid search + multimodal works exactly like text-only hybrid — pass `query_type="hybrid"`, but specify `vector_column_name` so vector search hits the right column:

```python
hits = (table.search("a photo of a cat", query_type="hybrid",
                     vector_column_name="image_embedding")
              .rerank(reranker=RRFReranker())
              .limit(10))
```

The FTS half hits the columns indexed via `create_fts_index(...)` (caption, title, etc.).

## Filtering on metadata + multimodal

Combine `where` with multimodal search the same way:

```python
hits = (table.search(image_vec, vector_column_name="image_embedding")
              .where("label = 'cat' AND created_at > '2025-01-01'", prefilter=True)
              .limit(10))
```

Build a scalar index on `label` (and `created_at`) if you query like this often:

```python
table.create_scalar_index("label")
```

## Video / audio

Examples like `examples/multimodal_video_search/insert.py` extract one frame every N seconds with OpenCV, embed each with CLIP, and store frame-level rows:

```python
table.add([
    {
        "vector": clip_embedding,
        "video_id": vid,
        "start_time": (frame_idx - 1) * 30,    # seconds
    }
    for clip_embedding, vid, frame_idx in zip(image_features, video_ids, frames)
])
```

For audio, the equivalent pattern uses Whisper transcription chunks plus CLAP/ImageBind embeddings. The schema is otherwise identical to text RAG — just include the temporal metadata you'll need to deep-link back to the source media.
