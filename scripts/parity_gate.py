"""Parity gate: Fireworks vs local sentence-transformers Nomic v1.5 embeddings.

Embeds the 14 sample-transcript chunks plus synthetic database-jargon
sentences twice — once via the Fireworks hosted API
(``nomic-ai/nomic-embed-text-v1.5``) and once locally via
sentence-transformers — then compares cosine similarity per text.

The gate passes when every text scores cosine >= PARITY_THRESHOLD (0.99).

It also determines how Fireworks treats the ``search_document: `` prefix by
embedding both raw and prefixed text on both sides and reporting which
combination agrees with the local prefixed baseline (the convention used by
``lecture_search/embed.py``).

Reads FIREWORKS_API_KEY from .env.local; never prints it.
"""  # noqa: INP001  # standalone script, not a package

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TypedDict

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lecture_search import process_sample_transcripts  # noqa: E402

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/embeddings"
FIREWORKS_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FIREWORKS_TIMEOUT_SECONDS = 120
PARITY_THRESHOLD = 0.99
RESULTS_PATH = ROOT_DIR / "data" / "parity_gate.json"

# Database-jargon sentences to push the sample from 14 chunks to 20+ texts.
SYNTHETIC_TEXTS = [
    ("The query planner reorders the join to exploit the build side of the hash join."),
    "MVCC keeps old tuple versions so readers never block writers.",
    "The buffer pool evicts pages using an LRU-K replacement policy.",
    ("WAL records are flushed to durable storage before the commit is acknowledged."),
    "A covering index lets the executor answer the query from the index alone.",
    (
        "Vectorized execution processes tuples in batches to amortize "
        "interpretation overhead."
    ),
    (
        "The LSM tree compacts sorted runs in the background to bound read "
        "amplification."
    ),
    (
        "Two-phase locking guarantees conflict-serializable schedules at the "
        "cost of deadlocks."
    ),
]

# Functional syntax: "pass" is a keyword and cannot be a class-level field.
PerTextResult = TypedDict(
    "PerTextResult",
    {"index": int, "text_preview": str, "cosine": float, "pass": bool},
)


class EmbeddingItem(TypedDict):
    """One element of the Fireworks embeddings response."""

    index: int
    embedding: list[float]


class FireworksResponse(TypedDict):
    """Fireworks embeddings API response payload."""

    data: list[EmbeddingItem]


def load_api_key() -> str:
    """Read FIREWORKS_API_KEY from .env.local. Never printed."""
    env_file = ROOT_DIR / ".env.local"
    if not env_file.exists():
        msg = ".env.local not found"
        raise FileNotFoundError(msg)
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("FIREWORKS_API_KEY="):
            return line.split("=", 1)[1].strip("\"'")
    msg = "FIREWORKS_API_KEY not found in .env.local"
    raise ValueError(msg)


def fireworks_embed(texts: list[str], api_key: str) -> np.ndarray:
    """Embed texts via the hosted Fireworks API, in input order."""
    response = httpx.post(
        FIREWORKS_URL,
        json={"model": FIREWORKS_MODEL, "input": texts},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=FIREWORKS_TIMEOUT_SECONDS,
    )
    _ = response.raise_for_status()
    payload: FireworksResponse = response.json()
    ordered = sorted(payload["data"], key=lambda item: item["index"])
    return np.array([item["embedding"] for item in ordered], dtype=np.float64)


def local_embed(texts: list[str]) -> np.ndarray:
    """Embed texts locally with sentence-transformers Nomic v1.5 (CPU)."""
    model = SentenceTransformer(
        FIREWORKS_MODEL,
        device="cpu",
        trust_remote_code=True,
    )
    vectors = model.encode(texts, convert_to_numpy=True)  # pyright: ignore[reportUnknownMemberType]  # sentence-transformers encode() overloads are partially Unknown
    return np.asarray(vectors, dtype=np.float64)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cosine-similarity matrix between rows of `a` and rows of `b`."""
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)  # pyright: ignore[reportUnknownMemberType]  # numpy norm() overloads are partially Unknown
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)  # pyright: ignore[reportUnknownMemberType]  # numpy norm() overloads are partially Unknown
    return (a / a_norms) @ (b / b_norms).T


def cosine_diagonal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-text cosine similarity (diagonal of the full matrix)."""
    return np.diag(cosine_matrix(a, b))  # pyright: ignore[reportUnknownMemberType]  # numpy diag() overloads are partially Unknown


def main() -> int:
    """Run the parity gate; exit 0 when every text meets the threshold."""
    api_key = load_api_key()
    _ = os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    chunk_map = process_sample_transcripts(ROOT_DIR / "data" / "sample")
    texts = [chunk.text for chunks in chunk_map.values() for chunk in chunks]
    texts.extend(SYNTHETIC_TEXTS)
    print(f"Embedding {len(texts)} texts")  # noqa: T201  # CLI progress output

    raw = texts
    prefixed = [f"search_document: {text}" for text in texts]

    fw_raw = fireworks_embed(raw, api_key)
    fw_prefixed = fireworks_embed(prefixed, api_key)
    local_raw = local_embed(raw)
    local_prefixed = local_embed(prefixed)

    # Prefix probe: which Fireworks request matches the local prefixed baseline?
    prefix_probe = {
        "fw_raw_vs_local_prefixed_mean": float(
            cosine_diagonal(fw_raw, local_prefixed).mean(),
        ),
        "fw_prefixed_vs_local_prefixed_mean": float(
            cosine_diagonal(fw_prefixed, local_prefixed).mean(),
        ),
        "fw_raw_vs_local_raw_mean": float(cosine_diagonal(fw_raw, local_raw).mean()),
    }
    print("Prefix probe (mean cosine over texts):")  # noqa: T201  # CLI report output
    for name, value in prefix_probe.items():
        print(f"  {name}: {value:.6f}")  # noqa: T201  # CLI report output

    fireworks_adds_prefix = (
        prefix_probe["fw_raw_vs_local_prefixed_mean"]
        > prefix_probe["fw_raw_vs_local_raw_mean"]
    )
    print(f"Fireworks applies the prefix itself: {fireworks_adds_prefix}")  # noqa: T201  # CLI report output

    # Gate: compare like-for-like. If Fireworks already prefixes, gate on
    # fw_raw vs local_prefixed; otherwise gate on fw_prefixed vs local_prefixed
    # (matching how lecture_search/embed.py prepends the prefix itself).
    fw_like_for_like = fw_raw if fireworks_adds_prefix else fw_prefixed
    similarities = cosine_diagonal(fw_like_for_like, local_prefixed)

    per_text: list[PerTextResult] = [
        {
            "index": i,
            "text_preview": text[:80],
            "cosine": float(similarities[i]),
            "pass": bool(similarities[i] >= PARITY_THRESHOLD),
        }
        for i, text in enumerate(texts)
    ]
    for row in per_text:
        status = "PASS" if row["pass"] else "FAIL"
        print(f"  [{status}] {row['cosine']:.6f}  {row['text_preview']!r}")  # noqa: T201  # CLI report output

    min_cosine = float(similarities.min())
    mean_cosine = float(similarities.mean())
    gate_pass = min_cosine >= PARITY_THRESHOLD
    print(f"min={min_cosine:.6f} mean={mean_cosine:.6f}")  # noqa: T201  # CLI report output
    print(f"Gate at {PARITY_THRESHOLD}: {'PASS' if gate_pass else 'FAIL'}")  # noqa: T201  # CLI report output

    _ = RESULTS_PATH.write_text(
        json.dumps(
            {
                "model": FIREWORKS_MODEL,
                "threshold": PARITY_THRESHOLD,
                "num_texts": len(texts),
                "fireworks_adds_prefix": fireworks_adds_prefix,
                "prefix_probe": prefix_probe,
                "min_cosine": min_cosine,
                "mean_cosine": mean_cosine,
                "gate_pass": gate_pass,
                "per_text": per_text,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"Wrote {RESULTS_PATH}")  # noqa: T201  # CLI progress output
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
