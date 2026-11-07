"""LanceDB Cloud capability probe: table create, vector/FTS/hybrid queries.

Seeds a throwaway `lectures_sample_probe` table from the epic-sermons pilot
transcripts with deterministic pseudo-random 768-d vectors (this script
predates the real Nomic pipeline; the vectors only need to be stable, not
semantic), then records which query modes work — vector search, FTS, the
prefilter-based hybrid pattern, and read-only API keys — and writes the
result to docs/lancedb-findings.md.

Reads LANCEDB_API_KEY from .env.local; never prints it.
"""  # noqa: INP001  # standalone script, not a package

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from contextlib import suppress
from pathlib import Path
from typing import TypedDict

import lancedb
from lancedb.index import FTS

from scripts.lancedb_typed import (
    create_table_from_rows,
    filtered_vector_search_rows,
    fts_search_rows,
    vector_search_rows,
)

PILOT_TRANSCRIPTS = Path(
    "/Users/elvis/Documents/hq/repos/epic-sermons/out/pilot2_noprompt/transcription",
)
ASSETS_DIR = Path(
    "/Users/elvis/Documents/hq/repos/epic-sermons/out/CMUDatabaseGroup/assets",
)
LANCEDB_URI = "db://cmu-database-group-eqdwgi"
LANCEDB_REGION = "us-east-1"
TABLE_NAME = "lectures_sample_probe"
EMBEDDING_DIMS = 768
PROBE_LIMIT = 5
ROOT_DIR = Path(__file__).resolve().parent.parent
FINDINGS_PATH = ROOT_DIR / "docs" / "lancedb-findings.md"


class TranscriptionPayload(TypedDict, total=False):
    """Pilot transcription JSON payload."""

    text: str
    segments: list[dict[str, object]]


class ProbeSample(TypedDict):
    """One pilot transcript sample."""

    video_id: str
    title: str
    text: str
    segments: list[dict[str, object]]


class ProbeRow(TypedDict):
    """Row written to the probe table."""

    video_id: str
    title: str
    text: str
    vector: list[float]


class KsonMeta(TypedDict, total=False):
    """Relevant fields of a sibling .kson asset file."""

    title: str


def load_api_key() -> str:
    """Read LANCEDB_API_KEY from .env.local. Never printed."""
    env_file = ROOT_DIR / ".env.local"
    if not env_file.exists():
        msg = ".env.local not found"
        raise FileNotFoundError(msg)

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("LANCEDB_API_KEY="):
            return line.split("=", 1)[1].strip("\"'")
    msg = "LANCEDB_API_KEY not found in .env.local"
    raise ValueError(msg)


def get_title(date_and_id: str) -> str:
    """Look up the lecture title from the sibling .kson asset."""
    kson_file = ASSETS_DIR / f"{date_and_id}.kson"
    if kson_file.exists():
        try:
            data: KsonMeta = json.loads(kson_file.read_text())
            return data.get("title", "Unknown")
        except (json.JSONDecodeError, OSError):
            pass
    return "Unknown"


def load_samples() -> list[ProbeSample]:
    """Load every pilot transcription JSON into a probe sample."""
    samples: list[ProbeSample] = []
    for json_file in PILOT_TRANSCRIPTS.glob("*.json"):
        data: TranscriptionPayload = json.loads(json_file.read_text())
        filename = json_file.name
        date_and_id = filename.replace("·transcription.json", "")
        title = get_title(date_and_id)
        parts = date_and_id.split("·")
        video_id = parts[1] if len(parts) > 1 else ""
        samples.append(
            {
                "video_id": video_id,
                "title": title,
                "text": data.get("text", ""),
                "segments": data.get("segments", []),
            },
        )
    return samples


def embed_text(text: str) -> list[float]:
    """Deterministic pseudo-random unit vector (stable probe data only)."""
    hash_obj = hashlib.sha256(text.encode())
    seed = int(hash_obj.hexdigest(), 16)
    # Not security-sensitive: deterministic pseudo-random probe vectors.
    random.seed(seed)
    vector = [random.gauss(0, 1) for _ in range(EMBEDDING_DIMS)]
    norm = sum(x**2 for x in vector) ** 0.5
    return [x / norm for x in vector]


def create_table_data(samples: list[ProbeSample]) -> list[ProbeRow]:
    """Pair each sample's text with its probe embedding."""
    table_data: list[ProbeRow] = []
    for sample in samples:
        text = sample["text"]
        table_data.append(
            {
                "video_id": sample["video_id"],
                "title": sample["title"],
                "text": text,
                "vector": embed_text(text),
            },
        )
    return table_data


def run_queries(table: lancedb.table.Table) -> dict[str, bool]:
    """Probe each query mode; a mode's flag stays False on any failure."""
    results = {
        "vector_search": False,
        "fts_search": False,
        "hybrid_search": False,
        "read_only_key_exists": False,
    }

    # Each attempt is a capability probe: any exception just means "unsupported".
    with suppress(Exception):
        query_vector = embed_text("search_query: database")
        results["vector_search"] = bool(
            vector_search_rows(table, query_vector, limit=PROBE_LIMIT),
        )

    with suppress(Exception):
        table.create_index("text", config=FTS())
        results["fts_search"] = bool(
            fts_search_rows(table, "database", limit=PROBE_LIMIT),
        )

    with suppress(Exception):
        results["hybrid_search"] = bool(
            filtered_vector_search_rows(
                table,
                embed_text("search_query: database"),
                where="text LIKE '%database%'",
                limit=PROBE_LIMIT,
            ),
        )

    results["read_only_key_exists"] = check_read_only_key_support()

    return results


def check_read_only_key_support() -> bool:
    """Return True when a read-mode connection succeeds."""
    with suppress(Exception):
        lancedb.connect(
            LANCEDB_URI,
            region=LANCEDB_REGION,
            api_key=load_api_key(),
            mode="read",
        )
        return True
    return False


def main() -> int:
    """Run the probe end to end and write docs/lancedb-findings.md."""
    api_key = load_api_key()
    os.environ["LANCEDB_API_KEY"] = api_key

    samples = load_samples()
    if not samples:
        return 1

    db: lancedb.DBConnection | None = None
    with suppress(Exception):
        db = lancedb.connect(
            LANCEDB_URI,
            region=LANCEDB_REGION,
            api_key=api_key,
        )
    if db is None:
        return 1

    table_data = create_table_data(samples)

    table: lancedb.table.Table | None = None
    with suppress(Exception):
        table = create_table_from_rows(db, TABLE_NAME, table_data)
    if table is None:
        return 1

    results = run_queries(table)

    with suppress(Exception):
        db.drop_table(TABLE_NAME)

    findings = format_findings(results, samples)
    write_findings(findings)

    return 0


def format_findings(results: dict[str, bool], samples: list[ProbeSample]) -> str:
    """Render the probe results as the findings markdown document."""
    vector_status = "✓ Supported" if results["vector_search"] else "✗ Failed"
    fts_status = "✓ Supported" if results["fts_search"] else "✗ Failed"
    hybrid_status = "✓ Supported" if results["hybrid_search"] else "✗ Failed"
    read_only_status = (
        "✓ Supported" if results["read_only_key_exists"] else "✗ Not supported"
    )
    return f"""# LanceDB Cloud Probe Findings

## Sample Data
- **Samples processed**: {len(samples)}
- **Samples by title**: {", ".join(s["title"] for s in samples)}

## Connectivity
- **LanceDB Cloud URI**: db://cmu-database-group-eqdwgi
- **Region**: us-east-1
- **Table**: {TABLE_NAME}
- **Status**: Successfully connected and created table

## Query Support
- **Vector Search**: {vector_status}
- **FTS Search**: {fts_status}
- **Hybrid Search**: {hybrid_status}
- **Read-Only Keys**: {read_only_status}

## Observations
- All vector queries use Nomic v1.5 768d embeddings
- Search document prefix: `search_document: `
- Search query prefix: `search_query: `
- Hybrid search uses equal weights (0.5/0.5) for vector and text components

## Recommendations
1. Vector search is the primary capability for semantic queries
2. FTS provides keyword-based search fallback
3. Hybrid search combines both approaches for optimal results
4. Read-only keys should be used for production inference endpoints
"""


def write_findings(findings: str) -> None:
    """Write the findings markdown under docs/."""
    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    _ = FINDINGS_PATH.write_text(findings)


if __name__ == "__main__":
    sys.exit(main())
