"""Modal search API for the CMU Database Group lecture corpus (cdgl-bcy).

The public notebook never touches LanceDB: LanceDB Cloud has no read-only
API keys (docs/lancedb-findings.md), so this Modal app ('cmu-lecture-search')
holds the key as a Modal secret and answers read-only search requests.

Endpoints (FastAPI, served via ``modal.asgi_app``):

- ``POST /search`` — hybrid search (vector + FTS, MRR reranker, same
  construction as ``notebooks/search.py``). Body:
  ``{query, top_k, fts_weight, date_from?, date_to?, video_ids?}``.
  Returns ``{"results": [...]}`` with rows containing only
  ``video_id, title, date, start_seconds, end_seconds, deep_link, text,
  score`` — never vectors, keys, URIs or stack traces.
- ``GET /lectures`` — ``{"lectures": [{video_id, title, date}]}`` for the
  notebook's lecture filter (cached per container).

Security posture: the endpoint is public and unauthenticated, so every
input is validated hard (see ``parse_search_request``) and WHERE clauses are
only ever built from values that survived that validation
(``build_where``). Query embedding applies the Nomic ``search_query: ``
prefix exactly once (``embed_query_once``), matching how the corpus was
embedded (tests/test_embed_prefix.py).

Deploy (never prints secrets)::

    MODAL_PROFILE=marcella modal deploy lecture_search/search_api.py

Config comes from the Modal secret ``lancedb-search`` (LANCEDB_URI,
LANCEDB_API_KEY, LANCEDB_REGION, LANCEDB_TABLE); see docs/search-api.md.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Protocol

import lancedb
from lancedb.embeddings import get_registry
from lancedb.rerankers import MRRReranker, Reranker

from lecture_search.embed import EMBEDDING_DIMS, QUERY_PREFIX

# Deploy-only dependencies: baked into the Modal image below, intentionally
# absent from the dev venv, so the pure request/response logic stays
# importable (and testable) without them. The two guards are separate on
# purpose: the local `modal deploy` process has modal but NOT fastapi (the
# FastAPI names are only touched inside `web()`, which runs in-container).
try:
    import modal
except ImportError:  # pragma: no cover - local dev has no modal installed
    modal = None  # type: ignore[assignment]

try:
    from fastapi import Body, FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - only the Modal image has fastapi
    Body = FastAPI = CORSMiddleware = JSONResponse = None  # type: ignore[assignment]

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
APP_NAME = "cmu-lecture-search"
DEFAULT_TABLE = "lectures_sample"
DEFAULT_REGION = "us-east-1"

# Hard request limits (public, unauthenticated, read-only endpoint).
MAX_QUERY_CHARS = 300
MAX_TOP_K = 50
DEFAULT_TOP_K = 10
DEFAULT_FTS_WEIGHT = 0.5
MAX_VIDEO_IDS = 50
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Container shape: small CPU service, aggressive limits, fast scale-down.
MAX_CONTAINERS = 5
MAX_CONCURRENT_INPUTS = 8
SCALEDOWN_WINDOW_SECONDS = 300
LECTURES_CACHE_TTL_SECONDS = 3600
# Upper bound on rows scanned to build the distinct lecture list.
LECTURE_SCAN_LIMIT = 200_000


class QueryBuilder(Protocol):
    """The slice of the LanceDB query builder this service composes."""

    def vector(self, vector: list[float]) -> QueryBuilder:
        """Set the query vector."""
        ...

    def text(self, text: str) -> QueryBuilder:
        """Set the FTS query text."""
        ...

    def rerank(self, reranker: Reranker) -> QueryBuilder:
        """Set the reranker."""
        ...

    def limit(self, limit: int) -> QueryBuilder:
        """Set the result limit."""
        ...

    def where(self, where: str) -> QueryBuilder:
        """Set a SQL WHERE clause."""
        ...

    def select(self, columns: list[str]) -> QueryBuilder:
        """Project the given columns."""
        ...

    def to_list(self) -> list[dict[str, object]]:
        """Materialize results as a list of row dicts."""
        ...


class SearchTable(Protocol):
    """The slice of a LanceDB table this service uses (read-only)."""

    def search(self, query_type: str = "auto") -> QueryBuilder:
        """Start a search (or, with no query, a scan)."""
        ...


class Embedder(Protocol):
    """Sentence-transformers registry embedder (prefixes applied by caller)."""

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed texts verbatim (prefixes are the caller's job)."""
        ...


class RequestValidationError(ValueError):
    """A /search payload failed validation; maps to HTTP 422.

    Messages are generic on purpose: they describe the rule that was
    violated, never echo the offending input back.
    """


@dataclass(frozen=True)
class SearchParams:
    """Validated, safe-to-use /search parameters."""

    query: str
    top_k: int
    fts_weight: float
    date_from: date | None
    date_to: date | None
    video_ids: tuple[str, ...]


def _parse_optional_date(value: object, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{field} must be an ISO date string (YYYY-MM-DD)"
        raise RequestValidationError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError:
        msg = f"{field} must be an ISO date string (YYYY-MM-DD)"
        raise RequestValidationError(msg) from None


def _parse_top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"top_k must be an integer between 1 and {MAX_TOP_K}"
        raise RequestValidationError(msg)
    if not 1 <= value <= MAX_TOP_K:
        msg = f"top_k must be an integer between 1 and {MAX_TOP_K}"
        raise RequestValidationError(msg)
    return value


def _parse_fts_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "fts_weight must be a number between 0 and 1"
        raise RequestValidationError(msg)
    weight = float(value)
    if not 0.0 <= weight <= 1.0:
        msg = "fts_weight must be a number between 0 and 1"
        raise RequestValidationError(msg)
    return weight


def _parse_video_ids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > MAX_VIDEO_IDS:
        msg = f"video_ids must be a list of at most {MAX_VIDEO_IDS} video IDs"
        raise RequestValidationError(msg)
    if any(
        not isinstance(video_id, str) or not VIDEO_ID_RE.fullmatch(video_id)
        for video_id in value
    ):
        msg = "video_ids must be 11-character YouTube video IDs"
        raise RequestValidationError(msg)
    return tuple(value)


def parse_search_request(payload: object) -> SearchParams:
    """Validate a raw JSON body into typed, safe-to-use SearchParams.

    Every field that can end up in a WHERE clause or a LanceDB call is
    constrained here: dates become ``datetime.date`` (only ever rendered via
    ``isoformat()``) and video_ids must match a strict regex, so nothing a
    user sends can be interpreted as SQL.
    """
    if not isinstance(payload, dict):
        msg = "request body must be a JSON object"
        raise RequestValidationError(msg)

    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        msg = "query must be a non-empty string"
        raise RequestValidationError(msg)
    if len(query) > MAX_QUERY_CHARS:
        msg = f"query must be at most {MAX_QUERY_CHARS} characters"
        raise RequestValidationError(msg)

    date_from = _parse_optional_date(payload.get("date_from"), "date_from")
    date_to = _parse_optional_date(payload.get("date_to"), "date_to")
    if date_from is not None and date_to is not None and date_from > date_to:
        msg = "date_from must not be after date_to"
        raise RequestValidationError(msg)

    return SearchParams(
        query=query,
        top_k=_parse_top_k(payload.get("top_k", DEFAULT_TOP_K)),
        fts_weight=_parse_fts_weight(payload.get("fts_weight", DEFAULT_FTS_WEIGHT)),
        date_from=date_from,
        date_to=date_to,
        video_ids=_parse_video_ids(payload.get("video_ids")),
    )


def build_where(params: SearchParams) -> str | None:
    """Build a WHERE clause from validated params only.

    ``date`` values come from ``datetime.date.isoformat()`` and video IDs
    have already matched ``^[A-Za-z0-9_-]{11}$``, so interpolated values can
    never break out of their string literals. Raw user input is never
    interpolated.
    """
    clauses: list[str] = []
    if params.date_from is not None:
        clauses.append(f"date >= '{params.date_from.isoformat()}'")
    if params.date_to is not None:
        clauses.append(f"date <= '{params.date_to.isoformat()}'")
    if params.video_ids:
        quoted = ", ".join(f"'{video_id}'" for video_id in params.video_ids)
        clauses.append(f"video_id IN ({quoted})")
    return " AND ".join(clauses) if clauses else None


def embed_query_once(embedder: Embedder, query: str) -> list[float]:
    """Embed a query with the Nomic ``search_query: `` prefix applied once."""
    vectors = embedder.generate_embeddings([QUERY_PREFIX + query])
    vector = [float(value) for value in vectors[0]]
    if len(vector) != EMBEDDING_DIMS:
        msg = "embedding dimension mismatch"
        raise RuntimeError(msg)
    return vector


def serialize_row(row: dict[str, object]) -> dict[str, object]:
    """Whitelist one result row: no vector, no internal columns, no keys."""
    score = row.get("_relevance_score")
    return {
        "video_id": str(row["video_id"]),
        "title": str(row["title"]),
        "date": str(row["date"]),
        "start_seconds": float(row["start_seconds"]),
        "end_seconds": float(row["end_seconds"]),
        "deep_link": str(row["deep_link"]),
        "text": str(row["text"]),
        "score": float(score) if score is not None else None,
    }


def run_hybrid_search(
    table: SearchTable,
    embedder: Embedder,
    params: SearchParams,
) -> list[dict[str, object]]:
    """Run the same hybrid search as notebooks/search.py, then serialize."""
    query_vector = embed_query_once(embedder, params.query)
    builder = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(params.query)
        .rerank(
            MRRReranker(
                weight_vector=1.0 - params.fts_weight,
                weight_fts=params.fts_weight,
            ),
        )
        .limit(params.top_k)
    )
    where = build_where(params)
    if where is not None:
        builder = builder.where(where)
    return [serialize_row(row) for row in builder.to_list()]


def summarize_lectures(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    """Dedupe scanned chunk rows into one {video_id, title, date} per lecture."""
    lectures: dict[str, dict[str, str]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        if video_id not in lectures:
            lectures[video_id] = {
                "video_id": video_id,
                "title": str(row["title"]),
                "date": str(row["date"]),
            }
    return sorted(
        lectures.values(),
        key=lambda lecture: (lecture["date"], lecture["title"]),
        reverse=True,
    )


if modal is not None:
    app = modal.App(APP_NAME)

    # Pins mirror the notebook header in notebooks/search.py (transformers
    # must stay on 4.x: Nomic v1.5's trust_remote_code modeling file uses
    # APIs removed in 5.x). fastapi serves the ASGI app. Model weights are
    # baked in via run_commands (not run_function) so the image build never
    # has to import this module — add_local_python_source only lands after.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "lancedb==0.40.0",
            "sentence-transformers==5.7.0",
            "transformers==4.57.6",
            "einops==0.8.2",
            "fastapi>=0.115,<1",
        )
        .env({"HF_HOME": "/hf-cache"})
        .run_commands(
            "python -c 'from huggingface_hub import snapshot_download; "
            f'snapshot_download("{MODEL_NAME}")\'',
        )
        .add_local_python_source("lecture_search")
    )

    @app.cls(
        image=image,
        secrets=[modal.Secret.from_name("lancedb-search")],
        cpu=1.0,
        memory=2048,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    )
    @modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
    class SearchService:
        """CPU container: Nomic embedder + LanceDB connection, loaded once."""

        embedder: Embedder
        table: SearchTable
        _lectures_cache: tuple[float, list[dict[str, str]]] | None

        @modal.enter()
        def load(self) -> None:
            """Load the model and connect to LanceDB once per container.

            Config arrives via the ``lancedb-search`` Modal secret.
            """
            self.embedder = (
                get_registry()
                .get("sentence-transformers")
                .create(
                    name=MODEL_NAME,
                    device="cpu",
                    trust_remote_code=True,
                )
            )
            db = lancedb.connect(
                os.environ["LANCEDB_URI"],
                region=os.environ.get("LANCEDB_REGION", DEFAULT_REGION),
                api_key=os.environ["LANCEDB_API_KEY"],
            )
            table_name = os.environ.get("LANCEDB_TABLE", DEFAULT_TABLE)
            self.table = db.open_table(table_name)  # type: ignore[assignment]
            self._lectures_cache = None

        def _lectures(self) -> list[dict[str, str]]:
            now = time.time()
            cache = self._lectures_cache
            ttl = LECTURES_CACHE_TTL_SECONDS
            if cache is not None and now - cache[0] < ttl:
                return cache[1]
            # LanceDB Cloud's RemoteTable has no .query(); an empty
            # .search() with select/limit acts as a full-table scan.
            rows = (
                self.table.search()
                .select(["video_id", "title", "date"])
                .limit(LECTURE_SCAN_LIMIT)
                .to_list()
            )
            lectures = summarize_lectures(rows)
            self._lectures_cache = (now, lectures)
            return lectures

        @modal.asgi_app()
        def web(self) -> FastAPI:
            """Build the read-only FastAPI app (CORS allow-all, no docs)."""
            api = FastAPI(
                title=APP_NAME,
                docs_url=None,
                redoc_url=None,
                openapi_url=None,
            )
            # Read-only public API; CORS allow-all is fine.
            api.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )

            @api.post("/search", response_model=None)
            def search(
                payload: Annotated[dict[str, object], Body()],
            ) -> JSONResponse | dict[str, list[dict[str, object]]]:
                try:
                    params = parse_search_request(payload)
                except RequestValidationError as exc:
                    return JSONResponse(status_code=422, content={"error": str(exc)})
                # Broad-but-enumerated catch: embedding, LanceDB and I/O
                # failures all map to a generic 500. Anything else propagates
                # to FastAPI's ServerErrorMiddleware, which returns a bare
                # "Internal Server Error" — still no stack trace or internals.
                try:
                    rows = run_hybrid_search(self.table, self.embedder, params)
                except (ValueError, RuntimeError, OSError, KeyError, TypeError):
                    return JSONResponse(
                        status_code=500,
                        content={"error": "search failed"},
                    )
                return {"results": rows}

            @api.get("/lectures", response_model=None)
            def lectures() -> JSONResponse | dict[str, list[dict[str, str]]]:
                try:
                    return {"lectures": self._lectures()}
                except (ValueError, RuntimeError, OSError, KeyError, TypeError):
                    return JSONResponse(
                        status_code=500,
                        content={"error": "failed to list lectures"},
                    )

            return api
