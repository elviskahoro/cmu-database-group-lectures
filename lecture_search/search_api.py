"""Modal search API for the CMU Database Group lecture corpus (cdgl-99e).

The public notebook never touches LanceDB: LanceDB Cloud has no read-only
API keys (docs/lancedb-findings.md), so this Modal app ('cmu-lecture-search')
holds the key as a Modal secret and answers read-only search requests.

**Embedding is client-side by design:** the caller (the molab notebook)
embeds the query itself — Nomic v1.5 with the ``search_query: `` prefix, on
a molab GPU if available else CPU, via ``lecture_search.embed.embed_query``
— and sends the vector in the request. The Modal image therefore carries
only ``lancedb`` + ``fastapi``: no sentence-transformers/transformers/
einops/torch and no baked model weights, so the image is small, cold starts
are fast, and no CPU time is billed for embedding.

Endpoints (FastAPI, served via ``modal.asgi_app``):

- ``POST /search`` — hybrid search (vector + FTS, MRR reranker, same
  construction as ``notebooks/search.py``). Body:
  ``{query, vector, top_k, fts_weight, date_from?, date_to?, video_ids?}``
  where ``query`` is the raw text (used for the FTS half) and ``vector`` is
  the client-embedded query (exactly 768 finite floats).
  Returns ``{"results": [...]}`` with rows containing only
  ``video_id, title, date, start_seconds, end_seconds, deep_link, text,
  score`` — never vectors, keys, URIs or stack traces.
- ``GET /lectures`` — ``{"lectures": [{video_id, title, date}]}`` for the
  notebook's lecture filter (cached per container).

Security posture: the endpoint is public and unauthenticated, so every
input is validated hard (see ``parse_search_request``) and WHERE clauses are
only ever built from values that survived that validation
(``build_where``). The request body is size-capped (``check_body_size``)
before it is parsed.

Deploy (never prints secrets)::

    MODAL_PROFILE=marcella modal deploy lecture_search/search_api.py

Config comes from the Modal secret ``lancedb-search`` (LANCEDB_URI,
LANCEDB_API_KEY, LANCEDB_REGION, LANCEDB_TABLE); see docs/search-api.md.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import lancedb
from lancedb.rerankers import MRRReranker, Reranker

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
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - only the Modal image has fastapi
    FastAPI = Request = CORSMiddleware = JSONResponse = None  # type: ignore[assignment]

APP_NAME = "cmu-lecture-search"
DEFAULT_TABLE = "lectures"
DEFAULT_REGION = "us-east-1"

# Nomic v1.5 embedding dimension. Duplicated from lecture_search.embed on
# purpose: importing lecture_search.embed here would pull sentence-
# transformers into the container, defeating the client-side-embedding
# design (the image must stay lancedb + fastapi only).
VECTOR_DIMS = 768

# Hard request limits (public, unauthenticated, read-only endpoint).
# A legitimate body is dominated by the 768-float vector (~15 KB as JSON),
# so 20 KB caps it with headroom and rejects vector-padded payloads.
MAX_BODY_BYTES = 20 * 1024
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
# LanceDB Cloud rejects empty-search scans with k > 10000 ("k cannot be
# bigger than 10000"), so the /lectures scan pages with offset() at this
# page size instead of issuing one LECTURE_SCAN_LIMIT-row query.
LECTURE_SCAN_PAGE_SIZE = 10_000


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

    def offset(self, offset: int) -> QueryBuilder:
        """Skip the first ``offset`` rows (scan pagination)."""
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


class RequestValidationError(ValueError):
    """A /search payload failed validation; maps to HTTP 422.

    Messages are generic on purpose: they describe the rule that was
    violated, never echo the offending input back.
    """


@dataclass(frozen=True)
class SearchParams:
    """Validated, safe-to-use /search parameters."""

    query: str
    vector: tuple[float, ...]
    top_k: int
    fts_weight: float
    date_from: date | None
    date_to: date | None
    video_ids: tuple[str, ...]


def check_body_size(body: bytes) -> None:
    """Reject oversized request bodies before parsing (see MAX_BODY_BYTES)."""
    if len(body) > MAX_BODY_BYTES:
        msg = f"request body must be at most {MAX_BODY_BYTES} bytes"
        raise RequestValidationError(msg)


def _parse_vector(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != VECTOR_DIMS:
        msg = f"vector must be a list of exactly {VECTOR_DIMS} numbers"
        raise RequestValidationError(msg)
    components: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            msg = "vector components must be numbers"
            raise RequestValidationError(msg)
        number = float(component)
        if not math.isfinite(number):
            msg = "vector components must be finite (no NaN or infinity)"
            raise RequestValidationError(msg)
        components.append(number)
    return tuple(components)


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
    user sends can be interpreted as SQL. The query vector is validated for
    shape and finiteness but is never interpreted as text.
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
        vector=_parse_vector(payload.get("vector")),
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
    params: SearchParams,
) -> list[dict[str, object]]:
    """Run the same hybrid search as notebooks/search.py, then serialize.

    The query vector arrives pre-embedded by the caller; this service never
    runs an embedding model.
    """
    builder = (
        table.search(query_type="hybrid")
        .vector(list(params.vector))
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


def scan_lecture_rows(
    table: SearchTable,
    page_size: int = LECTURE_SCAN_PAGE_SIZE,
    scan_limit: int = LECTURE_SCAN_LIMIT,
) -> list[dict[str, object]]:
    """Scan up to ``scan_limit`` chunk rows for the lecture list.

    LanceDB Cloud's RemoteTable has no ``.query()``; an empty ``.search()``
    with select/limit acts as a full-table scan, but the cloud service caps
    a single scan at 10000 rows, so this pages with ``offset()`` until a
    short page or ``scan_limit`` is reached.
    """
    rows: list[dict[str, object]] = []
    offset = 0
    while offset < scan_limit:
        page_limit = min(page_size, scan_limit - offset)
        page = (
            table.search()
            .select(["video_id", "title", "date"])
            .limit(page_limit)
            .offset(offset)
            .to_list()
        )
        rows.extend(page)
        if len(page) < page_limit:
            break
        offset += page_limit
    return rows


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

    # Slim image: no embedding model, no torch, no baked weights — the
    # caller embeds. lancedb + fastapi only, so builds and cold starts are
    # fast and cheap.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "lancedb==0.40.0",
            "fastapi>=0.115,<1",
        )
        .add_local_python_source("lecture_search")
    )

    @app.cls(
        image=image,
        secrets=[modal.Secret.from_name("lancedb-search")],
        cpu=1.0,
        memory=1024,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    )
    @modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
    class SearchService:
        """CPU container: LanceDB connection, opened once (no model)."""

        table: SearchTable
        _lectures_cache: tuple[float, list[dict[str, str]]] | None

        @modal.enter()
        def load(self) -> None:
            """Connect to LanceDB once per container.

            Config arrives via the ``lancedb-search`` Modal secret.
            """
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
            rows = scan_lecture_rows(self.table)
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
            async def search(
                request: Request,
            ) -> JSONResponse | dict[str, list[dict[str, object]]]:
                # Read the raw body first so the size cap applies before any
                # JSON parsing work happens.
                body = await request.body()
                try:
                    check_body_size(body)
                except RequestValidationError as exc:
                    return JSONResponse(status_code=422, content={"error": str(exc)})
                try:
                    payload: object = json.loads(body)
                except json.JSONDecodeError:
                    return JSONResponse(
                        status_code=422,
                        content={"error": "request body must be valid JSON"},
                    )
                try:
                    params = parse_search_request(payload)
                except RequestValidationError as exc:
                    return JSONResponse(status_code=422, content={"error": str(exc)})
                # Broad-but-enumerated catch: LanceDB and I/O failures all
                # map to a generic 500. Anything else propagates to FastAPI's
                # ServerErrorMiddleware, which returns a bare "Internal
                # Server Error" — still no stack trace or internals.
                try:
                    rows = run_hybrid_search(self.table, params)
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
