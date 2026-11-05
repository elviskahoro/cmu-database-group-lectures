# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.25.2",
#     "httpx==0.28.1",
#     "sentence-transformers==5.7.0",
#     # nomic-embed-text-v1.5's trust_remote_code modeling file uses the
#     # transformers 4.x API (get_extended_attention_mask); 5.x removes it.
#     "transformers==4.57.6",
#     "pytest==9.1.1",
#     # transformers' lazy Trainer import requires IPython once marimo is
#     # imported (marimo's presence flips transformers' IPython detection);
#     # without it, `import sentence_transformers` fails inside marimo.
#     "ipython==9.7.0",
#     # required by nomic-embed-text-v1.5's trust_remote_code modeling file
#     "einops==0.8.2",
# ]
# ///
# Hybrid search over the CMU Database Group lectures corpus via the public
# Modal search API (lecture_search/search_api.py, docs/search-api.md).
#
# The notebook never touches LanceDB (LanceDB Cloud has no read-only keys):
# it embeds the query locally — Nomic v1.5 on a GPU if available, else CPU —
# and POSTs {query, vector, top_k, fts_weight, filters} to the API, which
# runs the hybrid search and returns whitelisted rows.
#
# Interactive: uv run marimo run notebooks/search.py
# Headless:    uv run notebooks/search.py  (runs sample queries in script mode)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import httpx
    import marimo as mo

    return httpx, mo


@app.cell
def _():
    # Public read-only search API (Modal workspace "marcella"); see
    # docs/search-api.md. Not a secret.
    API_BASE_URL = (
        "https://marcella--cmu-lecture-search-searchservice-web.us-east.modal.run"
    )
    NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIMS = 768
    QUERY_PREFIX = "search_query: "
    # Modal cold starts (container boot + LanceDB connect) measured at ~7 s;
    # allow generous headroom before declaring the API unreachable.
    REQUEST_TIMEOUT_SECONDS = 120.0

    return (
        API_BASE_URL,
        EMBEDDING_DIMS,
        NOMIC_MODEL,
        QUERY_PREFIX,
        REQUEST_TIMEOUT_SECONDS,
    )


@app.cell
def _(EMBEDDING_DIMS, NOMIC_MODEL, QUERY_PREFIX, mo):
    @mo.cache
    def get_embedder():
        import torch
        from sentence_transformers import SentenceTransformer

        # Nomic v1.5 is small enough that the CPU fallback is fine when no
        # GPU is available (e.g. for viewers of a published molab app).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(NOMIC_MODEL, device=device, trust_remote_code=True)

    def embed_query(text: str) -> list[float]:
        # normalize_embeddings=True matches how the corpus was embedded
        # (lancedb's sentence-transformers registry, normalize=True).
        vector = get_embedder().encode(
            [QUERY_PREFIX + text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        embedding = [float(value) for value in vector]
        assert len(embedding) == EMBEDDING_DIMS
        return embedding

    return embed_query, get_embedder


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(API_BASE_URL, REQUEST_TIMEOUT_SECONDS, httpx, is_script_mode, mo):
    def fetch_lectures() -> tuple[list[dict[str, str]], str | None]:
        try:
            response = httpx.get(
                f"{API_BASE_URL}/lectures",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return list(response.json()["lectures"]), None
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            # Network/HTTP/JSON failure — surface a friendly error.
            return [], (
                "Could not load the lecture list from the search API "
                f"({type(exc).__name__}: {exc}). The API may be cold-starting "
                "— wait a few seconds and refresh, or try again shortly."
            )

    if is_script_mode:
        lectures, lectures_error = fetch_lectures()
        if lectures_error is not None:
            pass
    else:
        with mo.status.spinner(
            title="Loading the lecture list — the search API may be "
            "cold-starting (first load can take ~10 s)…",
        ):
            lectures, lectures_error = fetch_lectures()
    return lectures, lectures_error


@app.cell
def _(lectures, lectures_error, mo):
    lecture_options = {
        f"{lecture['title']} ({lecture['date']})": lecture["video_id"]
        for lecture in lectures
    }
    error_view = (
        mo.callout(mo.md(lectures_error), kind="warn")
        if lectures_error is not None
        else None
    )
    error_view
    return (lecture_options,)


@app.cell
def _(lecture_options, lectures, mo):
    # mo.ui.date_range defaults its value to (today, today), which would
    # silently filter every search to today's (nonexistent) lectures. Gate
    # the filter behind a checkbox and default the range to the full corpus
    # span instead.
    corpus_dates = sorted(lecture["date"] for lecture in lectures)
    first_date = corpus_dates[0] if corpus_dates else "2015-01-01"
    last_date = corpus_dates[-1] if corpus_dates else "2030-12-31"
    search_form = (
        mo.md(
            """
            **Search the lectures**

            {query}

            {top_k}

            {fts_weight}

            {use_date_filter}

            {date_range}

            {lecture_filter}
            """,
        )
        .batch(
            query=mo.ui.text(
                label="Query",
                value="how does dbt transform data in the warehouse",
                full_width=True,
            ),
            top_k=mo.ui.slider(
                start=1,
                stop=10,
                step=1,
                value=5,
                label="Top-k results",
            ),
            fts_weight=mo.ui.slider(
                start=0.0,
                stop=1.0,
                step=0.05,
                value=0.5,
                label="Hybrid weight (0 = pure vector, 1 = pure full-text)",
            ),
            use_date_filter=mo.ui.checkbox(label="Filter by date range"),
            date_range=mo.ui.date_range(
                start=first_date,
                stop=last_date,
                value=(first_date, last_date),
                label="Lecture date range (applies when the box above is checked)",
            ),
            lecture_filter=mo.ui.multiselect(
                options=lecture_options,
                label="Restrict to lectures (optional)",
            ),
        )
        .form(submit_button_label="Search", bordered=True)
    )
    search_form
    return (search_form,)


@app.cell
def _(
    API_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    embed_query,
    httpx,
):
    from datetime import date
    from typing import Any

    def build_payload(
        query: str,
        top_k: int,
        fts_weight: float,
        date_from: date | None = None,
        date_to: date | None = None,
        video_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assemble the POST /search body (see docs/search-api.md)."""
        payload: dict[str, Any] = {
            "query": query,
            "vector": embed_query(query),
            "top_k": top_k,
            "fts_weight": fts_weight,
        }
        if date_from is not None:
            payload["date_from"] = date_from.isoformat()
        if date_to is not None:
            payload["date_to"] = date_to.isoformat()
        if video_ids:
            payload["video_ids"] = video_ids
        return payload

    def post_search(payload: dict[str, Any]) -> list[dict[str, Any]]:
        response = httpx.post(
            f"{API_BASE_URL}/search",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return list(response.json()["results"])

    def make_snippet(text: str, max_chars: int = 240) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= max_chars:
            return collapsed
        return collapsed[:max_chars].rstrip() + "…"

    def format_timestamp(seconds: float) -> str:
        total = int(seconds)
        minutes, secs = divmod(total, 60)
        hours, mins = divmod(minutes, 60)
        return f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins}:{secs:02d}"

    return build_payload, format_timestamp, make_snippet, post_search


@app.cell
def _(format_timestamp, make_snippet, mo):

    def result_card(rank: int, row: dict):
        score = row.get("score")
        score_text = f"{float(score):.4f}" if score is not None else "n/a"
        return mo.md(
            f"**{rank}. {row['title']}** ({row['date']}) — "
            f"[▶ {format_timestamp(row['start_seconds'])}]({row['deep_link']}) "
            f"· score `{score_text}`\n\n> {make_snippet(row['text'])}",
        )

    return (result_card,)


@app.cell
def _(build_payload, httpx, is_script_mode, mo, post_search, search_form):

    SAMPLE_QUERIES = [
        "how does dbt transform data in the warehouse",
        "distributed SQL database replication",
        "YugabyteDB",
        "what is a B-tree index",
        "how does query optimization work",
    ]

    jobs: list[tuple[str, int, float, dict]] = []
    if is_script_mode:
        jobs = [(query, 5, 0.5, {}) for query in SAMPLE_QUERIES]
    elif search_form.value is not None:
        form_values = search_form.value
        filters: dict = {}
        if form_values.get("use_date_filter"):
            date_range = form_values.get("date_range")
            if date_range is not None:
                start, end = date_range
                if start is not None:
                    filters["date_from"] = start
                if end is not None:
                    filters["date_to"] = end
        video_ids = form_values.get("lecture_filter") or []
        if video_ids:
            filters["video_ids"] = list(video_ids)
        jobs = [
            (
                form_values["query"],
                int(form_values["top_k"]),
                float(form_values["fts_weight"]),
                filters,
            ),
        ]

    all_results: list[tuple[str, float, list[dict]]] = []
    search_error: str | None = None
    if jobs:

        def run_jobs() -> list[tuple[str, float, list[dict]]]:
            return [
                (
                    query,
                    fts_weight,
                    post_search(build_payload(query, top_k, fts_weight, **filters)),
                )
                for query, top_k, fts_weight, filters in jobs
            ]

        if is_script_mode:
            all_results = run_jobs()
        else:
            with mo.status.spinner(
                title="Searching — the API may be cold-starting on the first "
                "query (~10 s)…",
            ):
                try:
                    all_results = run_jobs()
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    # Network/HTTP/JSON failure — surface a friendly error.
                    search_error = (
                        "Search request failed "
                        f"({type(exc).__name__}: {exc}). The search API may be "
                        "cold-starting or temporarily unavailable — press "
                        "**Search** again in a few seconds."
                    )
    return all_results, search_error


@app.cell
def _(
    all_results,
    is_script_mode,
    mo,
    result_card,
    search_error,
    search_form,
):
    def render_results() -> object:
        if search_error is not None:
            return mo.callout(mo.md(search_error), kind="danger")
        if not all_results:
            if not is_script_mode and search_form.value is None:
                return mo.md("_Enter a query above and press **Search**._")
            return mo.md("_No results._")
        blocks: list[object] = []
        for query, fts_weight, rows in all_results:
            header = mo.md(
                f"### `{query}` — {len(rows)} result(s) "
                f"(hybrid fts_weight={fts_weight})",
            )
            cards = [result_card(rank, row) for rank, row in enumerate(rows, start=1)]
            blocks.append(mo.vstack([header, *cards]))
        return mo.vstack(blocks)

    results_view = render_results()

    if is_script_mode:
        for query, fts_weight, rows in all_results:
            print(f"\n=== {query!r} (fts_weight={fts_weight}) ===")
            for rank, row in enumerate(rows, start=1):
                score = row.get("score")
                score_text = f"{float(score):.4f}" if score is not None else "n/a"
                print(
                    f"{rank}. [{row['date']}] {row['title']} "
                    f"@ {int(row['start_seconds'])}s score={score_text}",
                )
                print(f"   {row['deep_link']}")
                print(f"   {' '.join(row['text'].split())[:120]}")

    results_view
    return


@app.cell
def _(format_timestamp, make_snippet):
    def test_make_snippet_truncates_long_text():
        snippet = make_snippet("word " * 100, max_chars=50)
        assert len(snippet) <= 51
        assert snippet.endswith("…")

    def test_make_snippet_collapses_whitespace():
        assert make_snippet("a\n\n  b\tc") == "a b c"

    def test_format_timestamp():
        assert format_timestamp(0.0) == "0:00"
        assert format_timestamp(65.9) == "1:05"
        assert format_timestamp(3723.0) == "1:02:03"

    return


if __name__ == "__main__":
    app.run()
