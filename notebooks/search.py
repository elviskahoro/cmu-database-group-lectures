# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.25.2",
#     "lancedb==0.40.0",
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
# Hybrid search over the CMU Database Group lectures_sample LanceDB table.
# Interactive: uv run marimo run notebooks/search.py
# Headless:    uv run notebooks/search.py  (runs sample queries in script mode)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo

    return Path, mo, os


@app.cell
def _(Path, os):
    REPO_ROOT = Path(__file__).resolve().parent.parent
    LANCEDB_URI = os.environ.get("LANCEDB_URI", "db://cmu-database-group-eqdwgi")
    LANCEDB_REGION = os.environ.get("LANCEDB_REGION", "us-east-1")
    TABLE_NAME = "lectures_sample"
    NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DIMS = 768
    QUERY_PREFIX = "search_query: "

    def load_api_key() -> str | None:
        key = os.environ.get("LANCEDB_API_KEY")
        if key:
            return key
        env_file = REPO_ROOT / ".env.local"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("LANCEDB_API_KEY="):
                    return line.split("=", 1)[1].strip("\"'")
        return None

    LANCEDB_API_KEY = load_api_key()

    return (
        EMBEDDING_DIMS,
        LANCEDB_API_KEY,
        LANCEDB_REGION,
        LANCEDB_URI,
        QUERY_PREFIX,
        TABLE_NAME,
    )


@app.cell
def _(LANCEDB_API_KEY, LANCEDB_URI, mo):
    mo.stop(
        not LANCEDB_URI,
        mo.md(
            "**Missing `LANCEDB_URI`.** Set it in the environment "
            "(e.g. `db://<project-slug>`) and rerun."
        ),
    )
    mo.stop(
        LANCEDB_API_KEY is None,
        mo.md(
            "**Missing `LANCEDB_API_KEY`.** Export it in the environment or add "
            "it to `.env.local` at the repo root, then rerun."
        ),
    )
    return


@app.cell
def _(EMBEDDING_DIMS, QUERY_PREFIX, mo):
    @mo.cache
    def get_embedder():
        from lancedb.embeddings import get_registry

        return get_registry().get("sentence-transformers").create(
            name="nomic-ai/nomic-embed-text-v1.5",
            device="cpu",
            trust_remote_code=True,
        )

    def embed_query(text: str) -> list[float]:
        prefixed = [QUERY_PREFIX + text]
        vector = get_embedder().generate_embeddings(prefixed)[0]
        embedding = [float(value) for value in vector]
        assert len(embedding) == EMBEDDING_DIMS
        return embedding

    return embed_query, get_embedder


@app.cell
def _(LANCEDB_API_KEY, LANCEDB_REGION, LANCEDB_URI, TABLE_NAME, mo):
    import lancedb

    db = lancedb.connect(LANCEDB_URI, region=LANCEDB_REGION, api_key=LANCEDB_API_KEY)
    table = db.open_table(TABLE_NAME)
    row_count = table.count_rows()
    mo.md(f"Connected to `{TABLE_NAME}` — **{row_count}** chunks indexed.")
    return table,


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(mo):
    search_form = (
        mo.md(
            """
            **Search the lectures**

            {query}

            {top_k}

            {fts_weight}
            """
        )
        .batch(
            query=mo.ui.text(
                label="Query",
                value="how does dbt transform data in the warehouse",
                full_width=True,
            ),
            top_k=mo.ui.slider(start=1, stop=10, step=1, value=5, label="Top-k results"),
            fts_weight=mo.ui.slider(
                start=0.0,
                stop=1.0,
                step=0.05,
                value=0.5,
                label="Hybrid weight (0 = pure vector, 1 = pure full-text)",
            ),
        )
        .form(submit_button_label="Search", bordered=True)
    )
    search_form
    return (search_form,)


@app.cell
def _(embed_query, mo, table):
    from typing import Any

    from lancedb.rerankers import MRRReranker

    def run_search(query: str, top_k: int, fts_weight: float) -> list[dict[str, Any]]:
        query_vector = embed_query(query)
        return (
            table.search(query_type="hybrid")
            .vector(query_vector)
            .text(query)
            .rerank(MRRReranker(weight_vector=1.0 - fts_weight, weight_fts=fts_weight))
            .limit(top_k)
            .to_list()
        )

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

    def result_card(rank: int, row: dict[str, Any]) -> Any:
        score = row.get("_relevance_score")
        score_text = f"{float(score):.4f}" if score is not None else "n/a"
        return mo.md(
            f"**{rank}. {row['title']}** ({row['date']}) — "
            f"[▶ {format_timestamp(row['start_seconds'])}]({row['deep_link']}) "
            f"· score `{score_text}`\n\n> {make_snippet(row['text'])}"
        )

    return format_timestamp, make_snippet, result_card, run_search


@app.cell
def _(is_script_mode, mo, run_search, search_form):
    SAMPLE_QUERIES = [
        "how does dbt transform data in the warehouse",
        "distributed SQL database replication",
        "YugabyteDB",
        "what is a B-tree index",
        "how does query optimization work",
    ]

    if is_script_mode:
        jobs = [(query, 5, 0.5) for query in SAMPLE_QUERIES]
    elif search_form.value is not None:
        form_values = search_form.value
        jobs = [
            (
                form_values["query"],
                int(form_values["top_k"]),
                float(form_values["fts_weight"]),
            )
        ]
    else:
        jobs = []

    all_results = [
        (query, fts_weight, run_search(query, top_k, fts_weight))
        for query, top_k, fts_weight in jobs
    ]
    return all_results,


@app.cell
def _(all_results, is_script_mode, mo, result_card, search_form):
    def render_results() -> object:
        if not all_results:
            if search_form.value is None:
                return mo.md("_Enter a query above and press **Search**._")
            return mo.md("_No results._")
        blocks: list[object] = []
        for query, fts_weight, rows in all_results:
            header = mo.md(
                f"### `{query}` — {len(rows)} result(s) "
                f"(hybrid fts_weight={fts_weight})"
            )
            cards = [result_card(rank, row) for rank, row in enumerate(rows, start=1)]
            blocks.append(mo.vstack([header, *cards]))
        return mo.vstack(blocks)

    results_view = render_results()

    if is_script_mode:
        for query, fts_weight, rows in all_results:
            print(f"\n=== {query!r} (fts_weight={fts_weight}) ===")
            for rank, row in enumerate(rows, start=1):
                score = row.get("_relevance_score")
                score_text = f"{float(score):.4f}" if score is not None else "n/a"
                print(
                    f"{rank}. [{row['date']}] {row['title']} "
                    f"@ {int(row['start_seconds'])}s score={score_text}"
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
