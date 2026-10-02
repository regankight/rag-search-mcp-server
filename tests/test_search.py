# ============================================================
# Tests for search.py and mcp_server.py.
#
# Three layers, deliberately kept separate:
#   - Fast unit tests using a tiny fake "model" (no real embeddings
#     loaded) for ranking, corpus loading, and the server's search logic.
#   - Chunking tests against the real embedding model's tokenizer — the
#     chunker's whole job is to agree with *that* tokenizer's limits, so a
#     mock could pass while the real one still truncates.
#   - Slow end-to-end tests: build the real index over the bundled corpus,
#     and talk to the server over real stdio with an MCP client.
#
# Run: pytest tests/ -v
# ============================================================

import asyncio
import json
import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mcp_server  # noqa: E402
from search import (  # noqa: E402
    DEFAULT_DOCS_DIR,
    EMBEDDING_MODEL_NAME,
    Index,
    _split_oversized_paragraph,
    build_index,
    chunk_text,
    compute_max_content_tokens,
    load_corpus,
    rank_chunks,
    retrieve_top_chunks,
    validate_chunk_fits,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


class _FakeModel:
    """Stands in for a loaded SentenceTransformer: encode() returns a
    pre-set vector per text, keyed by exact string match. Ranking only
    ever calls model.encode(query) — chunk embeddings are passed in
    separately — so this is all a test needs to exercise it without
    loading any real model."""

    def __init__(self, vectors_by_text):
        self._vectors = vectors_by_text

    def encode(self, text):
        return self._vectors[text]


# --- ranking ------------------------------------------------------------

def test_rank_chunks_scores_identical_vectors_one():
    ranked = rank_chunks("q", [np.array([1.0, 2.0, 3.0])], _FakeModel({"q": np.array([1.0, 2.0, 3.0])}))
    assert ranked[0][1] == pytest.approx(1.0)


def test_rank_chunks_scores_orthogonal_vectors_zero():
    ranked = rank_chunks("q", [np.array([0.0, 1.0])], _FakeModel({"q": np.array([1.0, 0.0])}))
    assert ranked[0][1] == pytest.approx(0.0)


def test_rank_chunks_scores_opposite_vectors_minus_one():
    ranked = rank_chunks("q", [np.array([-1.0, 0.0])], _FakeModel({"q": np.array([1.0, 0.0])}))
    assert ranked[0][1] == pytest.approx(-1.0)


def test_rank_chunks_ignores_vector_length():
    # Cosine similarity is about direction, not magnitude.
    ranked = rank_chunks("q", [np.array([10.0, 0.0]), np.array([0.1, 0.1])], _FakeModel({"q": np.array([1.0, 0.0])}))
    assert ranked[0][0] == 0 and ranked[0][1] == pytest.approx(1.0)


def test_rank_chunks_zero_vector_scores_zero_not_nan():
    ranked = rank_chunks("q", [np.array([0.0, 0.0]), np.array([1.0, 0.0])], _FakeModel({"q": np.array([1.0, 0.0])}))
    assert [i for i, _ in ranked] == [1, 0]
    assert ranked[1][1] == 0.0  # a real number, so ordering is well-defined


def test_rank_chunks_ties_keep_corpus_order():
    same = np.array([1.0, 0.0])
    ranked = rank_chunks("q", [same, same, same], _FakeModel({"q": same}))
    assert [i for i, _ in ranked] == [0, 1, 2]


def test_retrieve_top_chunks_ranks_by_similarity_descending():
    chunks = ["about cats", "about dogs", "about the french revolution"]
    chunk_embeddings = [
        np.array([1.0, 0.0]),  # cats
        np.array([0.9, 0.1]),  # dogs — close to cats
        np.array([0.0, 1.0]),  # unrelated
    ]
    model = _FakeModel({"tell me about cats": np.array([1.0, 0.0])})

    results = retrieve_top_chunks("tell me about cats", chunks, chunk_embeddings, model, top_n=2)

    assert [r[1] for r in results] == ["about cats", "about dogs"]
    assert results[0][2] > results[1][2]


def test_retrieve_top_chunks_respects_top_n():
    chunks = ["a", "b", "c", "d"]
    chunk_embeddings = [np.array([1.0]), np.array([0.8]), np.array([0.5]), np.array([0.1])]
    model = _FakeModel({"q": np.array([1.0])})

    assert len(retrieve_top_chunks("q", chunks, chunk_embeddings, model, top_n=2)) == 2


def test_retrieve_top_chunks_ids_are_readable_and_positional():
    chunks = ["x", "y"]
    chunk_embeddings = [np.array([1.0]), np.array([-1.0])]
    model = _FakeModel({"q": np.array([1.0])})

    results = retrieve_top_chunks("q", chunks, chunk_embeddings, model, top_n=2)

    assert {chunk_id for chunk_id, _text, _score in results} == {"chunk_0000", "chunk_0001"}


# --- corpus loading -----------------------------------------------------

def test_load_corpus_keeps_files_separate_and_ignores_non_txt(tmp_path):
    (tmp_path / "a.txt").write_text("First file.")
    (tmp_path / "b.txt").write_text("Second file.")
    (tmp_path / "ignored.md").write_text("Should not be read.")

    assert load_corpus(str(tmp_path)) == [("a.txt", "First file."), ("b.txt", "Second file.")]


def test_load_corpus_collapses_hard_wraps_but_keeps_paragraph_breaks(tmp_path):
    (tmp_path / "a.txt").write_text("one line\nwrapped here\n\nsecond paragraph")

    (_name, text), = load_corpus(str(tmp_path))

    assert text == "one line wrapped here\n\nsecond paragraph"


def test_load_corpus_raises_on_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_corpus("/no/such/directory")


def test_load_corpus_raises_when_no_txt_files(tmp_path):
    (tmp_path / "readme.md").write_text("not a txt file")
    with pytest.raises(FileNotFoundError):
        load_corpus(str(tmp_path))


# --- chunking (real tokenizer) -------------------------------------------
# Ported from pdf-rag-from-scratch's tests, trimmed to the behaviors that
# matter for this repo: paragraphs that fit stay whole, oversized ones are
# split into overlapping, word-aligned, within-budget windows, and the
# pre-embedding guard rejects anything that would be silently truncated.

@pytest.fixture(scope="module")
def model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="module")
def max_content_tokens(model):
    return compute_max_content_tokens(model)


def token_count(text, model, add_special_tokens=False):
    return len(model.tokenizer.encode(text, add_special_tokens=add_special_tokens))


_COMMON_WORDS = (
    "the quick brown fox jumps over lazy dog while a small cat "
    "sleeps near warm sunny window during long slow afternoon"
).split()


def make_paragraph(word_count):
    """Real dictionary words, each a single token in this model's
    vocabulary, so word count and token count stay aligned 1:1."""
    return " ".join(_COMMON_WORDS[i % len(_COMMON_WORDS)] for i in range(word_count))


def stress_paragraph(word_count):
    """Words like "item47" that WordPiece splits into several sub-word
    tokens — the shape of text where a raw token-index cut can land
    inside a word."""
    return " ".join(f"item{i}" for i in range(word_count))


def test_paragraph_that_fits_is_kept_whole(model, max_content_tokens):
    para = "Supervised learning uses labeled data to train a model."
    assert chunk_text(para, model.tokenizer, max_content_tokens, overlap_tokens=10) == [para]


def test_adjacent_short_paragraphs_each_become_their_own_chunk(model, max_content_tokens):
    paragraphs = ["Regression predicts a number.", "Clustering groups similar points."]
    chunks = chunk_text("\n\n".join(paragraphs), model.tokenizer, max_content_tokens, overlap_tokens=10)
    assert chunks == paragraphs


def test_oversized_paragraph_is_split_within_budget(model, max_content_tokens):
    chunks = chunk_text(make_paragraph(max_content_tokens * 3), model.tokenizer, max_content_tokens, overlap_tokens=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert token_count(chunk, model, add_special_tokens=True) <= model.max_seq_length


def test_oversized_paragraph_chunks_overlap(model, max_content_tokens):
    overlap_tokens = 20
    chunks = chunk_text(make_paragraph(max_content_tokens * 2), model.tokenizer, max_content_tokens, overlap_tokens)

    assert len(chunks) >= 2
    # One word == one token here and both chunks are full length, so the
    # overlap is exactly the last N words of one == the first N of the next.
    assert chunks[0].split()[-overlap_tokens:] == chunks[1].split()[:overlap_tokens]


def test_oversized_paragraph_splits_only_on_word_boundaries(model, max_content_tokens):
    para = stress_paragraph(max_content_tokens * 3)
    chunks = chunk_text(para, model.tokenizer, max_content_tokens, overlap_tokens=15)

    assert len(chunks) > 1
    cursor = 0
    for chunk in chunks:
        idx = para.index(chunk, cursor)
        end = idx + len(chunk)
        assert idx == 0 or para[idx - 1].isspace(), f"chunk starts mid-word: {chunk[:30]!r}"
        assert end == len(para) or para[end].isspace(), f"chunk ends mid-word: {chunk[-30:]!r}"
        cursor = idx


def test_zero_overlap_chunks_are_contiguous(model, max_content_tokens):
    para = stress_paragraph(max_content_tokens * 3)
    chunks = chunk_text(para, model.tokenizer, max_content_tokens, overlap_tokens=0)

    cursor, prev_end = 0, None
    for chunk in chunks:
        idx = para.index(chunk, cursor)
        if prev_end is not None:
            assert para[prev_end:idx] == " "  # no gap, no overlap, nothing dropped
        prev_end, cursor = idx + len(chunk), idx


def test_single_giant_word_fallback_covers_the_whole_word():
    # Real text can't reach this branch with this tokenizer (WordPiece caps
    # word length), so exercise it directly with hand-built offsets.
    para = "0123456789abcdefghijklmnopqrst"
    offsets = [(i, i + 1) for i in range(30)]

    chunks = _split_oversized_paragraph(para, offsets, 10, overlap_tokens=2)

    assert chunks == ["0123456789", "89abcdefgh", "ghijklmnop", "opqrst"]


def test_validate_chunk_fits_rejects_oversized_and_accepts_normal(model):
    validate_chunk_fits("A short chunk that comfortably fits the model.", model)
    with pytest.raises(ValueError):
        validate_chunk_fits(make_paragraph(model.max_seq_length * 3), model)


# --- mcp_server.search(): the tool's logic, decorator-free ---------------

def _fake_index(chunks, embeddings, vectors_by_text, sources=None):
    return Index(chunks, embeddings, _FakeModel(vectors_by_text), sources or [f"doc{i}.txt" for i in range(len(chunks))])


def test_search_returns_text_source_and_score(monkeypatch):
    index = _fake_index(
        ["about cats", "about dogs"],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        {"cats": np.array([1.0, 0.0])},
        sources=["cats.txt", "dogs.txt"],
    )
    monkeypatch.setattr(mcp_server, "get_index", lambda: index)

    assert mcp_server.search("cats", top_k=1) == [
        {"chunk_id": "chunk_0000", "source": "cats.txt", "text": "about cats", "score": 1.0}
    ]


def test_search_clamps_top_k_to_corpus_size(monkeypatch):
    index = _fake_index(["a", "b"], [np.array([1.0]), np.array([0.5])], {"q": np.array([1.0])})
    monkeypatch.setattr(mcp_server, "get_index", lambda: index)

    assert len(mcp_server.search("q", top_k=50)) == 2


def test_search_returns_at_least_one_result_for_top_k_below_one(monkeypatch):
    index = _fake_index(["a"], [np.array([1.0])], {"q": np.array([1.0])})
    monkeypatch.setattr(mcp_server, "get_index", lambda: index)

    assert len(mcp_server.search("q", top_k=0)) == 1


def test_get_index_builds_once_under_concurrent_first_calls(monkeypatch):
    calls = []

    def slow_build(_docs_dir):
        calls.append(1)
        time.sleep(0.2)  # long enough for the other threads to arrive
        return "the-index"

    monkeypatch.setattr(mcp_server, "build_index", slow_build)
    monkeypatch.setattr(mcp_server, "_index", None)

    results = []
    threads = [threading.Thread(target=lambda: results.append(mcp_server.get_index())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert results == ["the-index"] * 8


# --- slow end-to-end: real model, real corpus, real stdio ----------------

@pytest.mark.slow
def test_build_index_and_search_bundled_sample_docs():
    """A query about meaning, worded differently from the passage, should
    surface the right document — the thing a fake-model test can't verify."""
    index = build_index(DEFAULT_DOCS_DIR)

    assert len(index.chunks) >= 5
    assert set(index.sources) >= {"hybrid_search.txt", "observability.txt"}

    ranked = rank_chunks(
        "catching quality getting worse over time before users complain",
        index.embeddings,
        index.model,
        top_n=1,
    )
    assert index.sources[ranked[0][0]] == "observability.txt"


@pytest.mark.slow
def test_server_works_over_real_stdio():
    """Starts the server as a subprocess and talks to it with an MCP
    client — the protocol layer (tool registration, schema, call) that
    the unit tests above deliberately bypass."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=[os.path.join(REPO_ROOT, "mcp_server.py")]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                result = await session.call_tool(
                    "search_documents", {"query": "what is a vector embedding?", "top_k": 2}
                )
                return tools, result

    tools, result = asyncio.run(run())

    assert [t.name for t in tools] == ["search_documents"]
    assert tools[0].input_schema["required"] == ["query"]
    hits = [json.loads(c.text) for c in result.content]
    assert len(hits) == 2
    assert hits[0]["source"] == "vector_embeddings.txt"
    assert set(hits[0]) == {"chunk_id", "source", "text", "score"}
