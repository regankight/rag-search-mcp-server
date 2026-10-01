# ============================================================
# Tests for search.py and the mcp_server.search() wrapper.
#
# Two layers, deliberately kept separate:
#   - Fast unit tests using a tiny fake "model" (no real embeddings
#     loaded) to check ranking/formatting logic in milliseconds.
#   - One real integration test that loads the actual embedding model and
#     the bundled sample_docs/, to catch anything the fake can't (model
#     load failures, tokenizer edge cases, the corpus actually existing).
#
# Run: pytest tests/ -v
# ============================================================

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search import DEFAULT_DOCS_DIR, build_index, cosine_similarity, load_corpus_text, retrieve_top_chunks


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


class _FakeModel:
    """Stands in for a loaded SentenceTransformer: encode() returns a
    pre-set vector per text, keyed by exact string match. retrieve_top_
    chunks only ever calls model.encode(query) — chunk embeddings are
    passed in separately — so this is all a test needs to exercise it
    without loading any real model."""

    def __init__(self, vectors_by_text):
        self._vectors = vectors_by_text

    def encode(self, text):
        return self._vectors[text]


def test_retrieve_top_chunks_ranks_by_similarity_descending():
    chunks = ["about cats", "about dogs", "about the french revolution"]
    chunk_embeddings = [
        np.array([1.0, 0.0]),   # cats
        np.array([0.9, 0.1]),   # dogs — close to cats
        np.array([0.0, 1.0]),   # unrelated
    ]
    model = _FakeModel({"tell me about cats": np.array([1.0, 0.0])})

    results = retrieve_top_chunks("tell me about cats", chunks, chunk_embeddings, model, top_n=2)

    assert [r[1] for r in results] == ["about cats", "about dogs"]
    assert results[0][2] > results[1][2]


def test_retrieve_top_chunks_respects_top_n():
    chunks = ["a", "b", "c", "d"]
    chunk_embeddings = [np.array([1.0]), np.array([0.8]), np.array([0.5]), np.array([0.1])]
    model = _FakeModel({"q": np.array([1.0])})

    results = retrieve_top_chunks("q", chunks, chunk_embeddings, model, top_n=2)

    assert len(results) == 2


def test_retrieve_top_chunks_ids_are_readable_and_positional():
    chunks = ["x", "y"]
    chunk_embeddings = [np.array([1.0]), np.array([-1.0])]
    model = _FakeModel({"q": np.array([1.0])})

    results = retrieve_top_chunks("q", chunks, chunk_embeddings, model, top_n=2)

    ids = {chunk_id for chunk_id, _text, _score in results}
    assert ids == {"chunk_0000", "chunk_0001"}


def test_load_corpus_text_joins_files_with_blank_line_separator(tmp_path):
    (tmp_path / "a.txt").write_text("First file.")
    (tmp_path / "b.txt").write_text("Second file.")
    (tmp_path / "ignored.md").write_text("Should not be read.")

    text = load_corpus_text(str(tmp_path))

    assert "First file." in text
    assert "Second file." in text
    assert "Should not be read." not in text
    assert "\n\n" in text


def test_load_corpus_text_raises_on_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_corpus_text("/no/such/directory")


def test_load_corpus_text_raises_when_no_txt_files(tmp_path):
    (tmp_path / "readme.md").write_text("not a txt file")
    with pytest.raises(FileNotFoundError):
        load_corpus_text(str(tmp_path))


# --- Real integration test: actual model, actual bundled corpus --------

@pytest.mark.slow
def test_build_index_and_search_bundled_sample_docs():
    """Loads the real embedding model and the real sample_docs/ shipped
    with this repo, and checks that a query about hybrid search actually
    surfaces the hybrid-search document above unrelated ones — the thing
    a fake-model test can't verify."""
    chunks, chunk_embeddings, model = build_index(DEFAULT_DOCS_DIR)

    assert len(chunks) >= 5  # at least one chunk per sample doc

    results = retrieve_top_chunks(
        "how does combining keyword and vector search work?",
        chunks,
        chunk_embeddings,
        model,
        top_n=1,
    )

    assert "reciprocal rank fusion" in results[0][1].lower() or "hybrid" in results[0][1].lower()


# --- mcp_server.search(): the tool's logic, decorator-free -------------

import mcp_server  # noqa: E402  (import after sys.path.insert above)


def test_mcp_server_search_formats_results_as_dicts(monkeypatch):
    chunks = ["about cats", "about dogs"]
    chunk_embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    model = _FakeModel({"cats": np.array([1.0, 0.0])})
    monkeypatch.setattr(mcp_server, "get_index", lambda: (chunks, chunk_embeddings, model))

    results = mcp_server.search("cats", top_k=1)

    assert results == [{"chunk_id": "chunk_0000", "text": "about cats", "score": 1.0}]


def test_mcp_server_search_clamps_top_k_to_corpus_size(monkeypatch):
    chunks = ["a", "b"]
    chunk_embeddings = [np.array([1.0]), np.array([0.5])]
    model = _FakeModel({"q": np.array([1.0])})
    monkeypatch.setattr(mcp_server, "get_index", lambda: (chunks, chunk_embeddings, model))

    results = mcp_server.search("q", top_k=50)

    assert len(results) == 2  # clamped to corpus size, not 50


def test_mcp_server_search_clamps_top_k_below_one():
    # top_k=0 (or negative) should still return at least one result, not
    # an empty list or a crash from retrieve_top_chunks(top_n=0) slicing.
    chunks = ["a"]
    chunk_embeddings = [np.array([1.0])]
    model = _FakeModel({"q": np.array([1.0])})

    import mcp_server as m
    m._index = (chunks, chunk_embeddings, model)
    try:
        results = m.search("q", top_k=0)
        assert len(results) == 1
    finally:
        m._index = None  # don't leak state into other tests
