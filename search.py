# ============================================================
# Local semantic search over a small text corpus.
#
# Chunking/embedding/retrieval logic is adapted from pdf-rag-from-scratch
# (github.com/regankight/pdf-rag-from-scratch) — same tokenizer-aware
# chunking and cosine-similarity retrieval, re-pointed at a directory of
# plain-text files instead of a single PDF. That drops the PDF-extraction
# dependency and the need for anyone cloning this repo to supply their own
# source document: sample_docs/ ships with the repo, so the server runs
# immediately after `pip install`.
# ============================================================

import os
import re
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    # Imported lazily in build_index(): sentence-transformers pulls in torch,
    # which takes ~2s to import. Importing this module (and so starting the
    # MCP server) shouldn't pay that before the protocol handshake.
    from sentence_transformers import SentenceTransformer

DEFAULT_OVERLAP_TOKENS = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, local — no API key
DEFAULT_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")


class Index(NamedTuple):
    """Everything a search needs: chunk text, its embeddings, the model
    that produced them, and which source file each chunk came from
    (parallel to `chunks`)."""

    chunks: list
    embeddings: np.ndarray
    model: "SentenceTransformer"
    sources: list


def load_corpus(docs_dir):
    """Read every .txt file in docs_dir, sorted for a deterministic chunk
    order. Returns [(filename, text), ...] — files stay separate (rather
    than being joined into one string) so every chunk can be traced back
    to the file it came from.

    Single newlines inside a paragraph are collapsed to spaces, matching
    what pdf-rag-from-scratch's PDF extractor did: hard-wrapped source
    files would otherwise return passages with mid-sentence line breaks.
    Blank lines (paragraph breaks) are preserved for the chunker."""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"No such directory: {docs_dir}")
    docs = []
    for name in sorted(os.listdir(docs_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(docs_dir, name), "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append((name, re.sub(r"(?<!\n)\n(?!\n)", " ", text)))
    if not docs:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")
    return docs


def _find_words(para):
    """(start_char, end_char) for each whitespace-delimited run, in the
    same splitting semantics as str.split(). A token's offset span never
    straddles a whitespace gap, so every token falls entirely within
    exactly one of these spans."""
    return [m.span() for m in re.finditer(r"\S+", para)]


def _split_oversized_paragraph(para, offsets, max_content_tokens, overlap_tokens):
    """Split one over-budget paragraph into overlapping chunks whose cuts
    land on whole-word boundaries, while guaranteeing every chunk's real
    token count stays within max_content_tokens. See
    pdf-rag-from-scratch/rag_project.py for the full rationale (this
    repo's sample_docs are short enough that this path is rarely
    exercised, but it stays in for correctness on larger corpora)."""
    words = _find_words(para)

    token_word_idx = []
    word_first_token = [0] * len(words)
    word_token_counts = [0] * len(words)
    w = 0
    for i, (tok_start, _tok_end) in enumerate(offsets):
        while w + 1 < len(words) and tok_start >= words[w][1]:
            w += 1
        if not token_word_idx or token_word_idx[-1] != w:
            word_first_token[w] = i
        token_word_idx.append(w)
        word_token_counts[w] += 1

    chunks = []
    n_words = len(words)
    start_w = 0
    while start_w < n_words:
        end_w = start_w
        token_total = 0
        while end_w < n_words and token_total + word_token_counts[end_w] <= max_content_tokens:
            token_total += word_token_counts[end_w]
            end_w += 1

        if end_w == start_w:
            # A lone word already exceeds the budget — slice it directly
            # by token offsets, looping until all of its tokens are used.
            first_tok = word_first_token[start_w]
            last_tok = first_tok + word_token_counts[start_w]
            tok_start = first_tok
            while tok_start < last_tok:
                tok_end = min(tok_start + max_content_tokens, last_tok)
                char_start = offsets[tok_start][0]
                char_end = offsets[tok_end - 1][1]
                chunks.append(para[char_start:char_end])
                if tok_end == last_tok:
                    break
                tok_start += max_content_tokens - overlap_tokens
            end_w = start_w + 1
        else:
            char_start = words[start_w][0]
            char_end = words[end_w - 1][1]
            chunks.append(para[char_start:char_end])

        if end_w >= n_words:
            break

        new_start_w = end_w
        back_tokens = 0
        while new_start_w > start_w + 1 and back_tokens < overlap_tokens:
            new_start_w -= 1
            back_tokens += word_token_counts[new_start_w]
        start_w = new_start_w

    return chunks


def chunk_text(text, tokenizer, max_content_tokens, overlap_tokens):
    """Paragraph-aware, tokenizer-aware chunking. Sizes are measured in the
    embedding model's own tokens, not words. A paragraph that fits is kept
    whole; an oversized one is split into overlapping, word-boundary-aligned
    windows."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        encoding = tokenizer(para, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]

        if len(offsets) <= max_content_tokens:
            chunks.append(para)
        else:
            chunks.extend(_split_oversized_paragraph(para, offsets, max_content_tokens, overlap_tokens))

    return chunks


def compute_max_content_tokens(model):
    """The token budget available for chunk *content*: the model's
    max_seq_length minus however many special tokens its tokenizer adds
    automatically."""
    reserved = model.tokenizer.num_special_tokens_to_add(pair=False)
    return model.max_seq_length - reserved


def validate_chunk_fits(chunk, model):
    """Guard against a chunk silently reaching the embedding model
    truncated. Raises ValueError (loud, not silent) if a chunk would
    exceed max_seq_length."""
    token_count = len(model.tokenizer.encode(chunk, add_special_tokens=True))
    if token_count > model.max_seq_length:
        raise ValueError(
            f"Chunk has {token_count} tokens (including special tokens), "
            f"which exceeds {EMBEDDING_MODEL_NAME}'s max_seq_length of "
            f"{model.max_seq_length}. Chunk preview: {chunk[:80]!r}..."
        )


def format_chunk_id(index):
    """chunk_0000, chunk_0001, ... — readable IDs derived from list
    position. Stable only for a fixed (corpus, embedding model, overlap)
    combination."""
    return f"chunk_{index:04d}"


def load_chunks(docs_dir, model, overlap_tokens=DEFAULT_OVERLAP_TOKENS):
    """Load every .txt file in docs_dir and chunk each one against a given
    (already-loaded) embedding model's tokenizer and token budget. Files
    are chunked separately, so a chunk never spans two source files.
    Returns (chunks, sources), parallel lists: sources[i] is the filename
    chunks[i] came from."""
    max_content_tokens = compute_max_content_tokens(model)
    chunks, sources = [], []
    for name, text in load_corpus(docs_dir):
        file_chunks = chunk_text(text, model.tokenizer, max_content_tokens, overlap_tokens)
        chunks.extend(file_chunks)
        sources.extend([name] * len(file_chunks))
    return chunks, sources


def build_index(docs_dir=DEFAULT_DOCS_DIR, overlap_tokens=DEFAULT_OVERLAP_TOKENS, embedding_model_name=EMBEDDING_MODEL_NAME):
    """Load chunks, verify each one actually fits the model, and embed
    them. Returns an Index."""
    from sentence_transformers import SentenceTransformer  # deferred, see top of file

    model = SentenceTransformer(embedding_model_name)
    chunks, sources = load_chunks(docs_dir, model, overlap_tokens)
    for chunk in chunks:
        validate_chunk_fits(chunk, model)
    chunk_embeddings = model.encode(chunks, show_progress_bar=False)
    return Index(chunks, chunk_embeddings, model, sources)


def rank_chunks(query, chunk_embeddings, model, top_n=5):
    """Returns the top_n (chunk_index, cosine_score) pairs, best first.

    Vectorized: one matrix-vector product over L2-normalized embeddings
    instead of a Python loop recomputing every norm per chunk. A
    zero-length vector scores 0.0 rather than producing NaN (which would
    make the sort order undefined). Ties keep corpus order (stable sort).
    Takes the model explicitly rather than reading a module-level global,
    so callers (including tests) control which model a call runs against."""
    embeddings = np.asarray(chunk_embeddings, dtype=float)
    query_embedding = np.asarray(model.encode(query), dtype=float)

    def unit(vectors):
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)

    scores = unit(embeddings) @ unit(query_embedding)
    order = np.argsort(-scores, kind="stable")[:top_n]
    return [(int(i), float(scores[i])) for i in order]


def retrieve_top_chunks(query, chunks, chunk_embeddings, model, top_n=5):
    """Returns the top_n matches as (chunk_id, chunk_text, score) tuples,
    ranked by cosine similarity descending."""
    return [
        (format_chunk_id(i), chunks[i], score)
        for i, score in rank_chunks(query, chunk_embeddings, model, top_n)
    ]
