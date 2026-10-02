# ============================================================
# Minimal MCP server exposing one tool: semantic search over a local
# document corpus.
#
# The index (embedding model + embedded chunks) is built once per process.
# `python mcp_server.py` starts building it in a background thread right
# away, so the server can answer the client's handshake without waiting
# for the model to load (the first run also downloads it). The heavy
# import (torch, via sentence-transformers) is deferred into that thread
# too. A search that arrives before the build finishes simply waits for
# it. Importing this module (e.g. in tests) builds nothing, downloads
# nothing, and doesn't import torch.
#
# Uses the MCP Python SDK's v2 API (mcp>=2,<3): the server class moved
# from mcp.server.fastmcp.FastMCP to mcp.server.MCPServer in the SDK's
# July 2026 2.0 release. See README "Design decisions" for why this
# matters and what breaks on the wrong version.
# ============================================================

import threading

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from search import DEFAULT_DOCS_DIR, build_index, rank_chunks, format_chunk_id

mcp = MCPServer("rag-search")

_index = None  # a search.Index, built on first use
_index_lock = threading.Lock()


def get_index():
    """Build the index on first call and cache it for the life of the
    process (one server process serves one corpus; MCP clients start one
    process per connection).

    The lock matters: the SDK runs sync tool functions on worker threads,
    so two searches arriving together would otherwise both see an empty
    cache and each load the embedding model. Checked again inside the lock
    so the second caller reuses the first caller's result."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = build_index(DEFAULT_DOCS_DIR)
    return _index


def search(query, top_k=5):
    """The actual search logic, kept as a plain function so it can be
    unit-tested directly without going through the MCP tool-call
    machinery. Returns a list of {chunk_id, source, text, score} dicts,
    ranked by relevance descending. Raises ValueError for a blank query:
    embedding an empty string still ranks every passage, and the near-zero
    scores would look like (meaningless) hits to a caller."""
    if not query.strip():
        raise ValueError("query must not be empty or whitespace")
    index = get_index()
    top_k = max(1, min(top_k, len(index.chunks)))
    ranked = rank_chunks(query, index.embeddings, index.model, top_n=top_k)
    return [
        {
            "chunk_id": format_chunk_id(i),
            "source": index.sources[i],
            "text": index.chunks[i],
            "score": round(score, 4),
        }
        for i, score in ranked
    ]


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search the local document corpus and return the most semantically
    relevant passages, best first. Each result has the passage text, the
    source file it came from, and a cosine-similarity score (higher is
    more relevant; in practice roughly 0-1 for this model, though cosine
    can range from -1 to 1). top_k is clamped to between 1 and the number
    of passages in the corpus; queries longer than the embedding model's
    256-token limit are truncated. Use this to find passages that answer or
    relate to a natural-language question — it does not generate an answer
    itself, only retrieves supporting text."""
    try:
        return search(query, top_k)
    except ValueError as exc:
        # The SDK hides ordinary exceptions from the model (it sees only
        # "Error executing tool ..."); ToolError is how to say *why*.
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    # Warm the index in the background; errors surface on the next search
    # (get_index retries) rather than killing the server.
    threading.Thread(target=get_index, daemon=True).start()
    mcp.run(transport="stdio")
