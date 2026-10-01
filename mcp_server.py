# ============================================================
# Minimal MCP server exposing one tool: semantic search over a local
# document corpus.
#
# The index (embedding model + embedded chunks) is built once, lazily, on
# the first search call — not per request, and not at import time (so
# importing this module for tests never triggers a model download/load).
#
# Uses the MCP Python SDK's v2 API (mcp>=2,<3): the server class moved
# from mcp.server.fastmcp.FastMCP to mcp.server.MCPServer in the SDK's
# July 2026 2.0 release. See README "Design decisions" for why this
# matters and what breaks on the wrong version.
# ============================================================

from mcp.server import MCPServer

from search import DEFAULT_DOCS_DIR, build_index, retrieve_top_chunks

mcp = MCPServer("rag-search")

_index = None  # (chunks, chunk_embeddings, model), built on first use


def get_index():
    """Build and cache the index on first call. A module-level cache
    (rather than e.g. a class) is enough here: this process serves one
    corpus for its whole lifetime, and MCP spins up one server process
    per client connection."""
    global _index
    if _index is None:
        _index = build_index(DEFAULT_DOCS_DIR)
    return _index


def search(query, top_k=5):
    """The actual search logic, kept as a plain function so it can be
    unit-tested directly without going through the MCP tool-call
    machinery. Returns a list of {chunk_id, text, score} dicts, ranked by
    relevance descending."""
    chunks, chunk_embeddings, model = get_index()
    top_k = max(1, min(top_k, len(chunks)))
    matches = retrieve_top_chunks(query, chunks, chunk_embeddings, model, top_n=top_k)
    return [
        {"chunk_id": chunk_id, "text": text, "score": round(float(score), 4)}
        for chunk_id, text, score in matches
    ]


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search the local document corpus and return the most semantically
    relevant passages, ranked by cosine-similarity score (0-1, higher is
    more relevant). Use this to find passages that answer or relate to a
    natural-language question — it does not generate an answer itself,
    only retrieves supporting text."""
    return search(query, top_k)


if __name__ == "__main__":
    mcp.run(transport="stdio")
