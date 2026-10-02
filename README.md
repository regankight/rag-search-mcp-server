# RAG Search MCP Server

A minimal [MCP](https://modelcontextprotocol.io) server that exposes semantic search over a local
document corpus as a single tool an agent can call: `search_documents(query, top_k)`.

No external services, no API keys, no database. Clone it, install dependencies, point an MCP
client at it, and it works immediately against the small bundled corpus.

## Why this project exists

An agent is only as useful as the tools it can reach. This project demonstrates the other half of
retrieval work — not building a RAG pipeline, but **exposing one as agent-callable infrastructure**
via the protocol real agent clients (Claude Desktop, Claude Code, and others) actually speak.

It deliberately reuses retrieval logic already built and tested in
[`pdf-rag-from-scratch`](https://github.com/regankight/pdf-rag-from-scratch) — tokenizer-aware
chunking, local embeddings, cosine-similarity retrieval — rather than re-deriving it, and re-points
it at a directory of plain-text files instead of a single PDF so the repo needs no external corpus
supplied by whoever clones it.

## What it does

```text
query
  ↓
embed query (same model used to embed the corpus)
  ↓
cosine similarity against every chunk
  ↓
top_k chunks, ranked, with scores
```

One tool, one job: retrieval. It does not generate an answer — that's a deliberate scope boundary,
not a missing feature (see [Design decisions](#design-decisions)).

## Install and run

```bash
git clone https://github.com/regankight/rag-search-mcp-server.git
cd rag-search-mcp-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run it directly (stdio transport — what MCP clients expect):

```bash
python mcp_server.py
```

To use it from an MCP client (e.g. Claude Desktop or Claude Code), point the client's MCP config at
this command, e.g.:

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "/absolute/path/to/rag-search-mcp-server/.venv/bin/python",
      "args": ["/absolute/path/to/rag-search-mcp-server/mcp_server.py"]
    }
  }
}
```

The first call downloads the embedding model (`all-MiniLM-L6-v2`, ~90MB) from Hugging Face and
caches it locally — that call will be slow; every call after is fast.

## Example

Asking an MCP-connected agent to search for *"how does combining keyword and vector search work?"*
returns:

```json
[
  {
    "chunk_id": "chunk_0001",
    "text": "Hybrid search combines two different ways of finding relevant documents...",
    "score": 0.5463
  }
]
```

— correctly surfacing the hybrid-search document over the four other unrelated ones in the sample
corpus, by meaning rather than keyword overlap (the query shares almost no exact words with the
matched passage).

## Design decisions

### Why a bundled sample corpus instead of requiring the user's own documents
An MCP server as a portfolio piece only proves something if a reviewer can actually run it. Making
them supply their own PDF/corpus before they can see it work is friction that most people won't
push through. `sample_docs/` ships five short, self-written passages on RAG-adjacent topics
(retrieval, hybrid search, observability, evaluation, embeddings) specifically so the demo works
out of the box — clone, install, run, query, see a correct result. Point `DEFAULT_DOCS_DIR` in
`search.py` at any other directory of `.txt` files to search your own corpus instead.

### Why these are original, self-written sample documents, not a real dataset
`pdf-rag-from-scratch`'s own benchmark PDF is gitignored there deliberately — its license for
redistribution was never verified, so it was never committed to begin with. Carrying it (or any
other real corpus with unclear reuse rights) into a second public repo just to have a bigger demo
corpus wasn't worth that risk for an artifact whose only job is to demonstrate that search works,
not to search anything in particular.

### Why no generation step
Adding an LLM call on top of retrieval would turn this into a second, smaller copy of
`pdf-rag-from-scratch` rather than a distinct artifact. The point of *this* repo is specifically
the MCP-exposure layer — giving an agent a callable search tool — not another retrieval pipeline.
`pdf-rag-from-scratch` already covers retrieval-to-answer; this repo covers retrieval-as-a-tool.

### Why a lazily-built, process-lifetime index instead of a vector database
The corpus is five short files. A linear cosine-similarity scan over the resulting handful of
chunks is simpler and faster than standing up a vector database for a dataset with no scaling
problem to solve. The index builds once, on the first tool call, and is kept in memory for the
life of the server process — reasonable for one corpus served to one client connection, which is
the scope this project targets.

### Why the retrieval logic is copied, not imported, from `pdf-rag-from-scratch`
Making this repo depend on another one (git submodule, `pip install` from a git URL) would mean a
reviewer has to clone two repos just to run one. Copying the small amount of needed logic keeps
this repo standalone and clone-and-run, at the cost of the two copies drifting if the original is
later changed — an acceptable tradeoff for a demo-scale artifact, not something to do for
production code shared across real services.

### Why `mcp>=2,<3` is pinned, and why that matters here specifically
The MCP Python SDK had a breaking change on 2026-07-28 (v2.0.0): the server class moved from
`mcp.server.fastmcp.FastMCP` to `mcp.server.MCPServer`, with the old import path removed outright,
not deprecated. Code written against the pre-July-2026 tutorials (`from mcp.server.fastmcp import
FastMCP`) fails immediately on a fresh install with `ModuleNotFoundError` once `pip install mcp`
resolves to 2.x. This repo targets the current v2 API (`from mcp.server import MCPServer`,
`mcp.run(transport="stdio")` with an explicit transport) and pins accordingly, rather than silently
breaking for the next person who clones it after the SDK moves again.

### Why `search()` is a plain function, separate from the `@mcp.tool()`-decorated one
`mcp_server.search_documents` is a thin wrapper around `mcp_server.search()`. Testing against the
decorated tool directly would couple the test suite to the MCP SDK's decorator internals; testing
the plain function underneath doesn't. The decorator's only job is exposing `search()` over the
protocol — correct by inspection, not something worth testing twice.

### Why two tiers of tests
Fast tests use a stub model (fixed vectors keyed by exact text) so ranking and formatting logic run
in milliseconds with no model load. One slower test loads the real embedding model and the real
bundled corpus and checks an actual query surfaces the right document — catching anything the stub
can't, like a tokenizer edge case or the sample corpus failing to load.

## Project structure

```text
rag-search-mcp-server/
├── mcp_server.py      # the MCP server: index cache + the search_documents tool
├── search.py           # chunking / embedding / cosine-similarity retrieval (adapted from pdf-rag-from-scratch)
├── sample_docs/         # 5 short, self-written passages the demo searches by default
├── tests/
│   └── test_search.py
├── requirements.txt
├── pytest.ini
├── LICENSE
└── README.md
```

## Running the tests

```bash
pytest tests/ -v
```

## What this project demonstrates

- Building an MCP server against the current (post-2026-07-28) MCP Python SDK API
- Exposing retrieval as an agent-callable tool, not just a pipeline stage
- Reusing proven retrieval logic across projects without inter-repo coupling
- Scoping a demo to be runnable with zero external setup
- Testing tool logic independent of the protocol/decorator layer it's served through

## Scope

This is a deliberately small, single-tool MCP server for demonstration purposes. It is not a
general-purpose document-search service: no auth, no multi-corpus support, no persistence beyond
the process lifetime, no concurrent-write handling. Adding those would solve problems this project
doesn't have.
