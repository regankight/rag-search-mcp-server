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
top_k chunks, ranked, each with its source file and score
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

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90MB) from Hugging Face and
caches it locally. The server loads the model and builds the index in a background thread at
startup, so it answers the client's handshake immediately; a search that arrives before the index
is ready waits for it. Later runs start from the local cache.

## Example

Asking an MCP-connected agent to search for *"catching quality getting worse over time before users
complain"* returns (text shortened here):

```json
[
  {
    "chunk_id": "chunk_0002",
    "source": "observability.txt",
    "text": "Observability for a language model application means being able to see...",
    "score": 0.3443
  },
  {
    "chunk_id": "chunk_0000",
    "source": "evaluation_frameworks.txt",
    "text": "Evaluating a retrieval system means measuring whether the documents it...",
    "score": 0.192
  }
]
```

The observability passage ranks first by a wide margin. It talks about noticing "quality drift
before a user complains" — the query and the passage share only a couple of generic words
(`quality`, `before`); the match comes from meaning, not from the query's wording. (Numbers are from
a real run against the bundled corpus; scores shift slightly with the embedding model.)

## Design decisions

### Why a bundled sample corpus instead of requiring the user's own documents
An MCP server as a portfolio piece only proves something if a reviewer can actually run it. Making
them supply their own PDF/corpus before they can see it work is friction that most people won't
push through. `sample_docs/` ships five short, self-written passages on RAG-adjacent topics
(retrieval, hybrid search, observability, evaluation, embeddings) specifically so the demo works
out of the box — clone, install, run, query, see a correct result. Point `DEFAULT_DOCS_DIR` in
`search.py` at any other directory of `.txt` files to search your own corpus instead.

### Why these are original, self-written sample documents, not a real dataset
`pdf-rag-from-scratch`'s own benchmark PDF isn't committed there (`data/*.pdf` is gitignored), and
its license for redistribution was never verified. Carrying it (or any
other real corpus with unclear reuse rights) into a second public repo just to have a bigger demo
corpus wasn't worth that risk for an artifact whose only job is to demonstrate that search works,
not to search anything in particular.

### Why no generation step
Adding an LLM call on top of retrieval would turn this into a second, smaller copy of
`pdf-rag-from-scratch` rather than a distinct artifact. The point of *this* repo is specifically
the MCP-exposure layer — giving an agent a callable search tool — not another retrieval pipeline.
`pdf-rag-from-scratch` already covers retrieval-to-answer; this repo covers retrieval-as-a-tool.

### Why an in-memory index instead of a vector database
The corpus is five short files. A linear cosine-similarity scan over the resulting handful of
chunks is simpler and faster than standing up a vector database for a dataset with no scaling
problem to solve. The index builds once per server process and is kept in memory for the
life of that process — reasonable for one corpus served to one client connection, which is
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

### Why the index is built in a background thread, behind a lock
Loading the model (and, on a first run, downloading it) takes seconds to tens of seconds. Doing it
inside the first tool call risks a client timeout on exactly the call a reviewer makes first; doing
it before the server starts would delay the protocol handshake instead. So `python mcp_server.py`
starts the build in a background thread and serves immediately. A search that arrives early waits
on the same build. The lock exists because the MCP SDK runs sync tool functions on worker threads:
without it, two searches arriving together would each load the model. A test starts eight threads
at once and asserts the index is built exactly once. Importing the module builds nothing, so tests
never trigger a download.

### Why each result carries its source file
A passage with only a positional `chunk_id` can't be cited or checked — and a retrieval tool exists
so an agent can ground an answer in something. Files are chunked separately (a chunk never spans
two files), so each chunk maps to exactly one source filename, returned alongside the text.

### Why `search()` is a plain function, separate from the `@mcp.tool()`-decorated one
`mcp_server.search_documents` is a thin wrapper around `mcp_server.search()`. Testing against the
decorated tool directly would couple the test suite to the MCP SDK's decorator internals; testing
the plain function underneath doesn't. The decorator's only job is exposing `search()` over the
protocol — correct by inspection, not something worth testing twice.

### Why three tiers of tests
Fast tests use a stub model (fixed vectors keyed by exact text), so ranking, corpus loading, and
the server's search logic run in milliseconds with no model load. Chunking tests use the real
embedding model's tokenizer, because the chunker's whole job is to agree with that tokenizer's
limits — a mock could pass while the real one still truncates. Two slower end-to-end tests build
the index over the real bundled corpus and start the server as a subprocess, talking to it with an
MCP client over real stdio — the tool-registration and protocol layer that the unit tests
deliberately bypass, and the layer the SDK's July 2026 rename actually broke.

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
