# RAG Search MCP Server

A small [MCP](https://modelcontextprotocol.io) server with one tool, `search_documents(query, top_k)`,
that lets an agent search a local set of text files by meaning.

There are no external services, API keys, or databases. Clone it, install the dependencies, point an
MCP client at it, and it works against the bundled sample corpus.

## Why I built it

An agent can only use the tools it's given. I'd already built retrieval pipelines, so this project
covers the other half: making retrieval something an agent can call, through the protocol agent
clients (Claude Desktop, Claude Code, and others) already speak.

The search code comes from my [`pdf-rag-from-scratch`](https://github.com/regankight/pdf-rag-from-scratch)
project: tokenizer-aware chunking, local embeddings, cosine-similarity ranking. I re-pointed it at a
folder of `.txt` files instead of a PDF, so nobody has to supply their own corpus to try it.

## What it does

```text
query
  ↓
embed the query (same model that embedded the corpus)
  ↓
cosine similarity against every chunk
  ↓
top_k chunks, best first, each with its source file and score
```

It only retrieves. It doesn't write an answer; that's the agent's job (see
[Design decisions](#design-decisions)).

## Install and run

Needs Python 3.10 or newer (the MCP SDK's minimum). I've only run it on 3.14.

```bash
git clone https://github.com/regankight/rag-search-mcp-server.git
cd rag-search-mcp-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run it directly (MCP clients talk to it over stdio):

```bash
python mcp_server.py
```

To use it from an MCP client such as Claude Desktop or Claude Code, point the client's config at that
command:

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

The first run downloads the embedding model (`all-MiniLM-L6-v2`, about 90MB) from Hugging Face and
caches it. The server loads the model and builds the index in a background thread as soon as it
starts, so it answers the client's handshake in about 0.3 seconds instead of waiting for the model.
A search that arrives before the index is ready waits for it.

## Example

Searching for *"catching quality getting worse over time before users complain"* returns (text
shortened):

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

The observability passage wins by a wide margin. It talks about noticing "quality drift before a user
complains", and shares only two generic words with the query (`quality`, `before`), so the match comes
from meaning, not wording. These numbers come from a real run on the bundled corpus; scores shift
slightly with the embedding model.

## Design decisions

**Bundled sample corpus.** A server only proves something if a reviewer can run it. Asking them to
bring their own documents first is friction most people won't push through. `sample_docs/` has five
short passages on RAG-related topics (retrieval, hybrid search, observability, evaluation,
embeddings) so a fresh clone gives a correct result straight away. To search your own files, point
`DEFAULT_DOCS_DIR` in `search.py` at a directory of `.txt` files.

**Self-written samples, not a real dataset.** The benchmark PDF used in `pdf-rag-from-scratch` isn't
committed there (`data/*.pdf` is gitignored), and its redistribution license was never checked. I
wrote the samples myself instead of carrying content with unclear reuse rights into a second public
repo. Their only job is to show that search works.

**Retrieval only, no generation.** Adding an LLM call would make this a smaller copy of
`pdf-rag-from-scratch`. The point here is the MCP layer: giving an agent a search tool it can call.

**In-memory index, no vector database.** Five short files make a handful of chunks, so comparing the
query against every chunk is simpler and fast enough. A vector database would solve a scaling problem
this project doesn't have. The index is built once per server process and kept in memory.

**Copied search code, not imported.** Depending on `pdf-rag-from-scratch` (submodule, or `pip install`
from a git URL) would mean cloning two repos to run one. Copying the small amount of code needed keeps
this repo standalone. The cost is that the copies can drift apart, which is fine for a demo and not
something I'd do for shared production code.

**`mcp>=2,<3` is pinned.** The MCP Python SDK made a breaking change in v2.0.0 (2026-07-28): the server
class moved from `mcp.server.fastmcp.FastMCP` to `mcp.server.MCPServer`, and the old import path was
removed, not deprecated. Tutorials written before then fail on a fresh install with
`ModuleNotFoundError`. This repo uses the v2 API (`from mcp.server import MCPServer`, with an explicit
`mcp.run(transport="stdio")`) and pins the major version so it doesn't break the same way when the SDK
changes again.

**Index built in a background thread, behind a lock.** Loading the model takes seconds, and the first
run also downloads it. Doing that inside the first tool call risks a client timeout on the call a
reviewer is most likely to make first. Doing it before the server starts would delay the handshake. So
`python mcp_server.py` starts the build in the background and serves immediately, and an early search
waits on the same build. Even importing the embedding library is slow, because `sentence-transformers`
pulls in torch (about 2.5s), so that import is deferred into the build too. A test checks that
importing the server never loads torch. The lock is there because the SDK runs sync tool functions on
worker threads: without it, two searches arriving together would each load the model. A test starts
eight threads at once and asserts the index is built exactly once.

**Blank queries are an error.** Embedding an empty string still produces a vector, so every passage
gets ranked and returned with a near-zero score that looks like a result but isn't one. The tool
rejects blank queries instead. The error is raised as the SDK's `ToolError`, because an ordinary
exception is hidden from the client (it would see only "Error executing tool search_documents"), and
the agent wouldn't learn what to fix. A test checks the message from the client's side over real
stdio. The tool description also states two behaviors the schema can't show: `top_k` is clamped to
between 1 and the corpus size, and queries over the embedding model's 256-token limit are truncated.

**Results include the source file.** A passage with only a positional `chunk_id` can't be cited or
checked, and a retrieval tool exists so an agent can ground an answer in something. Files are chunked
separately, so each chunk comes from exactly one file, and that filename is returned with the text.

**`search()` is separate from the decorated tool.** `search_documents` is a thin wrapper around
`search()`. Tests call the plain function, so they don't depend on the SDK's decorator internals. The
wrapper's only other job is converting a plain error into a `ToolError`.

**Three kinds of tests.** Fast tests use a stub model (fixed vectors keyed by exact text), so ranking,
corpus loading, and the search logic run in milliseconds without loading a model. Chunking tests use
the real embedding model's tokenizer, because the chunker has to agree with that tokenizer's limits
and a mock could pass while the real one truncates. Slower end-to-end tests build the index over the
real corpus and start the server as a subprocess, talking to it with an MCP client over real stdio.
That covers tool registration and the protocol layer, which the unit tests skip and which the SDK's
July 2026 rename actually broke.

## Project structure

```text
rag-search-mcp-server/
├── mcp_server.py       # the MCP server: index cache and the search_documents tool
├── search.py           # chunking, embedding, cosine-similarity ranking (adapted from pdf-rag-from-scratch)
├── sample_docs/        # five short passages the demo searches by default
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

## Scope

This is a small, single-tool server meant to show how retrieval gets exposed to an agent. It isn't a
general document-search service: no authentication, one fixed corpus directory, and the index lives
only as long as the server process.
