# studyforge

Multi-agent RAG study assistant for studying at two universities at once.
Local embeddings (bge-m3) and reranking (bge-reranker-v2-m3), Postgres + pgvector hybrid search,
LlamaIndex data plane. Agents, Chainlit UI, MCP server and calendar are coming in later phases —
see [the design spec](docs/superpowers/specs/2026-09-29-studyforge-design.md).

## Run

```bash
docker compose up -d --wait
uv sync
# put course files in notes/<uni>/<course>/ (gitignored — never committed)
uv run ingest
uv run ask "What is a p-value?" --uni NTNU --course STAT200
```

Folder conventions: `notes/<uni>/<course>/...`; subfolders named `exams/`, `code/`, `admin/` set the document type;
files directly under `notes/<uni>/` or in `notes/<uni>/admin/` belong to the whole institution.

## Test

```bash
uv run pytest                 # unit + integration (needs Postgres)
uv run pytest -m eval evals -s   # retrieval quality on samples/ (downloads models)
```
