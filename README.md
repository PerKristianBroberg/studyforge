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

## Agents

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
uv run --env-file .env chat                      # auto-routes to the best expert
uv run --env-file .env chat --expert UiO/INF1010 # talk to one expert
```

Tracing (optional, local only): `docker compose --env-file .env -f infra/langfuse/docker-compose.yml up -d`,
then open http://127.0.0.1:3000. The `LANGFUSE_INIT_*` values in `.env` create the org, project, API keys and login
on first start. Each chat turn is one `chat-turn` trace (session = conversation), with `route-question`,
`retrieve-notes` (retriever) and `generate-quiz` steps.

## Web UI

```bash
uv run --env-file .env chainlit run src/studyforge/ui.py --host 127.0.0.1
```

Open http://127.0.0.1:8000: pick an expert (or Auto), click citations to open sources, drop files to add them to a course,
`/library` lists what's indexed. Models and expert personas warm up in the background at startup.
