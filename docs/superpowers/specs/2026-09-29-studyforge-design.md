# studyforge — Design Spec

**Date:** 2026-09-29
**Status:** Approved in brainstorming, pending written-spec review

## 1. Purpose

A personal AI study assistant for a student enrolled at **two institutions at once**. All course material is ingested into a local RAG store; a set of Claude-powered expert agents helps study it (tutoring, quizzes, spaced review, exam prep, code and dataset explanation, study planning).

It is also a **portfolio project**: the goal is to demonstrate sound AI-architecture choices using established tools, not to hand-write everything. Success means:

1. The student actually uses it to study (answers are grounded in their notes, with citations).
2. The architecture is legible: a diagram, visible agent traces, and measured retrieval/answer quality in CI.

## 2. Constraints

- **Privacy:** source files, embeddings, database, and traces stay on the local PC. Only the user's question plus the retrieved excerpts (≈6 chunks) are sent to Anthropic. Only calendar events are sent to Google.
- **Platform:** Windows 11, local machine. Services run via Docker Compose.
- **Language:** Python only (backend + UI). Notes may be Norwegian and/or English.
- **Inputs:** PDF, PPTX, DOCX, Markdown/Obsidian, code files, CSV/XLSX, past exams, institution admin documents. No OCR/handwriting.

## 3. Stack

| Layer | Tool |
|---|---|
| Parsing | Docling (PDF/PPTX/DOCX, tables, slide structure) |
| Ingestion + retrieval | LlamaIndex (IngestionPipeline with docstore dedupe, PGVectorStore hybrid, CodeSplitter) |
| Embeddings | `BAAI/bge-m3`, local via sentence-transformers |
| Reranker | `BAAI/bge-reranker-v2-m3`, local |
| Database | Postgres 16 + pgvector (Docker Compose) |
| Agents | LangGraph (supervisor → expert template), `langchain-anthropic` |
| LLMs | Claude Sonnet (`claude-sonnet-5-5`) for experts; Claude Haiku (`claude-haiku-4-5`) for routing |
| Spaced repetition | FSRS (`py-fsrs`) |
| Backend | FastAPI |
| UI | Chainlit, mounted in FastAPI |
| MCP | FastMCP server over the same backend |
| Calendar | Google Calendar API (two-way sync) |
| Observability | Langfuse, self-hosted (Docker Compose) |
| Evals | Ragas + pytest, run in GitHub Actions |
| Tooling | `uv` for Python env/deps |

Architectural split to highlight: **LlamaIndex owns the data plane, LangGraph owns the reasoning plane.**

## 4. Architecture

```
 Sources: notes/<uni>/<course>/{...,exams/,code/,data/,admin/}
        │
 Ingestion (Docling → LlamaIndex pipeline → bge-m3, local)
        │
 Postgres ┌ chunks: pgvector + full-text, metadata {uni, course, doc_type, path, page}
          ├ datasets: one schema per course, read-only role
          ├ calendar_events, fsrs_cards, study_plan, weak_spots
          ├ LangGraph checkpoints (conversation state)
          └ Chainlit data layer (chat history)
        │
 Retrieval service: hybrid search (top 30) → reranker (top 6) → citations; scope filter
        │
 LangGraph: Supervisor (Haiku) → Expert(scope, persona) with shared tools
        │
 FastAPI ├ Chainlit UI
         ├ FastMCP server
         └ Google Calendar sync
 Cross-cutting: Langfuse traces + cost per query; Ragas evals in CI
```

## 5. Experts

**Core idea:** a *capability* is a **tool**; an *expert* is a **scope + persona**. Every expert can use every tool, always filtered to its scope. No per-subject-per-capability agents.

| Expert kind | Scope | Source |
|---|---|---|
| Course expert | one course | auto, one per `notes/<uni>/<course>/` |
| Uni expert | one institution (incl. `admin/` docs, that uni's deadlines) | auto, one per `notes/<uni>/` |
| Topic expert | list of course tags (spans both unis) | `experts.yaml` |
| Study coach / planner | all | built-in |
| Exam strategist | all, emphasises `exams/` and `weak_spots` | built-in |
| Connections finder | all, retrieves across courses to link concepts | built-in |

Persona for course/uni experts is generated once at ingest time from that scope's material (a short summary of topics and level) and stored; regenerated when the scope's content changes.

**Shared tools**

| Tool | Does |
|---|---|
| `search_notes(query, scope)` | Hybrid retrieval + rerank, returns cited chunks |
| `make_quiz(topic, scope, n)` | Generates questions from retrieved chunks; records results to `weak_spots` |
| `review_cards(scope)` | Returns due FSRS cards; grades update FSRS state and `weak_spots` |
| `query_dataset(sql, course)` | Read-only SELECT with statement timeout on that course's schema |
| `explain_code(query, scope)` | Retrieves code chunks and explains them |
| `calendar_read(range)` / `calendar_write(event)` | Local calendar table; synced to Google |

**Routing:** the user picks an expert via Chainlit chat profiles, or picks "Auto" and the supervisor chooses one. State is kept per thread via LangGraph checkpoints.

## 6. Interface (Chainlit)

- **Chat profiles** = expert picker, grouped Overall / Uni A / Uni A courses / Uni B / Uni B courses; generated from folders + `experts.yaml`.
- **Steps** show supervisor routing and every tool call (portfolio visibility).
- **Citations** as Elements; clicking opens the source passage (PDF opens at page in side panel).
- **Welcome message** acts as the dashboard: due cards, next deadlines (both unis), today's study block, weakest topics.
- **Starters:** "Quiz me on weak spots", "Plan this week", "What's due?", "Explain a past exam question".
- **Custom elements** (React + shadcn): flashcard, quiz, week-view calendar (side panel).
- **`/library` command:** ingested files per uni/course with ingest status.
- **Full calendar view** = Google Calendar. App-created events are colour-coded per uni and tagged (extended property `unihelper=true`) so personal events are never modified.

## 7. Data flow

**Ingest (`uv run ingest`):**
1. Walk `notes/`; derive `uni`, `course` from path; `doc_type` from subfolder (`exams`, `code`, `data`, `admin`) else from extension.
2. Hash each file; skip unchanged; for changed files delete old chunks and re-ingest.
3. Docs → Docling → chunks → bge-m3 → Postgres. Code → CodeSplitter. CSV/XLSX → tables in the course's dataset schema.
4. Regenerate personas for scopes whose content changed.
5. Print report: `N new, N updated, N failed: <file>: <reason>`.

**Query:** expert → tools with scope filter → hybrid top 30 → rerank top 6 → Claude answer with citations → Langfuse trace.

**Calendar:** exam/deadline dates entered by user or extracted from `admin/` docs (user confirms before saving). Study coach generates study blocks backwards from exams, weighted by `weak_spots` and FSRS load. Sync to Google both ways; last-write-wins on tagged events.

## 8. Error handling

| Case | Behaviour |
|---|---|
| File fails to parse | Log, skip, continue batch, appear in ingest report |
| No relevant chunks (rerank score below threshold) | Expert replies "not in your notes" and does not answer from general knowledge unless user asks |
| Claude API error | Retry with exponential backoff (3 tries), then clear message in chat |
| Google sync conflict | Last-write-wins; only tagged events are ever modified |
| Dataset query | Read-only role, SELECT only, 5 s statement timeout |

## 9. Testing

- **Unit (pytest):** path→metadata, chunking, file-hash skipping, FSRS scheduling, calendar tagging/conflict rule, SQL read-only guard.
- **Golden-set evals:** ~50 Q/A pairs written from real notes (`evals/golden.jsonl`), scored with Ragas (context precision/recall, faithfulness, answer relevancy) plus a "not in notes" refusal subset. Runs in GitHub Actions on a small committed sample corpus; CI fails on score regression versus stored baseline.
- **Smoke:** ingest `samples/` → ask one question → answer contains a citation.

## 10. Build phases

Each phase ends working and demoable.

1. **Foundation:** repo, `uv`, Docker Compose (Postgres+pgvector, Langfuse), config.
2. **Ingestion:** Docling + LlamaIndex pipeline, metadata, hashing, report.
3. **Retrieval + evals:** hybrid + rerank + citations; golden set + Ragas in CI.
4. **Agents v1:** LangGraph supervisor, expert template, course/uni experts, `search_notes`, `make_quiz`; Langfuse tracing.
5. **Chainlit UI:** chat profiles, steps, citations, starters.
6. **MCP server:** FastMCP exposing experts/tools.
7. **Review + weak spots:** FSRS cards, flashcard/quiz elements, `weak_spots`.
8. **Calendar:** local calendar, Study coach planning, Google two-way sync, week-view element, welcome dashboard.
9. **Remaining experts/tools:** Exam strategist, Connections finder, topic experts (`experts.yaml`), `explain_code`, `query_dataset`.
10. **Portfolio polish:** README with architecture diagram, eval results, demo GIF.

## 11. Later (not in scope)

Auto-ingest file watcher · voice mode · concept graph view · Anki export · lecture recording transcription/summaries · custom Next.js dashboard.
