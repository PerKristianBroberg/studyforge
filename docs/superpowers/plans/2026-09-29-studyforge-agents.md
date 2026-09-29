# studyforge Plan 2 — Agents (Supervisor, Experts, Tools, Tracing, Answer Evals)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chat with auto-discovered course/uni experts that answer from the student's notes with citations, quiz on request, remember the conversation per thread, are traced in a self-hosted Langfuse, and are scored for faithfulness in an eval gate.

**Architecture:** LangGraph owns the reasoning plane. A parent `StateGraph` has a `route` node and one node per expert. Each expert node is a LangChain `create_agent` (Claude Sonnet + tools `search_notes`, `make_quiz`) whose tools are closed over that expert's `Scope`, so an expert cannot see outside it. `route` passes through an explicitly chosen expert or asks a Claude Haiku router, falling back to the `general` expert. Personas are generated once per scope by Haiku and cached in Postgres, keyed by the scope's file hashes. Conversation state lives in a Postgres LangGraph checkpointer.

**Tech Stack:** langgraph 1.2, langchain 1.4 (`create_agent`), langchain-anthropic 1.7, langgraph-checkpoint-postgres 3.1 + psycopg 3, langfuse 4.15 (self-hosted server v3), ragas 0.4 (dev), existing Plan 1 stack.

**Spec:** `docs/superpowers/specs/2026-09-29-studyforge-design.md` (implements §10 phase 4; §5 experts for kinds course/uni/general; Langfuse and Ragas moved here from Plan 1).

**Scope notes:**
- Only course, uni and one `general` (all notes) expert. Topic experts, Study coach, Exam strategist and Connections finder are phase 9.
- `make_quiz` generates quizzes; grading, FSRS and `weak_spots` are phase 7.
- The `general` expert is the seed of the overall experts and the router's fallback.

## Global Constraints

- Everything from Plan 1's Global Constraints still holds.
- Expert model `claude-sonnet-5-5`; router, persona and eval-judge model `claude-haiku-4-5`. Both come from `Settings` (`expert_model`, `router_model`).
- Only the user's message plus retrieved passages go to Anthropic. Langfuse runs locally, with every port bound to `127.0.0.1`.
- `ANTHROPIC_API_KEY` and `LANGFUSE_*` live in `.env` (gitignored) and are loaded with `uv run --env-file .env ...`. Unit tests never need a key; they use fake chat models.
- LangGraph node names must not contain `:` or `|`. Expert node names are sanitised.
- Experts answer only from `search_notes` results and cite `[path p.N]`. When there are no results they say "Not in your notes."

## Review Focus

1. **Router returns an unknown or hallucinated expert id** → fall back to `general`, never crash. Test: Task 5 `test_router_garbage_falls_back_to_general`.
2. **A course expert's tools searching outside its course** → tools must use the expert's scope. Test: Task 3 `test_search_notes_uses_bound_scope` and Task 5 `test_explicit_expert_runs_with_its_scope`.
3. **Two folders whose ids sanitise to the same node name** (e.g. `Uni A` and `uni-a`) → fail loudly at startup, not by silently merging experts. Test: Task 5 `test_duplicate_node_names_rejected`.
4. **Missing `ANTHROPIC_API_KEY`** → a clear message and exit, not a stack trace mid-conversation. Test: Task 6 `test_chat_without_api_key_exits_with_message`.
5. **Expert answers from general knowledge when the notes have nothing** → the refusal cases in the answer eval must say "not in your notes". Test: Task 8 `test_answer_quality` (refusal assertions).

---

## File Structure

```
pyproject.toml                    + langgraph, langchain, langchain-anthropic, checkpoint-postgres, psycopg, langfuse; dev: ragas
.gitignore                        + .env
src/studyforge/config.py          + expert_model, router_model, pg_conn
src/studyforge/experts.py         Expert, GENERAL, discover()
src/studyforge/tools.py           search_notes / make_quiz tool factory, Quiz models
src/studyforge/personas.py        persona(): Haiku-written persona cached per scope hash
src/studyforge/graph.py           build_graph(), make_router(), RULES
src/studyforge/app.py             build_app(), run_config() (tracing), `chat` CLI
infra/langfuse/docker-compose.yml official Langfuse compose, ports bound to 127.0.0.1
tests/test_experts.py, tests/test_tools.py, tests/test_personas.py, tests/test_graph.py, tests/test_app.py
evals/test_answer_quality.py      Ragas faithfulness + refusal gate (real Claude)
.github/workflows/ci.yml          + llm_eval step (runs only when ANTHROPIC_API_KEY secret exists)
```

---

### Task 1: Dependencies and settings

**Files:**
- Modify: `pyproject.toml`, `.gitignore`, `src/studyforge/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.expert_model: str = "claude-sonnet-5-5"`, `Settings.router_model: str = "claude-haiku-4-5"`, property `Settings.pg_conn -> str` (psycopg-3 DSN `postgresql://user:pw@host:port/db`). Pytest marker `llm_eval`.

- [ ] **Step 1: Add dependencies**

Run: `uv add langgraph langchain langchain-anthropic langgraph-checkpoint-postgres "psycopg[binary]" langfuse && uv add --group dev ragas`
Expected: resolves and installs; `uv.lock` updated.

- [ ] **Step 2: Update pytest config in `pyproject.toml`** — replace the `markers` and `addopts` lines under `[tool.pytest.ini_options]` with:

```toml
markers = [
    "eval: retrieval-quality evals with real models (slow)",
    "llm_eval: answer-quality evals calling the Claude API (needs ANTHROPIC_API_KEY)",
]
addopts = "-m 'not eval and not llm_eval'"
```

- [ ] **Step 3: Add `.env` to `.gitignore`** — append under the private-material block:

```gitignore
.env
```

- [ ] **Step 4: Write the failing test** — append to `tests/test_config.py`:

```python
def test_agent_settings():
    s = Settings()
    assert s.expert_model == "claude-sonnet-5-5"
    assert s.router_model == "claude-haiku-4-5"
    assert s.pg_conn == "postgresql://studyforge:studyforge@localhost:5433/studyforge"


def test_env_file_is_gitignored():
    root = Path(__file__).parent.parent
    assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=root).returncode == 0
```

- [ ] **Step 5: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL: `AttributeError: 'Settings' object has no attribute 'expert_model'`.

- [ ] **Step 6: Implement** — in `src/studyforge/config.py`, add the two fields after `rerank_min_score` and the property after `db_url`:

```python
    expert_model: str = "claude-sonnet-5-5"
    router_model: str = "claude-haiku-4-5"  # routing, personas, eval judge
```

```python
    @property
    def pg_conn(self) -> str:  # psycopg 3 DSN for the LangGraph checkpointer
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
```

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest -q`
Expected: all pass (28).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/studyforge/config.py tests/test_config.py
git commit -m "build: agent dependencies and settings"
```

---

### Task 2: Expert discovery

**Files:**
- Create: `src/studyforge/experts.py`
- Test: `tests/test_experts.py`

**Interfaces:**
- Consumes: `scan` (paths), `Scope` (retrieval).
- Produces:
  - `@dataclass(frozen=True) Expert(id: str, name: str, kind: str, scope: Scope)` with property `node -> str` (`"expert_" + lowercase id with non-word runs replaced by "_"`). `kind` ∈ `{"general", "uni", "course"}`. ids: `"general"`, `"<uni>"`, `"<uni>/<course>"`.
  - `GENERAL: Expert` = `Expert("general", "General (all notes)", "general", Scope())`.
  - `discover(root: Path) -> list[Expert]`: `[GENERAL, *uni experts sorted, *course experts sorted]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_experts.py`

```python
from studyforge.experts import GENERAL, Expert, discover
from studyforge.retrieval import Scope


def touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def test_discover_uni_and_course_experts(tmp_path):
    touch(tmp_path, "UiO/INF1010/a.md")
    touch(tmp_path, "UiO/admin/rules.md")
    touch(tmp_path, "NTNU/TDT4100/b.md")
    experts = discover(tmp_path)
    assert [e.id for e in experts] == ["general", "NTNU", "UiO", "NTNU/TDT4100", "UiO/INF1010"]
    assert experts[0] is GENERAL
    uio_inf = experts[-1]
    assert uio_inf.kind == "course" and uio_inf.scope == Scope(uni="UiO", course="INF1010")
    assert experts[2].scope == Scope(uni="UiO")


def test_node_names_are_langgraph_safe():
    e = Expert("UiO/INF 1010", "x", "course", Scope(uni="UiO", course="INF 1010"))
    assert e.node == "expert_uio_inf_1010"
    assert GENERAL.node == "expert_general"


def test_missing_notes_dir_gives_only_general(tmp_path):
    assert discover(tmp_path / "nope") == [GENERAL]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_experts.py -q`
Expected: FAIL: `ModuleNotFoundError: No module named 'studyforge.experts'`.

- [ ] **Step 3: Implement** — `src/studyforge/experts.py`

```python
import re
from dataclasses import dataclass
from pathlib import Path

from studyforge.paths import scan
from studyforge.retrieval import Scope


@dataclass(frozen=True)
class Expert:
    id: str
    name: str
    kind: str  # "general" | "uni" | "course"
    scope: Scope

    @property
    def node(self) -> str:
        # LangGraph reserves ':' and '|' in node names
        return "expert_" + re.sub(r"\W+", "_", self.id).strip("_").lower()


GENERAL = Expert("general", "General (all notes)", "general", Scope())


def discover(root: Path) -> list[Expert]:
    unis, courses = set(), set()
    for _, meta in scan(root) if root.is_dir() else []:
        unis.add(meta.uni)
        if meta.course:
            courses.add((meta.uni, meta.course))
    return [
        GENERAL,
        *(Expert(u, f"{u} expert", "uni", Scope(uni=u)) for u in sorted(unis)),
        *(Expert(f"{u}/{c}", f"{c} expert ({u})", "course", Scope(uni=u, course=c)) for u, c in sorted(courses)),
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_experts.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/studyforge/experts.py tests/test_experts.py
git commit -m "feat: discover uni and course experts from the notes tree"
```

---

### Task 3: Scoped tools

**Files:**
- Create: `src/studyforge/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `retrieval.search(query, scope, s) -> list[Hit]`, `Hit.cite()`.
- Produces:
  - `NOT_IN_NOTES: str` (starts with `"NOT_IN_NOTES"`).
  - Pydantic `QuizQuestion(question: str, choices: list[str], answer: int, source: str)`, `Quiz(questions: list[QuizQuestion])`.
  - `format_hits(hits: list[Hit]) -> str`: `"[1] <cite>\n<text>\n\n[2] ..."` or `NOT_IN_NOTES`.
  - `make_tools(scope: Scope, s: Settings, quiz_llm: BaseChatModel) -> list[BaseTool]`: `[search_notes(query: str), make_quiz(topic: str, n: int = 5)]`. `make_quiz` returns `Quiz` JSON or `NOT_IN_NOTES`.

- [ ] **Step 1: Write the failing tests** — `tests/test_tools.py`

```python
import json

from studyforge.config import Settings
from studyforge.retrieval import Hit, Scope
from studyforge.tools import NOT_IN_NOTES, Quiz, QuizQuestion, format_hits, make_tools

HIT = Hit("A heap is a complete binary tree.", 0.9, "UiO/INF1010/heaps.pdf", 3, "UiO", "INF1010")


class StubQuizLLM:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is Quiz
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return Quiz(questions=[QuizQuestion(question="What is a heap?", choices=["a", "b", "c", "d"],
                                            answer=0, source="UiO/INF1010/heaps.pdf p.3")])


def test_format_hits():
    assert format_hits([HIT]) == "[1] UiO/INF1010/heaps.pdf p.3\nA heap is a complete binary tree."
    assert format_hits([]) == NOT_IN_NOTES


def test_search_notes_uses_bound_scope(monkeypatch):
    seen = []
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: seen.append(scope) or [HIT])
    scope = Scope(uni="UiO", course="INF1010")
    search_notes, _ = make_tools(scope, Settings(), StubQuizLLM())
    out = search_notes.invoke({"query": "heaps"})
    assert seen == [scope] and "heaps.pdf p.3" in out


def test_make_quiz_uses_notes_and_returns_json(monkeypatch):
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: [HIT])
    llm = StubQuizLLM()
    _, make_quiz = make_tools(Scope(uni="UiO", course="INF1010"), Settings(), llm)
    quiz = json.loads(make_quiz.invoke({"topic": "heaps", "n": 1}))
    assert quiz["questions"][0]["source"] == "UiO/INF1010/heaps.pdf p.3"
    assert "complete binary tree" in llm.prompts[0]


def test_make_quiz_without_notes_does_not_call_llm(monkeypatch):
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: [])
    llm = StubQuizLLM()
    _, make_quiz = make_tools(Scope(), Settings(), llm)
    assert make_quiz.invoke({"topic": "photosynthesis"}) == NOT_IN_NOTES
    assert llm.prompts == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tools.py -q`
Expected: FAIL: `ModuleNotFoundError: No module named 'studyforge.tools'`.

- [ ] **Step 3: Implement** — `src/studyforge/tools.py`

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from studyforge import retrieval
from studyforge.config import Settings
from studyforge.retrieval import Hit, Scope

NOT_IN_NOTES = "NOT_IN_NOTES: nothing relevant in the student's notes for this scope."


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="exactly 4 answer options")
    answer: int = Field(description="0-based index of the correct choice")
    source: str = Field(description="citation of the passage used, e.g. 'UiO/INF1010/x.pdf p.3'")


class Quiz(BaseModel):
    questions: list[QuizQuestion]


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return NOT_IN_NOTES
    return "\n\n".join(f"[{i}] {h.cite()}\n{h.text}" for i, h in enumerate(hits, 1))


def make_tools(scope: Scope, s: Settings, quiz_llm: BaseChatModel) -> list[BaseTool]:
    """Tools closed over one expert's scope, so the expert cannot search outside it."""

    @tool
    def search_notes(query: str) -> str:
        """Search the student's notes in this expert's scope. Returns numbered passages with citations, or NOT_IN_NOTES."""
        return format_hits(retrieval.search(query, scope, s))

    @tool
    def make_quiz(topic: str, n: int = 5) -> str:
        """Create an n-question multiple-choice quiz on a topic, using only the student's notes. Returns JSON."""
        hits = retrieval.search(topic, scope, s)
        if not hits:
            return NOT_IN_NOTES
        quiz = quiz_llm.with_structured_output(Quiz).invoke(
            f"Write {n} multiple-choice questions (4 choices each) testing understanding of '{topic}', "
            f"using ONLY these notes. Cite the passage each question comes from.\n\n{format_hits(hits)}"
        )
        return quiz.model_dump_json()

    return [search_notes, make_quiz]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_tools.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/studyforge/tools.py tests/test_tools.py
git commit -m "feat: scoped search_notes and make_quiz tools"
```

---

### Task 4: Cached personas

**Files:**
- Create: `src/studyforge/personas.py`
- Test: `tests/test_personas.py`

**Interfaces:**
- Consumes: `Expert`, `Manifest(s).all()`, `engine(url)`, `retrieval.search`.
- Produces: `persona(expert: Expert, s: Settings, llm, material: Callable[[Expert, Settings], str] = default_material) -> str`. It generates with `llm.invoke(prompt).text` and caches in table `personas(expert_id text primary key, key text, text text)`. `key` = sha256 of the sorted file shas in the expert's scope, so the persona regenerates when that scope's files change. `in_scope(path: str, scope: Scope) -> bool`.

- [ ] **Step 1: Write the failing tests** — `tests/test_personas.py`

```python
import sqlalchemy as sa
from langchain_core.messages import AIMessage

from studyforge.experts import Expert
from studyforge.personas import in_scope, persona
from studyforge.retrieval import Scope
from studyforge.store import Manifest, engine

INF = Expert("UiO/INF1010", "INF1010 expert (UiO)", "course", Scope(uni="UiO", course="INF1010"))


class CountingLLM:
    calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return AIMessage(content=f"You are the INF1010 expert v{self.calls}.")


def test_in_scope():
    assert in_scope("UiO/INF1010/a.pdf", Scope(uni="UiO", course="INF1010"))
    assert not in_scope("NTNU/INF1010/a.pdf", Scope(uni="UiO", course="INF1010"))
    assert not in_scope("UiO/admin/rules.pdf", Scope(uni="UiO", course="INF1010"))
    assert in_scope("UiO/admin/rules.pdf", Scope(uni="UiO"))
    assert in_scope("NTNU/x.md", Scope())


def test_persona_cached_until_scope_changes(db_settings):
    with engine(db_settings.db_url).begin() as c:
        c.execute(sa.text("DROP TABLE IF EXISTS personas"))
    m = Manifest(db_settings)
    m.put("UiO/INF1010/a.md", "sha-a", ["UiO/INF1010/a.md"])
    llm = CountingLLM()
    material = lambda e, s: "heaps, trees"

    assert persona(INF, db_settings, llm, material) == "You are the INF1010 expert v1."
    assert persona(INF, db_settings, llm, material) == "You are the INF1010 expert v1."
    assert llm.calls == 1

    m.put("NTNU/TDT4100/b.md", "sha-b", ["NTNU/TDT4100/b.md"])  # outside scope: still cached
    persona(INF, db_settings, llm, material)
    assert llm.calls == 1

    m.put("UiO/INF1010/a.md", "sha-a2", ["UiO/INF1010/a.md"])  # inside scope changed
    assert persona(INF, db_settings, llm, material) == "You are the INF1010 expert v2."
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_personas.py -q`
Expected: FAIL: `ModuleNotFoundError: No module named 'studyforge.personas'`.

- [ ] **Step 3: Implement** — `src/studyforge/personas.py`

```python
import hashlib
from typing import Callable

import sqlalchemy as sa

from studyforge import retrieval
from studyforge.config import Settings
from studyforge.experts import Expert
from studyforge.retrieval import Scope
from studyforge.store import Manifest, engine

PROMPT = """You are writing the persona for a study assistant that is an expert on: {name}.
File names and excerpts from the student's notes in this scope:

{material}

In 3-4 sentences, written as "You are ...", describe the subject area, the main topics covered
and the level (introductory/advanced). No bullet points."""


def in_scope(path: str, scope: Scope) -> bool:
    parts = path.split("/")
    if scope.uni and parts[0] != scope.uni:
        return False
    return not scope.course or (len(parts) > 2 and parts[1] == scope.course)


def default_material(expert: Expert, s: Settings) -> str:
    files = [p for p in Manifest(s).all() if in_scope(p, expert.scope)]
    hits = retrieval.search(f"main topics of {expert.name}", expert.scope, s)
    excerpts = "\n\n".join(h.text for h in hits)[:4000]
    return "Files:\n" + "\n".join(files[:50]) + "\n\nExcerpts:\n" + excerpts


def persona(expert: Expert, s: Settings, llm, material: Callable[[Expert, Settings], str] = default_material) -> str:
    shas = sorted(sha for path, (sha, _) in Manifest(s).all().items() if in_scope(path, expert.scope))
    key = hashlib.sha256("".join(shas).encode()).hexdigest()
    eng = engine(s.db_url)
    with eng.begin() as c:
        c.execute(sa.text("CREATE TABLE IF NOT EXISTS personas (expert_id text PRIMARY KEY, key text, text text)"))
        row = c.execute(sa.text("SELECT text FROM personas WHERE expert_id = :e AND key = :k"),
                        {"e": expert.id, "k": key}).first()
    if row:
        return row[0]
    text = llm.invoke(PROMPT.format(name=expert.name, material=material(expert, s))).text
    with eng.begin() as c:
        c.execute(sa.text(
            "INSERT INTO personas (expert_id, key, text) VALUES (:e, :k, :t) "
            "ON CONFLICT (expert_id) DO UPDATE SET key = :k, text = :t"
        ), {"e": expert.id, "k": key, "t": text})
    return text
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_personas.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/studyforge/personas.py tests/test_personas.py
git commit -m "feat: Haiku-written expert personas cached per scope hash"
```

---

### Task 5: Supervisor graph

**Files:**
- Create: `src/studyforge/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `Expert`, `GENERAL`, `make_tools`.
- Produces:
  - `RULES: str` (appended to every expert's system prompt).
  - `class State(MessagesState)` with `requested: str` (`"auto"` or an expert id) and `expert_id: str` (set by `route`).
  - `build_graph(experts: list[Expert], personas: dict[str, str], expert_llm, router: Callable[[list[BaseMessage]], str], s: Settings, checkpointer=None, quiz_llm=None) -> CompiledStateGraph`. Input `{"messages": [...], "requested": "auto" | id}`; output state has `expert_id` and `messages`. Raises `ValueError` on duplicate node names.
  - `make_router(llm, experts: list[Expert]) -> Callable[[list[BaseMessage]], str]` (Haiku structured output `Route(expert_id: str)`).

- [ ] **Step 1: Write the failing tests** — `tests/test_graph.py`

```python
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from studyforge.config import Settings
from studyforge.experts import GENERAL, Expert
from studyforge.graph import build_graph
from studyforge.retrieval import Hit, Scope

UIO = Expert("UiO", "UiO expert", "uni", Scope(uni="UiO"))
INF = Expert("UiO/INF1010", "INF1010 expert (UiO)", "course", Scope(uni="UiO", course="INF1010"))
EXPERTS = [GENERAL, UIO, INF]


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*answers):
    """Each answer: first call search_notes, then reply with the text."""
    msgs = []
    for i, text in enumerate(answers):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "search_notes", "args": {"query": "q"}, "id": f"c{i}"}]))
        msgs.append(AIMessage(content=text))
    return FakeToolModel(messages=iter(msgs))


@pytest.fixture
def seen_scopes(monkeypatch):
    seen = []
    hit = Hit("A heap is a tree.", 0.9, "UiO/INF1010/h.md", None, "UiO", "INF1010")
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: seen.append(scope) or [hit])
    return seen


def run(graph, text, requested, thread="t1"):
    return graph.invoke({"messages": [("user", text)], "requested": requested},
                        {"configurable": {"thread_id": thread}})


def test_explicit_expert_runs_with_its_scope(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("Heaps are trees [UiO/INF1010/h.md]."),
                    lambda m: pytest.fail("router must not run"), Settings(), InMemorySaver())
    out = run(g, "What is a heap?", "UiO/INF1010")
    assert out["expert_id"] == "UiO/INF1010"
    assert seen_scopes == [INF.scope]
    assert out["messages"][-1].text == "Heaps are trees [UiO/INF1010/h.md]."


def test_auto_uses_router(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("ok"), lambda m: "UiO", Settings(), InMemorySaver())
    assert run(g, "exam rules?", "auto")["expert_id"] == "UiO"
    assert seen_scopes == [UIO.scope]


def test_router_garbage_falls_back_to_general(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("ok"), lambda m: "made-up-expert", Settings(), InMemorySaver())
    assert run(g, "hi", "auto")["expert_id"] == "general"


def test_thread_memory(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("first", "second"), lambda m: "general", Settings(), InMemorySaver())
    run(g, "question one", "general")
    out = run(g, "question two", "general")
    texts = [m.text for m in out["messages"]]
    assert "question one" in texts and "first" in texts and texts[-1] == "second"


def test_duplicate_node_names_rejected():
    clash = Expert("uio", "x", "uni", Scope(uni="uio"))
    with pytest.raises(ValueError, match="expert_uio"):
        build_graph([GENERAL, UIO, clash], {}, scripted(), lambda m: "general", Settings())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph.py -q`
Expected: FAIL: `ModuleNotFoundError: No module named 'studyforge.graph'`.

- [ ] **Step 3: Implement** — `src/studyforge/graph.py`

```python
from collections import Counter
from typing import Callable

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from studyforge.config import Settings
from studyforge.experts import GENERAL, Expert
from studyforge.tools import make_tools

RULES = """

Rules:
- Call search_notes before answering any question about course content.
- Answer only from the returned passages and cite them inline like [UiO/INF1010/x.pdf p.3].
- If search_notes returns NOT_IN_NOTES, reply "Not in your notes." and offer general knowledge only if the student asks for it.
- Use make_quiz when the student asks to be quizzed or tested.
- Reply in the student's language (Norwegian or English)."""


class State(MessagesState):
    requested: str  # "auto" or an Expert.id
    expert_id: str


class Route(BaseModel):
    expert_id: str = Field(description="id of the best expert for the student's message")


def make_router(llm, experts: list[Expert]) -> Callable[[list[BaseMessage]], str]:
    menu = "\n".join(f"- {e.id}: {e.name}" for e in experts)
    structured = llm.with_structured_output(Route)

    def router(messages: list[BaseMessage]) -> str:
        return structured.invoke(
            "Pick the expert best suited to the student's latest message. Prefer the most specific "
            f"course expert; use 'general' if unsure or if it spans courses.\nExperts:\n{menu}\n\n"
            f"Message: {messages[-1].text}"
        ).expert_id

    return router


def build_graph(experts: list[Expert], personas: dict[str, str], expert_llm,
                router: Callable[[list[BaseMessage]], str], s: Settings,
                checkpointer=None, quiz_llm=None):
    dupes = [n for n, k in Counter(e.node for e in experts).items() if k > 1]
    if dupes:
        raise ValueError(f"expert folders collide on node names: {dupes}")
    by_id = {e.id: e for e in experts}

    def route(state: State) -> Command:
        wanted = state.get("requested") or "auto"
        eid = wanted if wanted in by_id else router(state["messages"])
        eid = eid if eid in by_id else GENERAL.id
        return Command(goto=by_id[eid].node, update={"expert_id": eid})

    g = StateGraph(State)
    g.add_node("route", route, destinations=tuple(e.node for e in experts))
    for e in experts:
        prompt = personas.get(e.id, f"You are a study assistant for {e.name}.") + RULES
        tools = make_tools(e.scope, s, quiz_llm or expert_llm)
        g.add_node(e.node, create_agent(expert_llm, tools, system_prompt=prompt, name=e.node))
        g.add_edge(e.node, END)
    g.add_edge(START, "route")
    return g.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_graph.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/studyforge/graph.py tests/test_graph.py
git commit -m "feat: LangGraph supervisor routing to scoped expert agents"
```

---

### Task 6: App wiring and `chat` CLI

**Files:**
- Create: `src/studyforge/app.py`
- Modify: `pyproject.toml` (`[project.scripts]` add `chat = "studyforge.app:main"`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `discover`, `persona`, `build_graph`, `make_router`, `Settings`.
- Produces:
  - `build_app(s: Settings, checkpointer) -> tuple[CompiledStateGraph, list[Expert]]` (real Claude models).
  - `run_config(thread_id: str) -> dict`: `{"configurable": {"thread_id": ...}}` plus `"callbacks": [langfuse CallbackHandler()]` when `LANGFUSE_PUBLIC_KEY` is set.
  - `main()`: console script `chat [--expert ID|auto] [--thread ID]`, a REPL printing `[expert_id] answer`. It exits with a clear message if `ANTHROPIC_API_KEY` is unset.

- [ ] **Step 1: Write the failing tests** — `tests/test_app.py`

```python
import pytest

from studyforge import app


def test_run_config_without_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert app.run_config("t1") == {"configurable": {"thread_id": "t1"}}


def test_run_config_with_langfuse(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
    cfg = app.run_config("t1")
    assert type(cfg["callbacks"][0]).__name__ == "LangchainCallbackHandler"


def test_chat_without_api_key_exits_with_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["chat"])
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        app.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_app.py -q`
Expected: FAIL: `ImportError: cannot import name 'app' from 'studyforge'`.

- [ ] **Step 3: Implement** — `src/studyforge/app.py`

```python
import argparse
import os
import uuid

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.postgres import PostgresSaver

from studyforge.config import Settings
from studyforge.experts import discover
from studyforge.graph import build_graph, make_router
from studyforge.personas import persona


def build_app(s: Settings, checkpointer):
    experts = discover(s.notes_dir)
    expert_llm = ChatAnthropic(model=s.expert_model, max_tokens=4096)
    fast_llm = ChatAnthropic(model=s.router_model, max_tokens=1024)
    personas = {e.id: persona(e, s, fast_llm) for e in experts if e.kind != "general"}
    graph = build_graph(experts, personas, expert_llm, make_router(fast_llm, experts), s, checkpointer)
    return graph, experts


def run_config(thread_id: str) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):  # tracing is opt-in; traces stay in the local Langfuse
        from langfuse.langchain import CallbackHandler
        cfg["callbacks"] = [CallbackHandler()]
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your study experts")
    ap.add_argument("--expert", default="auto", help="expert id (e.g. UiO/INF1010) or 'auto'")
    ap.add_argument("--thread", default=None, help="conversation id to resume")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Put it in .env and run: uv run --env-file .env chat")
    s = Settings.from_env()
    thread = args.thread or uuid.uuid4().hex[:8]
    with PostgresSaver.from_conn_string(s.pg_conn) as checkpointer:
        checkpointer.setup()
        graph, experts = build_app(s, checkpointer)
        print(f"thread {thread} | experts: {', '.join(e.id for e in experts)} | empty line quits")
        while q := input("> ").strip():
            out = graph.invoke({"messages": [("user", q)], "requested": args.expert}, run_config(thread))
            print(f"[{out['expert_id']}] {out['messages'][-1].text}\n")
```

Add to `[project.scripts]` in `pyproject.toml`:

```toml
chat = "studyforge.app:main"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv sync && uv run pytest tests/test_app.py -q`
Expected: 3 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Manual smoke with the real API** (needs `.env` containing `ANTHROPIC_API_KEY=...`)

Run (uses the eval tables so `samples/` never mixes into the real index):
`export STUDYFORGE_NOTES_DIR=samples STUDYFORGE_CHUNKS_TABLE=eval_chunks STUDYFORGE_MANIFEST_TABLE=eval_manifest && uv run --env-file .env ingest && uv run --env-file .env chat`
Then type: `What does a p-value mean?`, then `Quiz me on sorting with 2 questions`, then `Who won the 1998 World Cup?`, then an empty line.
Expected: the first answer is routed to `[uni-b/STAT200]` (or `[uni-b]`) and cites `uni-b/STAT200/hypothesis-testing.md`. The second returns 2 multiple-choice questions. The third says "Not in your notes."

- [ ] **Step 7: Commit**

```bash
git add src/studyforge/app.py tests/test_app.py pyproject.toml uv.lock
git commit -m "feat: chat CLI wiring experts, personas, router and Postgres memory"
```

---

### Task 7: Self-hosted Langfuse tracing

**Files:**
- Create: `infra/langfuse/docker-compose.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_config` (Task 6) already attaches the handler when `LANGFUSE_PUBLIC_KEY` is set.
- Produces: a local Langfuse at `http://127.0.0.1:3000`.

- [ ] **Step 1: Fetch the official compose file and bind the web port to localhost**

Run: `mkdir -p infra/langfuse && curl -fsSL https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml -o infra/langfuse/docker-compose.yml && sed -i 's/- 3000:3000/- 127.0.0.1:3000:3000/; s/- "3000:3000"/- "127.0.0.1:3000:3000"/' infra/langfuse/docker-compose.yml && grep -n "3000" infra/langfuse/docker-compose.yml`
Expected: the web port line reads `127.0.0.1:3000:3000`. Also check with `grep -nE "^\s+- \"?[0-9]+:" infra/langfuse/docker-compose.yml`: any port mapping without `127.0.0.1:` must be prefixed with it the same way.

- [ ] **Step 2: Start it**

Run: `docker compose -f infra/langfuse/docker-compose.yml up -d --wait`
Expected: all services healthy. Open http://127.0.0.1:3000, sign up locally, create a project, and create API keys.

- [ ] **Step 3: Configure keys** — add to `.env` (gitignored):

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://127.0.0.1:3000
```

- [ ] **Step 4: Verify a trace**

Run (same exports as Task 6 Step 6): `uv run --env-file .env chat --expert uni-b/STAT200`, ask `What is R-squared?`, then quit.
Expected: in Langfuse → Traces there is a trace with the `route` node, the `expert_uni_b_stat200` agent, the `search_notes` tool call with its passages, token usage and cost.

- [ ] **Step 5: Document in `README.md`** — append:

````markdown
## Agents

```bash
cp .env.example .env   # or create .env with ANTHROPIC_API_KEY=...
uv run --env-file .env chat                     # auto-routes to the best expert
uv run --env-file .env chat --expert UiO/INF1010
```

Tracing (optional, local): `docker compose -f infra/langfuse/docker-compose.yml up -d`, create keys at
http://127.0.0.1:3000 and add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` to `.env`.
````

Also create `.env.example`:

```
ANTHROPIC_API_KEY=
# optional local tracing
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://127.0.0.1:3000
```

- [ ] **Step 6: Commit**

```bash
git add infra/langfuse/docker-compose.yml README.md .env.example
git commit -m "feat: self-hosted Langfuse tracing"
```

---

### Task 8: Answer-quality eval (Ragas) and CI

**Files:**
- Create: `evals/test_answer_quality.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `build_app`, `ingest.run`, `evals/golden.jsonl`, `Settings`.
- Produces: `uv run --env-file .env pytest -m llm_eval evals -s`, which prints mean faithfulness and refusals and fails below thresholds. It skips when `ANTHROPIC_API_KEY` is unset.

- [ ] **Step 1: Write the eval** — `evals/test_answer_quality.py`

```python
import asyncio
import json
import os
from pathlib import Path

import pytest
from anthropic import AsyncAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from studyforge import ingest
from studyforge.app import build_app
from studyforge.config import Settings

ROOT = Path(__file__).parent.parent
MIN_FAITHFULNESS = 0.8  # ponytail: fixed threshold; move to stored baseline once the golden set grows


@pytest.mark.llm_eval
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_answer_quality():
    s = Settings(chunks_table="eval_chunks", manifest_table="eval_manifest", notes_dir=ROOT / "samples")
    assert not ingest.run(s.notes_dir, s).failed
    graph, _ = build_app(s, InMemorySaver())
    judge = Faithfulness(llm=llm_factory(s.router_model, provider="anthropic",
                                         client=AsyncAnthropic(), max_tokens=4096))

    cases = [json.loads(line) for line in (ROOT / "evals/golden.jsonl").read_text("utf-8").splitlines() if line]
    scores, refused, refusals = [], 0, 0
    for i, c in enumerate(cases):
        out = graph.invoke({"messages": [("user", c["q"])], "requested": "auto"},
                           {"configurable": {"thread_id": f"eval-{i}"}})
        answer = out["messages"][-1].text
        contexts = [m.text for m in out["messages"] if m.type == "tool"]
        if c["expect"] is None:
            refusals += 1
            refused += "not in your notes" in answer.lower()
            continue
        score = asyncio.run(judge.ascore(user_input=c["q"], response=answer, retrieved_contexts=contexts)).value
        scores.append(score)
        print(f"{score:.2f}  [{out['expert_id']}] {c['q']}")

    mean = sum(scores) / len(scores)
    print(f"faithfulness={mean:.2f}  refusals={refused}/{refusals}")
    assert mean >= MIN_FAITHFULNESS
    assert refused == refusals
```

- [ ] **Step 2: Run it**

Run: `uv run --env-file .env pytest -m llm_eval evals -s`
Expected: PASS with `faithfulness=0.8x+ refusals=3/3`. If `llm_factory` rejects `max_tokens`, check `ragas/llms/base.py::llm_factory`'s kwargs handling and pass the token limit the way it expects; ledger the ruling. If a refusal fails, print the answer. It usually means RULES needs tightening, not the threshold.

- [ ] **Step 3: Verify the skip path**

Run: `uv run pytest -m llm_eval evals -q -rs` (without `--env-file`)
Expected: `1 skipped` with reason `needs ANTHROPIC_API_KEY`.

- [ ] **Step 4: Add to CI** — append to the `steps:` of the `test` job in `.github/workflows/ci.yml`:

```yaml
      - run: uv run pytest -m llm_eval evals -s -rs
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

(Without the repo secret the test skips; with it, CI sends only the public `samples/` passages to Anthropic.)

- [ ] **Step 5: Commit**

```bash
git add evals/test_answer_quality.py .github/workflows/ci.yml
git commit -m "feat: Ragas faithfulness and refusal eval gate"
```
