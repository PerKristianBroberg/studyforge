# studyforge Plan 3 — Chainlit UI (chat, experts, sources, upload)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web chat at http://127.0.0.1:8000 where the student picks an expert (or Auto), watches the agent steps, clicks citations to open the source (PDFs at the cited page), drops files to add them to a course, and lists the library.

**Architecture:** Chainlit (Python, no separate frontend) calls the existing `app.ask()` in a worker thread (`cl.make_async`). The LangGraph graph and a single Postgres checkpointer connection are built once and cached, and rebuilt when an upload creates a new course. Chat profiles are generated from `discover()`. Chainlit's `LangchainCallbackHandler` renders every graph node and tool call as a Step. Uploads go through a pure, tested `uploads.py` (name/extension/size/path-traversal checks), then land in `notes/<uni>/<course>/` and trigger the incremental `ingest.run`. Citation parsing lives in a pure, tested `sources.py`.

**Tech Stack:** chainlit 2.12, existing Plan 1–2 stack.

**Spec:** `docs/superpowers/specs/2026-09-29-studyforge-design.md` §6 (phase 5). Implements chat profiles, Steps, citations/elements, starters, `/library`, upload from chat.
**Deferred:** welcome-message dashboard and week-view (need phases 7–8 data); flashcard/quiz custom elements (phase 7); Chainlit persisted thread history and login (LangGraph already keeps conversation memory per session).

## Global Constraints

- Everything from Plans 1–2 still holds. The UI binds to `127.0.0.1` only.
- Uploads: only `paths.SUPPORTED_EXTS`, max 50 MB per file, filenames without path separators, target must resolve inside `notes_dir`. Existing files with the same name are replaced (the new version is re-ingested).
- `uni`/`course` typed by the user must be a single path segment of letters/digits/space/`._()-` and must not start with `.` or `~$`.

## Review Focus

1. **Path traversal via filename or course name** (`../../x.pdf`, `..\\x.pdf`, `UiO/../..`) → rejected, nothing written outside `notes/`. Tests: Task 1 `test_rejects_traversal_names`, `test_parse_scope_rejects_bad_input`.
2. **Unsupported or oversized upload** (`.exe`, a 60 MB PDF) → rejected with a reason in chat, other files in the same drop still saved. Tests: Task 1 `test_rejects_bad_extension_and_size`; UI loop continues per file (Task 3 code).
3. **Citation formats the model varies** (`[x.md]`, `[x.pdf p.3]`, several in one bracket separated by `;`) → each is parsed. Test: Task 2 `test_cited_handles_pages_and_multiple`.
4. **A cited file that no longer exists** (deleted after ingest) → the source still shows as text from the retrieved passage, and the UI doesn't crash. Test: Task 2 `test_passages_parse_tool_output` plus the Task 3 fallback to `cl.Text`.
5. **Tracing tags per interface** → web turns are tagged `interface:web`. Test: Task 3 `test_ask_passes_interface_and_extra_callbacks`.

---

### Task 1: Safe uploads

**Files:** Create `src/studyforge/uploads.py`; Test `tests/test_uploads.py`

**Interfaces:**
- Produces: `class UploadRejected(ValueError)`; `MAX_BYTES = 50 * 1024 * 1024`; `parse_scope(text: str) -> tuple[str, str]` (`"UiO/INF1010"` → `("UiO", "INF1010")`); `save_upload(notes_dir: Path, uni: str, course: str, filename: str, src: Path, max_bytes: int = MAX_BYTES) -> Path` (copies, returns the target path).

- [ ] **Step 1: Write the failing tests** — `tests/test_uploads.py`

```python
import pytest

from studyforge.uploads import UploadRejected, parse_scope, save_upload


def src_file(tmp_path, name="a.pdf", size=10):
    p = tmp_path / "upload-tmp" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


def test_saves_into_course_folder(tmp_path):
    notes = tmp_path / "notes"
    target = save_upload(notes, "UiO", "INF1010", "Uke 1 – æøå.pdf", src_file(tmp_path))
    assert target == notes / "UiO" / "INF1010" / "Uke 1 – æøå.pdf"
    assert target.read_bytes() == b"x" * 10


def test_rejects_traversal_names(tmp_path):
    notes = tmp_path / "notes"
    for bad in ["../../evil.pdf", "..\\evil.pdf", "sub/evil.pdf", ".hidden.pdf", "~$lock.docx", ""]:
        with pytest.raises(UploadRejected):
            save_upload(notes, "UiO", "INF1010", bad, src_file(tmp_path))
    for uni, course in [("..", "INF1010"), ("UiO", ".."), ("UiO", "a/b")]:
        with pytest.raises(UploadRejected):
            save_upload(notes, uni, course, "ok.pdf", src_file(tmp_path))
    assert not (tmp_path / "evil.pdf").exists()


def test_rejects_bad_extension_and_size(tmp_path):
    notes = tmp_path / "notes"
    with pytest.raises(UploadRejected, match="type"):
        save_upload(notes, "UiO", "INF1010", "virus.exe", src_file(tmp_path, "virus.exe"))
    with pytest.raises(UploadRejected, match="MB"):
        save_upload(notes, "UiO", "INF1010", "big.pdf", src_file(tmp_path, size=11), max_bytes=10)


def test_parse_scope():
    assert parse_scope(" UiO / INF1010 ") == ("UiO", "INF1010")


def test_parse_scope_rejects_bad_input():
    for bad in ["", "UiO", "UiO/INF1010/x", "../INF1010", "UiO/..", "UiO/.git"]:
        with pytest.raises(UploadRejected):
            parse_scope(bad)
```

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_uploads.py -q` → FAIL: `ModuleNotFoundError: No module named 'studyforge.uploads'`.

- [ ] **Step 3: Implement** — `src/studyforge/uploads.py`

```python
import re
import shutil
from pathlib import Path

from studyforge.paths import SUPPORTED_EXTS

MAX_BYTES = 50 * 1024 * 1024
SEGMENT = re.compile(r"[\w .()\-–]+")  # \w covers æøå


class UploadRejected(ValueError):
    pass


def _segment(part: str, what: str) -> str:
    part = part.strip()
    if not SEGMENT.fullmatch(part) or part.startswith((".", "~$")):
        raise UploadRejected(f"invalid {what}: {part!r}")
    return part


def parse_scope(text: str) -> tuple[str, str]:
    parts = text.split("/")
    if len(parts) != 2:
        raise UploadRejected("expected <uni>/<course>, e.g. UiO/INF1010")
    return _segment(parts[0], "uni"), _segment(parts[1], "course")


def save_upload(notes_dir: Path, uni: str, course: str, filename: str, src: Path,
                max_bytes: int = MAX_BYTES) -> Path:
    name = _segment(filename, "file name")
    if Path(name).suffix.lower() not in SUPPORTED_EXTS:
        raise UploadRejected(f"file type not supported: {name}")
    if src.stat().st_size > max_bytes:
        raise UploadRejected(f"{name} is larger than {max_bytes // (1024 * 1024)} MB")
    root = notes_dir.resolve()
    target = (root / _segment(uni, "uni") / _segment(course, "course") / name).resolve()
    if not target.is_relative_to(root):  # belt and braces on top of the segment checks
        raise UploadRejected(f"invalid path: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    return target
```

- [ ] **Step 4: Run to verify pass**: `uv run pytest tests/test_uploads.py -q` → 5 passed. (If `big.pdf` with `max_bytes=10` reports `0 MB`, that is expected: the message uses whole MB. The test only matches "MB".)

- [ ] **Step 5: Commit**: `git add src/studyforge/uploads.py tests/test_uploads.py && git commit -m "feat: safe upload saving into notes/<uni>/<course>"`

---

### Task 2: Citation parsing

**Files:** Create `src/studyforge/sources.py`; Test `tests/test_sources.py`

**Interfaces:**
- Produces: `cited(answer: str) -> list[str]`: unique citation labels in order of appearance, e.g. `["uni-b/STAT200/regression.md", "UiO/A/x.pdf p.3"]`. `split_label(label: str) -> tuple[str, int | None]`. `passages(tool_texts: list[str]) -> dict[str, str]`: label → passage text, parsed from `search_notes` output (`"[1] <label>\n<text>\n\n[2] ..."`).

- [ ] **Step 1: Write the failing tests** — `tests/test_sources.py`

```python
from studyforge.sources import cited, passages, split_label


def test_cited_handles_pages_and_multiple():
    answer = ("R² is explained variance [uni-b/STAT200/regression.md]. See also "
              "[UiO/A/slides.pdf p.3; UiO/A/notes.md] and again [uni-b/STAT200/regression.md].")
    assert cited(answer) == ["uni-b/STAT200/regression.md", "UiO/A/slides.pdf p.3", "UiO/A/notes.md"]


def test_cited_ignores_non_citations():
    assert cited("Use [1] or [this link] or a list [a, b].") == []


def test_split_label():
    assert split_label("UiO/A/slides.pdf p.3") == ("UiO/A/slides.pdf", 3)
    assert split_label("UiO/A/notes.md") == ("UiO/A/notes.md", None)


def test_passages_parse_tool_output():
    tool = "[1] UiO/A/slides.pdf p.3\nStacks are LIFO.\nMore text.\n\n[2] UiO/A/notes.md\nQueues are FIFO."
    assert passages([tool, "NOT_IN_NOTES: nothing"]) == {
        "UiO/A/slides.pdf p.3": "Stacks are LIFO.\nMore text.",
        "UiO/A/notes.md": "Queues are FIFO.",
    }
```

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_sources.py -q` → FAIL: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/studyforge/sources.py`

```python
import re

LABEL = re.compile(r"[^\[\];]+?\.\w{1,5}(?: p\.\d+)?")  # "<path>.<ext>" optionally " p.<n>"
BRACKET = re.compile(r"\[([^\[\]]+)\]")
ENTRY = re.compile(r"^\[\d+\] (.+)$", re.M)


def cited(answer: str) -> list[str]:
    labels = []
    for group in BRACKET.findall(answer):
        for part in group.split(";"):
            part = part.strip()
            if "/" in part and LABEL.fullmatch(part) and part not in labels:
                labels.append(part)
    return labels


def split_label(label: str) -> tuple[str, int | None]:
    path, _, page = label.partition(" p.")
    return path, int(page) if page else None


def passages(tool_texts: list[str]) -> dict[str, str]:
    found = {}
    for text in tool_texts:
        heads = list(ENTRY.finditer(text))
        for head, nxt in zip(heads, heads[1:] + [None]):
            body = text[head.end():nxt.start() if nxt else len(text)]
            found[head.group(1).strip()] = body.strip()
    return found
```

- [ ] **Step 4: Run to verify pass**: `uv run pytest tests/test_sources.py -q` → 4 passed.

- [ ] **Step 5: Commit**: `git add src/studyforge/sources.py tests/test_sources.py && git commit -m "feat: parse citations and retrieved passages"`

---

### Task 3: Chainlit app

**Files:** Modify `src/studyforge/app.py`, `tests/test_tracing.py`; Create `src/studyforge/ui.py`, `.chainlit/config.toml` (generated then edited), `chainlit.md`; Modify `pyproject.toml` (dep), `README.md`.

**Interfaces:**
- Consumes: `app.build_app`, `app.ask`, `discover`, `ingest.run`, `Manifest`, `uploads`, `sources`.
- Produces: `app.ask(graph, question, requested, thread_id, interface="cli", extra_callbacks=()) -> dict`. Callbacks = tracing callbacks + `extra_callbacks`. The web UI is started with `uv run --env-file .env chainlit run src/studyforge/ui.py --host 127.0.0.1`.

- [ ] **Step 1: Add dependency**: `uv add chainlit` → installs chainlit 2.12.x.

- [ ] **Step 2: Write the failing test** — append to `tests/test_tracing.py`:

```python
def test_ask_passes_interface_and_extra_callbacks(recorded):
    seen = {}

    class Graph:
        def invoke(self, state, config):
            seen["callbacks"] = config.get("callbacks")
            return {"expert_id": "general", "messages": [AIMessage(content="ok")]}

    marker = object()
    app.ask(Graph(), "hi", "auto", "t9", interface="web", extra_callbacks=[marker])
    assert recorded["attrs"] == [{"session_id": "t9", "tags": ["interface:web"]}]
    assert marker in seen["callbacks"]
```

- [ ] **Step 3: Run to verify failure**: `uv run pytest tests/test_tracing.py -q` → FAIL: `TypeError: ask() got an unexpected keyword argument 'extra_callbacks'`.

- [ ] **Step 4: Implement in `src/studyforge/app.py`**: change `ask` to:

```python
def ask(graph, question: str, requested: str, thread_id: str, interface: str = "cli",
        extra_callbacks=()) -> dict:
    """One chat turn = one Langfuse trace (input: question, output: answer); turns group by session = thread."""
    with tracing.observe("chat-turn", "agent", input=question,
                         session_id=thread_id, tags=[f"interface:{interface}"]) as turn:
        cfg = run_config(thread_id)
        cfg["callbacks"] = [*cfg.get("callbacks", []), *extra_callbacks]
        out = graph.invoke({"messages": [("user", question)], "requested": requested}, cfg)
        turn.update(output=out["messages"][-1].text,
                    metadata={"expert_id": out["expert_id"], "requested": requested})
    return out
```

Run: `uv run pytest -q` → all pass.

- [ ] **Step 5: Generate the Chainlit config**: `uv run chainlit init` → creates `.chainlit/config.toml` and `chainlit.md`. In `.chainlit/config.toml` set:
  - under `[features.spontaneous_file_upload]`: `enabled = true`, `accept = [".pdf", ".pptx", ".docx", ".md", ".txt", ".py", ".java", ".js", ".ts", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".r", ".sql"]`, `max_files = 20`, `max_size_mb = 50`
  - under `[UI]`: `name = "studyforge"`

  Replace `chainlit.md` with:

```markdown
# studyforge

Pick an expert in the top-left (or **Auto**), ask about your notes, and click a citation to open the source.
Drop PDF/PPTX/DOCX/Markdown/code files into the chat to add them to a course. Type `/library` to see what's indexed.
```

- [ ] **Step 6: Write `src/studyforge/ui.py`**

```python
"""Chainlit UI: uv run --env-file .env chainlit run src/studyforge/ui.py --host 127.0.0.1"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import chainlit as cl
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

from studyforge import app, ingest
from studyforge.config import Settings
from studyforge.sources import cited, passages, split_label
from studyforge.store import Manifest
from studyforge.uploads import UploadRejected, parse_scope, save_upload

S = Settings.from_env()
AUTO = "Auto"
OTHER = "__other__"


@lru_cache(maxsize=1)
def _saver() -> PostgresSaver:
    # ponytail: one shared connection; single local user. Use a psycopg ConnectionPool if this ever serves many.
    saver = PostgresSaver(Connection.connect(S.pg_conn, autocommit=True, prepare_threshold=0))
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def _graph():
    return app.build_app(S, _saver())  # (graph, experts); cleared when an upload adds a course


@cl.set_chat_profiles
async def profiles(user=None, language=None):
    _, experts = await cl.make_async(_graph)()
    return [
        cl.ChatProfile(name=AUTO, markdown_description="Routes each question to the best expert.", default=True),
        *(cl.ChatProfile(name=e.id, display_name=e.name,
                         markdown_description=f"Answers only from **{e.id}** notes.") for e in experts),
    ]


@cl.set_starters
async def starters(user=None, language=None):
    return [
        cl.Starter(label="What's in my notes?", message="Give me a short overview of the topics in my notes."),
        cl.Starter(label="Quiz me", message="Quiz me with 3 questions on an important topic from my notes."),
        cl.Starter(label="Library", message="/library"),
    ]


def _library() -> str:
    by_course = defaultdict(list)
    for path in sorted(Manifest(S).all()):
        by_course["/".join(path.split("/")[:2])].append(path.split("/", 2)[-1])
    if not by_course:
        return "No notes indexed yet. Drop files here or put them in `notes/<uni>/<course>/`."
    return "\n".join(f"**{k}** ({len(v)} files)\n" + "\n".join(f"- {f}" for f in v) for k, v in by_course.items())


async def _pick_scope(expert: str) -> tuple[str, str] | None:
    if expert.count("/") == 1:
        return tuple(expert.split("/"))
    _, experts = _graph()
    courses = [e.id for e in experts if e.kind == "course" and (expert in (AUTO, "general") or e.id.startswith(expert + "/"))]
    res = await cl.AskActionMessage(
        content="Which course are these files for?",
        actions=[*(cl.Action(name="course", payload={"id": c}, label=c) for c in courses),
                 cl.Action(name="course", payload={"id": OTHER}, label="Another course…")],
    ).send()
    if not res:
        return None
    if res["payload"]["id"] != OTHER:
        return tuple(res["payload"]["id"].split("/"))
    typed = await cl.AskUserMessage(content="Type it as `<uni>/<course>`, e.g. `UiO/INF1010`.").send()
    try:
        return parse_scope(typed["output"]) if typed else None
    except UploadRejected as e:
        await cl.Message(content=f"Not saved: {e}").send()
        return None


async def _handle_uploads(files, expert: str) -> None:
    scope = await _pick_scope(expert)
    if not scope:
        await cl.Message(content="Upload cancelled.").send()
        return
    uni, course = scope
    saved, lines = 0, []
    for f in files:
        try:
            save_upload(S.notes_dir, uni, course, f.name, Path(f.path))
            saved += 1
        except UploadRejected as e:
            lines.append(f"- rejected {e}")
    report = await cl.make_async(ingest.run)(S.notes_dir, S) if saved else None
    if report and report.new:
        _graph.cache_clear()  # a new course needs a new expert; shows up in the picker on the next chat
    await cl.Message(content="\n".join([f"Saved {saved} file(s) to **{uni}/{course}**.",
                                        *( [f"Ingest: {report}"] if report else [] ), *lines])).send()


def _source_elements(answer: str, tool_texts: list[str]) -> list:
    found, elements = passages(tool_texts), []
    for label in cited(answer):
        path, page = split_label(label)
        file = S.notes_dir / path
        if file.suffix.lower() == ".pdf" and file.exists():
            elements.append(cl.Pdf(name=label, path=str(file), page=page or 1, display="side"))
        else:  # non-PDF, or the file is gone: show the retrieved passage
            elements.append(cl.Text(name=label, content=found.get(label, "(passage not available)"), display="side"))
    return elements


@cl.on_message
async def on_message(msg: cl.Message):
    expert = cl.user_session.get("chat_profile") or AUTO
    files = [e for e in msg.elements if getattr(e, "path", None)]
    if files:
        await _handle_uploads(files, expert)
    text = msg.content.strip()
    if not text:
        return
    if text == "/library":
        await cl.Message(content=_library()).send()
        return
    graph, _ = await cl.make_async(_graph)()
    out = await cl.make_async(app.ask)(
        graph, text, "auto" if expert == AUTO else expert, cl.context.session.thread_id,
        interface="web", extra_callbacks=[cl.LangchainCallbackHandler()],
    )
    answer = out["messages"][-1].text
    tool_texts = [m.text for m in out["messages"] if m.type == "tool"]
    await cl.Message(content=answer, author=out["expert_id"],
                     elements=_source_elements(answer, tool_texts)).send()
```

- [ ] **Step 7: Run the UI**

Run (background): `uv run --env-file .env chainlit run src/studyforge/ui.py --host 127.0.0.1 --port 8000 --headless`
Then: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000` → `200`.

Manual check in the browser at http://127.0.0.1:8000 (with the Plan 2 exports for `samples/` or real `notes/`):
1. The profile picker lists Auto, general, each uni and each course expert.
2. "What does a p-value mean?" in Auto shows Steps (route, expert, search_notes), the answer is authored by the STAT200 expert, and clicking the citation opens the passage in the side panel.
3. Drop a `.md` file with Auto selected → course picker → "Saved 1 file(s)…" and an ingest report. `/library` lists it.
4. Drop a `.exe` → rejected by Chainlit's file picker, or "rejected …" in chat.

- [ ] **Step 8: README** — append:

````markdown
## Web UI

```bash
uv run --env-file .env chainlit run src/studyforge/ui.py --host 127.0.0.1
```

Open http://127.0.0.1:8000: pick an expert (or Auto), click citations to open sources, drop files to add them to a course,
`/library` lists what's indexed.
````

- [ ] **Step 9: Commit**

```bash
git add src/studyforge/app.py src/studyforge/ui.py tests/test_tracing.py .chainlit/config.toml chainlit.md pyproject.toml uv.lock README.md
git commit -m "feat: Chainlit UI with expert profiles, steps, sources and uploads"
```
