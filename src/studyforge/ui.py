"""Chainlit UI: uv run --env-file .env chainlit run src/studyforge/ui.py --host 127.0.0.1"""
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import chainlit as cl
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

from studyforge import app, calendar_api, ingest, library_api
from studyforge.config import Settings
from studyforge.experts import discover
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


_build_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cached_graph():
    return app.build_app(S, _saver())  # (graph, experts); cleared when an upload adds a course


def _graph():
    with _build_lock:  # lru_cache alone lets two first requests build (and load the models) twice
        return _cached_graph()


# the /library page (public/library.html + JSON API) lives on Chainlit's own FastAPI server
from chainlit.server import app as _server  # noqa: E402

library_api.register(_server, S, on_change=lambda: _cached_graph.cache_clear())
calendar_api.register(_server, S)


@cl.on_app_startup
async def warm_up():
    # load models + personas in the background so the first message doesn't wait for them
    threading.Thread(target=_graph, daemon=True).start()


@cl.set_chat_profiles
async def profiles(user=None, language=None):
    experts = discover(S.notes_dir)  # folders only: instant, doesn't wait for the graph
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
    experts = discover(S.notes_dir)
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
        _cached_graph.cache_clear()  # a new course needs a new expert; shows up in the picker on the next chat
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
    await cl.Message(content=answer, author=_author(out["expert_id"]),
                     elements=_source_elements(answer, tool_texts)).send()


AVATARS = Path("public/avatars")
MONOGRAM = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="32" fill="#EAF1EC"/><text x="32" y="40" text-anchor="middle"
font-family="Inter, system-ui, sans-serif" font-size="{size}" font-weight="500"
fill="#3F6B55">{text}</text></svg>"""


def _author(expert_id: str) -> str:
    """Short author name (Chainlit avatar ids can't contain '/'), with a monogram avatar on disk."""
    if expert_id == "general":
        return "studyforge"  # falls back to the favicon mark
    name = expert_id.split("/")[-1]
    avatar = AVATARS / f"{name.strip().lower().replace(' ', '_').replace('.', '_')}.svg"
    if not avatar.exists():
        text = (name[:3] if len(name) <= 6 else name[:2]).upper()
        AVATARS.mkdir(parents=True, exist_ok=True)
        avatar.write_text(MONOGRAM.format(text=text, size=22 if len(text) == 3 else 26), encoding="utf-8")
    return name
