"""Chainlit UI: uv run --env-file .env chainlit run src/studyforge/ui.py --host 127.0.0.1"""
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import chainlit as cl
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

from studyforge import app, ingest
from studyforge.config import Settings
from studyforge.experts import discover
from studyforge.sources import cited, passages, split_label
from studyforge.store import Manifest
from studyforge.library import add_subject, delete_file, delete_subject, tree
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


# --- Subjects panel (public/elements/Subjects.jsx) ---------------------------------------------

UPLOAD_ACCEPT = {
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "text/plain": [".md", ".txt", ".py", ".java", ".js", ".ts", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".r", ".sql"],
}


@cl.on_chat_start
async def start():
    await cl.context.emitter.set_commands([
        {"id": "Subjects", "icon": "folder", "description": "Add subjects and upload notes", "button": True, "persistent": False},
    ])
    await _show_subjects()


async def _show_subjects(status: str | None = None) -> None:
    el = cl.CustomElement(name="Subjects", props={"tree": tree(S.notes_dir), "status": status}, display="inline")
    cl.user_session.set("subjects_el", el)
    await cl.Message(content="", author="studyforge", elements=[el]).send()


async def _refresh_subjects(status: str) -> None:
    el = cl.user_session.get("subjects_el")
    if not el:
        return await _show_subjects(status)
    el.props = {"tree": tree(S.notes_dir), "status": status}
    await el.update()


async def _reindex() -> str:
    report = await cl.make_async(ingest.run)(S.notes_dir, S)
    if report.new or report.removed:
        _cached_graph.cache_clear()  # subjects changed: experts are rebuilt for the next chat
    return str(report).replace("\n", " ")


@cl.action_callback("subjects_add")
async def on_add(action: cl.Action):
    try:
        folder = add_subject(S.notes_dir, action.payload["uni"], action.payload["course"])
        await _refresh_subjects(f"Added {folder.parent.name}/{folder.name}. Upload notes to it to create its expert.")
    except UploadRejected as e:
        await _refresh_subjects(f"Not added: {e}")


@cl.action_callback("subjects_upload")
async def on_upload(action: cl.Action):
    uni, course, doc_type = action.payload["uni"], action.payload["course"], action.payload["doc_type"]
    files = await cl.AskFileMessage(
        content=f"Upload **{doc_type}** for **{uni}/{course}** — PDF, PPTX, DOCX, Markdown or code.",
        accept=UPLOAD_ACCEPT, max_size_mb=50, max_files=20, timeout=600,
    ).send()
    if not files:
        return await _refresh_subjects("Upload cancelled.")
    saved, rejected = 0, []
    for f in files:
        try:
            save_upload(S.notes_dir, uni, course, f.name, Path(f.path), doc_type=doc_type)
            saved += 1
        except UploadRejected as e:
            rejected.append(str(e))
    status = f"Saved {saved} file(s) to {uni}/{course}. " + (await _reindex() if saved else "")
    await _refresh_subjects(status + (" Rejected: " + "; ".join(rejected) if rejected else ""))


@cl.action_callback("subjects_delete_file")
async def on_delete_file(action: cl.Action):
    try:
        delete_file(S.notes_dir, action.payload["path"])
        await _refresh_subjects(f"Deleted {action.payload['path']}. " + await _reindex())
    except UploadRejected as e:
        await _refresh_subjects(f"Not deleted: {e}")


@cl.action_callback("subjects_delete_subject")
async def on_delete_subject(action: cl.Action):
    uni, course = action.payload["uni"], action.payload["course"]
    try:
        delete_subject(S.notes_dir, uni, course)
        await _refresh_subjects(f"Deleted subject {uni}/{course}. " + await _reindex())
    except UploadRejected as e:
        await _refresh_subjects(f"Not deleted: {e}")


@cl.on_message
async def on_message(msg: cl.Message):
    if msg.command == "Subjects":
        return await _show_subjects()
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
