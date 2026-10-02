"""/library page + JSON API for managing notes/<uni>/<subject>/ from the browser.

Mutating endpoints only accept requests whose Origin is this machine: without that, any website open in the
browser could POST/DELETE to 127.0.0.1 and wipe the notes (Chainlit's CORS allows all origins)."""
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from studyforge import ingest
from studyforge.config import Settings
from studyforge.library import add_subject, delete_file, delete_subject, tree
from studyforge.store import Manifest
from studyforge.uploads import UploadRejected, save_upload, save_zip

PAGE = Path("public/library.html")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class NewSubject(BaseModel):
    uni: str
    course: str


def require_local_origin(request: Request) -> None:
    if urlparse(request.headers.get("origin", "")).hostname not in LOCAL_HOSTS:
        raise HTTPException(403, "requests must come from the studyforge page on this machine")


class Indexer:
    """Runs ingest one at a time off the request thread, so big uploads return at once and a crash mid-batch
    doesn't lose the upload. A request while running is queued as one more pass (it picks up every change)."""

    def __init__(self, s: Settings, on_change: Callable[[], None], background: bool):
        self.s, self.on_change, self.background = s, on_change, background
        self.lock, self.pending = threading.Lock(), False
        self.state = {"running": False, "done": 0, "total": 0, "current": "", "report": "", "error": ""}
        self.started = 0.0
        self.failed: dict[str, str] = {}  # path -> reason, from the last finished pass

    def status(self) -> dict:
        # elapsed counts from the start of the current pass; no time-left estimate, since file times vary 10x
        return {**self.state, "elapsed": round(time.monotonic() - self.started) if self.state["running"] else 0}

    def request(self) -> None:
        with self.lock:
            if self.state["running"]:
                self.pending = True
                return
            self.state.update(running=True, done=0, total=0, current="", error="")
            self.started = time.monotonic()
        if self.background:
            threading.Thread(target=self._work, daemon=True).start()
        else:
            self._work()

    def _progress(self, done: int, total: int, path: str) -> None:
        self.state.update(done=done, total=total, current=path)

    def _work(self) -> None:
        while True:
            try:
                report = ingest.run(self.s.notes_dir, self.s, progress=self._progress)
                self.state["report"] = str(report)
                self.failed = dict(getattr(report, "failed", []))
                self.on_change()
            except Exception as e:  # surfaced on the page; files stay on disk for the next run
                self.state["error"] = f"{type(e).__name__}: {e}"
            with self.lock:
                if not self.pending:
                    self.state["running"] = False
                    return
                self.pending = False
                self.started = time.monotonic()


def register(app: FastAPI, s: Settings, on_change: Callable[[], None] = lambda: None,
             background: bool = True, indexed: Callable[[], set[str]] | None = None) -> Indexer:
    router = APIRouter()
    indexer = Indexer(s, on_change, background)
    indexed = indexed or (lambda: set(Manifest(s).all()))

    def listing() -> list[dict]:
        """tree() plus each file's index status: indexing | failed | indexed | waiting."""
        unis, done, st = tree(s.notes_dir), indexed(), indexer.status()
        for u in unis:
            for f in [*u["files"], *(f for sub in u["subjects"] for f in sub["files"])]:
                # ponytail: "indexed" = in the manifest; a file edited on disk since then still shows indexed
                if st["running"] and f["path"] == st["current"]:
                    f["status"] = "indexing"
                elif f["path"] in indexer.failed:
                    f.update(status="failed", error=indexer.failed[f["path"]])
                else:
                    f["status"] = "indexed" if f["path"] in done else "waiting"
        return unis

    def changed() -> str:
        indexer.request()
        return indexer.status()["report"] if not background else ""

    def rejected(e: UploadRejected):
        return HTTPException(400, str(e))

    @router.get("/library")
    def page():
        return FileResponse(PAGE)

    @router.get("/api/library")
    def get_tree():
        return {"tree": listing()}

    @router.post("/api/library/subjects")
    def create(body: NewSubject, request: Request):
        require_local_origin(request)
        try:
            add_subject(s.notes_dir, body.uni, body.course)
        except UploadRejected as e:
            raise rejected(e)
        return {"tree": listing()}

    @router.post("/api/library/upload")
    def upload(request: Request, uni: str = Form(...), course: str = Form(...), doc_type: str = Form("notes"),
               files: list[UploadFile] = File(...)):
        require_local_origin(request)
        saved, bad = 0, []
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                src = Path(tmp) / "upload"
                with src.open("wb") as out:
                    shutil.copyfileobj(f.file, out)
                try:
                    if (f.filename or "").lower().endswith(".zip"):
                        extracted, skipped = save_zip(s.notes_dir, uni, course, src, doc_type=doc_type)
                        saved += len(extracted)
                        bad += skipped
                    else:
                        save_upload(s.notes_dir, uni, course, f.filename or "", src, doc_type=doc_type)
                        saved += 1
                except UploadRejected as e:
                    bad.append(f"{f.filename}: {e}")
        report = changed() if saved else ""
        return {"saved": saved, "rejected": bad, "report": report, "indexing": indexer.status(),
                "tree": listing()}

    @router.get("/api/library/indexing")
    def indexing():
        return indexer.status()

    @router.delete("/api/library/file")
    def remove_file(path: str, request: Request):
        require_local_origin(request)
        try:
            delete_file(s.notes_dir, path)
        except UploadRejected as e:
            raise rejected(e)
        return {"report": changed(), "tree": listing()}

    @router.delete("/api/library/subject")
    def remove_subject(uni: str, course: str, request: Request):
        require_local_origin(request)
        try:
            delete_subject(s.notes_dir, uni, course)
        except UploadRejected as e:
            raise rejected(e)
        return {"report": changed(), "tree": listing()}

    # Chainlit registers a catch-all GET for its SPA, so our routes must come first
    for route in reversed(router.routes):
        app.router.routes.insert(0, route)
    return indexer
