"""/library page + JSON API for managing notes/<uni>/<subject>/ from the browser.

Mutating endpoints only accept requests whose Origin is this machine: without that, any website open in the
browser could POST/DELETE to 127.0.0.1 and wipe the notes (Chainlit's CORS allows all origins)."""
import shutil
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from studyforge import ingest
from studyforge.config import Settings
from studyforge.library import add_subject, delete_file, delete_subject, tree
from studyforge.uploads import UploadRejected, save_upload

PAGE = Path("public/library.html")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class NewSubject(BaseModel):
    uni: str
    course: str


def require_local_origin(request: Request) -> None:
    if urlparse(request.headers.get("origin", "")).hostname not in LOCAL_HOSTS:
        raise HTTPException(403, "requests must come from the studyforge page on this machine")


def register(app: FastAPI, s: Settings, on_change: Callable[[], None] = lambda: None) -> None:
    router = APIRouter()

    def changed() -> str:
        report = str(ingest.run(s.notes_dir, s))
        on_change()
        return report

    def rejected(e: UploadRejected):
        return HTTPException(400, str(e))

    @router.get("/library")
    def page():
        return FileResponse(PAGE)

    @router.get("/api/library")
    def get_tree():
        return {"tree": tree(s.notes_dir)}

    @router.post("/api/library/subjects")
    def create(body: NewSubject, request: Request):
        require_local_origin(request)
        try:
            add_subject(s.notes_dir, body.uni, body.course)
        except UploadRejected as e:
            raise rejected(e)
        return {"tree": tree(s.notes_dir)}

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
                    save_upload(s.notes_dir, uni, course, f.filename or "", src, doc_type=doc_type)
                    saved += 1
                except UploadRejected as e:
                    bad.append(f"{f.filename}: {e}")
        return {"saved": saved, "rejected": bad, "report": changed() if saved else "", "tree": tree(s.notes_dir)}

    @router.delete("/api/library/file")
    def remove_file(path: str, request: Request):
        require_local_origin(request)
        try:
            delete_file(s.notes_dir, path)
        except UploadRejected as e:
            raise rejected(e)
        return {"report": changed(), "tree": tree(s.notes_dir)}

    @router.delete("/api/library/subject")
    def remove_subject(uni: str, course: str, request: Request):
        require_local_origin(request)
        try:
            delete_subject(s.notes_dir, uni, course)
        except UploadRejected as e:
            raise rejected(e)
        return {"report": changed(), "tree": tree(s.notes_dir)}

    # Chainlit registers a catch-all GET for its SPA, so our routes must come first
    for route in reversed(router.routes):
        app.router.routes.insert(0, route)
