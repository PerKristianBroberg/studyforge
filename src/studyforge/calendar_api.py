"""/calendar page + JSON API: events (manual or found in documents), availability, planning, done ticks."""
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Callable, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from studyforge.calendar_store import Calendar, Event
from studyforge.config import Settings
from studyforge.library import tree
from studyforge.library_api import require_local_origin
from studyforge.planner import WEEKDAYS, plan

PAGE = Path("public/calendar.html")
Kind = Literal["exam", "arbeidskrav", "deadline", "other"]


class NewEvent(BaseModel):
    uni: str
    course: str
    kind: Kind
    title: str
    due: date
    due_time: str | None = None


class StatusIn(BaseModel):
    status: Literal["suggested", "confirmed", "dismissed"]


class DoneIn(BaseModel):
    done: bool


def _valid_prefs(p: dict) -> dict:
    try:
        hours = {d: float(p["hours"].get(d, 0)) for d in WEEKDAYS}
        block, start = int(p["block_minutes"]), int(p["start_hour"])
    except (KeyError, TypeError, ValueError, AttributeError):
        raise HTTPException(400, "prefs need hours per weekday, block_minutes and start_hour")
    if not all(0 <= h <= 16 for h in hours.values()) or not 15 <= block <= 240 or not 0 <= start <= 23:
        raise HTTPException(400, "hours 0-16 per day, block 15-240 min, start hour 0-23")
    return {"hours": hours, "block_minutes": block, "start_hour": start}


def _default_find(s: Settings):
    llm = None

    def find(uni: str, course: str, today: date) -> list[Event]:
        nonlocal llm
        from langchain_anthropic import ChatAnthropic

        from studyforge.deadlines import find_deadlines
        llm = llm or ChatAnthropic(model=s.router_model, max_tokens=2048)
        return find_deadlines(uni, course, s, llm, today)

    return find


def register(app: FastAPI, s: Settings, find: Callable[[str, str, date], list[Event]] | None = None,
             today: Callable[[], date] = date.today) -> None:
    router = APIRouter()
    find = find or _default_find(s)

    def cal() -> Calendar:
        return Calendar(s)

    def subjects() -> list[dict]:
        return [{"uni": u["uni"], "course": sub["name"]} for u in tree(s.notes_dir) for sub in u["subjects"]]

    def topics() -> dict:
        # study-block topics: the subject's note files, e.g. "linked-lists.md" -> "Linked lists"
        out = {}
        for u in tree(s.notes_dir):
            for sub in u["subjects"]:
                names = [Path(f["name"]).stem.replace("-", " ").replace("_", " ") for f in sub["files"]
                         if f["type"] == "notes"]
                out[(u["uni"], sub["name"])] = [n[:1].upper() + n[1:] for n in names]
        return out

    def state() -> dict:
        c = cal()
        return {"today": today().isoformat(), "events": [asdict(e) for e in c.events()],
                "blocks": [asdict(b) for b in c.blocks()], "prefs": c.prefs(), "subjects": subjects()}

    @router.get("/calendar")
    def page():
        return FileResponse(PAGE)

    @router.get("/api/calendar")
    def get_state():
        return state()

    @router.post("/api/calendar/events")
    def add(body: NewEvent, request: Request):
        require_local_origin(request)
        cal().add_event(Event(None, body.uni, body.course, body.kind, body.title.strip(), body.due, body.due_time,
                              "confirmed", "added by you"))
        return state()

    @router.patch("/api/calendar/events/{event_id}")
    def set_status(event_id: int, body: StatusIn, request: Request):
        require_local_origin(request)
        cal().set_status(event_id, body.status)
        return state()

    @router.delete("/api/calendar/events/{event_id}")
    def delete(event_id: int, request: Request):
        require_local_origin(request)
        cal().delete_event(event_id)
        return state()

    @router.put("/api/calendar/prefs")
    def set_prefs(body: dict, request: Request):
        require_local_origin(request)
        cal().set_prefs(_valid_prefs(body))
        return state()

    @router.post("/api/calendar/plan")
    def make_plan(request: Request):
        require_local_origin(request)
        c, t = cal(), today()
        keep = [b for b in c.blocks() if b.done or b.day < t]
        c.replace_future_blocks(t, plan(c.events(), c.prefs(), t, topics(), keep))
        return state()

    @router.patch("/api/calendar/blocks/{block_id}")
    def set_done(block_id: int, body: DoneIn, request: Request):
        require_local_origin(request)
        cal().set_done(block_id, body.done)
        return state()

    @router.post("/api/calendar/find")
    def find_all(request: Request):
        require_local_origin(request)
        c, t, added = cal(), today(), 0
        known = {(e.uni, e.course, e.kind, e.due, e.title.lower()) for e in c.events()}  # incl. dismissed
        for sub in subjects():
            for e in find(sub["uni"], sub["course"], t):
                key = (e.uni, e.course, e.kind, e.due, e.title.lower())
                if key not in known:
                    known.add(key)
                    c.add_event(e)
                    added += 1
        return {"added": added, **state()}

    for route in reversed(router.routes):  # before Chainlit's catch-all GET
        app.router.routes.insert(0, route)
