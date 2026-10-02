"""Find exams, arbeidskrav and deadlines in course documents, as suggestions for the user to confirm.

Each indexed file version is read once (scan_new), straight from its stored chunks. Only chunks that name a
deadline AND contain something date-like go to Claude, so most files cost no AI call at all."""
import re
import threading
from datetime import date
from typing import Literal

import sqlalchemy as sa
from pydantic import BaseModel

from studyforge.calendar_store import Calendar, Event
from studyforge.config import Settings
from studyforge.retrieval import Hit
from studyforge.store import Manifest, engine
from studyforge.tools import format_hits

# Norwegian + English. A chunk must match both to be worth a Claude call.
KEYWORD = re.compile(
    r"frist|innlever|arbeidskrav|obligatorisk|oblig\b|eksamen|midtveis|prøve|"
    r"deadline|due\b|submi|assignment|exam|midterm|quiz\b", re.I)
_MONTH = r"(?:jan|feb|mar|apr|mai|may|jun|jul|aug|sep|okt|oct|nov|des|dec)[a-z]*\.?"
DATE = re.compile(
    rf"\b\d{{1,2}}\.\s?\d{{1,2}}\.|"            # 16.10. / 16.10.2026
    rf"\b\d{{1,2}}\.?\s+{_MONTH}|"              # 8. desember / 9 October
    rf"\b{_MONTH}\s+\d{{1,2}}\b|"               # October 9 / Dec 8
    rf"\b20\d\d-\d\d-\d\d\b|"                   # 2026-10-16
    rf"\b\d{{1,2}}/\d{{1,2}}\b|"                # 10/16
    rf"\b(?:uke|veke|week)\s?\d{{1,2}}\b", re.I)  # uke 41 / week 41
MAX_PASSAGES = 12
_scan_lock = threading.Lock()  # the indexer and the "find" button may both trigger a scan


def mentions_deadline(text: str) -> bool:
    return bool(KEYWORD.search(text) and DATE.search(text))


class Found(BaseModel):
    kind: Literal["exam", "arbeidskrav", "deadline", "other"]
    title: str
    date: str  # YYYY-MM-DD
    time: str | None = None
    source: str
    uncertain: bool = False


class FoundList(BaseModel):
    items: list[Found]


def find_deadlines(uni: str, course: str, hits: list[Hit], llm, today: date) -> list[Event]:
    hits = [h for h in hits if mentions_deadline(h.text)]
    if not hits:
        return []  # nothing date-like: don't spend a Claude call
    prompt = (
        f"Today is {today.isoformat()}. Below are passages from course documents for {uni} {course}.\n"
        "List every exam, arbeidskrav (mandatory assignment), and other deadline they state, with its date as "
        "YYYY-MM-DD and time HH:MM if given. Passages may be in Norwegian or English. Resolve week numbers "
        "('uke 42', 'week 42') and weekdays ('fredag i uke 41') to a calendar date in the current or next semester "
        "relative to today. Set uncertain=true when the date is inferred (e.g. from a week number) or ambiguous. "
        "Include exams (eksamen) as kind 'exam'. Use the passage citation (e.g. 'UiB/INF100/emneplan.pdf p.3') "
        "as source. Only include items stated in the passages; return an empty list if there are none.\n\n"
        + format_hits(hits[:MAX_PASSAGES])
    )
    found = llm.with_structured_output(FoundList, method="json_schema").invoke(
        prompt, config={"run_name": "find-deadlines"})
    events = []
    for f in found.items:
        try:
            due = date.fromisoformat(f.date)
        except ValueError:
            continue  # model couldn't pin a date
        if due >= today:
            events.append(Event(None, uni, course, f.kind, f.title.strip(), due, f.time, "suggested",
                                f.source, f.uncertain))
    return events


def _chunks(s: Settings, path: str) -> list[Hit]:
    """A file's indexed chunks, in order, as Hits (no search, no reranker)."""
    with engine(s.db_url).connect() as c:
        rows = c.execute(sa.text(f"SELECT text, metadata_ FROM data_{s.chunks_table} "
                                 "WHERE metadata_->>'path' = :p ORDER BY id"), {"p": path})
        return [Hit(t, 0.0, m["path"], m.get("page"), m["uni"], m.get("course")) for t, m in rows]


def scan_new(s: Settings, llm, today: date) -> int:
    """Look for deadlines in files indexed (or changed) since the last scan. Returns how many events were added."""
    with _scan_lock:
        cal = Calendar(s)
        seen = cal.scanned()
        known = {(e.uni, e.course, e.kind, e.due, e.title.lower()) for e in cal.events()}  # incl. dismissed
        added = 0
        for path, (sha, _) in sorted(Manifest(s).all().items()):
            if seen.get(path) == sha:
                continue
            hits = _chunks(s, path)
            if hits and hits[0].course:  # events belong to a course; uni-wide files are skipped
                for e in find_deadlines(hits[0].uni, hits[0].course, hits, llm, today):
                    key = (e.uni, e.course, e.kind, e.due, e.title.lower())
                    if key not in known:
                        known.add(key)
                        cal.add_event(e)
                        added += 1
            cal.mark_scanned(path, sha)  # after success only: a failed Claude call retries next time
        return added
