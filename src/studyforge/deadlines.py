"""Find exams, arbeidskrav and deadlines in a subject's documents, as suggestions for the user to confirm."""
from datetime import date
from typing import Callable, Literal

from pydantic import BaseModel

from studyforge import retrieval
from studyforge.calendar_store import Event
from studyforge.retrieval import Scope
from studyforge.tools import format_hits

# Norwegian + English phrasings of "when is it due"
QUERIES = [
    "arbeidskrav frist innlevering",
    "obligatorisk oppgave godkjent frist",
    "eksamen dato skriftlig eksamen",
    "midtveiseksamen semesterplan uke",
    "assignment deadline due date submission",
    "exam date final exam schedule",
]
MAX_PASSAGES = 12


class Found(BaseModel):
    kind: Literal["exam", "arbeidskrav", "deadline", "other"]
    title: str
    date: str  # YYYY-MM-DD
    time: str | None = None
    source: str
    uncertain: bool = False


class FoundList(BaseModel):
    items: list[Found]


def find_deadlines(uni: str, course: str, s, llm, today: date, search: Callable = None) -> list[Event]:
    search = search or retrieval.search
    hits, seen = [], set()
    for q in QUERIES:
        for h in search(q, Scope(uni=uni, course=course), s):
            if h.cite() not in seen:
                seen.add(h.cite())
                hits.append(h)
    if not hits:
        return []  # nothing to read: don't spend a Claude call
    prompt = (
        f"Today is {today.isoformat()}. Below are passages from course documents for {uni} {course}.\n"
        "List every exam, arbeidskrav (mandatory assignment), and other deadline they state, with its date as "
        "YYYY-MM-DD and time HH:MM if given. Resolve week numbers ('uke 42') and weekdays ('fredag i uke 41') to a "
        "calendar date in the current or next semester relative to today. Set uncertain=true when the date is "
        "inferred or ambiguous. Use the passage citation (e.g. 'UiB/INF100/emneplan.pdf p.3') as source. "
        "Only include items stated in the passages; return an empty list if there are none.\n\n"
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
