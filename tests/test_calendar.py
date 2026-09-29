from datetime import date

import pytest
import sqlalchemy as sa

from studyforge.calendar_store import DEFAULT_PREFS, Block, Calendar, Event
from studyforge.deadlines import Found, FoundList, find_deadlines
from studyforge.planner import plan
from studyforge.retrieval import Hit
from studyforge.store import engine

MON = date(2026, 10, 5)  # a Monday


def ev(id, kind, due, course="INF100", status="confirmed", title=None):
    return Event(id, "UiB", course, kind, title or f"{course} {kind}", due, None, status, None, False)


# ---------- planner (pure) ----------

PREFS = {"hours": {"mon": 3, "tue": 3, "wed": 3, "thu": 3, "fri": 3, "sat": 3, "sun": 0},
         "block_minutes": 90, "start_hour": 9}


def test_no_confirmed_events_no_plan():
    assert plan([ev(1, "exam", date(2026, 10, 20), status="suggested")], PREFS, MON) == []


def test_blocks_respect_hours_windows_and_due_dates():
    exam = ev(1, "exam", date(2026, 11, 20))  # 46 days away: only the last 28 days are eligible
    blocks = plan([exam], PREFS, MON)
    assert blocks and all(b.day < exam.due for b in blocks)
    assert min(b.day for b in blocks) >= date(2026, 10, 23)  # window start = due - 28 days
    assert not [b for b in blocks if b.day.weekday() == 6]  # Sunday has 0 hours
    per_day = {}
    for b in blocks:
        per_day[b.day] = per_day.get(b.day, 0) + 1
    assert max(per_day.values()) == 2  # 3 h / 90 min
    assert {b.start for b in blocks} == {"09:00", "10:30"}


def test_nearer_and_heavier_items_get_more_time_but_others_still_get_some():
    soon = ev(1, "exam", date(2026, 10, 16), course="INF100")
    later = ev(2, "exam", date(2026, 10, 30), course="STAT110")
    blocks = [b for b in plan([soon, later], PREFS, MON) if b.day < soon.due]
    by_course = {c: sum(b.course == c for b in blocks) for c in ("INF100", "STAT110")}
    assert by_course["INF100"] > by_course["STAT110"] > 0


def test_block_kinds_and_topics():
    exam = ev(1, "exam", date(2026, 10, 12))
    ak = ev(2, "arbeidskrav", date(2026, 10, 9), course="STAT110", title="Arbeidskrav 2")
    blocks = plan([exam, ak], PREFS, MON, topics={("UiB", "INF100"): ["Recursion", "Linked lists"]})
    assert {b.topic for b in blocks if b.event_id == 2} == {"Arbeidskrav 2"}
    assert all(b.kind == "assignment" for b in blocks if b.event_id == 2)
    last_days = [b for b in blocks if b.event_id == 1 and (exam.due - b.day).days <= 3]
    assert last_days and all(b.kind == "exam-prep" for b in last_days)
    early = [b for b in blocks if b.event_id == 1 and (exam.due - b.day).days > 3]
    assert {b.topic for b in early} <= {"Recursion", "Linked lists"}


def test_kept_slots_are_not_double_booked():
    exam = ev(1, "exam", date(2026, 10, 9))
    kept = [Block(7, MON, "09:00", 90, "UiB", "INF100", "study", "done already", 1, True)]
    blocks = plan([exam], PREFS, MON, keep=kept)
    assert (MON, "09:00") not in {(b.day, b.start) for b in blocks}


# ---------- store (Postgres) ----------

@pytest.fixture
def cal(db_settings):
    s = db_settings.__class__(**{**db_settings.__dict__, "calendar_prefix": "test_cal"})
    with engine(s.db_url).begin() as c:
        c.execute(sa.text("DROP TABLE IF EXISTS test_cal_events, test_cal_blocks, test_cal_prefs"))
    yield Calendar(s)
    with engine(s.db_url).begin() as c:
        c.execute(sa.text("DROP TABLE IF EXISTS test_cal_events, test_cal_blocks, test_cal_prefs"))


def test_events_crud(cal):
    e = cal.add_event(ev(None, "exam", date(2026, 12, 1), status="suggested"))
    assert e.id and cal.events()[0].title == "INF100 exam"
    cal.set_status(e.id, "confirmed")
    assert cal.events()[0].status == "confirmed"
    cal.delete_event(e.id)
    assert cal.events() == []


def test_prefs_default_and_update(cal):
    assert cal.prefs() == DEFAULT_PREFS
    cal.set_prefs({**DEFAULT_PREFS, "block_minutes": 60})
    assert cal.prefs()["block_minutes"] == 60


def test_replace_future_blocks_keeps_done_and_past(cal):
    past = Block(None, date(2026, 10, 1), "09:00", 90, "UiB", "INF100", "study", "a", None, False)
    done = Block(None, date(2026, 10, 7), "09:00", 90, "UiB", "INF100", "study", "b", None, True)
    todo = Block(None, date(2026, 10, 8), "09:00", 90, "UiB", "INF100", "study", "c", None, False)
    cal.replace_future_blocks(MON, [past, done, todo])
    new = Block(None, date(2026, 10, 9), "09:00", 90, "UiB", "INF100", "study", "d", None, False)
    cal.replace_future_blocks(MON, [new])
    assert sorted(b.topic for b in cal.blocks()) == ["a", "b", "d"]
    b = next(b for b in cal.blocks() if b.topic == "d")
    cal.set_done(b.id, True)
    assert next(x for x in cal.blocks() if x.topic == "d").done


# ---------- deadline finding ----------

class StubLLM:
    def __init__(self, items):
        self.items, self.prompts = items, []

    def with_structured_output(self, schema, method="function_calling"):
        assert schema is FoundList and method == "json_schema"
        return self

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return FoundList(items=self.items)


HIT = Hit("Arbeidskrav 2 leveres innen fredag i uke 41.", 0.8, "UiB/INF100/admin/emneplan.pdf", 3, "UiB", "INF100")


def test_find_deadlines_turns_passages_into_suggestions():
    llm = StubLLM([
        Found(kind="arbeidskrav", title="Arbeidskrav 2", date="2026-10-09", time="23:59",
              source="UiB/INF100/admin/emneplan.pdf p.3", uncertain=True),
        Found(kind="exam", title="Old exam", date="2025-12-01", time=None, source="x", uncertain=False),  # past
        Found(kind="deadline", title="Garbage", date="uke 42", time=None, source="x", uncertain=False),   # unparseable
    ])
    found = find_deadlines("UiB", "INF100", None, llm, MON, search=lambda q, scope, s: [HIT])
    assert [(e.kind, e.title, e.due, e.due_time, e.status, e.uncertain) for e in found] == [
        ("arbeidskrav", "Arbeidskrav 2", date(2026, 10, 9), "23:59", "suggested", True)]
    assert "uke 41" in llm.prompts[0] and "2026-10-05" in llm.prompts[0]


def test_find_deadlines_without_passages_does_not_call_llm():
    llm = StubLLM([])
    assert find_deadlines("UiB", "INF100", None, llm, MON, search=lambda q, scope, s: []) == []
    assert llm.prompts == []


def test_find_deadlines_keeps_every_section_of_the_same_file():
    exam = Hit("Skriftlig eksamen 8. desember 2026 kl. 09:00.", 0.7, "UiB/INF100/admin/emneplan.pdf", 3, "UiB", "INF100")
    llm = StubLLM([])
    find_deadlines("UiB", "INF100", None, llm, MON, search=lambda q, scope, s: [HIT, exam])
    # same file + page, different chunks: both must reach the model (the exam was being dropped)
    assert "uke 41" in llm.prompts[0] and "8. desember" in llm.prompts[0]
