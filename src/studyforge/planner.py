"""Deterministic study planner: fills available hours with blocks for confirmed exams and deadlines."""
from collections import Counter
from datetime import date, timedelta

from studyforge.calendar_store import Block, Event

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WINDOW = {"exam": 28, "arbeidskrav": 14, "deadline": 10, "other": 7}  # days before due when work starts
WEIGHT = {"exam": 2.0, "arbeidskrav": 1.5, "deadline": 1.0, "other": 0.5}
EXAM_PREP_DAYS = 3


def plan(events: list[Event], prefs: dict, today: date,
         topics: dict[tuple[str, str], list[str]] | None = None, keep: list[Block] = ()) -> list[Block]:
    """Blocks from `today` until the last confirmed event. `keep` = blocks that stay (done/past): their slots are skipped."""
    active = [e for e in events if e.status == "confirmed" and e.due > today]
    if not active:
        return []
    topics = topics or {}
    taken = {(b.day, b.start) for b in keep}
    given, topic_turn, out = Counter(), Counter(), []
    size, first_hour = prefs["block_minutes"], prefs["start_hour"]

    day = today
    while day < max(e.due for e in active):
        per_day = int(prefs["hours"].get(WEEKDAYS[day.weekday()], 0) * 60 // size)
        for i in range(per_day):
            minutes_in = i * size
            start = f"{first_hour + minutes_in // 60:02d}:{minutes_in % 60:02d}"
            if (day, start) in taken:
                continue
            eligible = [e for e in active if day < e.due and (e.due - day).days <= WINDOW[e.kind]]
            if not eligible:
                break

            def score(e: Event) -> tuple:
                days_left = (e.due - day).days
                closeness = (WINDOW[e.kind] - days_left + 1) / WINDOW[e.kind]
                # nearer + heavier first; the more an item already got, the more room others get
                return WEIGHT[e.kind] * closeness / (1 + given[e.id]), -e.due.toordinal()

            e = max(eligible, key=score)
            given[e.id] += 1
            out.append(Block(None, day, start, size, e.uni, e.course, *_label(e, day, topics, topic_turn), e.id))
        day += timedelta(days=1)
    return out


def _label(e: Event, day: date, topics: dict, turn: Counter) -> tuple[str, str]:
    if e.kind == "exam" and (e.due - day).days <= EXAM_PREP_DAYS:
        return "exam-prep", "Past exam papers"
    if e.kind in ("arbeidskrav", "deadline"):
        return "assignment", e.title
    options = topics.get((e.uni, e.course)) or ["Review notes"]
    topic = options[turn[(e.uni, e.course)] % len(options)]
    turn[(e.uni, e.course)] += 1
    return "study", topic
