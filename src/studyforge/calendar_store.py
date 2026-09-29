"""Calendar persistence: events (exams, arbeidskrav, deadlines), planned study blocks, availability prefs."""
import json
from dataclasses import dataclass
from datetime import date

import sqlalchemy as sa

from studyforge.config import Settings
from studyforge.store import engine

KINDS = ("exam", "arbeidskrav", "deadline", "other")
STATUSES = ("suggested", "confirmed", "dismissed")
DEFAULT_PREFS = {"hours": {"mon": 3, "tue": 3, "wed": 3, "thu": 3, "fri": 2, "sat": 3, "sun": 0},
                 "block_minutes": 90, "start_hour": 9}


@dataclass
class Event:
    id: int | None
    uni: str
    course: str
    kind: str
    title: str
    due: date
    due_time: str | None
    status: str
    source: str | None
    uncertain: bool = False


@dataclass
class Block:
    id: int | None
    day: date
    start: str  # "HH:MM"
    minutes: int
    uni: str
    course: str
    kind: str  # study | assignment | exam-prep
    topic: str
    event_id: int | None
    done: bool = False


class Calendar:
    def __init__(self, s: Settings):
        self.eng = engine(s.db_url)
        p = s.calendar_prefix  # from Settings, not user input
        self.ev, self.bl, self.pr = f"{p}_events", f"{p}_blocks", f"{p}_prefs"
        with self.eng.begin() as c:
            c.execute(sa.text(f"""CREATE TABLE IF NOT EXISTS {self.ev} (
                id serial PRIMARY KEY, uni text NOT NULL, course text NOT NULL, kind text NOT NULL, title text NOT NULL,
                due date NOT NULL, due_time text, status text NOT NULL, source text, uncertain boolean NOT NULL DEFAULT false)"""))
            c.execute(sa.text(f"""CREATE TABLE IF NOT EXISTS {self.bl} (
                id serial PRIMARY KEY, day date NOT NULL, start text NOT NULL, minutes int NOT NULL, uni text NOT NULL,
                course text NOT NULL, kind text NOT NULL, topic text NOT NULL, event_id int, done boolean NOT NULL DEFAULT false)"""))
            c.execute(sa.text(f"CREATE TABLE IF NOT EXISTS {self.pr} (key text PRIMARY KEY, value jsonb NOT NULL)"))

    # events
    def events(self) -> list[Event]:
        with self.eng.connect() as c:
            rows = c.execute(sa.text(f"SELECT id, uni, course, kind, title, due, due_time, status, source, uncertain "
                                     f"FROM {self.ev} ORDER BY due, id"))
            return [Event(*r) for r in rows]

    def add_event(self, e: Event) -> Event:
        with self.eng.begin() as c:
            new_id = c.execute(sa.text(
                f"INSERT INTO {self.ev} (uni, course, kind, title, due, due_time, status, source, uncertain) "
                "VALUES (:uni, :course, :kind, :title, :due, :due_time, :status, :source, :uncertain) RETURNING id"),
                {k: v for k, v in e.__dict__.items() if k != "id"}).scalar_one()
        return Event(**{**e.__dict__, "id": new_id})

    def set_status(self, event_id: int, status: str) -> None:
        with self.eng.begin() as c:
            c.execute(sa.text(f"UPDATE {self.ev} SET status = :s WHERE id = :i"), {"s": status, "i": event_id})

    def delete_event(self, event_id: int) -> None:
        with self.eng.begin() as c:
            c.execute(sa.text(f"DELETE FROM {self.bl} WHERE event_id = :i AND NOT done"), {"i": event_id})
            c.execute(sa.text(f"DELETE FROM {self.ev} WHERE id = :i"), {"i": event_id})

    # prefs
    def prefs(self) -> dict:
        with self.eng.connect() as c:
            row = c.execute(sa.text(f"SELECT value FROM {self.pr} WHERE key = 'prefs'")).first()
        return row[0] if row else DEFAULT_PREFS

    def set_prefs(self, prefs: dict) -> None:
        with self.eng.begin() as c:
            c.execute(sa.text(f"INSERT INTO {self.pr} (key, value) VALUES ('prefs', CAST(:v AS jsonb)) "
                              "ON CONFLICT (key) DO UPDATE SET value = CAST(:v AS jsonb)"), {"v": json.dumps(prefs)})

    # blocks
    def blocks(self) -> list[Block]:
        with self.eng.connect() as c:
            rows = c.execute(sa.text(f"SELECT id, day, start, minutes, uni, course, kind, topic, event_id, done "
                                     f"FROM {self.bl} ORDER BY day, start"))
            return [Block(*r) for r in rows]

    def replace_future_blocks(self, today: date, new: list[Block]) -> None:
        """Re-planning: keep past and done blocks, replace everything else."""
        with self.eng.begin() as c:
            c.execute(sa.text(f"DELETE FROM {self.bl} WHERE day >= :t AND NOT done"), {"t": today})
            for b in new:
                c.execute(sa.text(
                    f"INSERT INTO {self.bl} (day, start, minutes, uni, course, kind, topic, event_id, done) "
                    "VALUES (:day, :start, :minutes, :uni, :course, :kind, :topic, :event_id, :done)"),
                    {k: v for k, v in b.__dict__.items() if k != "id"})

    def set_done(self, block_id: int, done: bool) -> None:
        with self.eng.begin() as c:
            c.execute(sa.text(f"UPDATE {self.bl} SET done = :d WHERE id = :i"), {"d": done, "i": block_id})
