import dataclasses
from datetime import date

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studyforge import calendar_api
from studyforge.calendar_store import Event
from studyforge.store import engine

LOCAL = {"Origin": "http://127.0.0.1:8000"}
TODAY = date(2026, 10, 5)


@pytest.fixture
def client(db_settings, tmp_path):
    s = dataclasses.replace(db_settings, calendar_prefix="test_cal", notes_dir=tmp_path / "notes")
    (s.notes_dir / "UiB" / "INF100").mkdir(parents=True)
    (s.notes_dir / "UiB" / "INF100" / "recursion.md").write_text("x", encoding="utf-8")
    drop = sa.text("DROP TABLE IF EXISTS test_cal_events, test_cal_blocks, test_cal_prefs")
    with engine(s.db_url).begin() as c:
        c.execute(drop)
    calls = []

    def fake_find(uni, course, today):
        calls.append((uni, course))
        return [Event(None, uni, course, "arbeidskrav", "Arbeidskrav 1", date(2026, 10, 16), "23:59",
                      "suggested", "UiB/INF100/emneplan.pdf p.2", False)]

    app = FastAPI()
    calendar_api.register(app, s, find=fake_find, today=lambda: TODAY)
    c = TestClient(app)
    c.find_calls = calls
    yield c
    with engine(s.db_url).begin() as conn:
        conn.execute(drop)


def test_page_and_initial_state(client):
    assert "Calendar" in client.get("/calendar").text
    state = client.get("/api/calendar").json()
    assert state["today"] == "2026-10-05" and state["events"] == [] and state["blocks"] == []
    assert state["subjects"] == [{"uni": "UiB", "course": "INF100"}]


def test_manual_exam_prefs_plan_and_done(client):
    r = client.post("/api/calendar/events", headers=LOCAL, json={
        "uni": "UiB", "course": "INF100", "kind": "exam", "title": "INF100 exam", "due": "2026-10-20"})
    assert r.status_code == 200 and r.json()["events"][0]["status"] == "confirmed"
    prefs = {"hours": {"mon": 3, "tue": 0, "wed": 0, "thu": 0, "fri": 0, "sat": 0, "sun": 0},
             "block_minutes": 90, "start_hour": 9}
    assert client.put("/api/calendar/prefs", headers=LOCAL, json=prefs).status_code == 200
    blocks = client.post("/api/calendar/plan", headers=LOCAL).json()["blocks"]
    assert blocks and {b["day"] for b in blocks} == {"2026-10-05", "2026-10-12", "2026-10-19"}  # Mondays only
    assert blocks[0]["topic"] == "Recursion"  # topic from the subject's note files
    r = client.patch(f"/api/calendar/blocks/{blocks[0]['id']}", headers=LOCAL, json={"done": True})
    assert next(b for b in r.json()["blocks"] if b["id"] == blocks[0]["id"])["done"] is True


def test_find_suggests_once_and_confirm(client):
    r = client.post("/api/calendar/find", headers=LOCAL).json()
    assert r["added"] == 1 and client.find_calls == [("UiB", "INF100")]
    [s] = r["events"]
    assert s["status"] == "suggested" and s["source"].endswith("p.2")
    assert client.post("/api/calendar/find", headers=LOCAL).json()["added"] == 0  # no duplicates
    r = client.patch(f"/api/calendar/events/{s['id']}", headers=LOCAL, json={"status": "confirmed"})
    assert r.json()["events"][0]["status"] == "confirmed"
    r = client.delete(f"/api/calendar/events/{s['id']}", headers=LOCAL)
    assert r.json()["events"] == []


def test_guards(client):
    evil = {"Origin": "https://evil.example"}
    assert client.post("/api/calendar/plan", headers=evil).status_code == 403
    assert client.put("/api/calendar/prefs", headers=LOCAL, json={"hours": {"mon": 40}}).status_code == 400
    assert client.post("/api/calendar/events", headers=LOCAL, json={
        "uni": "UiB", "course": "INF100", "kind": "party", "title": "x", "due": "2026-10-20"}).status_code == 422
    assert client.patch("/api/calendar/events/1", headers=LOCAL, json={"status": "maybe"}).status_code == 422
