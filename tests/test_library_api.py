import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studyforge import library_api
from studyforge.config import Settings

LOCAL = {"Origin": "http://127.0.0.1:8000"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    changes = []
    monkeypatch.setattr(library_api.ingest, "run", lambda root, s: "1 new, 0 updated, 0 unchanged, 0 removed, 0 failed")
    app = FastAPI()
    library_api.register(app, Settings(notes_dir=tmp_path / "notes"), on_change=lambda: changes.append(1))
    c = TestClient(app)
    c.changes = changes
    c.notes = tmp_path / "notes"
    return c


def test_page_and_empty_tree(client):
    assert "Library" in client.get("/library").text
    assert client.get("/api/library").json() == {"tree": []}


def test_add_subject_upload_and_delete(client):
    r = client.post("/api/library/subjects", json={"uni": "UiO", "course": "INF1010"}, headers=LOCAL)
    assert r.status_code == 200 and r.json()["tree"][0]["subjects"][0]["name"] == "INF1010"

    r = client.post("/api/library/upload", headers=LOCAL,
                    data={"uni": "UiO", "course": "INF1010", "doc_type": "exams"},
                    files=[("files", ("2024.md", b"# exam", "text/markdown")),
                           ("files", ("virus.exe", b"MZ", "application/octet-stream"))])
    body = r.json()
    assert r.status_code == 200
    assert (client.notes / "UiO/INF1010/exams/2024.md").read_bytes() == b"# exam"
    assert body["saved"] == 1 and "virus.exe" in body["rejected"][0]
    assert body["report"].startswith("1 new") and client.changes

    r = client.request("DELETE", "/api/library/file", params={"path": "UiO/INF1010/exams/2024.md"}, headers=LOCAL)
    assert r.status_code == 200 and not (client.notes / "UiO/INF1010/exams/2024.md").exists()
    r = client.request("DELETE", "/api/library/subject", params={"uni": "UiO", "course": "INF1010"}, headers=LOCAL)
    assert r.status_code == 200 and not (client.notes / "UiO/INF1010").exists()


def test_rejects_other_websites(client):
    evil = {"Origin": "https://evil.example"}
    assert client.post("/api/library/subjects", json={"uni": "UiO", "course": "X"}, headers=evil).status_code == 403
    assert client.request("DELETE", "/api/library/subject", params={"uni": "UiO", "course": "X"},
                          headers=evil).status_code == 403
    assert client.post("/api/library/subjects", json={"uni": "UiO", "course": "X"}).status_code == 403  # no Origin


def test_bad_paths_are_400(client):
    r = client.request("DELETE", "/api/library/file", params={"path": "../../etc/passwd"}, headers=LOCAL)
    assert r.status_code == 400
    r = client.post("/api/library/subjects", json={"uni": "..", "course": "x"}, headers=LOCAL)
    assert r.status_code == 400
