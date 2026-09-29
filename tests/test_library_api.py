import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studyforge import library_api
from studyforge.config import Settings

LOCAL = {"Origin": "http://127.0.0.1:8000"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    changes = []
    monkeypatch.setattr(library_api.ingest, "run",
                        lambda root, s, progress=None: "1 new, 0 updated, 0 unchanged, 0 removed, 0 failed")
    app = FastAPI()
    library_api.register(app, Settings(notes_dir=tmp_path / "notes"), on_change=lambda: changes.append(1),
                         background=False)
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


def test_zip_upload(client):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ch1/solutions.md", "# CH1")
        zf.writestr("readme.exe", "MZ")
    r = client.post("/api/library/upload", headers=LOCAL, data={"uni": "NHH", "course": "BED3E", "doc_type": "notes"},
                    files=[("files", ("course.zip", buf.getvalue(), "application/zip"))])
    body = r.json()
    assert r.status_code == 200 and body["saved"] == 1 and "readme.exe" in body["rejected"][0]
    assert (client.notes / "NHH/BED3E/ch1/solutions.md").exists()


def test_background_indexing_returns_immediately(tmp_path, monkeypatch):
    import threading
    import time
    release = threading.Event()

    def slow_ingest(root, s, progress=None):
        progress(1, 2, "NHH/BED3E/a.md")
        release.wait(5)
        progress(2, 2, "NHH/BED3E/b.md")
        return "2 new, 0 updated, 0 unchanged, 0 removed, 0 failed"

    monkeypatch.setattr(library_api.ingest, "run", slow_ingest)
    app = FastAPI()
    library_api.register(app, Settings(notes_dir=tmp_path / "notes"))
    c = TestClient(app)
    r = c.post("/api/library/upload", headers=LOCAL, data={"uni": "NHH", "course": "BED3E"},
               files=[("files", ("a.md", b"# A", "text/markdown"))]).json()
    assert r["saved"] == 1 and r["indexing"]["running"] is True  # returned before indexing finished
    time.sleep(0.2)
    s = c.get("/api/library/indexing").json()
    assert s["running"] and s["done"] == 1 and s["total"] == 2 and s["current"] == "NHH/BED3E/a.md"
    release.set()
    for _ in range(50):
        s = c.get("/api/library/indexing").json()
        if not s["running"]:
            break
        time.sleep(0.1)
    assert not s["running"] and s["report"].startswith("2 new")
