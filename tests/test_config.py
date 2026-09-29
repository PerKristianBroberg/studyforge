import subprocess
from pathlib import Path

from studyforge.config import Settings


def test_defaults_match_docker_compose():
    s = Settings()
    assert s.db_url == "postgresql+psycopg2://studyforge:studyforge@localhost:5433/studyforge"
    assert s.embed_dim == 1024


def test_from_env_overrides_and_casts(monkeypatch):
    monkeypatch.setenv("STUDYFORGE_DB_PORT", "6000")
    monkeypatch.setenv("STUDYFORGE_RERANK_MIN_SCORE", "0.3")
    monkeypatch.setenv("STUDYFORGE_NOTES_DIR", "C:/somewhere/notes")
    s = Settings.from_env()
    assert s.db_port == 6000
    assert s.rerank_min_score == 0.3
    assert s.notes_dir == Path("C:/somewhere/notes")


def test_notes_dir_is_gitignored():
    root = Path(__file__).parent.parent
    r = subprocess.run(["git", "check-ignore", "-q", "notes/uni/course/secret.pdf"], cwd=root)
    assert r.returncode == 0, "notes/ must be gitignored"


def test_agent_settings():
    s = Settings()
    assert s.expert_model == "claude-sonnet-5-5"
    assert s.router_model == "claude-haiku-4-5"
    assert s.pg_conn == "postgresql://studyforge:studyforge@localhost:5433/studyforge"


def test_env_file_is_gitignored():
    root = Path(__file__).parent.parent
    assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=root).returncode == 0
