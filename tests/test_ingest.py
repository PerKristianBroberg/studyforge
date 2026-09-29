from pathlib import Path

import pytest
import sqlalchemy as sa

from studyforge import ingest
from studyforge.store import Manifest, engine


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def chunks_for(s, path):
    with engine(s.db_url).connect() as c:
        return [r[0] for r in c.execute(
            sa.text(f"SELECT text FROM data_{s.chunks_table} WHERE metadata_->>'path' = :p"), {"p": path}
        )]


def test_new_unchanged_updated_removed(tmp_path, db_settings, mock_embed):
    a = write(tmp_path, "UiO/A/a.md", "# Heaps\nA heap is a tree.")
    write(tmp_path, "UiO/A/code/f.py", "def f():\n    return 1\n")
    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert (r.new, r.updated, r.unchanged, r.removed, r.failed) == (2, 0, 0, 0, [])
    assert chunks_for(db_settings, "UiO/A/a.md")

    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert (r.new, r.unchanged) == (0, 2)

    a.write_text("# Tries\nA trie stores strings.", encoding="utf-8")
    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert r.updated == 1
    texts = " ".join(chunks_for(db_settings, "UiO/A/a.md"))
    assert "trie" in texts and "heap" not in texts

    a.unlink()
    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert r.removed == 1
    assert chunks_for(db_settings, "UiO/A/a.md") == []


def test_parse_failure_keeps_old_chunks(tmp_path, db_settings, mock_embed, monkeypatch):
    f = write(tmp_path, "UiO/A/a.md", "Original content about graphs.")
    ingest.run(tmp_path, db_settings, mock_embed)
    f.write_text("changed", encoding="utf-8")

    def boom(file, meta):
        raise RuntimeError("corrupt file")

    monkeypatch.setattr(ingest, "load", boom)
    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert r.failed == [("UiO/A/a.md", "RuntimeError: corrupt file")]
    assert "graphs" in " ".join(chunks_for(db_settings, "UiO/A/a.md"))
    assert "FAILED UiO/A/a.md: RuntimeError: corrupt file" in str(r)


def test_file_with_no_text_is_reported_failed(tmp_path, db_settings, mock_embed):
    write(tmp_path, "UiO/A/empty.md", "   \n")
    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert r.new == 0
    assert r.failed[0][0] == "UiO/A/empty.md"
    assert "no extractable text" in r.failed[0][1]


def test_missing_notes_dir_refuses_instead_of_wiping(tmp_path, db_settings, mock_embed):
    write(tmp_path, "UiO/A/a.md", "Graphs have vertices.")
    ingest.run(tmp_path, db_settings, mock_embed)
    with pytest.raises(FileNotFoundError):
        ingest.run(tmp_path / "typo", db_settings, mock_embed)
    assert chunks_for(db_settings, "UiO/A/a.md")


def test_split_failure_keeps_old_chunks(tmp_path, db_settings, mock_embed, monkeypatch):
    f = write(tmp_path, "UiO/A/a.md", "Original content about graphs.")
    ingest.run(tmp_path, db_settings, mock_embed)
    f.write_text("changed", encoding="utf-8")

    def boom(docs):
        raise ValueError("Could not parse code")

    monkeypatch.setattr(ingest, "_splitters", boom)
    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert r.failed and "graphs" in " ".join(chunks_for(db_settings, "UiO/A/a.md"))


def test_crash_before_manifest_write_does_not_duplicate(tmp_path, db_settings, mock_embed):
    write(tmp_path, "UiO/A/a.md", "Graphs have vertices.")
    ingest.run(tmp_path, db_settings, mock_embed)
    n = len(chunks_for(db_settings, "UiO/A/a.md"))
    Manifest(db_settings).remove("UiO/A/a.md")  # as if the process died before manifest.put
    ingest.run(tmp_path, db_settings, mock_embed)
    assert len(chunks_for(db_settings, "UiO/A/a.md")) == n


def test_no_changes_does_not_load_embedding_model(tmp_path, db_settings, mock_embed, monkeypatch):
    write(tmp_path, "UiO/A/a.md", "Graphs have vertices.")
    ingest.run(tmp_path, db_settings, mock_embed)

    def no_load(name):
        raise AssertionError("embedding model loaded with nothing to embed")

    monkeypatch.setattr(ingest, "embed_model", no_load)
    assert ingest.run(tmp_path, db_settings).unchanged == 1


def test_unreadable_file_is_reported_not_fatal(tmp_path, db_settings, mock_embed, monkeypatch):
    write(tmp_path, "UiO/A/a.md", "Graphs have vertices.")
    write(tmp_path, "UiO/A/b.md", "Trees are graphs.")
    real = Path.read_bytes

    def locked(self):
        if self.name == "a.md":
            raise PermissionError("file is locked")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", locked)
    r = ingest.run(tmp_path, db_settings, mock_embed)
    assert r.new == 1 and r.failed[0][0] == "UiO/A/a.md"
