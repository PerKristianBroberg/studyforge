import sqlalchemy as sa

from studyforge import ingest
from studyforge.store import engine


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
