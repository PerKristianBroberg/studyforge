import pytest

from studyforge.uploads import UploadRejected, parse_scope, save_upload


def src_file(tmp_path, name="a.pdf", size=10):
    p = tmp_path / "upload-tmp" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


def test_saves_into_course_folder(tmp_path):
    notes = tmp_path / "notes"
    target = save_upload(notes, "UiO", "INF1010", "Uke 1 – æøå.pdf", src_file(tmp_path))
    assert target == notes / "UiO" / "INF1010" / "Uke 1 – æøå.pdf"
    assert target.read_bytes() == b"x" * 10


def test_rejects_traversal_names(tmp_path):
    notes = tmp_path / "notes"
    for bad in ["../../evil.pdf", "..\\evil.pdf", "sub/evil.pdf", ".hidden.pdf", "~$lock.docx", ""]:
        with pytest.raises(UploadRejected):
            save_upload(notes, "UiO", "INF1010", bad, src_file(tmp_path))
    for uni, course in [("..", "INF1010"), ("UiO", ".."), ("UiO", "a/b")]:
        with pytest.raises(UploadRejected):
            save_upload(notes, uni, course, "ok.pdf", src_file(tmp_path))
    assert not (tmp_path / "evil.pdf").exists()


def test_rejects_bad_extension_and_size(tmp_path):
    notes = tmp_path / "notes"
    with pytest.raises(UploadRejected, match="type"):
        save_upload(notes, "UiO", "INF1010", "virus.exe", src_file(tmp_path, "virus.exe"))
    with pytest.raises(UploadRejected, match="MB"):
        save_upload(notes, "UiO", "INF1010", "big.pdf", src_file(tmp_path, size=11), max_bytes=10)


def test_parse_scope():
    assert parse_scope(" UiO / INF1010 ") == ("UiO", "INF1010")


def test_parse_scope_rejects_bad_input():
    for bad in ["", "UiO", "UiO/INF1010/x", "../INF1010", "UiO/..", "UiO/.git"]:
        with pytest.raises(UploadRejected):
            parse_scope(bad)


def _zip(tmp_path, entries):
    import zipfile
    z = tmp_path / "upload.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return z


def test_zip_extracts_supported_files_keeping_folders(tmp_path):
    from studyforge.uploads import save_zip
    z = _zip(tmp_path, {"week1/intro.md": "a", "exams/2024.pdf": "b", "photo.jpg": "c",
                        "__MACOSX/._intro.md": "d", ".DS_Store": "e", "~$lock.docx": "f"})
    notes = tmp_path / "notes"
    saved, skipped = save_zip(notes, "NHH", "BED3E", z, doc_type="notes")
    assert sorted(p.relative_to(notes).as_posix() for p in saved) == ["NHH/BED3E/exams/2024.pdf", "NHH/BED3E/week1/intro.md"]
    assert any("photo.jpg" in s for s in skipped) and not (notes / "NHH/BED3E/photo.jpg").exists()


def test_zip_type_folder_and_traversal(tmp_path):
    from studyforge.uploads import save_zip
    z = _zip(tmp_path, {"sol.pdf": "a", "../../evil.md": "x", "a/../../evil2.md": "y"})
    notes = tmp_path / "notes"
    saved, skipped = save_zip(notes, "NHH", "BED3E", z, doc_type="exams")
    assert [p.relative_to(notes).as_posix() for p in saved] == ["NHH/BED3E/exams/sol.pdf"]
    assert len(skipped) == 2 and not list(tmp_path.rglob("evil*"))


def test_zip_bomb_limits(tmp_path):
    from studyforge.uploads import save_zip
    z = _zip(tmp_path, {f"n{i}.md": "x" * 100 for i in range(5)})
    with pytest.raises(UploadRejected, match="files"):
        save_zip(tmp_path / "notes", "NHH", "BED3E", z, max_files=3)
    with pytest.raises(UploadRejected, match="MB"):
        save_zip(tmp_path / "notes", "NHH", "BED3E", z, max_total=200)
    assert not (tmp_path / "notes").exists()  # nothing extracted when a limit is hit
