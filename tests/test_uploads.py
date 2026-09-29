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
