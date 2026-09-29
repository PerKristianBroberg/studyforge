import pytest

from studyforge.library import add_subject, delete_file, delete_subject, tree
from studyforge.uploads import UploadRejected, save_upload


def touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def test_tree_counts_types_and_keeps_empty_subjects(tmp_path):
    touch(tmp_path, "UiO/INF1010/week1.md")
    touch(tmp_path, "UiO/INF1010/exams/2024.pdf")
    touch(tmp_path, "UiO/INF1010/code/Main.java")
    touch(tmp_path, "UiO/admin/rules.pdf")
    touch(tmp_path, "UiO/INF1010/~$lock.docx")  # junk, ignored
    (tmp_path / "NTNU" / "TDT4100").mkdir(parents=True)  # just added, no files yet
    t = tree(tmp_path)
    assert [u["uni"] for u in t] == ["NTNU", "UiO"]
    ntnu, uio = t
    assert ntnu["subjects"] == [{"name": "TDT4100", "counts": {}, "files": []}]
    inf = uio["subjects"][0]
    assert inf["name"] == "INF1010"
    assert inf["counts"] == {"notes": 1, "exams": 1, "code": 1}
    assert {f["path"] for f in inf["files"]} == {"UiO/INF1010/week1.md", "UiO/INF1010/exams/2024.pdf",
                                                "UiO/INF1010/code/Main.java"}
    assert [f["path"] for f in uio["files"]] == ["UiO/admin/rules.pdf"]  # uni-level (admin) files
    assert "admin" not in [s["name"] for s in uio["subjects"]]


def test_tree_of_missing_dir_is_empty(tmp_path):
    assert tree(tmp_path / "nope") == []


def test_add_subject(tmp_path):
    assert add_subject(tmp_path, " UiO ", "INF1010") == tmp_path / "UiO" / "INF1010"
    assert (tmp_path / "UiO" / "INF1010").is_dir()
    for uni, course in [("..", "x"), ("UiO", "../x"), ("UiO", "admin"), ("", "x"), ("UiO", "a/b")]:
        with pytest.raises(UploadRejected):
            add_subject(tmp_path, uni, course)


def test_upload_into_type_subfolder(tmp_path):
    src = tmp_path / "src.pdf"
    src.write_bytes(b"x")
    notes = tmp_path / "notes"
    assert save_upload(notes, "UiO", "INF1010", "e.pdf", src, doc_type="exams") == notes / "UiO/INF1010/exams/e.pdf"
    assert save_upload(notes, "UiO", "INF1010", "n.pdf", src, doc_type="notes") == notes / "UiO/INF1010/n.pdf"
    with pytest.raises(UploadRejected):
        save_upload(notes, "UiO", "INF1010", "n.pdf", src, doc_type="../../x")


def test_delete_file_only_inside_notes(tmp_path):
    notes = tmp_path / "notes"
    touch(notes, "UiO/INF1010/week1.md")
    touch(tmp_path, "outside.md")
    delete_file(notes, "UiO/INF1010/week1.md")
    assert not (notes / "UiO/INF1010/week1.md").exists()
    for bad in ["../outside.md", "UiO/INF1010", "UiO/INF1010/missing.md", "C:/Windows/win.ini"]:
        with pytest.raises(UploadRejected):
            delete_file(notes, bad)
    assert (tmp_path / "outside.md").exists()


def test_delete_subject(tmp_path):
    notes = tmp_path / "notes"
    touch(notes, "UiO/INF1010/exams/2024.pdf")
    touch(notes, "UiO/MAT1100/a.md")
    delete_subject(notes, "UiO", "INF1010")
    assert not (notes / "UiO/INF1010").exists() and (notes / "UiO/MAT1100/a.md").exists()
    for uni, course in [("UiO", ".."), ("..", "notes"), ("UiO", "missing")]:
        with pytest.raises(UploadRejected):
            delete_subject(notes, uni, course)
    assert (notes / "UiO/MAT1100/a.md").exists()
