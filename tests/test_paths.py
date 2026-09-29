from pathlib import Path

from studyforge.paths import FileMeta, classify, scan


def touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    return p


def test_course_file(tmp_path):
    f = touch(tmp_path, "UiO/INF1010/week1/lecture.pdf")
    assert classify(tmp_path, f) == FileMeta("UiO/INF1010/week1/lecture.pdf", "UiO", "INF1010", "notes")


def test_doc_type_folders(tmp_path):
    exam = touch(tmp_path, "UiO/INF1010/Exams/2024.pdf")
    code = touch(tmp_path, "UiO/INF1010/misc/Main.java")
    admin = touch(tmp_path, "UiO/admin/exam-rules.pdf")
    uni_level = touch(tmp_path, "UiO/studieplan.docx")
    assert classify(tmp_path, exam).doc_type == "exams"
    assert classify(tmp_path, code).doc_type == "code"
    assert classify(tmp_path, admin) == FileMeta("UiO/admin/exam-rules.pdf", "UiO", None, "admin")
    assert classify(tmp_path, uni_level) == FileMeta("UiO/studieplan.docx", "UiO", None, "notes")


def test_skips_junk_and_handles_norwegian_paths(tmp_path):
    junk = [
        touch(tmp_path, "root-file.md"),
        touch(tmp_path, "UiO/INF1010/~$lecture.docx"),
        touch(tmp_path, "UiO/.obsidian/workspace.md"),
        touch(tmp_path, "UiO/INF1010/data.csv"),
        touch(tmp_path, "UiO/INF1010/photo.jpg"),
    ]
    for f in junk:
        assert classify(tmp_path, f) is None, f
    nb = touch(tmp_path, "NTNU/FIL1001/Etikk og æøå.md")
    assert classify(tmp_path, nb) == FileMeta("NTNU/FIL1001/Etikk og æøå.md", "NTNU", "FIL1001", "notes")


def test_scan_sorted_and_filtered(tmp_path):
    touch(tmp_path, "b/C2/z.md")
    touch(tmp_path, "a/C1/y.md")
    touch(tmp_path, "a/C1/~$lock.docx")
    assert [m.path for _, m in scan(tmp_path)] == ["a/C1/y.md", "b/C2/z.md"]
