"""The notes/ folder as subjects: list, add, delete. All paths are validated to stay inside notes_dir."""
import shutil
from collections import Counter
from pathlib import Path

from studyforge.paths import DOC_TYPE_DIRS, classify
from studyforge.uploads import UploadRejected, _segment


def _visible_dirs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "~$")))


def _files(notes_dir: Path, folder: Path) -> list[dict]:
    found = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and (meta := classify(notes_dir, f)):
            found.append({"path": meta.path, "name": f.name, "type": meta.doc_type, "course": meta.course})
    return found


def tree(notes_dir: Path) -> list[dict]:
    """[{uni, subjects: [{name, counts, files}], files}] — empty subject folders included."""
    if not notes_dir.is_dir():
        return []
    unis = []
    for uni_dir in sorted(_visible_dirs(notes_dir), key=lambda d: d.name.lower() == "other"):  # "Other" last
        subjects = []
        for sub in _visible_dirs(uni_dir):
            if sub.name.lower() in DOC_TYPE_DIRS:
                continue  # e.g. UiO/admin/ holds uni-level files, not a subject
            files = [{k: v for k, v in f.items() if k != "course"} for f in _files(notes_dir, sub)]
            subjects.append({"name": sub.name, "counts": dict(Counter(f["type"] for f in files)), "files": files})
        uni_files = [{k: v for k, v in f.items() if k != "course"}
                     for f in _files(notes_dir, uni_dir) if f["course"] is None]
        unis.append({"uni": uni_dir.name, "subjects": subjects, "files": uni_files})
    return unis


def add_subject(notes_dir: Path, uni: str, course: str) -> Path:
    uni, course = _segment(uni, "university"), _segment(course, "subject")
    if course.lower() in DOC_TYPE_DIRS:
        raise UploadRejected(f"'{course}' is reserved for document types")
    folder = notes_dir / uni / course
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def delete_file(notes_dir: Path, rel_path: str) -> None:
    root = notes_dir.resolve()
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise UploadRejected(f"not a file in your notes: {rel_path!r}")
    target.unlink()


def delete_subject(notes_dir: Path, uni: str, course: str) -> None:
    root = notes_dir.resolve()
    folder = (root / _segment(uni, "university") / _segment(course, "subject")).resolve()
    if not folder.is_relative_to(root) or folder.parent.parent != root or not folder.is_dir():
        raise UploadRejected(f"no such subject: {uni}/{course}")
    shutil.rmtree(folder)
