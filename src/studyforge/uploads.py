import re
import shutil
from pathlib import Path

from studyforge.paths import SUPPORTED_EXTS

MAX_BYTES = 50 * 1024 * 1024
SEGMENT = re.compile(r"[\w .()\-–]+")  # \w covers æøå


class UploadRejected(ValueError):
    pass


def _segment(part: str, what: str) -> str:
    part = part.strip()
    if not SEGMENT.fullmatch(part) or part.startswith((".", "~$")):
        raise UploadRejected(f"invalid {what}: {part!r}")
    return part


def parse_scope(text: str) -> tuple[str, str]:
    parts = text.split("/")
    if len(parts) != 2:
        raise UploadRejected("expected <uni>/<course>, e.g. UiO/INF1010")
    return _segment(parts[0], "uni"), _segment(parts[1], "course")


TYPE_DIRS = {"notes": None, "exams": "exams", "code": "code", "admin": "admin"}  # doc type -> subfolder


def save_upload(notes_dir: Path, uni: str, course: str, filename: str, src: Path,
                max_bytes: int = MAX_BYTES, doc_type: str = "notes") -> Path:
    name = _segment(filename, "file name")
    if Path(name).suffix.lower() not in SUPPORTED_EXTS:
        raise UploadRejected(f"file type not supported: {name}")
    if doc_type not in TYPE_DIRS:
        raise UploadRejected(f"unknown type: {doc_type!r}")
    if src.stat().st_size > max_bytes:
        raise UploadRejected(f"{name} is larger than {max_bytes // (1024 * 1024)} MB")
    root = notes_dir.resolve()
    folder = root / _segment(uni, "uni") / _segment(course, "course")
    target = (folder / TYPE_DIRS[doc_type] / name if TYPE_DIRS[doc_type] else folder / name).resolve()
    if not target.is_relative_to(root):  # belt and braces on top of the segment checks
        raise UploadRejected(f"invalid path: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    return target
