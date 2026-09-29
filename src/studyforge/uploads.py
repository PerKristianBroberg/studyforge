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
ZIP_MAX_TOTAL = 500 * 1024 * 1024
ZIP_MAX_FILES = 500
ZIP_JUNK = ("__MACOSX", ".DS_Store", "Thumbs.db")


def save_zip(notes_dir: Path, uni: str, course: str, zip_src: Path, doc_type: str = "notes",
             max_total: int = ZIP_MAX_TOTAL, max_files: int = ZIP_MAX_FILES) -> tuple[list[Path], list[str]]:
    """Extract supported files from a zip into notes/<uni>/<course>/[type]/..., keeping inner folders.
    Limits are checked before anything is written (zip bombs); every entry is path-checked (zip slip)."""
    import zipfile
    from pathlib import PurePosixPath

    if doc_type not in TYPE_DIRS:
        raise UploadRejected(f"unknown type: {doc_type!r}")
    root = notes_dir.resolve()
    base = root / _segment(uni, "uni") / _segment(course, "course")
    base = base / TYPE_DIRS[doc_type] if TYPE_DIRS[doc_type] else base
    try:
        zf = zipfile.ZipFile(zip_src)
    except zipfile.BadZipFile:
        raise UploadRejected("not a valid zip file")
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > max_files:
            raise UploadRejected(f"zip has {len(members)} files; the limit is {max_files} files")
        if sum(m.file_size for m in members) > max_total:
            raise UploadRejected(f"zip unpacks to more than {max_total // (1024 * 1024)} MB")
        saved, skipped = [], []
        for m in members:
            parts = PurePosixPath(m.filename.replace("\\", "/")).parts
            if any(p in ZIP_JUNK for p in parts):
                continue  # OS metadata, not worth reporting
            try:
                segs = [_segment(p, "path") for p in parts]
                if Path(segs[-1]).suffix.lower() not in SUPPORTED_EXTS:
                    raise UploadRejected("file type not supported")
                target = base.joinpath(*segs).resolve()
                if not target.is_relative_to(root):
                    raise UploadRejected("path escapes notes/")
            except UploadRejected as e:
                skipped.append(f"{m.filename}: {e}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            saved.append(target)
    return saved, skipped


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
