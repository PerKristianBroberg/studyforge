from dataclasses import dataclass
from pathlib import Path

DOCLING_EXTS = {".pdf", ".pptx", ".docx"}
TEXT_EXTS = {".md", ".txt"}
# extension -> tree-sitter-language-pack name; extend as courses need
CODE_LANGS = {
    ".py": "python", ".java": "java", ".js": "javascript", ".ts": "typescript",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp", ".go": "go",
    ".rs": "rust", ".r": "r", ".sql": "sql",
}
SUPPORTED_EXTS = DOCLING_EXTS | TEXT_EXTS | set(CODE_LANGS)
DOC_TYPE_DIRS = {"exams", "code", "data", "admin"}


@dataclass(frozen=True)
class FileMeta:
    path: str
    uni: str
    course: str | None
    doc_type: str


def classify(root: Path, file: Path) -> FileMeta | None:
    parts = file.relative_to(root).parts
    ext = file.suffix.lower()
    if len(parts) < 2 or ext not in SUPPORTED_EXTS:
        return None
    if any(p.startswith((".", "~$")) for p in parts):
        return None
    dirs = [p.lower() for p in parts[1:-1]]
    course = parts[1] if dirs and dirs[0] not in DOC_TYPE_DIRS else None
    default = "code" if ext in CODE_LANGS else "notes"
    doc_type = next((d for d in dirs if d in DOC_TYPE_DIRS), default)
    return FileMeta(Path(*parts).as_posix(), parts[0], course, doc_type)


def scan(root: Path) -> list[tuple[Path, FileMeta]]:
    found = [(f, classify(root, f)) for f in root.rglob("*") if f.is_file()]
    return sorted(((f, m) for f, m in found if m), key=lambda fm: fm[1].path)
