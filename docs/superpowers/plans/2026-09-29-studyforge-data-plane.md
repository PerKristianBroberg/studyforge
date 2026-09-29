# studyforge Plan 1 — Data Plane (Foundation, Ingestion, Retrieval, Evals)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest a `notes/<uni>/<course>/` tree of PDF/PPTX/DOCX/Markdown/code files into local Postgres+pgvector with local bge-m3 embeddings, and retrieve scoped, reranked, cited passages — with a retrieval-quality eval gate in CI.

**Architecture:** LlamaIndex owns the data plane. Docling parses office/PDF files to Markdown (one LlamaIndex `Document` per page/slide so citations carry page numbers). A small `ingest_manifest` table stores each file's SHA-256 so unchanged files are skipped *before* the slow Docling parse. `PGVectorStore` in hybrid mode returns dense top-30 ∪ keyword top-30; a local cross-encoder reranks to top-6 and drops passages below a score floor ("not in your notes").

**Tech Stack:** Python 3.12, uv, Docker Compose (pgvector/pgvector:pg17), docling 2.131, llama-index-core 0.14, llama-index-vector-stores-postgres 0.9, llama-index-embeddings-huggingface 0.8 (`BAAI/bge-m3`), llama-index-postprocessor-sbert-rerank 0.6 (`BAAI/bge-reranker-v2-m3`), tree-sitter-language-pack, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-29-studyforge-design.md` (this plan implements spec §10 phases 1–3).

**Deliberately moved out of this plan (spec deviations, each lands in the plan that first needs it):**
- Langfuse → Plan 2 (agents); nothing in this plan calls an LLM, so there is nothing to trace.
- CSV/XLSX → dataset tables → the plan that builds `query_dataset`; CSV/XLSX files are ignored by the scanner until then.
- Ragas (faithfulness/answer relevancy) → Plan 2; they need generated answers. This plan's eval is retrieval-only (hit@6, MRR, refusal), deterministic and API-key-free.
- Persona generation → Plan 2.

## Global Constraints

- Python `>=3.12`; dependency management with `uv` only (`uv sync`, `uv run ...`).
- Postgres runs in Docker, bound to `127.0.0.1:5433` only (never exposed on the LAN).
- `notes/` is the user's private corpus and must never be committed (`.gitignore`). CI and evals use only the committed `samples/` corpus.
- Embeddings and reranking run locally; this plan makes **no** network calls to any LLM API.
- Embedding model `BAAI/bge-m3`, dim `1024`. Reranker `BAAI/bge-reranker-v2-m3`. Retrieval: hybrid `top_k=30` (dense and sparse each), rerank `top_n=6`.
- Postgres full-text config is `simple` (language-agnostic; notes are Norwegian and English).
- OCR is disabled (spec: no handwriting/scans).
- torch is installed from the CPU wheel index (no NVIDIA GPU on the dev machine; keeps CI small).
- All settings come from `studyforge.config.Settings`, overridable via `STUDYFORGE_<FIELD>` env vars.

## Review Focus

1. **Private notes leaking into git** — `notes/**` must be ignored. Test: Task 1 `test_notes_dir_is_gitignored`.
2. **Scanned/image-only PDF yields no text** — must appear in the report as failed with a clear reason, not be silently "ingested" with zero chunks. Test: Task 5 `test_file_with_no_text_is_reported_failed`.
3. **A previously ingested file that later fails to parse** (corrupted re-save, Office lock) must keep its old chunks, not lose them. Test: Task 5 `test_parse_failure_keeps_old_chunks`.
4. **Junk files in the tree** — Office lock files `~$x.docx`, Obsidian `.obsidian/` config, files at the notes root with no uni — must be skipped, and Norwegian characters in paths must work. Test: Task 2 `test_skips_junk_and_handles_norwegian_paths`.
5. **Same course code at both institutions** — a course-scoped search must also filter by uni. Test: Task 6 `test_course_scope_does_not_leak_across_unis`.

---

## File Structure

```
pyproject.toml                     deps, scripts (ingest, ask), pytest config, torch CPU index
docker-compose.yml                 Postgres+pgvector on 127.0.0.1:5433
.gitignore                         notes/, venv, caches
README.md                          how to run (full portfolio README is Plan 10)
src/studyforge/__init__.py
src/studyforge/config.py           Settings dataclass + from_env()
src/studyforge/paths.py            path → FileMeta (uni, course, doc_type); scan()
src/studyforge/loaders.py          file → list[Document] (Docling / text / code)
src/studyforge/store.py            vector_store(), embed_model(), Manifest table
src/studyforge/ingest.py           run() → Report; `ingest` CLI
src/studyforge/retrieval.py        Scope, Hit, search(); `ask` CLI
tests/conftest.py                  db_settings fixture (isolated tables), mock embed
tests/test_config.py
tests/test_paths.py
tests/test_loaders.py
tests/test_store.py
tests/test_ingest.py
tests/test_retrieval.py
samples/...                        small public corpus for evals/CI
evals/golden.jsonl                 questions → expected source file (or null = refuse)
evals/test_retrieval_quality.py    hit@6 / MRR / refusal gate (real models)
.github/workflows/ci.yml
```

---

### Task 1: Project scaffold and settings

**Files:**
- Create: `pyproject.toml`, `docker-compose.yml`, `.gitignore`, `src/studyforge/__init__.py`, `src/studyforge/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `studyforge.config.Settings` — frozen dataclass with fields `db_host: str`, `db_port: int`, `db_name: str`, `db_user: str`, `db_password: str`, `chunks_table: str`, `manifest_table: str`, `notes_dir: Path`, `embed_model: str`, `embed_dim: int`, `rerank_model: str`, `top_k: int`, `top_n: int`, `rerank_min_score: float`; classmethod `Settings.from_env() -> Settings`; property `db_url -> str`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "studyforge"
version = "0.1.0"
description = "Multi-agent RAG study assistant for studying at two universities"
requires-python = ">=3.12"
dependencies = [
    "docling>=2.131",
    "llama-index-core>=0.14.25,<0.15",
    "llama-index-vector-stores-postgres>=0.9",
    "llama-index-embeddings-huggingface>=0.8",
    "llama-index-postprocessor-sbert-rerank>=0.6",
    "tree-sitter-language-pack",
    "torch>=2.6",
]

[project.scripts]
ingest = "studyforge.ingest:main"
ask = "studyforge.retrieval:main"

[dependency-groups]
dev = ["pytest>=8", "python-docx", "python-pptx"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/studyforge"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["eval: retrieval-quality evals with real models (slow)"]
addopts = "-m 'not eval'"
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: studyforge
      POSTGRES_PASSWORD: studyforge
      POSTGRES_DB: studyforge
    ports:
      - "127.0.0.1:5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U studyforge"]
      interval: 5s
      retries: 10

volumes:
  pgdata:
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
# private course material — never commit
notes/
evals/*.local.jsonl

.venv/
__pycache__/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Write the failing tests** — `tests/test_config.py`

```python
import subprocess
from pathlib import Path

from studyforge.config import Settings


def test_defaults_match_docker_compose():
    s = Settings()
    assert s.db_url == "postgresql+psycopg2://studyforge:studyforge@localhost:5433/studyforge"
    assert s.embed_dim == 1024


def test_from_env_overrides_and_casts(monkeypatch):
    monkeypatch.setenv("STUDYFORGE_DB_PORT", "6000")
    monkeypatch.setenv("STUDYFORGE_RERANK_MIN_SCORE", "0.3")
    monkeypatch.setenv("STUDYFORGE_NOTES_DIR", "C:/somewhere/notes")
    s = Settings.from_env()
    assert s.db_port == 6000
    assert s.rerank_min_score == 0.3
    assert s.notes_dir == Path("C:/somewhere/notes")


def test_notes_dir_is_gitignored():
    root = Path(__file__).parent.parent
    r = subprocess.run(["git", "check-ignore", "-q", "notes/uni/course/secret.pdf"], cwd=root)
    assert r.returncode == 0, "notes/ must be gitignored"
```

- [ ] **Step 5: Run to verify failure**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'studyforge.config'`. (`uv sync` downloads torch CPU + docling; first run takes several minutes.)

- [ ] **Step 6: Implement** — `src/studyforge/__init__.py` (empty file) and `src/studyforge/config.py`

```python
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "studyforge"
    db_user: str = "studyforge"
    db_password: str = "studyforge"
    chunks_table: str = "chunks"  # PGVectorStore prefixes this: real table is data_chunks
    manifest_table: str = "ingest_manifest"
    notes_dir: Path = Path("notes")
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 30
    top_n: int = 6
    # ponytail: calibration knob — below this rerank score a passage counts as "not in your notes".
    # Tune against the refusal cases in evals/golden.jsonl.
    rerank_min_score: float = 0.05

    @classmethod
    def from_env(cls) -> "Settings":
        overrides = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(f"STUDYFORGE_{f.name.upper()}")
            if raw is not None:
                overrides[f.name] = type(f.default)(raw)
        return cls(**overrides)

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
```

Note: `type(f.default)` for `notes_dir` is `WindowsPath`/`PosixPath`, which accepts a string — that is intended.

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 8: Start Postgres and verify**

Run: `docker compose up -d --wait && docker compose exec db psql -U studyforge -c "select 1"`
Expected: a result row with `1`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock docker-compose.yml .gitignore src tests
git commit -m "feat: project scaffold, settings, local Postgres"
```

---

### Task 2: Path classification and scanning

**Files:**
- Create: `src/studyforge/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DOCLING_EXTS: set[str]` = `{".pdf", ".pptx", ".docx"}`, `TEXT_EXTS: set[str]` = `{".md", ".txt"}`, `CODE_LANGS: dict[str, str]` (extension → tree-sitter language name).
  - `@dataclass(frozen=True) FileMeta(path: str, uni: str, course: str | None, doc_type: str)` — `path` is POSIX-style relative to the notes root; `doc_type` ∈ `{"notes", "code", "exams", "admin", "data"}`.
  - `classify(root: Path, file: Path) -> FileMeta | None`
  - `scan(root: Path) -> list[tuple[Path, FileMeta]]` sorted by `FileMeta.path`.

Rules: `notes/<uni>/<file>` → uni-level (course `None`). `notes/<uni>/<course>/.../<file>` → course-level, unless the second part is a doc-type folder (`notes/<uni>/admin/x.pdf` → course `None`, doc_type `admin`). doc_type = first folder below uni named `exams|code|data|admin` (case-insensitive), else `code` for code extensions, else `notes`. Skip: files directly in root, any path part starting with `.` or `~$`, unsupported extensions.

- [ ] **Step 1: Write the failing tests** — `tests/test_paths.py`

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'studyforge.paths'`.

- [ ] **Step 3: Implement** — `src/studyforge/paths.py`

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_paths.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/studyforge/paths.py tests/test_paths.py
git commit -m "feat: classify note files by uni/course/doc_type"
```

---

### Task 3: File loaders

**Files:**
- Create: `src/studyforge/loaders.py`
- Test: `tests/test_loaders.py`

**Interfaces:**
- Consumes: `FileMeta`, `DOCLING_EXTS`, `CODE_LANGS` from `studyforge.paths`.
- Produces: `load(file: Path, meta: FileMeta) -> list[llama_index.core.Document]`.
  - Every Document's metadata has `path`, `uni`, `course`, `doc_type`; code docs add `language`; paged docs (PDF, PPTX) add `page` (1-based).
  - Document ids are deterministic: `meta.path` for single-document files, `f"{meta.path}#p{n}"` for pages. Ingest relies on this to delete old chunks.
  - Pages with no text are dropped; a file with no text at all returns `[]`.
  - Obsidian wikilinks are flattened: `[[target|alias]]` → `alias`, `[[target]]` → `target`.

- [ ] **Step 1: Write the failing tests** — `tests/test_loaders.py`

```python
from pathlib import Path

import docx
import pptx

from studyforge.loaders import load
from studyforge.paths import classify


def meta_for(root: Path, rel: str):
    return classify(root, root / rel)


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_markdown_flattens_wikilinks(tmp_path):
    f = write(tmp_path, "UiO/INF1010/a.md", "See [[Quicksort|quick sort]] and [[Heaps]].")
    [doc] = load(f, meta_for(tmp_path, "UiO/INF1010/a.md"))
    assert doc.text == "See quick sort and Heaps."
    assert doc.id_ == "UiO/INF1010/a.md"
    assert doc.metadata == {"path": "UiO/INF1010/a.md", "uni": "UiO", "course": "INF1010", "doc_type": "notes"}


def test_code_keeps_text_and_language(tmp_path):
    src = "def f(x):\n    return [[x]]\n"
    f = write(tmp_path, "UiO/INF1010/code/f.py", src)
    [doc] = load(f, meta_for(tmp_path, "UiO/INF1010/code/f.py"))
    assert doc.text == src  # wikilink regex must not touch code
    assert doc.metadata["language"] == "python"
    assert doc.metadata["doc_type"] == "code"


def test_docx_is_single_document(tmp_path):
    f = tmp_path / "UiO/INF1010/notes.docx"
    f.parent.mkdir(parents=True)
    d = docx.Document()
    d.add_heading("Binary trees", level=1)
    d.add_paragraph("A binary tree node has at most two children.")
    d.save(f)
    [doc] = load(f, meta_for(tmp_path, "UiO/INF1010/notes.docx"))
    assert "at most two children" in doc.text
    assert "page" not in doc.metadata


def test_pptx_one_document_per_slide(tmp_path):
    f = tmp_path / "UiO/INF1010/slides.pptx"
    f.parent.mkdir(parents=True)
    deck = pptx.Presentation()
    for title in ["Stacks are LIFO", "Queues are FIFO"]:
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = title
    deck.save(f)
    docs = load(f, meta_for(tmp_path, "UiO/INF1010/slides.pptx"))
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert [d.id_ for d in docs] == ["UiO/INF1010/slides.pptx#p1", "UiO/INF1010/slides.pptx#p2"]
    assert "LIFO" in docs[0].text and "FIFO" in docs[1].text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'studyforge.loaders'`.

- [ ] **Step 3: Implement** — `src/studyforge/loaders.py`

```python
import re
from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import Document

from studyforge.paths import CODE_LANGS, DOCLING_EXTS, FileMeta

WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    # spec: no handwriting/scans, so OCR stays off (much faster on CPU)
    pdf = PdfFormatOption(pipeline_options=PdfPipelineOptions(do_ocr=False))
    return DocumentConverter(format_options={InputFormat.PDF: pdf})


def load(file: Path, meta: FileMeta) -> list[Document]:
    base = {"path": meta.path, "uni": meta.uni, "course": meta.course, "doc_type": meta.doc_type}
    ext = file.suffix.lower()
    if ext in DOCLING_EXTS:
        return _load_docling(file, base)
    text = file.read_text(encoding="utf-8", errors="replace")
    if ext in CODE_LANGS:
        return [Document(id_=meta.path, text=text, metadata={**base, "language": CODE_LANGS[ext]})]
    text = WIKILINK.sub(lambda m: m.group(2) or m.group(1), text)
    return [Document(id_=meta.path, text=text, metadata=base)] if text.strip() else []


def _load_docling(file: Path, base: dict) -> list[Document]:
    doc = _converter().convert(file).document
    if not doc.pages:  # DOCX has no pages
        text = doc.export_to_markdown(image_placeholder="")
        return [Document(id_=base["path"], text=text, metadata=base)] if text.strip() else []
    docs = []
    for n in sorted(doc.pages):  # PDF pages / PPTX slides, 1-based
        text = doc.export_to_markdown(page_no=n, image_placeholder="")
        if text.strip():
            docs.append(Document(id_=f"{base['path']}#p{n}", text=text, metadata={**base, "page": n}))
    return docs
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_loaders.py -v`
Expected: 4 passed.

- [ ] **Step 5: Manual PDF check** (PDF conversion downloads Docling's layout model on first use, so it is not in the unit suite)

Run: `uv run python -c "from pathlib import Path; from studyforge.paths import classify; from studyforge.loaders import load; import sys; p=Path(sys.argv[1]); d=load(p, classify(p.parents[2], p)); print(len(d), d[0].metadata, d[0].text[:300])" "notes/<uni>/<course>/<some>.pdf"`
Expected: page count > 0, `page: 1` in metadata, readable Markdown text. (Put any real PDF under `notes/<uni>/<course>/` first; `notes/` is gitignored.)

- [ ] **Step 6: Commit**

```bash
git add src/studyforge/loaders.py tests/test_loaders.py
git commit -m "feat: load PDF/PPTX/DOCX via Docling, markdown and code as text"
```

---

### Task 4: Vector store, embeddings, and ingest manifest

**Files:**
- Create: `src/studyforge/store.py`, `tests/conftest.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces:
  - `vector_store(s: Settings) -> PGVectorStore` (hybrid search on, `simple` text config, jsonb metadata).
  - `embed_model(name: str) -> HuggingFaceEmbedding` (cached per name).
  - `engine(url: str) -> sqlalchemy.Engine` (cached per URL).
  - `class Manifest(s: Settings)` with `all() -> dict[str, tuple[str, list[str]]]` (path → (sha, doc_ids)), `put(path: str, sha: str, doc_ids: list[str]) -> None`, `remove(path: str) -> None`.
  - Test fixtures (in `tests/conftest.py`): `db_settings` → `Settings` using isolated tables `test_chunks` / `test_manifest`, dropped before and after each test, skipping with a clear message if Postgres is down; `mock_embed` → `MockEmbedding(embed_dim=1024)`.

- [ ] **Step 1: Write fixtures** — `tests/conftest.py`

```python
import pytest
import sqlalchemy as sa
from llama_index.core.embeddings import MockEmbedding

from studyforge.config import Settings
from studyforge.store import engine


def _drop(s: Settings) -> None:
    with engine(s.db_url).begin() as c:
        c.execute(sa.text(f"DROP TABLE IF EXISTS data_{s.chunks_table}, {s.manifest_table}"))


@pytest.fixture
def db_settings():
    s = Settings(chunks_table="test_chunks", manifest_table="test_manifest")
    try:
        _drop(s)
    except sa.exc.OperationalError:
        pytest.skip("Postgres not running — start it with: docker compose up -d")
    yield s
    _drop(s)


@pytest.fixture
def mock_embed():
    return MockEmbedding(embed_dim=1024)
```

- [ ] **Step 2: Write the failing tests** — `tests/test_store.py`

```python
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from studyforge.store import Manifest, vector_store


def test_manifest_roundtrip(db_settings):
    m = Manifest(db_settings)
    assert m.all() == {}
    m.put("UiO/A/x.md", "sha1", ["UiO/A/x.md"])
    m.put("UiO/A/x.md", "sha2", ["UiO/A/x.md#p1", "UiO/A/x.md#p2"])  # upsert
    assert m.all() == {"UiO/A/x.md": ("sha2", ["UiO/A/x.md#p1", "UiO/A/x.md#p2"])}
    m.remove("UiO/A/x.md")
    assert m.all() == {}


def test_vector_store_add_and_delete_by_ref_doc(db_settings):
    vs = vector_store(db_settings)
    node = TextNode(
        text="hello", embedding=[0.1] * 1024, metadata={"path": "p"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},  # ref_doc_id = doc-1
    )
    vs.add([node])
    assert len(vs.get_nodes()) == 1
    vs.delete("doc-1")
    assert vs.get_nodes() == []
```

- [ ] **Step 3: Run to verify failure**

Run: `docker compose up -d --wait && uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'studyforge.store'`.

- [ ] **Step 4: Implement** — `src/studyforge/store.py`

```python
from functools import lru_cache

import sqlalchemy as sa
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from studyforge.config import Settings


def vector_store(s: Settings) -> PGVectorStore:
    return PGVectorStore.from_params(
        host=s.db_host, port=str(s.db_port), database=s.db_name,
        user=s.db_user, password=s.db_password,
        table_name=s.chunks_table, embed_dim=s.embed_dim,
        hybrid_search=True, text_search_config="simple",  # Norwegian + English
        use_jsonb=True,
    )


@lru_cache
def embed_model(name: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=name)


@lru_cache
def engine(url: str) -> sa.Engine:
    return sa.create_engine(url)


class Manifest:
    """path -> (file sha256, doc ids). Lets ingest skip unchanged files before parsing."""

    def __init__(self, s: Settings):
        self.engine = engine(s.db_url)
        self.table = s.manifest_table  # from Settings, not user input
        with self.engine.begin() as c:
            c.execute(sa.text(
                f"CREATE TABLE IF NOT EXISTS {self.table} "
                "(path text PRIMARY KEY, sha text NOT NULL, doc_ids text[] NOT NULL)"
            ))

    def all(self) -> dict[str, tuple[str, list[str]]]:
        with self.engine.connect() as c:
            rows = c.execute(sa.text(f"SELECT path, sha, doc_ids FROM {self.table}"))
            return {path: (sha, list(ids)) for path, sha, ids in rows}

    def put(self, path: str, sha: str, doc_ids: list[str]) -> None:
        with self.engine.begin() as c:
            c.execute(sa.text(
                f"INSERT INTO {self.table} (path, sha, doc_ids) VALUES (:p, :s, :d) "
                "ON CONFLICT (path) DO UPDATE SET sha = :s, doc_ids = :d"
            ), {"p": path, "s": sha, "d": doc_ids})

    def remove(self, path: str) -> None:
        with self.engine.begin() as c:
            c.execute(sa.text(f"DELETE FROM {self.table} WHERE path = :p"), {"p": path})
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/studyforge/store.py tests/conftest.py tests/test_store.py
git commit -m "feat: pgvector hybrid store, local embeddings, ingest manifest"
```

---

### Task 5: Incremental ingestion and `ingest` CLI

**Files:**
- Create: `src/studyforge/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `scan` (Task 2), `load` (Task 3), `vector_store`, `embed_model`, `Manifest`, `engine` (Task 4), `Settings`.
- Produces:
  - `@dataclass Report(new: int, updated: int, unchanged: int, removed: int, failed: list[tuple[str, str]])`; `str(report)` → first line `"N new, N updated, N unchanged, N removed, N failed"`, then one `"  FAILED <path>: <reason>"` line per failure.
  - `run(root: Path, s: Settings | None = None, embed: BaseEmbedding | None = None) -> Report`.
  - `main()` — console script `ingest [notes_dir]`.

Behaviour: removed files → delete their doc ids from the vector store and the manifest. Unchanged sha → skip without parsing. New/changed → load; if load raises or returns `[]`, record a failure and **leave existing chunks untouched**; otherwise delete old doc ids, embed and insert the new docs, then update the manifest. Code docs are split with `CodeSplitter`; everything else with `MarkdownNodeParser` then `SentenceSplitter(512, 64)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_ingest.py`

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL — `ImportError: cannot import name 'ingest' from 'studyforge'`.

- [ ] **Step 3: Implement** — `src/studyforge/ingest.py`

```python
import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser, SentenceSplitter

from studyforge.config import Settings
from studyforge.loaders import load
from studyforge.paths import scan
from studyforge.store import Manifest, embed_model, vector_store


@dataclass
class Report:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        head = (f"{self.new} new, {self.updated} updated, {self.unchanged} unchanged, "
                f"{self.removed} removed, {len(self.failed)} failed")
        return "\n".join([head, *(f"  FAILED {p}: {why}" for p, why in self.failed)])


def _splitters(docs: list[Document]) -> list:
    lang = docs[0].metadata.get("language")
    if lang:
        return [CodeSplitter(language=lang, max_chars=1500)]
    return [MarkdownNodeParser(), SentenceSplitter(chunk_size=512, chunk_overlap=64)]


def run(root: Path, s: Settings | None = None, embed: BaseEmbedding | None = None) -> Report:
    s = s or Settings.from_env()
    embed = embed or embed_model(s.embed_model)
    vs, manifest, report = vector_store(s), Manifest(s), Report()
    known = manifest.all()
    files = scan(root)

    on_disk = {meta.path for _, meta in files}
    for path, (_, doc_ids) in known.items():
        if path not in on_disk:
            for doc_id in doc_ids:
                vs.delete(doc_id)
            manifest.remove(path)
            report.removed += 1

    for file, meta in files:
        sha = hashlib.sha256(file.read_bytes()).hexdigest()
        old = known.get(meta.path)
        if old and old[0] == sha:
            report.unchanged += 1
            continue
        try:
            docs = load(file, meta)
            if not docs:
                raise ValueError("no extractable text (scanned PDF? OCR is disabled)")
            # doc ids are deterministic per path/page, so this also cleans up a half-finished earlier run
            for doc_id in old[1] if old else []:
                vs.delete(doc_id)
            IngestionPipeline(transformations=[*_splitters(docs), embed], vector_store=vs).run(documents=docs)
        except Exception as e:  # one bad file must not stop the batch
            report.failed.append((meta.path, f"{type(e).__name__}: {e}"))
            continue
        manifest.put(meta.path, sha, [d.id_ for d in docs])
        if old:
            report.updated += 1
        else:
            report.new += 1
    return report


def main() -> None:
    s = Settings.from_env()
    ap = argparse.ArgumentParser(description="Ingest notes/<uni>/<course>/ into the local vector store")
    ap.add_argument("notes_dir", nargs="?", type=Path, default=s.notes_dir)
    args = ap.parse_args()
    print(run(args.notes_dir, s))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass (16).

- [ ] **Step 6: Commit**

```bash
git add src/studyforge/ingest.py tests/test_ingest.py
git commit -m "feat: incremental ingestion with hash skip and failure report"
```

---

### Task 6: Scoped hybrid retrieval with reranking and `ask` CLI

**Files:**
- Create: `src/studyforge/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `vector_store`, `embed_model` (Task 4), `ingest.run` (Task 5, tests only), `Settings`.
- Produces (Plan 2's `search_notes` tool wraps these):
  - `@dataclass(frozen=True) Scope(uni: str | None = None, course: str | None = None, doc_types: tuple[str, ...] | None = None)`.
  - `@dataclass(frozen=True) Hit(text: str, score: float, path: str, page: int | None, uni: str, course: str | None)` with `cite() -> str` → `"UiO/A/x.pdf p.3"` or `"UiO/A/x.md"`.
  - `search(query: str, scope: Scope = Scope(), s: Settings | None = None, embed: BaseEmbedding | None = None, rerank: BaseNodePostprocessor | None = None) -> list[Hit]` — at most `s.top_n` hits, best first, all with `score >= s.rerank_min_score`; `[]` means "not in your notes".
  - `main()` — console script `ask "question" [--uni U] [--course C]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_retrieval.py`

```python
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from studyforge import ingest
from studyforge.retrieval import Hit, Scope, search


class FixedScoreRerank(BaseNodePostprocessor):
    """Stands in for the cross-encoder: keyword hits score 0.9, everything else 0.0."""
    keyword: str

    def _postprocess_nodes(self, nodes, query_bundle=None):
        for n in nodes:
            n.score = 0.9 if self.keyword in n.node.get_content().lower() else 0.0
        return sorted(nodes, key=lambda n: n.score, reverse=True)[:6]


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def corpus(tmp_path, s, embed):
    write(tmp_path, "UiO/MAT1100/limits.md", "Epsilon-delta definition of a limit.")
    write(tmp_path, "NTNU/MAT1100/limits.md", "Limit laws and squeeze theorem.")
    write(tmp_path, "NTNU/TDT4100/java.md", "Java interfaces and limit of inheritance.")
    ingest.run(tmp_path, s, embed)


def test_course_scope_does_not_leak_across_unis(tmp_path, db_settings, mock_embed):
    corpus(tmp_path, db_settings, mock_embed)
    hits = search("limit", Scope(uni="NTNU", course="MAT1100"), db_settings, mock_embed,
                  FixedScoreRerank(keyword="limit"))
    assert hits and {h.path for h in hits} == {"NTNU/MAT1100/limits.md"}


def test_uni_scope_and_citation(tmp_path, db_settings, mock_embed):
    corpus(tmp_path, db_settings, mock_embed)
    hits = search("limit", Scope(uni="NTNU"), db_settings, mock_embed, FixedScoreRerank(keyword="limit"))
    assert {h.uni for h in hits} == {"NTNU"}
    assert hits[0].cite() in {"NTNU/MAT1100/limits.md", "NTNU/TDT4100/java.md"}
    assert Hit("t", 1.0, "UiO/A/x.pdf", 3, "UiO", "A").cite() == "UiO/A/x.pdf p.3"


def test_low_scores_mean_not_in_notes(tmp_path, db_settings, mock_embed):
    corpus(tmp_path, db_settings, mock_embed)
    assert search("photosynthesis", Scope(), db_settings, mock_embed,
                  FixedScoreRerank(keyword="photosynthesis")) == []


def test_doc_type_filter(tmp_path, db_settings, mock_embed):
    write(tmp_path, "UiO/A/exams/2024.md", "Exam: prove the limit exists.")
    write(tmp_path, "UiO/A/notes.md", "Notes on the limit.")
    ingest.run(tmp_path, db_settings, mock_embed)
    hits = search("limit", Scope(doc_types=("exams",)), db_settings, mock_embed,
                  FixedScoreRerank(keyword="limit"))
    assert [h.path for h in hits] == ["UiO/A/exams/2024.md"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'studyforge.retrieval'`.

- [ ] **Step 3: Implement** — `src/studyforge/retrieval.py`

```python
import argparse
from dataclasses import dataclass
from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from studyforge.config import Settings
from studyforge.store import embed_model, vector_store


@dataclass(frozen=True)
class Scope:
    uni: str | None = None
    course: str | None = None
    doc_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Hit:
    text: str
    score: float
    path: str
    page: int | None
    uni: str
    course: str | None

    def cite(self) -> str:
        return f"{self.path} p.{self.page}" if self.page else self.path


def _filters(scope: Scope) -> MetadataFilters | None:
    fs = [MetadataFilter(key=k, value=v) for k, v in (("uni", scope.uni), ("course", scope.course)) if v]
    if scope.doc_types:
        fs.append(MetadataFilter(key="doc_type", value=list(scope.doc_types), operator=FilterOperator.IN))
    return MetadataFilters(filters=fs) if fs else None


@lru_cache
def reranker(model: str, top_n: int) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(model=model, top_n=top_n)


def search(query: str, scope: Scope = Scope(), s: Settings | None = None,
           embed: BaseEmbedding | None = None, rerank: BaseNodePostprocessor | None = None) -> list[Hit]:
    s = s or Settings.from_env()
    embed = embed or embed_model(s.embed_model)
    rerank = rerank or reranker(s.rerank_model, s.top_n)
    index = VectorStoreIndex.from_vector_store(vector_store(s), embed_model=embed)
    retriever = index.as_retriever(
        vector_store_query_mode="hybrid", similarity_top_k=s.top_k, sparse_top_k=s.top_k,
        filters=_filters(scope),
    )
    nodes = retriever.retrieve(query)  # dense top_k ∪ keyword top_k, deduped
    if not nodes:
        return []
    ranked = rerank.postprocess_nodes(nodes, query_str=query)
    return [
        Hit(n.node.get_content(), n.score or 0.0, m["path"], m.get("page"), m["uni"], m.get("course"))
        for n in ranked if (n.score or 0.0) >= s.rerank_min_score
        for m in [n.node.metadata]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Search your notes")
    ap.add_argument("query")
    ap.add_argument("--uni")
    ap.add_argument("--course")
    args = ap.parse_args()
    hits = search(args.query, Scope(args.uni, args.course))
    if not hits:
        print("Not in your notes.")
    for h in hits:
        print(f"[{h.score:.2f}] {h.cite()}\n    {h.text[:200].replace(chr(10), ' ')}\n")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/studyforge/retrieval.py tests/test_retrieval.py
git commit -m "feat: scoped hybrid retrieval with reranking and citations"
```

---

### Task 7: Sample corpus and retrieval-quality eval gate

**Files:**
- Create: `samples/uni-a/ALG101/sorting.md`, `samples/uni-a/ALG101/graphs.md`, `samples/uni-a/ALG101/code/quicksort.py`, `samples/uni-a/admin/exam-rules.md`, `samples/uni-b/STAT200/regression.md`, `samples/uni-b/STAT200/hypothesis-testing.md`, `samples/uni-b/STAT200/exams/2025-exam.md`, `samples/uni-b/FIL110/etikk.md`, `evals/golden.jsonl`, `evals/test_retrieval_quality.py`

**Interfaces:**
- Consumes: `ingest.run`, `search`, `Scope`, `Settings`, `engine`.
- Produces: `uv run pytest -m eval evals -s` → prints hit@6 / MRR / refusal and fails below thresholds. `evals/golden.jsonl` line format: `{"q": str, "expect": str | null, "uni"?: str, "course"?: str}`; `expect: null` means the correct result is "not in your notes".

- [ ] **Step 1: Create the sample corpus**

`samples/uni-a/ALG101/sorting.md`
```markdown
# Sorting algorithms

Merge sort splits the array in halves, sorts each half recursively and merges them.
It runs in O(n log n) time in every case and is stable, but needs O(n) extra memory.

Quicksort picks a pivot and partitions the array into elements smaller and larger than the pivot.
Average time is O(n log n); worst case O(n^2) happens when the pivot is always the smallest or largest element.
Randomised pivot choice makes the worst case unlikely.

Insertion sort is O(n^2) but fast on small or nearly sorted inputs.
```

`samples/uni-a/ALG101/graphs.md`
```markdown
# Graph traversal

Breadth-first search (BFS) explores a graph level by level using a queue.
On an unweighted graph BFS finds shortest paths in O(V + E).

Depth-first search (DFS) uses a stack or recursion; it is used for topological sorting
and finding strongly connected components.

Dijkstra's algorithm finds shortest paths with non-negative edge weights using a priority queue.
Topological order is only defined for directed acyclic graphs. See [[sorting|sorting algorithms]] for heaps.
```

`samples/uni-a/ALG101/code/quicksort.py`
```python
import random


def quicksort(xs: list[int]) -> list[int]:
    """Randomised quicksort: expected O(n log n)."""
    if len(xs) <= 1:
        return xs
    pivot = random.choice(xs)
    smaller = [x for x in xs if x < pivot]
    equal = [x for x in xs if x == pivot]
    larger = [x for x in xs if x > pivot]
    return quicksort(smaller) + equal + quicksort(larger)
```

`samples/uni-a/admin/exam-rules.md`
```markdown
# Exam regulations

Written exams last four hours. No aids are permitted unless the course page says otherwise.
Students who are ill on the exam day must submit a medical certificate within three days
to be registered for the postponed exam in the following semester.
Grades range from A to F, where F is a fail.
```

`samples/uni-b/STAT200/regression.md`
```markdown
# Linear regression

Ordinary least squares chooses coefficients that minimise the sum of squared residuals.
R-squared is the fraction of variance in the response explained by the model.

Assumptions: linearity, independent errors, constant error variance (homoscedasticity)
and normally distributed errors for exact inference.
Multicollinearity between predictors inflates the variance of coefficient estimates.
```

`samples/uni-b/STAT200/hypothesis-testing.md`
```markdown
# Hypothesis testing

The p-value is the probability, assuming the null hypothesis is true, of observing data at least as extreme as what was observed.
A type I error rejects a true null hypothesis; its probability is the significance level alpha.
A type II error fails to reject a false null hypothesis; power equals one minus its probability.
A two-sample t-test compares the means of two independent groups.
```

`samples/uni-b/STAT200/exams/2025-exam.md`
```markdown
# STAT200 exam, spring 2025

Problem 1: A study reports p = 0.03 with alpha = 0.05. State the conclusion and explain what the p-value means.
Problem 2: Fit a simple linear regression of exam score on hours studied and interpret the slope and R-squared.
```

`samples/uni-b/FIL110/etikk.md`
```markdown
# Etikk

Utilitarisme sier at en handling er riktig hvis den gir størst mulig samlet lykke for alle berørte.
Pliktetikk, knyttet til Immanuel Kant, sier at noen handlinger er riktige eller gale uavhengig av konsekvensene.
Det kategoriske imperativ: handle bare etter den maksime du samtidig kan ville skal bli en allmenn lov.
Dydsetikk, fra Aristoteles, spør hva en god person med gode karaktertrekk ville gjort.
```

- [ ] **Step 2: Create `evals/golden.jsonl`**

```jsonl
{"q": "What is the worst-case running time of quicksort and when does it happen?", "expect": "uni-a/ALG101/sorting.md"}
{"q": "Which sorting algorithm is stable and always O(n log n)?", "expect": "uni-a/ALG101/sorting.md"}
{"q": "How do I find shortest paths in an unweighted graph?", "expect": "uni-a/ALG101/graphs.md"}
{"q": "What algorithm handles shortest paths with non-negative weights?", "expect": "uni-a/ALG101/graphs.md"}
{"q": "Show me an implementation of randomised quicksort", "expect": "uni-a/ALG101/code/quicksort.py"}
{"q": "What happens if I am sick on the exam day?", "expect": "uni-a/admin/exam-rules.md"}
{"q": "What does R-squared measure?", "expect": "uni-b/STAT200/regression.md"}
{"q": "What is the difference between type I and type II errors?", "expect": "uni-b/STAT200/hypothesis-testing.md"}
{"q": "What does a p-value mean?", "expect": "uni-b/STAT200/hypothesis-testing.md", "uni": "uni-b", "course": "STAT200"}
{"q": "Past exam question about interpreting a regression slope", "expect": "uni-b/STAT200/exams/2025-exam.md"}
{"q": "Hva er det kategoriske imperativ?", "expect": "uni-b/FIL110/etikk.md"}
{"q": "What does utilitarianism say makes an action right?", "expect": "uni-b/FIL110/etikk.md"}
{"q": "Who won the 1998 FIFA World Cup?", "expect": null}
{"q": "How do I bake sourdough bread?", "expect": null}
{"q": "What is the capital of Australia?", "expect": null}
```

- [ ] **Step 3: Write the eval** — `evals/test_retrieval_quality.py`

```python
import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from studyforge import ingest
from studyforge.config import Settings
from studyforge.retrieval import Scope, search
from studyforge.store import engine

ROOT = Path(__file__).parent.parent
# ponytail: fixed thresholds; switch to a stored baseline + regression diff once the golden set grows
MIN_HIT_AT_N = 0.9
MIN_MRR = 0.7


@pytest.mark.eval
def test_retrieval_quality():
    s = Settings(chunks_table="eval_chunks", manifest_table="eval_manifest")
    with engine(s.db_url).begin() as c:
        c.execute(sa.text(f"DROP TABLE IF EXISTS data_{s.chunks_table}, {s.manifest_table}"))
    report = ingest.run(ROOT / "samples", s)
    assert not report.failed, str(report)

    cases = [json.loads(line) for line in (ROOT / "evals/golden.jsonl").read_text("utf-8").splitlines() if line]
    answerable = [c for c in cases if c["expect"]]
    refusals = [c for c in cases if not c["expect"]]

    hits, rr, refused = 0, 0.0, 0
    for c in answerable:
        paths = [h.path for h in search(c["q"], Scope(c.get("uni"), c.get("course")), s)]
        if c["expect"] in paths:
            hits += 1
            rr += 1 / (paths.index(c["expect"]) + 1)
        else:
            print(f"MISS  {c['q']!r} -> {paths[:3]}")
    for c in refusals:
        got = search(c["q"], Scope(), s)
        refused += not got
        if got:
            print(f"NOT REFUSED  {c['q']!r} -> {[(h.path, round(h.score, 3)) for h in got[:3]]}")

    hit_rate, mrr = hits / len(answerable), rr / len(answerable)
    print(f"hit@{s.top_n}={hit_rate:.2f}  MRR={mrr:.2f}  refusals={refused}/{len(refusals)}")
    assert hit_rate >= MIN_HIT_AT_N
    assert mrr >= MIN_MRR
    assert refused == len(refusals)
```

- [ ] **Step 4: Run the eval with the real models**

Run: `uv run pytest -m eval evals -s`
Expected: PASS and a line like `hit@6=1.00  MRR=0.9x  refusals=3/3`. First run downloads bge-m3 and bge-reranker-v2-m3 (~4.5 GB total) to `~/.cache/huggingface`.

If refusals fail: the `NOT REFUSED` lines print the offending scores. Set `rerank_min_score` in `src/studyforge/config.py` just above the highest printed refusal score, while staying below the lowest score of a correct hit (print those with a temporary `print` if needed). Rerun until it passes. This is the calibration step, not a bug.

- [ ] **Step 5: Commit**

```bash
git add samples evals src/studyforge/config.py
git commit -m "feat: sample corpus and retrieval-quality eval gate"
```

---

### Task 8: CI and README

**Files:**
- Create: `.github/workflows/ci.yml`, `README.md`

**Interfaces:**
- Consumes: the whole test and eval suite.
- Produces: GitHub Actions running unit/integration tests and the eval gate on every push and PR.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg17
        env:
          POSTGRES_USER: studyforge
          POSTGRES_PASSWORD: studyforge
          POSTGRES_DB: studyforge
        ports: ["5433:5432"]
        options: >-
          --health-cmd "pg_isready -U studyforge"
          --health-interval 5s --health-timeout 5s --health-retries 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv sync
      - run: uv run pytest -v
      - uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-${{ hashFiles('src/studyforge/config.py') }}
      - run: uv run pytest -m eval evals -s
```

- [ ] **Step 2: Create `README.md`**

````markdown
# studyforge

Multi-agent RAG study assistant for studying at two universities at once.
Local embeddings (bge-m3) and reranking (bge-reranker-v2-m3), Postgres + pgvector hybrid search,
LlamaIndex data plane. Agents, Chainlit UI, MCP server and calendar are coming in later phases —
see [the design spec](docs/superpowers/specs/2026-09-29-studyforge-design.md).

## Run

```bash
docker compose up -d --wait
uv sync
# put course files in notes/<uni>/<course>/ (gitignored — never committed)
uv run ingest
uv run ask "What is a p-value?" --uni NTNU --course STAT200
```

Folder conventions: `notes/<uni>/<course>/...`; subfolders named `exams/`, `code/`, `admin/` set the document type;
files directly under `notes/<uni>/` or in `notes/<uni>/admin/` belong to the whole institution.

## Test

```bash
uv run pytest                 # unit + integration (needs Postgres)
uv run pytest -m eval evals -s   # retrieval quality on samples/ (downloads models)
```
````

- [ ] **Step 3: Push and verify CI**

Run: `git add .github README.md && git commit -m "ci: tests and retrieval eval gate" && git push && gh run watch --exit-status`
Expected: the `ci` workflow finishes green. The first run is slow (~10–15 min: torch, docling and model downloads).

---

### Task 9: Real-notes smoke test (manual, with the user)

**Files:** none (uses gitignored `notes/`).

- [ ] **Step 1:** Ask the user to copy one course from each institution into `notes/<uni>/<course>/` (include at least one PDF, one PPTX and one DOCX).
- [ ] **Step 2:** Run `uv run ingest`. Expected: report with `0 failed`, or failures whose reasons are understandable (e.g. scanned PDF).
- [ ] **Step 3:** Run `uv run ingest` again. Expected: everything `unchanged`, finishing in seconds (no Docling parsing).
- [ ] **Step 4:** Run 3 real questions with `uv run ask "..." --uni <U> --course <C>`, plus one off-topic question. Expected: relevant cited passages with page numbers for PDFs/slides; "Not in your notes." for the off-topic one. Note any misses as candidate lines for a private `evals/golden.local.jsonl` (gitignored).
