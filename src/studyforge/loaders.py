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
