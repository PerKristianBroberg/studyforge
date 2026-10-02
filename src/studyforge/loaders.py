import os
import re
from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (AcceleratorOptions, EasyOcrOptions, OcrMode, PdfPipelineOptions,
                                                TableFormerMode, TableStructureOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import Document

from studyforge.paths import CODE_LANGS, DOCLING_EXTS, FileMeta

WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


@lru_cache(maxsize=2)
def _converter(ocr: bool = False) -> DocumentConverter:
    # CPU-only: the accurate table model took minutes per table-heavy PDF; fast mode + all cores is the trade-off
    speed = {"table_structure_options": TableStructureOptions(mode=TableFormerMode.FAST),
             "accelerator_options": AcceleratorOptions(num_threads=os.cpu_count() or 4)}
    # OCR is slow on CPU, so it only runs as a fallback for PDFs without a text layer (scans, photos)
    opts = (PdfPipelineOptions(do_ocr=True, ocr_options=EasyOcrOptions(lang=["no", "en"], mode=OcrMode.FULL_PAGE),
                               **speed)
            if ocr else PdfPipelineOptions(do_ocr=False, **speed))
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


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
    docs = _docling_docs(_converter().convert(file).document, base)
    if not docs and file.suffix.lower() == ".pdf":  # no text layer: scanned or photographed pages
        docs = _docling_docs(_converter(ocr=True).convert(file).document, base)
    return docs


def _docling_docs(doc, base: dict) -> list[Document]:
    if not doc.pages:  # DOCX has no pages
        text = doc.export_to_markdown(image_placeholder="")
        return [Document(id_=base["path"], text=text, metadata=base)] if text.strip() else []
    docs = []
    for n in sorted(doc.pages):  # PDF pages / PPTX slides, 1-based
        text = doc.export_to_markdown(page_no=n, image_placeholder="")
        if text.strip():
            docs.append(Document(id_=f"{base['path']}#p{n}", text=text, metadata={**base, "page": n}))
    return docs
