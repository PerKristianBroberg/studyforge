import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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


def run(root: Path, s: Settings | None = None, embed: BaseEmbedding | None = None,
        progress: Callable[[int, int, str], None] | None = None) -> Report:
    if not root.is_dir():  # an empty scan would otherwise "remove" every ingested file
        raise FileNotFoundError(f"notes dir not found: {root}")
    s = s or Settings.from_env()
    vs, manifest, report = vector_store(s), Manifest(s), Report()
    known = manifest.all()
    files = scan(root)

    def remove(path: str, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            vs.delete(doc_id)
        manifest.remove(path)
        report.removed += 1

    on_disk = {meta.path for _, meta in files}
    for path, (_, doc_ids) in known.items():
        if path not in on_disk:
            remove(path, doc_ids)

    for i, (file, meta) in enumerate(files, 1):
        if progress:
            progress(i, len(files), meta.path)
        old = known.get(meta.path)
        try:
            try:
                sha = hashlib.sha256(file.read_bytes()).hexdigest()
            except FileNotFoundError:  # deleted (e.g. in the library) after the scan: a removal, not a failure
                if old:
                    remove(meta.path, old[1])
                continue
            if old and old[0] == sha:
                report.unchanged += 1
                continue
            docs = load(file, meta)
            if not docs:
                raise ValueError("no extractable text (scanned PDF? OCR is disabled)")
            embed = embed or embed_model(s.embed_model)  # lazy: a no-change run never loads it
            nodes = IngestionPipeline(transformations=[*_splitters(docs), embed]).run(documents=docs)
            # swap only after split+embed succeeded; deleting new ids too cleans up a half-finished earlier run
            for doc_id in {*(old[1] if old else []), *(d.id_ for d in docs)}:
                vs.delete(doc_id)
            vs.add(nodes)
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
