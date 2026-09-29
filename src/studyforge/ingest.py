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
