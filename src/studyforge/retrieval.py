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

    def __post_init__(self):
        if self.course and not self.uni:  # same course code can exist at both institutions
            raise ValueError("course scope requires uni")


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
