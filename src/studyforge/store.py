import threading
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


# lru_cache alone lets two threads (UI warm-up + background indexer) load the same model twice: 2x RAM
model_lock = threading.Lock()


def embed_model(name: str) -> HuggingFaceEmbedding:
    with model_lock:
        return _embed_model(name)


@lru_cache
def _embed_model(name: str) -> HuggingFaceEmbedding:
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
