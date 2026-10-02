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
    # ponytail: candidates per retriever; reranking ~2x top_k passages on CPU dominates answer time (30 took ~1 min)
    top_k: int = 10
    top_n: int = 6
    # ponytail: calibration knob — below this rerank score a passage counts as "not in your notes".
    # Tune against the refusal cases in evals/golden.jsonl.
    rerank_min_score: float = 0.05
    expert_model: str = "claude-sonnet-5-5"
    router_model: str = "claude-haiku-4-5"  # routing, personas, eval judge
    calendar_prefix: str = "cal"  # tables cal_events, cal_blocks, cal_prefs

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

    @property
    def pg_conn(self) -> str:  # psycopg 3 DSN for the LangGraph checkpointer
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
