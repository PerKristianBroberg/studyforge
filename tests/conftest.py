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
