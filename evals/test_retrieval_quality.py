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
