import asyncio
import json
import os
from pathlib import Path

import pytest
from anthropic import AsyncAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from studyforge import ingest
from studyforge.app import build_app
from studyforge.config import Settings

ROOT = Path(__file__).parent.parent
MIN_FAITHFULNESS = 0.8  # ponytail: fixed threshold; move to stored baseline once the golden set grows


@pytest.mark.llm_eval
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_answer_quality():
    s = Settings(chunks_table="eval_chunks", manifest_table="eval_manifest", notes_dir=ROOT / "samples")
    assert not ingest.run(s.notes_dir, s).failed
    graph, _ = build_app(s, InMemorySaver())
    judge_llm = llm_factory(s.router_model, provider="anthropic", client=AsyncAnthropic(), max_tokens=4096)
    # ponytail: ragas 0.4.3 always sends temperature/top_p, which anthropic SDK 1.9 rejects; drop when ragas fixes it
    for arg in ("temperature", "top_p"):
        judge_llm.model_args.pop(arg, None)
    judge = Faithfulness(llm=judge_llm)

    cases = [json.loads(line) for line in (ROOT / "evals/golden.jsonl").read_text("utf-8").splitlines() if line]
    scores, refused, refusals = [], 0, 0
    for i, c in enumerate(cases):
        out = graph.invoke({"messages": [("user", c["q"])], "requested": "auto"},
                           {"configurable": {"thread_id": f"eval-{i}"}})
        answer = out["messages"][-1].text
        contexts = [m.text for m in out["messages"] if m.type == "tool"]
        if c["expect"] is None:
            refusals += 1
            refused += "not in your notes" in answer.lower()
            continue
        score = asyncio.run(judge.ascore(user_input=c["q"], response=answer, retrieved_contexts=contexts)).value
        scores.append(score)
        print(f"{score:.2f}  [{out['expert_id']}] {c['q']}")

    mean = sum(scores) / len(scores)
    print(f"faithfulness={mean:.2f}  refusals={refused}/{refusals}")
    assert mean >= MIN_FAITHFULNESS
    assert refused == refusals
