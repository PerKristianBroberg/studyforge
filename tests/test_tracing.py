from contextlib import contextmanager

import pytest
from langchain_core.messages import AIMessage

from studyforge import app, tracing
from studyforge.config import Settings
from studyforge.retrieval import Hit, Scope
from studyforge.tools import make_tools


class FakeObs:
    def __init__(self, rec):
        self.rec = rec

    def update(self, **kw):
        self.rec["updates"].append(kw)


@pytest.fixture
def recorded(monkeypatch):
    """Enable tracing and capture what would be sent to Langfuse."""
    rec = {"observations": [], "attrs": [], "updates": []}

    class FakeClient:
        @contextmanager
        def start_as_current_observation(self, **kw):
            rec["observations"].append(kw)
            yield FakeObs(rec)

    @contextmanager
    def fake_propagate(**kw):
        rec["attrs"].append(kw)
        yield

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
    monkeypatch.setattr("langfuse.get_client", lambda: FakeClient())
    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagate)
    monkeypatch.setattr(tracing, "callbacks", lambda: [])
    return rec


class FakeGraph:
    def invoke(self, state, config):
        return {"expert_id": "UiO/INF1010", "messages": [AIMessage(content="Heaps are trees [x.md].")]}


def test_disabled_tracing_is_noop(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert tracing.callbacks() == []
    with tracing.observe("chat-turn", "agent", input="q", session_id="t") as obs:
        obs.update(output="a")  # must not raise or contact Langfuse


@pytest.mark.parametrize("host,base_url,expected", [
    (None, None, False),                              # langfuse SDK would default to cloud.langfuse.com
    ("https://cloud.langfuse.com", None, False),
    (None, "https://us.cloud.langfuse.com", False),
    ("http://127.0.0.1:3000", None, True),
    (None, "http://localhost:3000", True),
])
def test_tracing_only_enabled_for_local_langfuse(monkeypatch, host, base_url, expected):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    for var, val in (("LANGFUSE_HOST", host), ("LANGFUSE_BASE_URL", base_url)):
        if val:
            monkeypatch.setenv(var, val)
        else:
            monkeypatch.delenv(var, raising=False)
    assert tracing.enabled() is expected


def test_chat_turn_is_one_named_trace_per_turn_in_a_session(recorded):
    out = app.ask(FakeGraph(), "What is a heap?", "auto", "thread-1")
    assert out["expert_id"] == "UiO/INF1010"
    assert recorded["observations"] == [{"name": "chat-turn", "as_type": "agent", "input": "What is a heap?"}]
    assert recorded["attrs"] == [{"session_id": "thread-1", "tags": ["interface:cli"]}]
    assert recorded["updates"] == [{"output": "Heaps are trees [x.md].",
                                    "metadata": {"expert_id": "UiO/INF1010", "requested": "auto"}}]


def test_search_notes_emits_retriever_observation(recorded, monkeypatch):
    hit = Hit("A heap is a tree.", 0.91234, "UiO/INF1010/h.pdf", 2, "UiO", "INF1010")
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: [hit])
    search_notes, _ = make_tools(Scope(uni="UiO", course="INF1010"), Settings(), quiz_llm=None)
    search_notes.invoke({"query": "heaps"})
    [obs] = recorded["observations"]
    assert (obs["name"], obs["as_type"]) == ("retrieve-notes", "retriever")
    assert obs["input"] == {"query": "heaps", "scope": {"uni": "UiO", "course": "INF1010", "doc_types": None}}
    assert recorded["updates"] == [{"output": ["UiO/INF1010/h.pdf p.2"], "metadata": {"scores": [0.912]}}]


def test_ask_passes_interface_and_extra_callbacks(recorded):
    seen = {}

    class Graph:
        def invoke(self, state, config):
            seen["callbacks"] = config.get("callbacks")
            return {"expert_id": "general", "messages": [AIMessage(content="ok")]}

    marker = object()
    app.ask(Graph(), "hi", "auto", "t9", interface="web", extra_callbacks=[marker])
    assert recorded["attrs"] == [{"session_id": "t9", "tags": ["interface:web"]}]
    assert marker in seen["callbacks"]
