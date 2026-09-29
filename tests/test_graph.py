import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from studyforge.config import Settings
from studyforge.experts import GENERAL, Expert
from studyforge.graph import build_graph
from studyforge.retrieval import Hit, Scope

UIO = Expert("UiO", "UiO expert", "uni", Scope(uni="UiO"))
INF = Expert("UiO/INF1010", "INF1010 expert (UiO)", "course", Scope(uni="UiO", course="INF1010"))
EXPERTS = [GENERAL, UIO, INF]


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*answers):
    """Each answer: first call search_notes, then reply with the text."""
    msgs = []
    for i, text in enumerate(answers):
        msgs.append(AIMessage(content="", tool_calls=[{"name": "search_notes", "args": {"query": "q"}, "id": f"c{i}"}]))
        msgs.append(AIMessage(content=text))
    return FakeToolModel(messages=iter(msgs))


@pytest.fixture
def seen_scopes(monkeypatch):
    seen = []
    hit = Hit("A heap is a tree.", 0.9, "UiO/INF1010/h.md", None, "UiO", "INF1010")
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: seen.append(scope) or [hit])
    return seen


def run(graph, text, requested, thread="t1"):
    return graph.invoke({"messages": [("user", text)], "requested": requested},
                        {"configurable": {"thread_id": thread}})


def test_explicit_expert_runs_with_its_scope(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("Heaps are trees [UiO/INF1010/h.md]."),
                    lambda m: pytest.fail("router must not run"), Settings(), InMemorySaver())
    out = run(g, "What is a heap?", "UiO/INF1010")
    assert out["expert_id"] == "UiO/INF1010"
    assert seen_scopes == [INF.scope]
    assert out["messages"][-1].text == "Heaps are trees [UiO/INF1010/h.md]."


def test_auto_uses_router(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("ok"), lambda m: "UiO", Settings(), InMemorySaver())
    assert run(g, "exam rules?", "auto")["expert_id"] == "UiO"
    assert seen_scopes == [UIO.scope]


def test_router_garbage_falls_back_to_general(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("ok"), lambda m: "made-up-expert", Settings(), InMemorySaver())
    assert run(g, "hi", "auto")["expert_id"] == "general"


def test_thread_memory(seen_scopes):
    g = build_graph(EXPERTS, {}, scripted("first", "second"), lambda m: "general", Settings(), InMemorySaver())
    run(g, "question one", "general")
    out = run(g, "question two", "general")
    texts = [m.text for m in out["messages"]]
    assert "question one" in texts and "first" in texts and texts[-1] == "second"


def test_duplicate_node_names_rejected():
    clash = Expert("uio", "x", "uni", Scope(uni="uio"))
    with pytest.raises(ValueError, match="expert_uio"):
        build_graph([GENERAL, UIO, clash], {}, scripted(), lambda m: "general", Settings())


def test_router_uses_native_json_schema():
    from studyforge.graph import Route, make_router

    class StubLLM:
        def with_structured_output(self, schema, method="function_calling"):
            assert schema is Route and method == "json_schema"
            return self

        def invoke(self, prompt):
            return Route(expert_id="UiO")

    from langchain_core.messages import HumanMessage
    assert make_router(StubLLM(), EXPERTS)([HumanMessage("exam rules?")]) == "UiO"


def test_router_menu_includes_personas():
    from langchain_core.messages import HumanMessage
    from studyforge.graph import Route, make_router

    class StubLLM:
        prompt = ""

        def with_structured_output(self, schema, method="function_calling"):
            return self

        def invoke(self, prompt):
            StubLLM.prompt = prompt
            return Route(expert_id="UiO/INF1010")

    personas = {"UiO/INF1010": "You are an expert on algorithms: sorting, graphs and heaps."}
    make_router(StubLLM(), EXPERTS, personas)([HumanMessage("quicksort code?")])
    assert "sorting, graphs and heaps" in StubLLM.prompt  # course codes alone are opaque to the router
