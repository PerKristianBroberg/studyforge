import json

from studyforge.config import Settings
from studyforge.retrieval import Hit, Scope
from studyforge.tools import NOT_IN_NOTES, Quiz, QuizQuestion, format_hits, make_tools

HIT = Hit("A heap is a complete binary tree.", 0.9, "UiO/INF1010/heaps.pdf", 3, "UiO", "INF1010")


class StubQuizLLM:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is Quiz
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return Quiz(questions=[QuizQuestion(question="What is a heap?", choices=["a", "b", "c", "d"],
                                            answer=0, source="UiO/INF1010/heaps.pdf p.3")])


def test_format_hits():
    assert format_hits([HIT]) == "[1] UiO/INF1010/heaps.pdf p.3\nA heap is a complete binary tree."
    assert format_hits([]) == NOT_IN_NOTES


def test_search_notes_uses_bound_scope(monkeypatch):
    seen = []
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: seen.append(scope) or [HIT])
    scope = Scope(uni="UiO", course="INF1010")
    search_notes, _ = make_tools(scope, Settings(), StubQuizLLM())
    out = search_notes.invoke({"query": "heaps"})
    assert seen == [scope] and "heaps.pdf p.3" in out


def test_make_quiz_uses_notes_and_returns_json(monkeypatch):
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: [HIT])
    llm = StubQuizLLM()
    _, make_quiz = make_tools(Scope(uni="UiO", course="INF1010"), Settings(), llm)
    quiz = json.loads(make_quiz.invoke({"topic": "heaps", "n": 1}))
    assert quiz["questions"][0]["source"] == "UiO/INF1010/heaps.pdf p.3"
    assert "complete binary tree" in llm.prompts[0]


def test_make_quiz_without_notes_does_not_call_llm(monkeypatch):
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: [])
    llm = StubQuizLLM()
    _, make_quiz = make_tools(Scope(), Settings(), llm)
    assert make_quiz.invoke({"topic": "photosynthesis"}) == NOT_IN_NOTES
    assert llm.prompts == []
