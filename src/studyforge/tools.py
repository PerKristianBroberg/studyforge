from dataclasses import asdict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from studyforge import retrieval, tracing
from studyforge.config import Settings
from studyforge.retrieval import Hit, Scope

NOT_IN_NOTES = "NOT_IN_NOTES: nothing relevant in the student's notes for this scope."


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="exactly 4 answer options")
    answer: int = Field(description="0-based index of the correct choice")
    source: str = Field(description="citation of the passage used, e.g. 'UiO/INF1010/x.pdf p.3'")


class Quiz(BaseModel):
    questions: list[QuizQuestion]


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return NOT_IN_NOTES
    return "\n\n".join(f"[{i}] {h.cite()}\n{h.text}" for i, h in enumerate(hits, 1))


def make_tools(scope: Scope, s: Settings, quiz_llm: BaseChatModel) -> list[BaseTool]:
    """Tools closed over one expert's scope, so the expert cannot search outside it."""

    @tool
    def search_notes(query: str) -> str:
        """Search the student's notes in this expert's scope. Returns numbered passages with citations, or NOT_IN_NOTES."""
        with tracing.observe("retrieve-notes", "retriever", input={"query": query, "scope": asdict(scope)}) as obs:
            hits = retrieval.search(query, scope, s)
            obs.update(output=[h.cite() for h in hits], metadata={"scores": [round(h.score, 3) for h in hits]})
        return format_hits(hits)

    @tool
    def make_quiz(topic: str, n: int = 5) -> str:
        """Create an n-question multiple-choice quiz on a topic, using only the student's notes. Returns JSON."""
        hits = retrieval.search(topic, scope, s)
        if not hits:
            return NOT_IN_NOTES
        quiz = quiz_llm.with_structured_output(Quiz, method="json_schema").invoke(
            f"Write {n} multiple-choice questions (4 choices each) testing understanding of '{topic}', "
            f"using ONLY these notes. Cite the passage each question comes from.\n\n{format_hits(hits)}",
            config={"run_name": "generate-quiz"},
        )
        return quiz.model_dump_json()

    return [search_notes, make_quiz]
