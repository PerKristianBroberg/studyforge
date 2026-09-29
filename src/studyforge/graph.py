from collections import Counter
from typing import Callable

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from studyforge.config import Settings
from studyforge.experts import GENERAL, Expert
from studyforge.tools import make_tools

RULES = """

Rules:
- Call search_notes before answering any question about course content.
- Answer only from the returned passages and cite them inline like [UiO/INF1010/x.pdf p.3].
- If search_notes returns NOT_IN_NOTES, reply "Not in your notes." and offer general knowledge only if the student asks for it.
- Use make_quiz when the student asks to be quizzed or tested.
- Reply in the student's language (Norwegian or English)."""


class State(MessagesState):
    requested: str  # "auto" or an Expert.id
    expert_id: str


class Route(BaseModel):
    expert_id: str = Field(description="id of the best expert for the student's message")


def make_router(llm, experts: list[Expert]) -> Callable[[list[BaseMessage]], str]:
    menu = "\n".join(f"- {e.id}: {e.name}" for e in experts)
    structured = llm.with_structured_output(Route, method="json_schema")  # Sonnet 5.5 rejects forced tool_choice

    def router(messages: list[BaseMessage]) -> str:
        return structured.invoke(
            "Pick the expert best suited to the student's latest message. Prefer the most specific "
            f"course expert; use 'general' if unsure or if it spans courses.\nExperts:\n{menu}\n\n"
            f"Message: {messages[-1].text}"
        ).expert_id

    return router


def build_graph(experts: list[Expert], personas: dict[str, str], expert_llm,
                router: Callable[[list[BaseMessage]], str], s: Settings,
                checkpointer=None, quiz_llm=None):
    dupes = [n for n, k in Counter(e.node for e in experts).items() if k > 1]
    if dupes:
        raise ValueError(f"expert folders collide on node names: {dupes}")
    by_id = {e.id: e for e in experts}

    def route(state: State) -> Command:
        wanted = state.get("requested") or "auto"
        eid = wanted if wanted in by_id else router(state["messages"])
        eid = eid if eid in by_id else GENERAL.id
        return Command(goto=by_id[eid].node, update={"expert_id": eid})

    g = StateGraph(State)
    g.add_node("route", route, destinations=tuple(e.node for e in experts))
    for e in experts:
        prompt = personas.get(e.id, f"You are a study assistant for {e.name}.") + RULES
        tools = make_tools(e.scope, s, quiz_llm or expert_llm)
        g.add_node(e.node, create_agent(expert_llm, tools, system_prompt=prompt, name=e.node))
        g.add_edge(e.node, END)
    g.add_edge(START, "route")
    return g.compile(checkpointer=checkpointer)
