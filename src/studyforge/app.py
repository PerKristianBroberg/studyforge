import argparse
import os
import uuid

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.postgres import PostgresSaver

from studyforge.config import Settings
from studyforge.experts import discover
from studyforge.graph import build_graph, make_router
from studyforge.personas import persona


def build_app(s: Settings, checkpointer):
    experts = discover(s.notes_dir)
    expert_llm = ChatAnthropic(model=s.expert_model, max_tokens=4096)
    fast_llm = ChatAnthropic(model=s.router_model, max_tokens=1024)
    personas = {e.id: persona(e, s, fast_llm) for e in experts if e.kind != "general"}
    graph = build_graph(experts, personas, expert_llm, make_router(fast_llm, experts, personas), s, checkpointer)
    return graph, experts


def run_config(thread_id: str) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):  # tracing is opt-in; traces stay in the local Langfuse
        from langfuse.langchain import CallbackHandler
        cfg["callbacks"] = [CallbackHandler()]
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your study experts")
    ap.add_argument("--expert", default="auto", help="expert id (e.g. UiO/INF1010) or 'auto'")
    ap.add_argument("--thread", default=None, help="conversation id to resume")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Put it in .env and run: uv run --env-file .env chat")
    s = Settings.from_env()
    thread = args.thread or uuid.uuid4().hex[:8]
    with PostgresSaver.from_conn_string(s.pg_conn) as checkpointer:
        checkpointer.setup()
        graph, experts = build_app(s, checkpointer)
        print(f"thread {thread} | experts: {', '.join(e.id for e in experts)} | empty line quits")
        while q := input("> ").strip():
            out = graph.invoke({"messages": [("user", q)], "requested": args.expert}, run_config(thread))
            print(f"[{out['expert_id']}] {out['messages'][-1].text}\n")
