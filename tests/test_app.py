import pytest

from studyforge import app


def test_run_config_without_langfuse(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert app.run_config("t1") == {"configurable": {"thread_id": "t1"}}


def test_run_config_with_langfuse(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
    cfg = app.run_config("t1")
    assert type(cfg["callbacks"][0]).__name__ == "LangchainCallbackHandler"


def test_repl_survives_a_failed_turn(monkeypatch):
    from langchain_core.messages import AIMessage

    class FlakyGraph:
        calls = 0

        def invoke(self, state, config):
            FlakyGraph.calls += 1
            if FlakyGraph.calls == 1:
                raise RuntimeError("529 overloaded")
            return {"expert_id": "general", "messages": [AIMessage(content="second answer")]}

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    inputs, printed = iter(["q1", "q2", ""]), []
    app.repl(FlakyGraph(), "auto", "t1", read=lambda _: next(inputs), write=printed.append)
    assert any("529 overloaded" in p for p in printed)
    assert any("second answer" in p for p in printed)


def test_chat_without_api_key_exits_with_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["chat"])
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        app.main()
