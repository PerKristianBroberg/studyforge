import sqlalchemy as sa
from langchain_core.messages import AIMessage

from studyforge.experts import Expert
from studyforge.personas import in_scope, persona
from studyforge.retrieval import Scope
from studyforge.store import Manifest, engine

INF = Expert("UiO/INF1010", "INF1010 expert (UiO)", "course", Scope(uni="UiO", course="INF1010"))


class CountingLLM:
    calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return AIMessage(content=f"You are the INF1010 expert v{self.calls}.")


def test_in_scope():
    assert in_scope("UiO/INF1010/a.pdf", Scope(uni="UiO", course="INF1010"))
    assert not in_scope("NTNU/INF1010/a.pdf", Scope(uni="UiO", course="INF1010"))
    assert not in_scope("UiO/admin/rules.pdf", Scope(uni="UiO", course="INF1010"))
    assert in_scope("UiO/admin/rules.pdf", Scope(uni="UiO"))
    assert in_scope("NTNU/x.md", Scope())


def test_persona_cached_until_scope_changes(db_settings):
    with engine(db_settings.db_url).begin() as c:
        c.execute(sa.text("DROP TABLE IF EXISTS personas"))
    m = Manifest(db_settings)
    m.put("UiO/INF1010/a.md", "sha-a", ["UiO/INF1010/a.md"])
    llm = CountingLLM()
    material = lambda e, s: "heaps, trees"

    assert persona(INF, db_settings, llm, material) == "You are the INF1010 expert v1."
    assert persona(INF, db_settings, llm, material) == "You are the INF1010 expert v1."
    assert llm.calls == 1

    m.put("NTNU/TDT4100/b.md", "sha-b", ["NTNU/TDT4100/b.md"])  # outside scope: still cached
    persona(INF, db_settings, llm, material)
    assert llm.calls == 1

    m.put("UiO/INF1010/a.md", "sha-a2", ["UiO/INF1010/a.md"])  # inside scope changed
    assert persona(INF, db_settings, llm, material) == "You are the INF1010 expert v2."
