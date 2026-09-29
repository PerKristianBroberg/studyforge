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


def test_default_material_excludes_admin_docs(db_settings, monkeypatch):
    from studyforge.personas import default_material

    m = Manifest(db_settings)
    m.put("UiO/INF1010/heaps.md", "s1", ["UiO/INF1010/heaps.md"])
    m.put("UiO/admin/enrollment-letter.pdf", "s2", ["UiO/admin/enrollment-letter.pdf"])
    scopes = []
    monkeypatch.setattr("studyforge.retrieval.search", lambda q, scope, s: scopes.append(scope) or [])
    text = default_material(Expert("UiO", "UiO expert", "uni", Scope(uni="UiO")), db_settings)
    assert "heaps.md" in text and "enrollment-letter" not in text
    assert "admin" not in scopes[0].doc_types  # admin docs (letters, forms) never go to Anthropic via personas


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
