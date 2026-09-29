from studyforge.experts import GENERAL, Expert, discover
from studyforge.retrieval import Scope


def touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")


def test_discover_uni_and_course_experts(tmp_path):
    touch(tmp_path, "UiO/INF1010/a.md")
    touch(tmp_path, "UiO/admin/rules.md")
    touch(tmp_path, "NTNU/TDT4100/b.md")
    experts = discover(tmp_path)
    assert [e.id for e in experts] == ["general", "NTNU", "UiO", "NTNU/TDT4100", "UiO/INF1010"]
    assert experts[0] is GENERAL
    uio_inf = experts[-1]
    assert uio_inf.kind == "course" and uio_inf.scope == Scope(uni="UiO", course="INF1010")
    assert experts[2].scope == Scope(uni="UiO")


def test_node_names_are_langgraph_safe():
    e = Expert("UiO/INF 1010", "x", "course", Scope(uni="UiO", course="INF 1010"))
    assert e.node == "expert_uio_inf_1010"
    assert GENERAL.node == "expert_general"


def test_missing_notes_dir_gives_only_general(tmp_path):
    assert discover(tmp_path / "nope") == [GENERAL]
