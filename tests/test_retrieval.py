from llama_index.core.postprocessor.types import BaseNodePostprocessor

from studyforge import ingest
from studyforge.retrieval import Hit, Scope, search


class FixedScoreRerank(BaseNodePostprocessor):
    """Stands in for the cross-encoder: keyword hits score 0.9, everything else 0.0."""
    keyword: str

    def _postprocess_nodes(self, nodes, query_bundle=None):
        for n in nodes:
            n.score = 0.9 if self.keyword in n.node.get_content().lower() else 0.0
        return sorted(nodes, key=lambda n: n.score, reverse=True)[:6]


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def corpus(tmp_path, s, embed):
    write(tmp_path, "UiO/MAT1100/limits.md", "Epsilon-delta definition of a limit.")
    write(tmp_path, "NTNU/MAT1100/limits.md", "Limit laws and squeeze theorem.")
    write(tmp_path, "NTNU/TDT4100/java.md", "Java interfaces and limit of inheritance.")
    ingest.run(tmp_path, s, embed)


def test_course_scope_does_not_leak_across_unis(tmp_path, db_settings, mock_embed):
    corpus(tmp_path, db_settings, mock_embed)
    hits = search("limit", Scope(uni="NTNU", course="MAT1100"), db_settings, mock_embed,
                  FixedScoreRerank(keyword="limit"))
    assert hits and {h.path for h in hits} == {"NTNU/MAT1100/limits.md"}


def test_uni_scope_and_citation(tmp_path, db_settings, mock_embed):
    corpus(tmp_path, db_settings, mock_embed)
    hits = search("limit", Scope(uni="NTNU"), db_settings, mock_embed, FixedScoreRerank(keyword="limit"))
    assert {h.uni for h in hits} == {"NTNU"}
    assert hits[0].cite() in {"NTNU/MAT1100/limits.md", "NTNU/TDT4100/java.md"}
    assert Hit("t", 1.0, "UiO/A/x.pdf", 3, "UiO", "A").cite() == "UiO/A/x.pdf p.3"


def test_low_scores_mean_not_in_notes(tmp_path, db_settings, mock_embed):
    corpus(tmp_path, db_settings, mock_embed)
    assert search("photosynthesis", Scope(), db_settings, mock_embed,
                  FixedScoreRerank(keyword="photosynthesis")) == []


def test_doc_type_filter(tmp_path, db_settings, mock_embed):
    write(tmp_path, "UiO/A/exams/2024.md", "Exam: prove the limit exists.")
    write(tmp_path, "UiO/A/notes.md", "Notes on the limit.")
    ingest.run(tmp_path, db_settings, mock_embed)
    hits = search("limit", Scope(doc_types=("exams",)), db_settings, mock_embed,
                  FixedScoreRerank(keyword="limit"))
    assert [h.path for h in hits] == ["UiO/A/exams/2024.md"]
