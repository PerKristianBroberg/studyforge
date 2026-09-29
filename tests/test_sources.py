from studyforge.sources import cited, passages, split_label


def test_cited_handles_pages_and_multiple():
    answer = ("R² is explained variance [uni-b/STAT200/regression.md]. See also "
              "[UiO/A/slides.pdf p.3; UiO/A/notes.md] and again [uni-b/STAT200/regression.md].")
    assert cited(answer) == ["uni-b/STAT200/regression.md", "UiO/A/slides.pdf p.3", "UiO/A/notes.md"]


def test_cited_ignores_non_citations():
    assert cited("Use [1] or [this link] or a list [a, b].") == []


def test_split_label():
    assert split_label("UiO/A/slides.pdf p.3") == ("UiO/A/slides.pdf", 3)
    assert split_label("UiO/A/notes.md") == ("UiO/A/notes.md", None)


def test_passages_parse_tool_output():
    tool = "[1] UiO/A/slides.pdf p.3\nStacks are LIFO.\nMore text.\n\n[2] UiO/A/notes.md\nQueues are FIFO."
    assert passages([tool, "NOT_IN_NOTES: nothing"]) == {
        "UiO/A/slides.pdf p.3": "Stacks are LIFO.\nMore text.",
        "UiO/A/notes.md": "Queues are FIFO.",
    }
