import re
from dataclasses import dataclass
from pathlib import Path

from studyforge.paths import scan
from studyforge.retrieval import Scope


@dataclass(frozen=True)
class Expert:
    id: str
    name: str
    kind: str  # "general" | "uni" | "course"
    scope: Scope

    @property
    def node(self) -> str:
        # LangGraph reserves ':' and '|' in node names
        return "expert_" + re.sub(r"\W+", "_", self.id).strip("_").lower()


GENERAL = Expert("general", "General (all notes)", "general", Scope())


def discover(root: Path) -> list[Expert]:
    unis, courses = set(), set()
    for _, meta in scan(root) if root.is_dir() else []:
        unis.add(meta.uni)
        if meta.course:
            courses.add((meta.uni, meta.course))
    return [
        GENERAL,
        *(Expert(u, f"{u} expert", "uni", Scope(uni=u)) for u in sorted(unis)),
        *(Expert(f"{u}/{c}", f"{c} expert ({u})", "course", Scope(uni=u, course=c)) for u, c in sorted(courses)),
    ]
