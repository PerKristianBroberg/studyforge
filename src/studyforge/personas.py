import hashlib
from dataclasses import replace
from typing import Callable

import sqlalchemy as sa

from studyforge import retrieval
from studyforge.config import Settings
from studyforge.experts import Expert
from studyforge.retrieval import Scope
from studyforge.store import Manifest, engine

PROMPT = """You are writing the persona for a study assistant that is an expert on: {name}.
File names and excerpts from the student's notes in this scope:

{material}

In 3-4 sentences, written as "You are ...", describe the subject area, the main topics covered
and the level (introductory/advanced). No bullet points."""


def in_scope(path: str, scope: Scope) -> bool:
    parts = path.split("/")
    if scope.uni and parts[0] != scope.uni:
        return False
    return not scope.course or (len(parts) > 2 and parts[1] == scope.course)


def default_material(expert: Expert, s: Settings) -> str:
    # admin docs (letters, forms) stay out: persona material is sent to Anthropic without a user question
    files = [p for p in Manifest(s).all()
             if in_scope(p, expert.scope) and "admin" not in (d.lower() for d in p.split("/")[1:-1])]
    scope = replace(expert.scope, doc_types=("notes", "code", "exams"))
    hits = retrieval.search(f"main topics of {expert.name}", scope, s)
    excerpts = "\n\n".join(h.text for h in hits)[:4000]
    return "Files:\n" + "\n".join(files[:50]) + "\n\nExcerpts:\n" + excerpts


def persona(expert: Expert, s: Settings, llm, material: Callable[[Expert, Settings], str] = default_material) -> str:
    shas = sorted(sha for path, (sha, _) in Manifest(s).all().items() if in_scope(path, expert.scope))
    key = hashlib.sha256("".join(shas).encode()).hexdigest()
    eng = engine(s.db_url)
    with eng.begin() as c:
        c.execute(sa.text("CREATE TABLE IF NOT EXISTS personas (expert_id text PRIMARY KEY, key text, text text)"))
        row = c.execute(sa.text("SELECT text FROM personas WHERE expert_id = :e AND key = :k"),
                        {"e": expert.id, "k": key}).first()
    if row:
        return row[0]
    text = llm.invoke(PROMPT.format(name=expert.name, material=material(expert, s))).text
    with eng.begin() as c:
        c.execute(sa.text(
            "INSERT INTO personas (expert_id, key, text) VALUES (:e, :k, :t) "
            "ON CONFLICT (expert_id) DO UPDATE SET key = :k, text = :t"
        ), {"e": expert.id, "k": key, "t": text})
    return text
