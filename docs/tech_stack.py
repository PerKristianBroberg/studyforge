"""Tech-stack diagram for studyforge, drawn with mingrammer/diagrams.

Logos are pulled from the <symbol>s embedded in docs/architecture.svg, so both
images share one source of truth. Needs Graphviz (winget install Graphviz.Graphviz):

    pip install diagrams pymupdf && python docs/tech_stack.py
"""

import base64
import re
import tempfile
from pathlib import Path

import diagrams
import pymupdf
from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import User
from diagrams.onprem.container import Docker
from diagrams.onprem.database import Postgresql
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

DOCS = Path(__file__).parent
ICONS = Path(tempfile.mkdtemp())
SCALE = 0.6  # logo size inside its node; the transparent margin keeps label space intact


def save_icon(name, box, body):
    """Render an SVG fragment, centered and shrunk by SCALE, to ICONS/<name>.png."""
    m = (1 - SCALE) / 2 * 100
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'viewBox="0 0 100 100"><svg x="{m}" y="{m}" width="{SCALE * 100}" height="{SCALE * 100}" '
           f'viewBox="{box}">{body}</svg></svg>')
    page = pymupdf.open(stream=doc.encode(), filetype="svg")[0]
    page.get_pixmap(matrix=pymupdf.Matrix(2.56, 2.56), alpha=True).save(ICONS / f"{name}.png")


def png_body(data):
    return f'<image width="1" height="1" xlink:href="data:image/png;base64,{data}"/>'


svg = (DOCS / "architecture.svg").read_text(encoding="utf-8")
for name, box, body in re.findall(r'<symbol id="i-([^"]+)" viewBox="([^"]+)">(.*?)</symbol>', svg, re.S):
    if png := re.search(r'href="data:image/png;base64,([^"]+)"', body):
        box, body = "0 0 1 1", png_body(png.group(1))
    save_icon(name, box, body)

for cls in (User, Python, Docker, Postgresql, Fastapi, GithubActions):
    png = Path(diagrams.__file__).parent.parent / cls._icon_dir / cls._icon
    save_icon(cls.__name__.lower(), "0 0 1 1", png_body(base64.b64encode(png.read_bytes()).decode()))


def icon(label, name):
    return Custom(label, str(ICONS / f"{name}.png"))


graph = {"fontsize": "22", "pad": "0.5", "nodesep": "0.7", "ranksep": "1.0"}
cloud = Edge(color="darkorange", style="dashed", label="question + ≤6 passages")
trace = Edge(color="gray50", style="dotted")

with Diagram("studyforge — tech stack", filename=str(DOCS / "tech_stack"), show=False,
             direction="LR", graph_attr=graph):
    student = icon("Student", "user")

    with Cluster("Interfaces"):
        ui = icon("Chainlit\nchat UI", "chainlit")
        api = icon("FastAPI\nlibrary · calendar", "fastapi")

    with Cluster("Reasoning"):
        graph_ = icon("LangGraph\nsupervisor + experts", "langgraph")
        tools = icon("LangChain\ntools", "langchain")

    with Cluster("Anthropic API"):
        claude = icon("Claude\nSonnet 5.5 · Haiku 4.5", "claude")

    with Cluster("Retrieval"):
        rag = icon("LlamaIndex\nhybrid search", "llamaindex")
        rerank = icon("bge-reranker-v2-m3", "huggingface")

    with Cluster("Storage (Docker)"):
        db = icon("Postgres 17\n+ pgvector", "postgresql")

    with Cluster("Ingestion"):
        docling = icon("Docling\nPDF · PPTX · DOCX", "docling")
        embed = icon("bge-m3 embeddings\nPyTorch · CPU", "pytorch")

    with Cluster("Observability (Docker)"):
        langfuse = icon("Langfuse", "langfuse")

    with Cluster("Tooling & CI"):
        # invisible chain lays them out in a row instead of a tall column
        (icon("Python", "python") >> Edge(style="invis") >> icon("uv", "uv") >> Edge(style="invis")
         >> icon("Docker", "docker") >> Edge(style="invis") >> icon("pytest", "pytest")
         >> Edge(style="invis") >> icon("GitHub Actions", "githubactions"))

    student >> [ui, api]
    ui >> graph_ >> tools >> rag >> rerank
    graph_ >> cloud >> claude
    api >> docling >> embed >> db
    rag >> db
    graph_ >> Edge(style="dashed", label="checkpoints") >> db
    graph_ >> trace >> langfuse
