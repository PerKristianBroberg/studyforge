import re

LABEL = re.compile(r"[^\[\];]+?\.\w{1,5}(?: p\.\d+)?")  # "<path>.<ext>" optionally " p.<n>"
BRACKET = re.compile(r"\[([^\[\]]+)\]")
ENTRY = re.compile(r"^\[\d+\] (.+)$", re.M)


def cited(answer: str) -> list[str]:
    labels = []
    for group in BRACKET.findall(answer):
        for part in group.split(";"):
            part = part.strip()
            if "/" in part and LABEL.fullmatch(part) and part not in labels:
                labels.append(part)
    return labels


def split_label(label: str) -> tuple[str, int | None]:
    path, _, page = label.partition(" p.")
    return path, int(page) if page else None


def passages(tool_texts: list[str]) -> dict[str, str]:
    found = {}
    for text in tool_texts:
        heads = list(ENTRY.finditer(text))
        for head, nxt in zip(heads, heads[1:] + [None]):
            body = text[head.end():nxt.start() if nxt else len(text)]
            found[head.group(1).strip()] = body.strip()
    return found
