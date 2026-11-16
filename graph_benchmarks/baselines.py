from __future__ import annotations

import math
import re
from collections import Counter

from graph_memory.store import Store, event_line

STOP = {"the", "a", "an", "in", "on", "to", "of", "from", "go", "it", "is", "and", "was", "when", "task", "batch",
        "seen", "held", "opened", "x", "by", "with", "this", "bring", "put", "times"}


def _toks(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP]


def bm25(lines: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> list[str]:
    docs = [_toks(x) for x in lines]
    n = len(docs)
    if not n:
        return []
    avg = sum(map(len, docs)) / n
    df = Counter(w for d in docs for w in set(d))
    q = _toks(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / max(avg, 1e-9)))
        scores.append(s)
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    return [lines[i] for i in order if scores[i] > 0]


def cap(lines: list[str], limit: int, header: str = "") -> str:
    text = header
    for line in lines:
        add = ("\n" if text else "") + "- " + line
        if len(text) + len(add) > limit:
            break
        text += add
    return text


def _fit(lines: list[str], order: list[int], limit: int) -> list[int]:
    out, used = [], 0
    for i in order:
        add = (1 if out else 0) + 2 + len(lines[i])
        if used + add > limit:
            break
        out.append(i)
        used += add
    return out


def raw_brief(raw: list[str], query: str, budget: int) -> str:
    rank = {ln: i for i, ln in reversed(list(enumerate(raw)))}
    return cap([raw[i] for i in _fit(raw, [rank[x] for x in bm25(raw, query)], budget)], budget)


def flat_brief(s: Store, query: str, budget: int) -> str:
    lines = [event_line(s, e) for e in s.events]
    first = {ln: i for i, ln in reversed(list(enumerate(lines)))}
    return cap([lines[i] for i in _fit(lines, [first[x] for x in bm25(lines, query)], budget)], budget)
