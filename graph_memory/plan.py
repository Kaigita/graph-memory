from __future__ import annotations

import re
from typing import Iterable

from graph_memory.store import Store, norm, place_text, the, words

TAKE = re.compile(r"^(take|get|pick up|grab)\s+(.+)$")
DROP = re.compile(r"^(drop|put down|discard)\s+(.+)$")


class Planner:

    def __init__(self, s: Store, here: str, held: set[str], max_depth: int = 6):
        self.s, self.cur, self.have = s, here, set(held)
        self.max_depth = max_depth
        self.done: set[tuple[str, str]] = set()
        self.done_any: set[str] = set()
        self.lines: list[str] = []
        self.actions: list[str] = []

    def _need_for_path(self, path: list[tuple[str, str, str]]) -> tuple[list[str], list[str]]:
        have, anywhere = [], []
        for a, action, b in path:
            key = f"{a}|{action}"
            have += self.s.link_have(key)
            for g in self.s.link_do(key):
                have += self.s.act_have(g)
            have += self.s.enter_have(b)
            for g in self.s.enter_do(b):
                have += self.s.act_have(g)
                anywhere.append(g)
        return have, anywhere

    def _way(self, path: list[tuple[str, str, str]]) -> str:
        parts = []
        for a, action, b in path:
            pre = []
            for g in self.s.link_do(f"{a}|{action}"):
                if (a, g) not in self.done:
                    self.done.add((a, g))
                    pre.append(g)
                    self.actions.append(g)
            self.actions.append(action)
            parts.append(f"{'; '.join(pre) + '; ' if pre else ''}{action} (to {the(self.s.name(b))})")
        return ", ".join(parts)

    def obtain(self, x: str, depth: int = 0) -> None:
        s = self.s
        if x in self.have or depth > self.max_depth:
            return
        rec = s.where(x)
        self.have.add(x)
        if rec is None:
            self.lines.append(f"the {s.name(x)}: no record of where it is.")
            return
        dst = rec["place"]
        path = s.route(self.cur, dst)
        have, anywhere = self._need_for_path(path or [])
        have += s.get_have(x)
        for g in s.get_do(x):
            have += s.act_have(g)
        for y in dict.fromkeys(have):
            if y != x:
                self.obtain(y, depth + 1)
        for g in dict.fromkeys(anywhere):
            if x in s.act_have(g):
                continue
            if g not in self.done_any:
                self.done_any.add(g)
                self.actions.append(g)
                self.lines.append(f"do '{g}' (needed before entering a place on the way).")
        path = s.route(self.cur, dst)
        if path is None:
            way = f"no known way from {the(s.name(self.cur))} to {the(s.name(dst))}"
        elif not path:
            way = f"it is here in {the(s.name(dst))}"
        else:
            way = f"from {the(s.name(self.cur))}: " + self._way(path)
        pre = [g for g in s.get_do(x) if (dst, g) not in self.done]
        for g in pre:
            self.done.add((dst, g))
            self.actions.append(g)
        how = s.gets[x]["action"] if x in s.gets else f"take {s.name(x).split()[-1]}"
        self.actions.append(how)
        tail = f"; there: {'; '.join(pre)}; then '{how}'" if pre else f"; then '{how}'"
        self.lines.append(f"get the {s.name(x)} ({place_text(s, rec)}): {way}{tail}.")
        if path is not None:
            self.cur = dst


def plan(s: Store, target: str, here: str, held: set[str]) -> tuple[list[str], list[str]]:
    p = Planner(s, here, held)
    p.obtain(target)
    return [f"{i + 1}. {x}" for i, x in enumerate(p.lines)], p.actions


def route_text(s: Store, here: str, place: str) -> list[str]:
    path = s.route(here, place)
    if path is None:
        return [f"no known way from {the(s.name(here))} to {the(s.name(place))}."]
    if not path:
        return [f"you are in {the(s.name(place))}."]
    p = Planner(s, here, set())
    have, anywhere = p._need_for_path(path)
    lines = [f"to enter on the way, first: {'; '.join(dict.fromkeys(anywhere))}."] if anywhere else []
    if have:
        lines.append(f"on the way you need to hold: the {', the '.join(map(s.name, dict.fromkeys(have)))}.")
    return lines + [f"{the(s.name(place))}: from {the(s.name(here))}: {p._way(path)}."]


def plan_brief(s: Store, goal: str) -> list[str]:
    hit = s.find(goal, ("thing",))
    if hit is None:
        places = sorted({s.name(e) for e, k in s.kinds.items() if k == "place"})
        return [f"no record of the thing this task asks for. Recorded places: {', '.join(places)}."]
    return plan(s, hit[1], s.start or "", set(s.start_have))[0]


def plan_answer(s: Store, query: str, here: str, held: set[str]) -> list[str]:
    hit = s.find(query)
    if hit is None:
        return []
    kind, x = hit
    if kind == "thing":
        if x in held:
            return [f"you are holding the {s.name(x)}."]
        return plan(s, x, here, held)[0]
    return route_text(s, here, x)


def track(s: Store, turns: Iterable) -> tuple[str, set[str]]:
    places: dict[str, list[str]] = {}
    for e, k in s.kinds.items():
        if k == "place":
            places.setdefault(norm(s.names[e]), []).append(e)
    here, held = s.start or "", set(s.start_have)
    by_name: dict[str, list[str]] = {}
    for x in s.things():
        by_name.setdefault(norm(s.name(x)), []).append(x)

    def named(text: str) -> list[str]:
        n = norm(text)
        if n in by_name:
            return by_name[n]
        ws = set(words(n))
        return [x for x in s.things() if ws and ws <= set(words(s.name(x))) and words(s.name(x))[-1] in ws]

    for st in turns:
        for line in [x for x in st.observation.splitlines() if x.strip()][:3]:
            hits = places.get(norm(line), [])
            if len(hits) == 1:
                here = hits[0]
                break
        a = norm(st.action)
        hit = [x for x, g in s.gets.items() if norm(g["action"]) == a]
        m = TAKE.match(a)
        if not hit and m:
            hit = named(m.group(2))
        if len(hit) == 1:
            held.add(hit[0])
        d = DROP.match(a)
        if d:
            held -= set(named(d.group(2)))
    return here, held
