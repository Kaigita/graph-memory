from __future__ import annotations

import json
import math
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any

from graph_benchmarks import house as H

BRIEF_CHARS = 1200
TOOL_CHARS = 800
HEADER = "From earlier tasks in this house:"
POLICY = ("Your memory of this house: the first message includes what earlier tasks in this house recorded. Call "
          "`memory` with the name of an object, room, container or key to ask for more; it does not use an "
          "environment step. Things may have been moved since they were recorded.")
MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": ("Ask your memory of this house: give the name of an object, room, container or key; returns "
                        "what earlier tasks in this house recorded about it. Does not use an environment step."),
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string",
                                                "description": "An object, room, container or key name."}},
                       "required": ["query"]},
    },
}
GOAL_BRING = re.compile(r"^Bring the ([a-z ]+) to the ([a-z ]+)\.$")
GOAL_PUT = re.compile(r"^Put the ([a-z ]+) (?:in|on) the ([a-z ]+) in the ([a-z ]+)\.$")


def door_id(a: str, b: str) -> str:
    return "door:" + "|".join(sorted((a, b)))


def parse_goal(goal: str) -> dict:
    m = GOAL_BRING.match(goal.strip())
    if m:
        return {"kind": "bring", "target": m.group(1), "dest_room": m.group(2), "dest_container": None}
    m = GOAL_PUT.match(goal.strip())
    if m:
        return {"kind": "put", "target": m.group(1), "dest_container": m.group(2), "dest_room": m.group(3)}
    return {"kind": "?", "target": None, "dest_room": None, "dest_container": None}


class FactStore:
    def __init__(self) -> None:
        self.exits: dict[str, dict[str, str]] = {}
        self.visited: set[str] = set()
        self.locked: set[str] = set()
        self.containers: dict[str, dict] = {}
        self.snaps: dict[str, list[list]] = {}
        self.sights: dict[str, list[list]] = {}
        self.opens: dict[str, str] = {}
        self.misfit: set[tuple[str, str]] = set()
        self.episodes: list[str] = []

    def _sight(self, item: str, place: str, batch: int, eid: str) -> None:
        rows = self.sights.setdefault(item, [])
        if not any(r[1] == eid and r[2] == place for r in rows):
            rows.append([batch, eid, place])

    def _contents(self, c: str, items: list[str], batch: int, eid: str, touched: set[str]) -> None:
        seen = sorted(i for i in items if i not in touched)
        rows = [r for r in self.snaps.get(c, []) if not (r[0] == batch and r[1] == eid)]
        rows.append([batch, eid, seen, sorted(touched)])
        self.snaps[c] = rows
        for i in seen:
            self._sight(i, c, batch, eid)

    def _room(self, info: dict, batch: int, eid: str, touched: set[str]) -> str:
        room = info["room"]
        self.visited.add(room)
        ex = self.exits.setdefault(room, {})
        for d, (r2, lk) in info["exits"].items():
            ex[d] = r2
            self.exits.setdefault(r2, {})[H.OPP[d]] = room
            if lk:
                self.locked.add(door_id(room, r2))
        for c, (state, items) in info["containers"].items():
            kind = state if state in ("surface", "open") else "closable"
            self.containers[c] = {"room": room, "kind": kind}
            if state == "locked":
                self.locked.add(c)
            if items is not None:
                self._contents(c, items, batch, eid, touched)
        for i in info["floor"]:
            if i not in touched:
                self._sight(i, f"floor:{room}", batch, eid)
        return room

    def add_episode(self, ep: dict, batch: int, eid: str) -> None:
        if eid in self.episodes:
            return
        self.episodes.append(eid)
        touched: set[str] = set()
        room = None
        info = H.parse_room(ep.get("initial_obs") or "")
        if info:
            room = self._room(info, batch, eid, touched)
        for step in ep["steps"]:
            text = step["observation"] or ""
            m = H.TAKE_RE.search(text)
            if m:
                touched.add(m.group(1))
                continue
            info = H.parse_room(text)
            if info:
                room = self._room(info, batch, eid, touched)
                continue
            if room is None:
                continue
            m = H.OPEN_RE.search(text)
            if m:
                c = m.group(1)
                self.containers.setdefault(c, {"room": room, "kind": "closable"})
                self._contents(c, H.parse_items(m.group(2)), batch, eid, touched)
                continue
            m = H.UNLOCK_OK_RE.search(text)
            if m:
                lock = door_id(room, m.group(2)) if m.group(1) else m.group(2)
                self.opens[lock] = m.group(3)
                self.locked.add(lock)
                continue
            m = H.UNLOCK_BAD_RE.search(text)
            if m:
                lock = door_id(room, m.group(3)) if m.group(2) else m.group(3)
                self.misfit.add((m.group(1), lock))
                self.locked.add(lock)

    def place_room(self, place: str) -> str | None:
        if place.startswith("floor:"):
            return place[6:]
        return self.containers.get(place, {}).get("room")

    def belief(self, item: str) -> dict | None:
        rows = self.sights.get(item)
        if not rows:
            return None
        b, _, place = max(rows, key=lambda r: r[0])
        gone = None
        if not place.startswith("floor:"):
            for sb, _, contents, touched in self.snaps.get(place, []):
                if sb > b and item not in contents and item not in touched:
                    gone = max(gone or 0, sb)
        counts = Counter(r[2] for r in rows)
        last: dict[str, int] = {}
        for r in rows:
            last[r[2]] = max(last.get(r[2], 0), r[0])
        others = sorted((p for p in counts if p != place), key=lambda p: (-last[p], -counts[p], p))
        return {"place": place, "batch": b, "gone": gone, "count": counts[place],
                "others": [(p, last[p], counts[p]) for p in others]}

    def route(self, src: str, dst: str) -> list[tuple[str, str, str]] | None:
        if src == dst:
            return []
        best: dict[str, tuple[int, int]] = {src: (0, 0)}
        prev: dict[str, tuple[str, str]] = {}
        q: deque = deque([src])
        while q:
            r = q.popleft()
            for d in ("north", "south", "east", "west"):
                r2 = self.exits.get(r, {}).get(d)
                if r2 is None:
                    continue
                cost = (best[r][0] + (door_id(r, r2) in self.locked), best[r][1] + 1)
                if r2 not in best or cost < best[r2]:
                    best[r2], prev[r2] = cost, (r, d)
                    q.append(r2)
        if dst not in best:
            return None
        path, r = [], dst
        while r != src:
            p, d = prev[r]
            path.append((p, d, r))
            r = p
        return path[::-1]

    def lock_name(self, lock: str) -> str:
        if lock.startswith("door:"):
            a, b = lock[5:].split("|")
            return f"the door between the {a} and the {b}"
        room = self.containers.get(lock, {}).get("room")
        return f"the {lock}" + (f" in the {room}" if room else "")

    def to_json(self) -> dict:
        return {"exits": self.exits, "visited": sorted(self.visited), "locked": sorted(self.locked),
                "containers": self.containers, "snaps": self.snaps, "sights": self.sights, "opens": self.opens,
                "misfit": sorted(self.misfit), "episodes": self.episodes}

    @classmethod
    def from_json(cls, data: dict) -> "FactStore":
        s = cls()
        s.exits, s.visited, s.locked = data["exits"], set(data["visited"]), set(data["locked"])
        s.containers, s.snaps, s.sights = data["containers"], data["snaps"], data["sights"]
        s.opens, s.misfit, s.episodes = data["opens"], {tuple(x) for x in data["misfit"]}, data["episodes"]
        return s

    def stats(self) -> dict:
        return {"rooms": len(self.visited), "containers": len(self.containers), "items": len(self.sights),
                "opened": len(self.snaps), "keys_known": len(self.opens), "episodes": len(self.episodes)}


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


def flat_lines(s: FactStore) -> list[str]:
    out = []
    for item in sorted(s.sights):
        by_place: dict[str, list[int]] = {}
        for b, _, place in s.sights[item]:
            by_place.setdefault(place, []).append(b)
        for place, bs in sorted(by_place.items()):
            if place.startswith("floor:"):
                where = f"on the floor of the {place[6:]}"
            else:
                prep = "on" if s.containers.get(place, {}).get("kind") == "surface" else "in"
                where = f"{prep} the {place} in the {s.place_room(place)}"
            times = "once" if len(bs) == 1 else f"{len(bs)} times"
            out.append(f"{item}: seen {where} (last in task batch {max(bs)}; seen {times})")
    for c in sorted(s.snaps):
        b, _, contents, _ = max(s.snaps[c], key=lambda r: r[0])
        out.append(f"the {c} in the {s.place_room(c)}, looked into in task batch {b}: held "
                   f"{', '.join(contents) if contents else 'nothing'}")
    for c in sorted(s.containers):
        if c not in s.snaps:
            out.append(f"the {c} is in the {s.containers[c]['room']}" + (" (locked)" if c in s.locked else
                                                                       "; its contents were never seen"))
        elif c in s.locked:
            out.append(f"the {c} in the {s.containers[c]['room']} is locked")
    for r in sorted(s.exits):
        for d in ("north", "south", "east", "west"):
            r2 = s.exits[r].get(d)
            if r2:
                out.append(f"exit: from the {r} go {d} to the {r2}" +
                           (" (the door was locked)" if door_id(r, r2) in s.locked else ""))
    for lock, k in sorted(s.opens.items()):
        out.append(f"the {k} opens {s.lock_name(lock)}")
    for k, lock in sorted(s.misfit):
        out.append(f"the {k} does not fit {s.lock_name(lock)}")
    return out


def cap(lines: list[str], limit: int, header: str = "") -> str:
    text = header
    for line in lines:
        add = ("\n" if text else "") + "- " + line
        if len(text) + len(add) > limit:
            break
        text += add
    return text


def _where(s: FactStore, place: str) -> str:
    room = s.place_room(place)
    if place.startswith("floor:"):
        return f"on the floor of the {room}"
    prep = "on" if s.containers.get(place, {}).get("kind") == "surface" else "in"
    return f"{prep} the {place} in the {room}"


def _way(s: FactStore, src: str, dst: str) -> str:
    path = s.route(src, dst)
    return ", ".join(f"go {d} (to the {r2})" for _, d, r2 in path) if path else ""


def _way_text(s: FactStore, src: str, dst: str) -> str:
    if src == dst:
        return f"you are already in the {dst}"
    steps = _way(s, src, dst)
    return f"way from the {src}: {steps}" if steps else f"no known way from the {src} to the {dst} yet"


def _status(s: FactStore, item: str) -> list[str]:
    bel = s.belief(item)
    if bel is None:
        return [f"{item}: not seen in this house yet."]
    out = [f"{item}: last seen {_where(s, bel['place'])} (task batch {bel['batch']})."]
    if bel["gone"]:
        out.append(f"But in task batch {bel['gone']} the {bel['place']} was looked into and the {item} was not there.")
    if bel["others"]:
        out.append(f"{item} was also seen " + "; ".join(f"{_where(s, p)} (task batch {b})"
                                                        for p, b, _ in bel["others"][:3]) + ".")
    return out


def _needs(s: FactStore, place: str, src: str = H.START) -> list[str]:
    room = s.place_room(place)
    path = s.route(src, room) if room else None
    locks = [door_id(a, b) for a, _, b in (path or []) if door_id(a, b) in s.locked]
    if place in s.locked:
        locks.append(place)
    return locks


def _key_order(s: FactStore, place: str, seen: set[str], order: list, unknown: list) -> None:
    for lock in _needs(s, place):
        k = s.opens.get(lock)
        if k is None:
            if lock not in unknown:
                unknown.append(lock)
            continue
        if k in seen:
            continue
        seen.add(k)
        bel = s.belief(k)
        if bel is not None:
            _key_order(s, bel["place"], seen, order, unknown)
        order.append((k, lock, bel))


def _misfits(s: FactStore, lock: str) -> str:
    bad = sorted(k for k, ll in s.misfit if ll == lock)
    return f" (these did not fit: {', '.join(bad)})" if bad else ""


def _lock_line(s: FactStore, lock: str) -> str:
    k = s.opens.get(lock)
    return (f"{s.lock_name(lock).capitalize()} is locked; " +
            (f"the {k} opens it." if k else f"which key opens it is not known yet{_misfits(s, lock)}."))


def _itinerary(s: FactStore, item: str, cur: str) -> tuple[list[str], str]:
    out = _status(s, item)
    bel = s.belief(item)
    if bel is None:
        unseen = [c for c in s.containers if c not in s.snaps and s.containers[c]["kind"] == "closable"]
        routes = {c: s.route(cur, s.containers[c]["room"]) for c in unseen}
        unseen.sort(key=lambda c: (99 if routes[c] is None else len(routes[c]), c))
        if unseen:
            out.append("Containers never looked into: " + ", ".join(
                f"the {c} ({s.containers[c]['room']}" + (", locked" if c in s.locked else "") +
                (", behind a locked door" if [x for x in _needs(s, c, cur) if x != c] else "") + ")"
                for c in unseen[:6]) + ".")
        rooms = sorted(r for r in s.exits if r not in s.visited)
        if rooms:
            out.append("Rooms never visited: " + ", ".join(f"the {r}" for r in rooms) + ".")
        return out, cur
    order: list = []
    unknown: list = []
    _key_order(s, bel["place"], {item}, order, unknown)
    for k, lock, kb in order:
        if kb is None:
            out.append(f"The {k} opens {s.lock_name(lock)}, but where it is is not known.")
            continue
        room = s.place_room(kb["place"])
        out.append(f"Get the {k} first (it opens {s.lock_name(lock)}): last seen {_where(s, kb['place'])} "
                   f"(task batch {kb['batch']}); {_way_text(s, cur, room)}.")
        cur = room
    for lock in unknown:
        out.append(_lock_line(s, lock))
    room = s.place_room(bel["place"])
    lock_note = ""
    if bel["place"] in s.locked and s.opens.get(bel["place"]):
        lock_note = f" Unlock the {bel['place']} with the {s.opens[bel['place']]}."
    out.append(f"{'Then get' if order else 'Get'} the {item}: {_way_text(s, cur, room)}.{lock_note}")
    return out, room


def graph_brief(s: FactStore, goal: str) -> list[str]:
    g = parse_goal(goal)
    if not g["target"]:
        return []
    out, cur = _itinerary(s, g["target"], H.START)
    if s.belief(g["target"]) is None or not g["dest_room"]:
        return out
    order: list = []
    _key_order(s, s.belief(g["target"])["place"], {g["target"]}, order, [])
    fetched = {k for k, _, kb in order if kb is not None}
    c = g["dest_container"]
    if g["kind"] == "bring":
        what = f"Then bring it to the {g['dest_room']}"
    else:
        prep = "on" if s.containers.get(c, {}).get("kind") == "surface" else "in"
        what = f"Then put it {prep} the {c} in the {g['dest_room']}"
    out.append(f"{what}: {_way_text(s, cur, g['dest_room'])}.")
    for lock in _needs(s, f"floor:{g['dest_room']}", cur):
        if s.opens.get(lock) not in fetched:
            out.append(_lock_line(s, lock))
    if c and s.containers.get(c, {}).get("kind") == "closable":
        out.append(f"The {c} is a closed container: open it before putting things in it.")
    return out


def graph_answer(s: FactStore, query: str, here: str) -> list[str]:
    q = " ".join(query.lower().replace(".", " ").split())
    names = ([(n, "item") for n in s.sights] + [(n, "item") for n in s.opens.values()] +
             [(n, "container") for n in s.containers] + [(n, "room") for n in s.exits])
    hits = sorted({n for n in names if n[0] in q}, key=lambda n: (-len(n[0]), n))
    if not hits:
        return ["Nothing recorded about that."]
    name, kind = hits[0]
    if kind == "item":
        out, _ = _itinerary(s, name, here)
        opened = [s.lock_name(lock) for lock, k in sorted(s.opens.items()) if k == name]
        if opened:
            out.append(f"The {name} opens {', '.join(opened)}.")
        return out
    if kind == "room":
        out = [f"The {name}: {_way_text(s, here, name)}."]
        out += [_lock_line(s, lock) for lock in _needs(s, f"floor:{name}", here)]
        return out
    room = s.containers[name]["room"]
    out = [f"The {name} is in the {room}."]
    if name in s.snaps:
        b, _, contents, _ = max(s.snaps[name], key=lambda r: r[0])
        out.append(f"When looked into in task batch {b} it held: {', '.join(contents) if contents else 'nothing'}.")
    else:
        out.append("Its contents were never seen.")
    out.append(f"The {room}: {_way_text(s, here, room)}.")
    out += [_lock_line(s, lock) for lock in _needs(s, name, here)]
    return out


class HouseMemory:
    tool_specs = [MEMORY_TOOL]
    policy = POLICY

    def __init__(self, mode: str, brief_chars: int = BRIEF_CHARS):
        assert mode in ("flat", "graph")
        self.mode = mode
        self.brief_chars = brief_chars
        self.stores: dict[str, FactStore] = {}

    def store(self, seed: int | str) -> FactStore:
        return self.stores.get(str(seed)) or FactStore()

    def brief(self, seed: int | str, goal: str) -> str:
        s = self.store(seed)
        lines = bm25(flat_lines(s), goal) if self.mode == "flat" else graph_brief(s, goal)
        text = cap(lines, self.brief_chars, HEADER)
        return "" if text == HEADER else text

    def answer(self, seed: int | str, query: str, here: str) -> str:
        s = self.store(seed)
        lines = bm25(flat_lines(s), query) if self.mode == "flat" else graph_answer(s, query, here)
        return cap(lines, TOOL_CHARS) or "Nothing recorded about that."

    def add(self, episodes: list[dict], tag: str) -> None:
        for ep in episodes:
            if ep["end_reason"] == "error":
                continue
            tid = ep["task_id"]
            self.stores.setdefault(str(H.task_seed(tid)), FactStore()).add_episode(ep, H.task_batch(tid),
                                                                                  f"{tid}#{tag}")

    def stats(self) -> dict[str, Any]:
        return {seed: s.stats() for seed, s in sorted(self.stores.items())}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"mode": self.mode, "stores": {k: s.to_json() for k, s in self.stores.items()}}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.stores = {k: FactStore.from_json(v) for k, v in data["stores"].items()}
