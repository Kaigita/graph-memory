from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

ARTICLES = re.compile(r"^(the|a|an|some)\s+")
WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "get", "take", "find", "where", "is", "to", "in", "on", "and", "task", "done",
        "as", "soon", "you", "are", "holding", "it", "be", "this", "that", "with", "at", "from", "for"}


def norm(name: str) -> str:
    return ARTICLES.sub("", " ".join(str(name).lower().split())).strip(" .,'\"")


def words(text: str) -> list[str]:
    return WORD.findall(str(text).lower())


def the(name: str) -> str:
    return name if name.lower().startswith(("the ", "a ", "an ")) else f"the {name}"


def _inter(obs: list[list[str]]) -> list[str]:
    if not obs:
        return []
    common = set(obs[0]).intersection(*map(set, obs[1:]))
    return [a for a in obs[0] if a in common]


class Store:
    def __init__(self) -> None:
        self.episodes: list[str] = []
        self.names: dict[str, str] = {}
        self.kinds: dict[str, str] = {}
        self.start: str | None = None
        self.start_have: list[str] = []
        self.links: dict[str, dict] = {}
        self.gets: dict[str, dict] = {}
        self.ats: dict[str, dict] = {}
        self.acts: dict[str, dict] = {}
        self.enters: dict[str, dict] = {}
        self.events: list[dict] = []
        self.aliases: dict[str, str] = {}
        self.rules: list[dict] = []
        self.seq = 0
        self.slot_names: dict[str, str] = {}
        self.slot_alias: dict[str, str] = {}

    def entity(self, eid: str, name: str, kind: str) -> str:
        if eid:
            self.names.setdefault(eid, name.strip() or eid)
            self.kinds.setdefault(eid, kind)
        return eid

    def add_link(self, a: str, action: str, b: str, have: list[str], do: list[str], src: str) -> None:
        m = self.links.setdefault(f"{a}|{action}", {"to": b, "have": [], "do": [], "src": []})
        m["to"] = b
        m["have"].append(list(dict.fromkeys(have)))
        m["do"].append(list(dict.fromkeys(do)))
        m["src"].append(src)

    def add_get(self, x: str, action: str, place: str, holder: str, have: list[str], do: list[str],
                src: str) -> None:
        g = self.gets.setdefault(x, {"have": [], "do": [], "src": []})
        g.update({"place": place, "holder": holder, "action": action})
        g["have"].append(list(dict.fromkeys(have)))
        g["do"].append(list(dict.fromkeys(do)))
        g["src"].append(src)

    def add_at(self, x: str, place: str, holder: str, src: str) -> None:
        self.ats[x] = {"place": place, "holder": holder, "src": src}

    def add_act(self, action: str, have: list[str], src: str) -> None:
        r = self.acts.setdefault(action, {"have": [], "src": []})
        r["have"].append(list(dict.fromkeys(have)))
        r["src"].append(src)

    def add_enter(self, place: str, have: list[str], do: list[str], src: str) -> None:
        r = self.enters.setdefault(place, {"have": [], "do": [], "src": []})
        r["have"].append(list(dict.fromkeys(have)))
        r["do"].append(list(dict.fromkeys(do)))
        r["src"].append(src)

    def add_event(self, subj: str, rel: str, obj: str, time: str, src: str, slot: str | None = None,
                  status: str = "set") -> None:
        e = {"s": subj, "r": rel, "o": obj, "time": time, "src": src}
        if slot is not None:
            self.seq += 1
            e.update({"slot": slot, "status": status, "seq": self.seq})
        self.events.append(e)

    def add_rule(self, subj: str, slot: str, on: str, on_value: str, new: str, time: str, src: str,
                 said: str = "") -> None:
        self.seq += 1
        r = {"s": subj, "slot": slot, "on": on, "on_value": on_value.strip(), "new": new.strip(), "time": time,
             "src": src, "seq": self.seq}
        if said:
            r["said"] = said
        self.rules.append(r)

    def forget(self, subj: str, slot: str, value_id: str, time: str, src: str) -> int:
        def hit(e: dict) -> bool:
            if e["s"] != subj:
                return False
            same_slot = bool(slot) and e.get("slot") == slot
            return (same_slot and (not value_id or e["o"] == value_id)) or (bool(value_id) and e["o"] == value_id)
        n = len(self.events)
        self.events = [e for e in self.events if not hit(e)]
        if slot:
            self.rules = [r for r in self.rules if not (r["s"] == subj and r["slot"] == slot)]
        self.add_event(subj, self.slot_names.get(slot, slot), "", time, src, slot=slot, status="forget")
        return n - len(self.events)

    def name(self, eid: str) -> str:
        return self.names.get(eid, eid)

    def link_do(self, key: str) -> list[str]:
        return _inter(self.links[key]["do"])

    def link_have(self, key: str) -> list[str]:
        return _inter(self.links[key]["have"])

    def get_do(self, x: str) -> list[str]:
        return _inter(self.gets[x]["do"]) if x in self.gets else []

    def get_have(self, x: str) -> list[str]:
        return _inter(self.gets[x]["have"]) if x in self.gets else []

    def act_have(self, action: str) -> list[str]:
        return _inter(self.acts[action]["have"]) if action in self.acts else []

    def enter_have(self, place: str) -> list[str]:
        return _inter(self.enters[place]["have"]) if place in self.enters else []

    def enter_do(self, place: str) -> list[str]:
        return _inter(self.enters[place]["do"]) if place in self.enters else []

    def where(self, x: str) -> dict | None:
        return self.gets.get(x) or self.ats.get(x)

    def things(self) -> list[str]:
        return sorted(set(self.gets) | set(self.ats), key=self.name)

    def route(self, src: str, dst: str) -> list[tuple[str, str, str]] | None:
        if src == dst:
            return []
        adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key, m in self.links.items():
            a, action = key.split("|", 1)
            adj[a].append((action, m["to"]))
        prev: dict[str, tuple[str, str]] = {src: ("", "")}
        q = deque([src])
        while q:
            r = q.popleft()
            for action, r2 in sorted(adj[r]):
                if r2 in prev:
                    continue
                prev[r2] = (r, action)
                if r2 == dst:
                    path, cur = [], dst
                    while cur != src:
                        p, c = prev[cur]
                        path.append((p, c, cur))
                        cur = p
                    return path[::-1]
                q.append(r2)
        return None

    def find(self, query: str, kinds: tuple[str, ...] = ("thing", "place")) -> tuple[str, str] | None:
        qs = set(words(query)) - STOP
        ql = " ".join(words(query))
        best, score = None, 0.0
        for eid, n in self.names.items():
            kind = self.kinds.get(eid, "thing")
            if kind not in kinds or (kind == "thing" and eid not in self.gets and eid not in self.ats):
                continue
            ws = set(words(n)) - STOP
            if not ws:
                continue
            full = " ".join(words(norm(n)))
            sc = 2 * len(qs & ws) + (3 if full and f" {full} " in f" {ql} " else 0) - 0.01 * len(ws - qs)
            if sc > score:
                best, score = (kind, eid), sc
        return best if score >= 2 else None

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        for k in ("rules", "seq", "slot_names", "slot_alias"):
            if not d[k]:
                d.pop(k)
        return d

    @classmethod
    def from_json(cls, data: dict) -> "Store":
        s = cls()
        s.__dict__.update(data)
        return s

    def digest(self) -> str:
        import hashlib
        import json
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]

    def stats(self) -> dict[str, Any]:
        return {"episodes": len(self.episodes), "places": sum(k == "place" for k in self.kinds.values()),
                "links": len(self.links), "things": len(set(self.gets) | set(self.ats)), "gets": len(self.gets),
                "gated": sum(bool(self.link_do(k)) for k in self.links) + sum(bool(self.get_do(x))
                                                                              for x in self.gets),
                "acts": len(self.acts), "enters": len(self.enters), "events": len(self.events),
                "rules": len(self.rules)}


def place_text(s: Store, rec: dict) -> str:
    p = the(s.name(rec["place"]))
    return f"in {the(rec['holder'])} in {p}" if rec.get("holder") else f"in {p}"


def flat_lines(s: Store) -> list[str]:
    out = []
    for x in s.things():
        if x in s.gets:
            g = s.gets[x]
            out.append(f"the {s.name(x)} was obtained {place_text(s, g)} by '{g['action']}'")
            if s.get_do(x):
                out.append(f"before '{g['action']}' in {the(s.name(g['place']))}, this was done there first: "
                           f"{'; '.join(s.get_do(x))}")
            if s.get_have(x):
                out.append(f"'{g['action']}' was done holding the {', the '.join(map(s.name, s.get_have(x)))}")
        else:
            out.append(f"the {s.name(x)} was seen {place_text(s, s.ats[x])}")
    for key in sorted(s.links):
        a, action = key.split("|", 1)
        out.append(f"move: from {the(s.name(a))}, '{action}' leads to {the(s.name(s.links[key]['to']))}")
        if s.link_do(key):
            out.append(f"before '{action}' from {the(s.name(a))}, this was done there first: "
                       f"{'; '.join(s.link_do(key))}")
        if s.link_have(key):
            out.append(f"'{action}' from {the(s.name(a))} was done holding the "
                       f"{', the '.join(map(s.name, s.link_have(key)))}")
    for action in sorted(s.acts):
        if s.act_have(action):
            out.append(f"'{action}' was done holding the {', the '.join(map(s.name, s.act_have(action)))}")
    for p in sorted(s.enters):
        need = [f"holding the {s.name(y)}" for y in s.enter_have(p)] + [f"'{a}' done" for a in s.enter_do(p)]
        if need:
            out.append(f"entering {the(s.name(p))} needed: {', '.join(need)}")
    for e in s.events:
        when = f" ({e['time']})" if e.get("time") else ""
        out.append(f"{s.name(e['s'])} {e['r']} {s.name(e['o'])}{when}")
    return out


def subject_text(s: Store, sid: str) -> str:
    n = s.name(sid)
    return "the user" if norm(n) in ("user", "i", "me") else n


def event_line(s: Store, e: dict) -> str:
    when = f" ({e['time']})" if e.get("time") else ""
    if "slot" not in e:
        return f"{s.name(e['s'])} {e['r']} {s.name(e['o'])}{when}"
    who = subject_text(s, e["s"])
    slot = s.slot_names.get(e["slot"], e["slot"])
    status = e.get("status", "set")
    if status == "forget":
        return f"{who} asked to remove the {slot or 'information'} from memory{when}"
    if not e["slot"]:
        return f"{who}: {s.name(e['o'])}{when}"
    if status == "ended":
        return f"{who}'s {slot}: {s.name(e['o'])} — no longer{when}"
    return f"{who}'s {slot}: {s.name(e['o'])}{when}"


def rule_line(s: Store, r: dict) -> str:
    who = subject_text(s, r["s"])
    slot, on = s.slot_names.get(r["slot"], r["slot"]), s.slot_names.get(r["on"], r["on"])
    named = f" ({r['on_value']})" if r.get("on_value") else ""
    then = f"the {slot} becomes {r['new']}" if r.get("new") else f"the {slot} would change too (new value not stated)"
    when = f" ({r['time']})" if r.get("time") else ""
    return f"{who}'s {slot} depends on {who}'s {on}{named}; if the {on} changes, {then}{when}"


def fact_lines(s: Store) -> list[str]:
    items = [((e.get("seq", 0), 0, i), event_line(s, e)) for i, e in enumerate(s.events)]
    items += [((r["seq"], 1, i), rule_line(s, r)) for i, r in enumerate(s.rules)]
    return [ln for _, ln in sorted(items)]
