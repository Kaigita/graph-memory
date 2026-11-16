from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date

from graph_memory.store import Store, event_line, norm, subject_text, words

TAU = 0.5
JOIN_K = 3
TIMELINE_MAX = 20
GROUP_MAX = 30
SLOT_K = 3
DEPTH = 4
PROFILE_CHARS = 4000
VALUE_CHARS = 150
STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "was", "were", "be", "been", "do", "does",
        "did", "has", "have", "had", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "many",
        "much", "for", "with", "at", "by", "from", "about", "as", "that", "this", "these", "those", "it", "its", "i",
        "me", "my", "mine", "you", "your", "he", "she", "his", "her", "they", "their", "them", "we", "our", "us",
        "there", "here", "any", "some", "ever", "all", "now", "currently", "current", "still", "so", "far", "can",
        "could", "would", "should", "will", "just", "also", "not", "no", "yes", "if", "than", "then", "into", "over",
        "after", "before", "since", "ago", "long", "between", "first", "last", "most", "recent", "latest", "total"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                         "dec"], 1)}
PRONOUNS = {"i", "me", "my", "mine", "myself"}
CUES = {
    "collect": re.compile(r"\b(how many|how often|number of|count|in total|total|list|which|what are|all|ever|what "
                          r"kinds?|what types?)\b"),
    "timeline": re.compile(r"\b(when|what date|what year|what month|before|after|first|last|how long|since|ago|"
                           r"between|days|weeks|months|years)\b"),
    "current": re.compile(r"\b(now|currently|current|latest|still|most recent|anymore|these days|updated)\b"),
    "join": re.compile(r"\bthe \w+(?: \w+)? (?:who|that|which|whose)\b"),
}
DIFF = re.compile(r"\b(how long|how many (?:days|weeks|months|years)|between|before|after|first|ago|since)\b")


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    for suf in ("ing", "ed", "ly", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def toks(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [toks(d) for d in docs]
        self.k1, self.b = k1, b
        self.n = len(self.docs)
        self.avg = sum(map(len, self.docs)) / max(1, self.n)
        self.df = Counter(w for d in self.docs for w in set(d))

    def scores(self, query: str, idx: list[int] | None = None) -> dict[int, float]:
        q = toks(query)
        out = {}
        for i in (range(self.n) if idx is None else idx):
            d = self.docs[i]
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(d) / max(self.avg,
                                                                                                           1e-9)))
            out[i] = s
        return out


def parse_date(text: str) -> tuple[date, str] | None:
    t = str(text).lower()
    try:
        m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", t)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "day"
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+"
                      r"(\d{4})", t)
        if m:
            return date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))), "day"
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+"
                      r"(\d{4})", t)
        if m:
            return date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2))), "day"
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(\d{4})", t)
        if m:
            return date(int(m.group(2)), MONTHS[m.group(1)], 15), "month"
        m = re.search(r"\b((?:19|20)\d{2})\b", t)
        if m:
            return date(int(m.group(1)), 7, 1), "year"
    except ValueError:
        return None
    return None


def _dkey(e: dict, order: int) -> tuple:
    d = parse_date(e.get("time", ""))
    return (0, d[0].toordinal(), order) if d else (1, 0, order)


def seeds(s: Store, query: str) -> set[str]:
    ql = f" {' '.join(words(query))} "
    out = set()
    for eid, n in s.names.items():
        w = words(norm(n))
        if not w or all(x in STOP for x in w) or len(" ".join(w)) < 3:
            continue
        if f" {' '.join(w)} " in ql:
            out.add(eid)
    if PRONOUNS & set(words(query)):
        out |= {eid for eid, n in s.names.items() if norm(n) in ("user", "the user", "i")}
    return out


class Reader:

    def __init__(self, s: Store, propagate: bool = True, inferred: bool = True, profile_chars: int = 0):
        self.s = s
        self.lines = [event_line(s, e) for e in s.events]
        self.bm = BM25(self.lines)
        self.propagate = propagate
        self.profile_chars = profile_chars or PROFILE_CHARS
        self.slotted: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, e in enumerate(s.events):
            if e.get("slot"):
                self.slotted[(e["s"], e["slot"])].append(i)
        for idx in self.slotted.values():
            idx.sort(key=lambda i: s.events[i]["seq"])
        self.deps: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        rules = sorted((r for r in getattr(s, "rules", []) if inferred or r.get("said") != "no"),
                       key=lambda r: r["seq"])
        srcs = [self._source(r) for r in rules]
        tgts = [self._target(r, on, rules, srcs) for r, on in zip(rules, srcs)]
        known = defaultdict(set)
        for r, on, tgt in zip(rules, srcs, tgts):
            if (r["s"], on) in self.slotted:
                known[(r["s"], tgt)].add(on)
        for r, on, tgt in zip(rules, srcs, tgts):
            key = (r["s"], tgt)
            if (r["s"], on) not in self.slotted and len(known[key]) == 1:
                on = next(iter(known[key]))
            if on == tgt:
                continue
            d = self.deps[key].setdefault(on, {"new": "", "on_value": "", "first": r["seq"], "rules": []})
            d["new"] = r.get("new") or d["new"]
            d["on_value"] = d["on_value"] or r.get("on_value", "")
            d["rules"].append(r)
        self.keys = sorted(set(self.slotted) | set(self.deps))
        self.slot_bm = BM25([self._slot_doc(k) for k in self.keys]) if self.keys else None

    def _source(self, r: dict) -> str:
        if (r["s"], r["on"]) in self.slotted:
            return r["on"]
        v = norm(r.get("on_value", ""))
        for exact in (True, False):
            best = None
            for (subj, slot), idx in self.slotted.items():
                if not v or subj != r["s"] or slot == r["slot"]:
                    continue
                for i in idx:
                    o = norm(self.s.name(self.s.events[i]["o"]))
                    ok = o == v if exact else (len(v) >= 4 and len(o) >= 4 and (v in o or o in v))
                    if o and ok and (best is None or self.s.events[i]["seq"] > best[1]):
                        best = (slot, self.s.events[i]["seq"])
            if best:
                return best[0]
        return self._by_words(r["s"], r["on"], exclude=r["slot"]) or r["on"]

    def _target(self, r: dict, on: str, rules: list[dict], srcs: list[str]) -> str:
        if (r["s"], r["slot"]) in self.slotted:
            return r["slot"]
        hit = self._by_words(r["s"], r["slot"], exclude=on)
        if hit:
            return hit
        same = {r2["slot"] for r2, on2 in zip(rules, srcs) if r2 is not r and r2["s"] == r["s"] and on2 == on
                and (r2["s"], r2["slot"]) in self.slotted}
        return next(iter(same)) if len(same) == 1 else r["slot"]

    def _by_words(self, subj: str, slot: str, exclude: str = "") -> str:
        a = words(self.s.slot_names.get(slot, slot))
        if not a:
            return ""
        scored = []
        for (s2, sid) in self.slotted:
            if s2 != subj or sid in (slot, exclude):
                continue
            b = words(self.s.slot_names.get(sid, sid))
            m = sum(1 for x in a if any(x == y or (min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x)))
                                        for y in b))
            scored.append((m / max(len(a), len(b)), sid))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.5 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
            return scored[0][1]
        return ""

    def _sname(self, key: tuple[str, str]) -> str:
        return self.s.slot_names.get(key[1], key[1])

    def _slot(self, key: tuple[str, str]) -> str:
        return f"{subject_text(self.s, key[0])}'s {self._sname(key)}"

    def _slot_doc(self, key: tuple[str, str]) -> str:
        vals = [self.s.name(self.s.events[i]["o"]) for i in self.slotted.get(key, [])]
        vals += [d["new"] for d in self.deps.get(key, {}).values() if d["new"]]
        return " ".join([self._sname(key)] * 2 + vals)

    def _timeline(self, key: tuple[str, str]) -> list[tuple[int, str | None]]:
        cur, out = None, []
        for i in self.slotted.get(key, []):
            e = self.s.events[i]
            st = e.get("status", "set")
            if st == "forget":
                cur = None
            elif st == "ended":
                cur = None if cur == e["o"] else cur
            else:
                cur = e["o"]
            out.append((i, cur))
        return out

    def _changed(self, key: tuple[str, str], t0: int, depth: int = 0,
                 seen: frozenset = frozenset()) -> tuple[int, dict] | None:
        if depth > DEPTH or key in seen:
            return None
        seen = seen | {key}
        best, v0, tl = None, None, self._timeline(key)
        for j, (i, cur) in enumerate(tl):
            e = self.s.events[i]
            if e["seq"] <= t0:
                v0 = cur
                continue
            if cur != v0:
                rec = e["src"].split(":", 1)[0]
                to = cur
                for i2, cur2 in tl[j + 1:]:
                    if self.s.events[i2]["src"].split(":", 1)[0] != rec:
                        break
                    to = cur2
                best = (e["seq"], {"kind": "direct", "key": key, "from": v0, "to": to, "idx": i})
                break
        for on, d in self.deps.get(key, {}).items():
            c = self._changed((key[0], on), max(t0, d["first"]), depth + 1, seen)
            if c and (best is None or c[0] < best[0]):
                best = (c[0], {"kind": "derived", "key": key, "via": c[1], "new": d["new"]})
        return best

    def resolve(self, key: tuple[str, str]) -> dict:
        tl = self._timeline(key)
        last = tl[-1][0] if tl else None
        out = {"key": key, "value": tl[-1][1] if tl else None, "stated": tl[-1][1] if tl else None,
               "status": self.s.events[last].get("status", "set") if last is not None else "", "changed": None}
        if not self.propagate or out["status"] == "forget":
            return out
        v_seq = self.s.events[last]["seq"] if last is not None else 0
        best = None
        for on, d in self.deps.get(key, {}).items():
            c = self._changed((key[0], on), max(v_seq, d["first"]))
            if c and (best is None or c[0] < best[0][0]):
                best = (c, d, on)
        if best:
            (q, why), d, on = best
            out.update({"value": d["new"] or None, "new": d["new"], "unknown": not d["new"], "on": on, "why": why,
                        "changed": q})
        return out

    def _vname(self, vid: str | None) -> str:
        return self.s.name(vid) if vid else ""

    def _history(self, key: tuple[str, str]) -> list[str]:
        out, cur = [], None
        for i, after in self._timeline(key):
            e = self.s.events[i]
            when = f" ({e['time']})" if e.get("time") else ""
            st = e.get("status", "set")
            if st == "forget":
                out.append(f"removed from memory at the user's request{when}")
            elif st == "ended":
                out.append(f"{self._vname(e['o'])} no longer{when}")
            elif e["o"] != cur:
                out.append(f"{self._vname(e['o'])}{when}")
            cur = after
        return out

    def _why(self, why: dict) -> str:
        if why["kind"] == "direct":
            e = self.s.events[why["idx"]]
            when = f" on {e['time']}" if e.get("time") else ""
            return (f"{self._slot(why['key'])} changed{when} ({self._vname(why['from']) or 'none'} → "
                    f"{self._vname(why['to']) or 'none'})")
        then = f"it becomes {why['new']}" if why["new"] else "no new value was stated"
        return f"{self._why(why['via'])}; {self._slot(why['key'])} depended on that, so it changed too ({then})"

    def _deps_text(self, key: tuple[str, str]) -> list[str]:
        out = []
        for on, d in self.deps.get(key, {}).items():
            named = f" ({d['on_value']})" if d["on_value"] else ""
            src = self.s.slot_names.get(on, on)
            then = (f"the {self._sname(key)} becomes {d['new']}" if d["new"]
                    else f"the {self._sname(key)} would change too (new value not stated)")
            out.append(f"it depends on {subject_text(self.s, key[0])}'s {src}{named}; if the {src} changes, {then}")
        return out

    def state_line(self, key: tuple[str, str]) -> str:
        r, name, hist = self.resolve(key), self._slot(key), self._history(key)
        if r["changed"] is not None:
            was = f" It was {'; then '.join(hist)}." if hist else ""
            if r["unknown"]:
                return (f"{name}: unknown now (no longer valid).{was} Why: {self._why(r['why'])}; the "
                        f"{self._sname(key)} depended on it and no new {self._sname(key)} was stated.")
            src = self.s.slot_names.get(r["on"], r["on"])
            return (f"{name}: {r['new']} (now).{was} Why: {self._why(r['why'])}; the user had said that if the {src} "
                    f"changes, the {self._sname(key)} becomes {r['new']}.")
        if r["status"] == "forget":
            return f"{name}: {hist[-1]}; nothing is recorded now."
        deps = "".join(f" Stated: {t}." for t in self._deps_text(key))
        if r["value"] is None:
            return f"{name}: none now ({'; then '.join(hist)}).{deps}" if hist else f"{name}: nothing recorded.{deps}"
        if len(hist) == 1:
            return f"{name}: {hist[0]}.{deps}"
        return f"{name}: {self._vname(r['value'])} (now). In order: {' → '.join(hist)}.{deps}"

    def retrieve_line(self, key: tuple[str, str], depth: int = 0, seen: frozenset = frozenset()) -> str:
        hist = self._history(key)
        text = f"{self._slot(key)}: {' → '.join(hist) if hist else 'nothing recorded'}."
        text += "".join(f" Stated: {t}." for t in self._deps_text(key))
        if depth < DEPTH and key not in seen:
            for on in self.deps.get(key, {}):
                src = (key[0], on)
                if src not in seen | {key}:
                    text += f"\n  {self.retrieve_line(src, depth + 1, seen | {key})}"
        return text

    def _slot_scores(self, q: str) -> dict[int, float]:
        ws = re.findall(r"[a-z]+", q.lower())
        return self.slot_bm.scores(q + " " + " ".join(a + b for a, b in zip(ws, ws[1:])))

    def match_slots(self, q: str) -> list[tuple[str, str]]:
        if not self.keys:
            return []
        sc = self._slot_scores(q)
        best = max(sc.values(), default=0.0)
        if best <= 0:
            return []
        keep = sorted((i for i, v in sc.items() if v >= TAU * best), key=lambda i: (-sc[i], i))[:SLOT_K]
        return [self.keys[i] for i in keep]

    def profile(self, q: str, skip: list[tuple[str, str]]) -> list[tuple[str, set[int]]]:
        sc = self._slot_scores(q) if self.keys else {}
        last = {k: max((self.s.events[i]["seq"] for i in self.slotted.get(k, [])), default=0) for k in self.keys}
        order = sorted((i for i, k in enumerate(self.keys) if k not in skip), key=lambda i: (-sc.get(i, 0.0),
                                                                                              -last[self.keys[i]]))
        out, used = [], 0
        for i in order:
            k = self.keys[i]
            r = self.resolve(k)
            if r["status"] == "forget" or (r["value"] is None and not r.get("unknown")):
                continue
            if r.get("unknown"):
                val = (f"unknown now (it depended on the {self.s.slot_names.get(r['on'], r['on'])}, which changed; "
                       f"no new value stated)")
            elif r["changed"] is not None:
                val = f"{r['new']} (changed with the {self.s.slot_names.get(r['on'], r['on'])})"
            else:
                val = self._vname(r["value"])
            val = val if len(val) <= VALUE_CHARS else val[: VALUE_CHARS - 1] + "…"
            cur = self._vname(r["value"]) if r["changed"] is None and r["value"] else ""
            prev: list[str] = []
            for j in reversed(self.slotted.get(k, [])):
                e = self.s.events[j]
                v = self._vname(e["o"])
                if e.get("status", "set") == "set" and v and v != cur and v not in prev:
                    prev.append(v)
            if prev:
                val += " (earlier: " + "; ".join(p if len(p) <= 60 else p[:59] + "…" for p in prev[:4]) + ")"
            who = "" if subject_text(self.s, k[0]) == "the user" else f"{subject_text(self.s, k[0])}'s "
            line = f"{who}{self._sname(k)}: {val}"
            if used + len(line) + 3 > self.profile_chars:
                break
            used += len(line) + 3
            out.append((line, set(self.slotted.get(k, []))))
        return out

    def _cand(self, sd: set[str]) -> list[int]:
        if not sd:
            return list(range(len(self.lines)))
        return [i for i, e in enumerate(self.s.events) if e["s"] in sd or e["o"] in sd]

    def _keep(self, sc: dict[int, float], tau: float) -> list[int]:
        best = max(sc.values(), default=0.0)
        if best <= 0:
            return []
        return [i for i, v in sc.items() if v >= tau * best]

    def _val(self, i: int) -> str:
        e = self.s.events[i]
        return self.s.name(e["o"]) + (f" ({e['time']})" if e.get("time") else "")

    def collect(self, q: str, sd: set[str]) -> list[tuple[str, set[int]]]:
        sc = self.bm.scores(q, self._cand(sd))
        keep = self._keep(sc, TAU)
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i in keep:
            e = self.s.events[i]
            groups[(e["s"], norm(e["r"]))].append(i)
        order = sorted(groups, key=lambda g: -max(sc[i] for i in groups[g]))
        out = [(f"{len(keep)} matching records in {len(order)} groups", set())] if keep else []
        for g in order:
            seen, items, idx = set(), [], set()
            for i in sorted(groups[g], key=lambda i: _dkey(self.s.events[i], i)):
                o = norm(self.s.name(self.s.events[i]["o"]))
                if o in seen:
                    continue
                seen.add(o)
                items.append(self._val(i))
                idx.add(i)
            out.append((f"{self.s.name(g[0])} — {self.s.events[groups[g][0]]['r']}: {'; '.join(items[:GROUP_MAX])} "
                        f"[{len(items)} distinct]", idx))
        return out

    def current(self, q: str, sd: set[str]) -> list[tuple[str, set[int]]]:
        cand = self._cand(sd)
        when = re.search(r"\bwhen\b(.+)", q.lower())
        main = q[: when.start()] if when and when.start() > 0 else q
        sc = self.bm.scores(main, cand)
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i in cand:
            if sc.get(i, 0) > 0:
                e = self.s.events[i]
                groups[(e["s"], norm(e["r"]))].append(i)
        order = sorted(groups, key=lambda g: -max(sc[i] for i in groups[g]))[:2]
        at = None
        if when and order and main is not q:
            csc = self.bm.scores(when.group(1), cand)
            inside = [i for i in cand if (self.s.events[i]["s"], norm(self.s.events[i]["r"])) != order[0]]
            best = max(inside, key=lambda i: csc.get(i, 0), default=None)
            if best is not None and csc.get(best, 0) > 0 and parse_date(self.s.events[best].get("time", "")):
                at = (best, parse_date(self.s.events[best]["time"])[0])
        out = []
        for g in order:
            seq = sorted(groups[g], key=lambda i: _dkey(self.s.events[i], i))
            last = self.s.events[seq[-1]]
            earlier = [self._val(i) for i in reversed(seq[:-1])][:8]
            out.append((f"{self.s.name(g[0])} {last['r']}: latest {self._val(seq[-1])}"
                        + (f"; earlier: {'; '.join(earlier)}" if earlier else ""), set(seq)))
            if at is not None:
                valid = [i for i in seq if (parse_date(self.s.events[i].get("time", "")) or (date.max, ""))[0] <= at[1]]
                if valid:
                    e = self.s.events[valid[-1]]
                    out.append((f"at the time of \"{self.lines[at[0]]}\": {self.s.name(g[0])} {e['r']} "
                                f"{self.s.name(e['o'])}", {valid[-1], at[0]}))
        return out

    def timeline(self, q: str, sd: set[str], now: str = "") -> list[tuple[str, set[int]]]:
        cand = self._cand(sd)
        sc = self.bm.scores(q, cand)
        keep = sorted(self._keep(sc, TAU), key=lambda i: -sc[i])[:TIMELINE_MAX]
        seq = sorted(keep, key=lambda i: _dkey(self.s.events[i], i))
        out = [(f"{self.s.events[i].get('time') or 'undated'}: {self.lines[i]}", {i}) for i in seq]
        if DIFF.search(q.lower()):
            dated: list[tuple[int, date]] = []
            for i in sorted(keep, key=lambda i: -sc[i]):
                d = parse_date(self.s.events[i].get("time", ""))
                if d and all(self.lines[i] != self.lines[j] for j, _ in dated):
                    dated.append((i, d[0]))
                if len(dated) == 2:
                    break
            nd = parse_date(now)
            head = []
            if dated and nd:
                i, a = dated[0]
                n = (nd[0] - a).days
                head.append((f"from \"{self.lines[i]}\" ({a.isoformat()}) to now ({nd[0].isoformat()}): {n} days "
                             f"(about {n / 7:.1f} weeks, {n / 30.44:.1f} months)", {i}))
            if len(dated) == 2:
                (i, a), (j, b) = sorted(dated, key=lambda x: x[1])
                n = (b - a).days
                head.append((f"between \"{self.lines[i]}\" ({a.isoformat()}) and \"{self.lines[j]}\" "
                             f"({b.isoformat()}): {n} days (about {n / 7:.1f} weeks, {n / 30.44:.1f} months); "
                             f"the first one came first", {i, j}))
            out = head + out
        return out

    def join(self, q: str, sd: set[str]) -> list[tuple[str, set[int]]]:
        frontier, seen_ent, used, out = set(sd), set(sd), set(), []
        for hop in range(1, 4):
            cand = [i for i, e in enumerate(self.s.events) if (e["s"] in frontier or e["o"] in frontier)
                    and i not in used]
            if not cand:
                break
            sc = self.bm.scores(q, cand)
            top = [i for i in sorted(cand, key=lambda i: -sc[i])[:JOIN_K] if sc[i] > 0]
            if not top:
                break
            out += [(f"hop {hop}: {self.lines[i]}", {i}) for i in top]
            used.update(top)
            nxt = set()
            for i in top:
                e = self.s.events[i]
                nxt |= {e["s"], e["o"]}
            frontier = nxt - seen_ent
            seen_ent |= nxt
            if not frontier:
                break
        return out

    def neighbor(self, q: str, sd: set[str]) -> list[int]:
        sc = self.bm.scores(q, self._cand(sd))
        return [i for i in sorted(sc, key=lambda i: -sc[i]) if sc[i] > 0]

    def ops(self, q: str) -> list[str]:
        ql = q.lower()
        found = [op for op in ("collect", "timeline", "current", "join") if CUES[op].search(ql)]
        return found or ["neighbor"]

    def brief(self, q: str, budget: int, now: str = "") -> tuple[str, dict]:
        sd = seeds(self.s, q)
        chosen = self.ops(q)
        titles = {"collect": "All matching records", "timeline": "Timeline", "current": "Current and earlier values",
                  "join": "Linked facts"}
        sections = []
        slots = self.match_slots(q)
        if slots and self.propagate:
            sections.append(("Current values, worked out from the records and the stated dependencies",
                             [(self.state_line(k), set(self.slotted.get(k, []))) for k in slots]))
        elif slots:
            sections.append(("Records and stated dependencies of the asked values",
                             [(self.retrieve_line(k), set(self.slotted.get(k, []))) for k in slots]))
        for op in chosen:
            if op == "current" and self.keys:
                continue
            if op != "neighbor":
                lines = self.timeline(q, sd, now) if op == "timeline" else getattr(self, op)(q, sd)
                if lines:
                    sections.append((titles[op], lines))
        if self.keys:
            prof = self.profile(q, slots)
            if prof:
                sections.append(("Everything else recorded" + (", values now" if self.propagate
                                                               else ", last stated values"), prof))
        used = {i for _, lines in sections for _, idx in lines for i in idx}
        rest, seen_text = [], set()
        for i in self.neighbor(q, sd) + self.neighbor(q, set()):
            if i not in used and self.lines[i] not in seen_text:
                seen_text.add(self.lines[i])
                rest.append((self.lines[i], {i}))
        sections.append(("Other related records", rest))
        text, n, shown = "", 0, set()
        for title, lines in sections:
            head = ("\n" if text else "") + f"{title}:"
            added = False
            for ln, idx in lines:
                add = f"\n- {ln}"
                room = budget - len(text) - (0 if added else len(head))
                if len(add) > room:
                    cut = add[: max(0, room - 2)]
                    if "; " not in cut or len(idx) < 2:
                        break
                    add = cut[: cut.rfind("; ")] + " …"
                    idx = {i for i in idx if self._val(i) in add}
                if not added:
                    text += head
                    added = True
                text += add
                n += 1
                shown |= idx
                if len(add) < len(f"\n- {ln}"):
                    break
        meta = {"ops": chosen, "seeds": sorted(self.s.name(x) for x in sd), "lines": n, "shown": sorted(shown)}
        if self.keys:
            meta["slots"] = [self._sname(k) for k in slots]
        return text, meta
