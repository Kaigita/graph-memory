from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path

from graph_memory.extract import Record, Turn, add_extracted
from graph_memory.store import Store

DATA = Path("data/locomo/locomo10.json")
WORLD = ("a long conversation between two people over many dated sessions; each step is one turn: the command is "
         "the speaker's name and the observation is what they said")
KINDS = ("events",)
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


@lru_cache(maxsize=None)
def load() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def sessions(conv: dict) -> list[dict]:
    c = conv["conversation"]
    out = []
    n = 1
    while f"session_{n}" in c:
        turns = []
        for t in c[f"session_{n}"]:
            text = t["text"]
            if t.get("blip_caption"):
                text += f" [shares a photo: {t['blip_caption']}]"
            turns.append({"speaker": t["speaker"], "dia_id": t["dia_id"], "text": text})
        out.append({"n": n, "date": c.get(f"session_{n}_date_time", ""), "turns": turns})
        n += 1
    return out


def questions(conv: dict) -> list[dict]:
    out = []
    for i, q in enumerate(conv["qa"]):
        ans = q.get("answer", q.get("adversarial_answer", ""))
        out.append({"qid": f"{conv['sample_id']}/{i}", "question": q["question"], "answer": str(ans),
                    "category": q["category"], "evidence": q.get("evidence", [])})
    return out


class Porter:

    def _cons(self, w: str, i: int) -> bool:
        c = w[i]
        if c in "aeiou":
            return False
        if c == "y":
            return i == 0 or not self._cons(w, i - 1)
        return True

    def _m(self, w: str) -> int:
        n, i, L = 0, 0, len(w)
        while i < L and self._cons(w, i):
            i += 1
        while i < L:
            while i < L and not self._cons(w, i):
                i += 1
            if i >= L:
                break
            n += 1
            while i < L and self._cons(w, i):
                i += 1
        return n

    def _vowel(self, w: str) -> bool:
        return any(not self._cons(w, i) for i in range(len(w)))

    def _dbl(self, w: str) -> bool:
        return len(w) >= 2 and w[-1] == w[-2] and self._cons(w, len(w) - 1)

    def _cvc(self, w: str) -> bool:
        return (len(w) >= 3 and self._cons(w, len(w) - 1) and not self._cons(w, len(w) - 2)
                and self._cons(w, len(w) - 3) and w[-1] not in "wxy")

    def _rep(self, w: str, rules: list[tuple[str, str]], m0: int) -> str:
        for suf, rep in rules:
            if w.endswith(suf):
                stem = w[: len(w) - len(suf)]
                return stem + rep if self._m(stem) > m0 else w
        return w

    def stem(self, w: str) -> str:
        if len(w) <= 2:
            return w
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]
        flag = False
        if w.endswith("eed"):
            if self._m(w[:-3]) > 0:
                w = w[:-1]
        elif w.endswith("ed") and self._vowel(w[:-2]):
            w, flag = w[:-2], True
        elif w.endswith("ing") and self._vowel(w[:-3]):
            w, flag = w[:-3], True
        if flag:
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif self._dbl(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif self._m(w) == 1 and self._cvc(w):
                w += "e"
        if w.endswith("y") and self._vowel(w[:-1]):
            w = w[:-1] + "i"
        w = self._rep(w, [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
                          ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
                          ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
                          ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
                          ("iviti", "ive"), ("biliti", "ble")], 0)
        w = self._rep(w, [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
                          ("ful", ""), ("ness", "")], 0)
        for suf in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou",
                    "ism", "ate", "iti", "ous", "ive", "ize"):
            if w.endswith(suf):
                stem = w[: len(w) - len(suf)]
                if self._m(stem) > 1 and (suf != "ion" or stem.endswith(("s", "t"))):
                    w = stem
                break
        if w.endswith("e"):
            stem = w[:-1]
            if self._m(stem) > 1 or (self._m(stem) == 1 and not self._cvc(stem)):
                w = stem
        if self._m(w) > 1 and self._dbl(w) and w.endswith("l"):
            w = w[:-1]
        return w


PS = Porter()


def normalize_answer(s: str) -> str:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p = [PS.stem(w) for w in normalize_answer(pred).split()]
    g = [PS.stem(w) for w in normalize_answer(gold).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_multi(pred: str, gold: str) -> float:
    ps = [x.strip() for x in pred.split(",")]
    gs = [x.strip() for x in gold.split(",")]
    return sum(max(f1_score(p, g) for p in ps) for g in gs) / len(gs)


def score(pred: str, answer: str, category: int) -> float:
    if category == 5:
        low = pred.lower()
        return float("no information available" in low or "not mentioned" in low)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        return f1_multi(pred, answer)
    return f1_score(pred, answer)


def split() -> tuple[list[str], list[str]]:
    ids = sorted((c["sample_id"] for c in load()), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    return ids[:2], ids[2:]


def conv(sample_id: str) -> dict:
    return next(c for c in load() if c["sample_id"] == sample_id)


def record(c: dict, sess: dict) -> Record:
    a, b = c["conversation"]["speaker_a"], c["conversation"]["speaker_b"]
    return Record(f"Session {sess['n']} between {a} and {b}, dated {sess['date']}.",
                  [Turn(x["speaker"], x["text"]) for x in sess["turns"]])


def add_session(store: Store, facts: dict, c: dict, sess: dict) -> None:
    ids = ["S0"] + [x["dia_id"] for x in sess["turns"]]
    mapped = dict(facts)
    mapped["events"] = [x | {"steps": [ids[i] for i in x["steps"] if 0 <= i < len(ids)]}
                        for x in facts.get("events", [])]
    add_extracted(store, mapped, f"{c['sample_id']}/s{sess['n']}")


def raw_turns(c: dict) -> list[str]:
    return [f"[{x['dia_id']}] ({s['date']}) {x['speaker']}: {x['text']}" for s in sessions(c) for x in s["turns"]]


def full_text(c: dict) -> str:
    out = []
    for s in sessions(c):
        out.append(f"=== Session {s['n']} ({s['date']})")
        out += [f"{x['speaker']}: {x['text']}" for x in s["turns"]]
    return "\n".join(out)
