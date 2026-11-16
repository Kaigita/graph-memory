from __future__ import annotations

import json
import re
from pathlib import Path

from graph_memory.extract import PARTS_S1, PARTS_S2, Record, Turn, add_extracted, extract, slot_key, value_key
from graph_memory.reader import Reader, parse_date
from graph_memory.store import Store, norm, words

ASSISTANT_CHARS = 300
KNOWN_CHARS = 60
KINDS = ("states", "rules")
VERSIONS = ("m1", "s1", "s2")
WORLD = ("a chat session between a user and an AI assistant; each step is one turn: the command is the role (user or "
         "assistant) and the observation is what they said (the assistant's turns are cut short). Record only what "
         "the user states about themselves, their life, their team or their project -- 'we' and 'our' mean the "
         "user's team: write 'the user' as the subject of all of these -- never the assistant's suggestions or "
         "general knowledge. Known slots of the user so far, each with its last stated value (write exactly these "
         "slot names when a fact gives a value of one of them): {known}")
DEPS = ("Kinds of value that known slots depend on and that hold no value yet (write exactly these names when a fact "
        "gives a value of one of them): {deps}")
USER = re.compile(r"^(the |our |my )?(user|users|user's team|users' team|team|we|i|me|us|project|team's project)$")


def session_record(turns: list[dict], timestamp: str = "", assistant_chars: int = ASSISTANT_CHARS) -> Record:
    return Record(f"Chat session dated {timestamp} between the user and an AI assistant.",
                  [Turn(t["role"], t["content"] if t["role"] == "user" else t["content"][:assistant_chars])
                   for t in turns])


def turn_texts(record: Record) -> list[str]:
    return [record.header.lower()] + [t.observation.lower() for t in record.turns]


def known(store: Store) -> list[tuple[str, str]]:
    val: dict[tuple[str, str], str | None] = {}
    for e in sorted((e for e in store.events if e.get("slot")), key=lambda e: e["seq"]):
        k = (e["s"], e["slot"])
        st = e.get("status", "set")
        if st == "forget" or (st == "ended" and val.get(k) == e["o"]):
            val[k] = None
        elif st == "set":
            val[k] = e["o"]
        else:
            val.setdefault(k, None)
    out = []
    for (s, slot), v in val.items():
        name = store.slot_names.get(slot, slot)
        text = store.name(v) if v else "(none now)"
        text = text if len(text) <= KNOWN_CHARS else text[: KNOWN_CHARS - 1] + "…"
        out.append((name, " ".join(text.split())))
    return out


def world(store: Store, version: str = "m1") -> str:
    ks = known(store)
    text = WORLD.format(known="; ".join(f"{n}: {v}" for n, v in ks) if ks else "(none yet)")
    if version == "m1":
        return text
    deps = dep_kinds(store)
    return text + ". " + DEPS.format(deps="; ".join(deps) if deps else "(none yet)")


def parts(version: str = "m1") -> dict | None:
    return None if version == "m1" else {"s1": PARTS_S1, "s2": PARTS_S2}[version]


def dep_kinds(store: Store) -> list[str]:
    have = {e["slot"] for e in store.events if e.get("slot")}
    out: list[str] = []
    for r in sorted(store.rules, key=lambda r: r["seq"]):
        n = store.slot_names.get(r["on"], r["on"])
        if r["on"] not in have and n not in out:
            out.append(n)
    return out


def slot_id(store: Store, name: str) -> str:
    k = slot_key(name)
    return store.slot_alias.get(k, "") if k else ""


def current(store: Store, subj: str, slot: str) -> str | None:
    cur = None
    for e in sorted((e for e in store.events if e.get("slot") == slot and e["s"] == subj), key=lambda e: e["seq"]):
        st = e.get("status", "set")
        if st == "forget" or (st == "ended" and cur == e["o"]):
            cur = None
        elif st == "set":
            cur = e["o"]
    return cur


def grounded(x: dict, kind: str, texts: list[str]) -> bool:
    steps = [i for i in x.get("steps", []) if isinstance(i, int) and 1 <= i < len(texts)]
    if not steps:
        return False
    if kind != "states" or not x.get("value"):
        return True
    near = " ".join(texts[j] for i in steps for j in (i - 1, i, i + 1) if 1 <= j < len(texts))
    w = re.findall(r"[a-z0-9]+", str(x["value"]).lower())
    return not w or sum(t in near for t in w) / len(w) >= 0.5


def grounded_s1(store: Store, x: dict, kind: str, texts: list[str]) -> bool:
    steps = [i for i in x.get("steps", []) if isinstance(i, int) and 1 <= i < len(texts)]
    if not steps:
        return False
    if kind != "states" or not x.get("value"):
        return True
    status = (x.get("status") or "set").strip().lower()
    slot = slot_id(store, x.get("slot", ""))
    if status == "ended" and slot and any(e.get("slot") == slot for e in store.events):
        return True
    if status in ("set", "") and value_key(x["value"]) not in {e["o"] for e in store.events if e.get("slot")}:
        return True
    return grounded(x, kind, texts)


def end_current(store: Store, x: dict) -> dict:
    if (x.get("status") or "").strip().lower() != "ended":
        return x
    slot = slot_id(store, x.get("slot", ""))
    subj = store.aliases.get(norm(x.get("subject", "")), norm(x.get("subject", "")))
    cur = current(store, subj, slot) if slot else None
    if cur is None or value_key(x.get("value", "")) == cur:
        return x
    name = store.name(cur)
    said = str(x.get("value", "")).replace("…", " ").strip()
    w, c = words(said), set(words(name))
    if not w or norm(name).startswith(norm(said)) or sum(t in c for t in w) / len(w) >= 0.5:
        return x | {"value": name}
    return x


def subject(name: str) -> str:
    return "the user" if USER.match(" ".join(str(name).lower().split()).strip(" .,'\"")) else name


def add_facts(store: Store, facts: dict, rec_id: str, day: str, texts: list[str], version: str = "m1") -> None:
    s1 = version != "m1"
    fixed = dict(facts)
    for kind in KINDS:
        rows = []
        for x in facts.get(kind, []):
            if not (grounded_s1(store, x, kind, texts) if s1 else grounded(x, kind, texts)):
                continue
            x = x | {"time": (x.get("time") or "").strip() or day, "subject": subject(x.get("subject", ""))}
            rows.append(end_current(store, x) if s1 and kind == "states" else x)
        fixed[kind] = rows
    add_extracted(store, fixed, rec_id)


def day_of(timestamp: str) -> str:
    d = parse_date(timestamp)
    return d[0].isoformat() if d and d[1] == "day" else ""


class ChatMemory:

    def __init__(self, llm=None, version: str = "m1", cache: str | Path | None = None, budget: int = 6000):
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}")
        self.llm, self.version, self.budget = llm, version, budget
        self.cache = Path(cache) if cache else None
        self.store = Store()
        self.sessions = 0

    async def add_session(self, turns: list[dict], timestamp: str = "") -> dict:
        rec = session_record(turns, timestamp)
        facts = await extract(rec, self.llm, self.cache, world(self.store, self.version), KINDS, Store(),
                              parts=parts(self.version))
        self.add_facts(facts, rec, timestamp)
        return facts

    def add_facts(self, facts: dict, record: Record, timestamp: str = "") -> None:
        add_facts(self.store, facts, f"session#{self.sessions}", day_of(timestamp), turn_texts(record), self.version)
        self.sessions += 1

    def recall(self, question: str, budget: int | None = None, propagate: bool = True) -> str:
        text, _ = Reader(self.store, propagate=propagate).brief(question, budget or self.budget)
        return text

    def save(self, path: str | Path) -> None:
        data = {"version": self.version, "sessions": self.sessions, "store": self.store.to_json()}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, llm=None, cache: str | Path | None = None) -> "ChatMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mem = cls(llm, data["version"], cache)
        mem.store = Store.from_json(data["store"])
        mem.sessions = data["sessions"]
        return mem
