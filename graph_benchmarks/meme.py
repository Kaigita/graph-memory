from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from graph_memory.chat import add_facts, session_record, turn_texts
from graph_memory.extract import Record
from graph_memory.store import Store

DATA = Path("data/meme/meme_filler32k.json")
TASKS = ("ER", "Agg", "Tr", "Del", "Cas", "Abs")
FILTERED = ("Cas", "Abs", "Del")
N_DEV = 5


@lru_cache(maxsize=1)
def load() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def by_id() -> dict[str, dict]:
    return {e["episode_id"]: e for e in load()}


def split() -> tuple[list[str], list[str]]:
    dev, formal = [], []
    for dom in ("pl_", "sw_"):
        ids = sorted((e for e in by_id() if e.startswith(dom)), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        dev += ids[:N_DEV]
        formal += ids[N_DEV:]
    return dev, formal


def points(ep: dict) -> tuple[int, int]:
    return ep["before_questions"]["position_after_session"] + 1, ep["after_questions"]["position_after_session"] + 1


def day(ts: str) -> str:
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", ts or "")
    return f"{m[1]}-{m[2]}-{m[3]}" if m else ""


def items(eid: str) -> list[dict]:
    ep = by_id()[eid]
    out = []
    for phase in ("before", "after"):
        for k, q in enumerate(ep[f"{phase}_questions"]["questions"]):
            ev = q["entity_values"]
            out.append({"qid": f"{eid}/{phase}/{k}", "unit": eid, "phase": phase, "task": q["task_type"].split(" (")[0],
                        "hop": q.get("hop", 0), "entity": list(ev)[0], "entity_values": ev,
                        "question": q["question"],
                        "gold": q.get("expected_answer") if phase == "before" else q.get("gold_answer")})
    return out


def turn_lines(sess: dict) -> list[str]:
    return [f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')}"
            for t in sess.get("conversation", [])]


def transcript(eid: str, upto: int) -> str:
    parts = []
    for sess in by_id()[eid]["sessions"][:upto]:
        parts.append(f"[Session: {sess.get('timestamp', '')}]")
        parts += turn_lines(sess)
    return "\n".join(parts)


def chunks(eid: str, upto: int, max_chars: int = 16000) -> list[str]:
    out = []
    for sess in by_id()[eid]["sessions"][:upto]:
        head = f"[Session: {sess.get('timestamp', 'unknown')}]"
        cur, size = [head], len(head)
        for line in turn_lines(sess):
            if size + len(line) > max_chars and len(cur) > 1:
                out.append("\n".join(cur))
                cur, size = [head], len(head)
            cur.append(line)
            size += len(line) + 1
        if len(cur) > 1:
            out.append("\n".join(cur))
    return out


ANSWER = """Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say you don't have that information.
Answer with ONLY the value. Do not explain or add context.

Context:
{context}

Question: {question}"""

BEFORE_COMMON = """You are a judge evaluating whether an AI agent's answer is semantically correct.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Does the agent's answer contain the correct information matching the gold answer?

Rules:
- Focus on semantic equivalence, not exact wording
- If the gold value is present in the agent's answer, it is correct — regardless of any additional information, future possibilities, or extra details the agent mentions
- "Dentist every 6 months; dermatologist monthly (if you change residence)" → gold is "dentist (every 6 months)" → YES (core answer correct, extra info is irrelevant)
- "40 minutes" = "40 min" → YES
- "dentist appointment every 6 months" = "dentist (every 6 months)" → YES
- If agent says "I don't know" and gold is a specific value → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

CASCADE_D = """You are a judge evaluating whether an AI agent correctly reports a CASCADE-DEPENDENT value after an upstream change.

QUESTION: {question}
GOLD (new cascaded value): {gold}
AGENT: {agent_answer}

The upstream entity changed, triggering a cascade. The agent should now report ONLY the new value.

Rules:
- Agent states the new value as the sole, definitive answer → YES
- Agent lists BOTH old and new values together → NO
- Agent states only the old value → NO
- Agent lists multiple options or hedges ("might", "considering") → NO
- Agent uses future tense ("will change to X", "will be X") instead of confirming as current → NO
- "I don't know" → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

CASCADE_U = """You are a judge evaluating whether an AI agent appropriately handles UNCERTAINTY after an upstream entity changed.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The upstream entity changed, making the previous answer potentially invalid. The agent should express uncertainty.

Rules:
- Agent says "I don't know" / "not sure" / "Unknown" / "None" / expresses uncertainty → YES
- Agent acknowledges the upstream change and questions validity → YES
- Agent confidently states the OLD value as definitive current fact → NO
- Agent provides the old value with no hedging or uncertainty → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

DELETION = """You are a judge evaluating whether an AI agent correctly handles a DELETED piece of information.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The information was explicitly deleted/removed. The agent should not return it.

Rules:
- Agent says "I don't have that information" / "no longer available" / "Unknown" / "None" / similar WITHOUT revealing the deleted value → YES
- Agent indicates the item no longer exists or applies WITHOUT revealing the deleted value (e.g., "No food allergies" when the deleted value was a specific allergy) → YES
- Agent reveals the deleted value in any way (even while saying it was deleted/removed) → NO
- Agent returns the deleted value as if it still exists → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

MULTIHOP = """You are a judge evaluating whether an AI agent's answer contains specific values.

QUESTION: {question}
TARGET VALUES: {gold_values}
AGENT: {agent_answer}

For EACH target value, check whether it appears (exact or paraphrased) in the agent's answer.

Rules:
- The value appears in the answer (exact or paraphrased) → true
- The value is absent → false
- "I don't know" or "I don't have that information" → all false

Output ONLY JSON: {{"results": [{{"value": "...", "present": true/false}}], "reason": "one sentence"}}"""

MULTIPLE_UPDATE = """You are a judge evaluating whether an AI agent correctly recalls a sequence of historical values for an entity that changed multiple times.

QUESTION: {question}
GOLD (ordered history values, comma-separated): {gold}
AGENT: {agent_answer}

The user's information changed multiple times. The question asks for the FULL HISTORY in chronological order (earliest to latest).

Rules:
- Agent's answer must contain ALL gold values in the EXACT order specified
- Order matters: the values must appear earliest-to-latest as in the gold
- Extra surrounding text is fine, but the gold sequence must be preserved
- Missing any value → NO
- Wrong order → NO
- Only some values → NO
- "I don't know" → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

AFTER_PROMPTS = {"Tr": MULTIPLE_UPDATE, "Cas": CASCADE_D, "Abs": CASCADE_U, "Del": DELETION, "Agg": MULTIHOP}


def judge_request(it: dict, answer: str) -> str | None:
    task, ev = it["task"], it["entity_values"]
    if task == "ER":
        return None
    if task == "Agg":
        gold_values = ", ".join(f'"{v}"' for v in ev.values())
        return MULTIHOP.format(question=it["question"], gold_values=gold_values, agent_answer=answer)
    value = ev[it["entity"]]
    if task == "Tr":
        history = value if isinstance(value, list) else [value]
        return MULTIPLE_UPDATE.format(question=it["question"], gold=", ".join(history), agent_answer=answer)
    template = BEFORE_COMMON if it["phase"] == "before" else AFTER_PROMPTS[task]
    return template.format(question=it["question"], gold=value, agent_answer=answer)


def er_pass(it: dict, answer: str) -> bool:
    gold = " ".join(str(it["entity_values"][it["entity"]]).lower().split())
    return gold in " ".join(answer.lower().split())


def parse_verdict(it: dict, content: str) -> bool:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
    if it["task"] == "Agg":
        res = data.get("results", []) if isinstance(data, dict) else []
        n = len(it["entity_values"])
        return len(res) >= n and all(isinstance(r, dict) and bool(r.get("present")) for r in res[:n])
    return bool(data.get("correct")) if isinstance(data, dict) else False


def score(rows: list[dict]) -> dict:
    before = {(r["unit"], r["entity"]): r["pass"] for r in rows if r["phase"] == "before"}
    out: dict[str, list[int]] = {t: [] for t in TASKS}
    for r in rows:
        if r["phase"] != "after":
            continue
        ok = bool(r["pass"])
        if r["task"] in FILTERED:
            ok = ok and before.get((r["unit"], r["entity"]), False)
        out[r["task"]].append(int(ok))
    acc = {t: (sum(v) / len(v) if v else float("nan")) for t, v in out.items()}
    acc["overall"] = sum(acc[t] for t in TASKS) / len(TASKS)
    acc["n"] = {t: len(v) for t, v in out.items()}
    return acc


def record(eid: str, k: int) -> Record:
    sess = by_id()[eid]["sessions"][k]
    return session_record(sess["conversation"], sess.get("timestamp", ""))


def add_session(store: Store, facts: dict, eid: str, k: int, version: str = "m1") -> None:
    sess = by_id()[eid]["sessions"][k]
    add_facts(store, facts, f"{eid}#{k}", day(sess.get("timestamp", "")), turn_texts(record(eid, k)), version)


def build_store(eid: str, upto: int, facts_of, version: str = "m1") -> Store:
    store = Store()
    for k in range(upto):
        f = facts_of(eid, k)
        if f is not None:
            add_session(store, f, eid, k, version)
    return store
