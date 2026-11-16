from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from graph_memory.extract import Record, Turn, add_extracted
from graph_memory.reader import MONTHS, parse_date
from graph_memory.store import Store

DATA = Path("data/longmemeval/longmemeval_s_cleaned.json")
N_DEV = 100
_BASE = ("a chat session between a user and an AI assistant; each step is one turn: the command is the role (user "
         "or assistant) and the observation is what they said. The user has no name: write 'the user' as the "
         "subject for anything about the user")
WORLD = _BASE + (
        ". Completeness: record EVERY specific detail the user states about themselves and their life -- each name, "
        "place, item, number, amount, price, count, duration, score, size and date is its own fact; a turn often "
        "holds several facts, never merge or skip them. When the user reports a change (moved, switched, replaced, "
        "upgraded, now, no longer, a new personal best), record the new state as its own fact. For the assistant's "
        "turns, record the specific names, items, numbers and recommendations it gave (subject: the assistant). "
        "Time: write every time as an absolute date resolved against the session date at the top of the record: "
        "YYYY-MM-DD for a day, the month name and year for a month ('July 2022'), the year alone for a year "
        "('2020'), followed by the original words in parentheses when they were relative, e.g. '2023-05-13 (last "
        "week)', 'July 2022 (last summer)', '2020 (a few years ago)', '2023-05-27 (this weekend)'; 'recently', "
        "'lately', 'today' and 'just' mean the session date")
KINDS = ("events",)
CHARS = 1000
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")


@lru_cache(maxsize=1)
def load() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))


def split() -> tuple[list[str], list[str]]:
    ids = sorted((q["question_id"] for q in load()), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    return ids[:N_DEV], ids[N_DEV:]


SAMPLE = {"knowledge-update": (10, 25), "temporal-reasoning": (10, 25), "multi-session": (10, 25),
          "single-session-user": (5, 13), "single-session-assistant": (5, 12)}


def sample() -> tuple[list[str], list[str]]:
    dev, formal = split()
    t = {q["question_id"]: q["question_type"] for q in load()}
    pick = lambda ids, k: [q for typ, n in SAMPLE.items() for q in [x for x in ids if t[x] == typ][: n[k]]]
    return pick(dev, 0), pick(formal, 1)


@lru_cache(maxsize=1)
def by_id() -> dict[str, dict]:
    return {q["question_id"]: q for q in load()}


@lru_cache(maxsize=1)
def sessions() -> dict[str, tuple[str, list[dict]]]:
    out = {}
    for q in load():
        for sid, d, s in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            out.setdefault(sid, (d, s))
    return out


def judge_prompt(qtype: str, question: str, answer: str, response: str, abstention: bool) -> str:
    if not abstention:
        if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
            template = ("I will give you a question, a correct answer, and a response from a model. Please answer yes "
                        "if the response contains the correct answer. Otherwise, answer no. If the response is "
                        "equivalent to the correct answer or contains all the intermediate steps to get the correct "
                        "answer, you should also answer yes. If the response only contains a subset of the "
                        "information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
                        "Model Response: {}\n\nIs the model response correct? Answer yes or no only.")
        elif qtype == "temporal-reasoning":
            template = ("I will give you a question, a correct answer, and a response from a model. Please answer yes "
                        "if the response contains the correct answer. Otherwise, answer no. If the response is "
                        "equivalent to the correct answer or contains all the intermediate steps to get the correct "
                        "answer, you should also answer yes. If the response only contains a subset of the "
                        "information required by the answer, answer no. In addition, do not penalize off-by-one "
                        "errors for the number of days. If the question asks for the number of days/weeks/months, "
                        "etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is "
                        "18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
                        "Model Response: {}\n\nIs the model response correct? Answer yes or no only.")
        elif qtype == "knowledge-update":
            template = ("I will give you a question, a correct answer, and a response from a model. Please answer yes "
                        "if the response contains the correct answer. Otherwise, answer no. If the response contains "
                        "some previous information along with an updated answer, the response should be considered "
                        "as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\n"
                        "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no "
                        "only.")
        elif qtype == "single-session-preference":
            template = ("I will give you a question, a rubric for desired personalized response, and a response from a "
                        "model. Please answer yes if the response satisfies the desired response. Otherwise, answer "
                        "no. The model does not need to reflect all the points in the rubric. The response is correct "
                        "as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: "
                        "{}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no "
                        "only.")
        else:
            raise NotImplementedError(qtype)
    else:
        template = ("I will give you an unanswerable question, an explanation, and a response from a model. Please "
                    "answer yes if the model correctly identifies the question as unanswerable. The model could say "
                    "that the information is incomplete, or some other information is given but the asked "
                    "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model "
                    "correctly identify the question as unanswerable? Answer yes or no only.")
    return template.format(question, answer, response)


def v3_time(time: str, day: str) -> str:
    def month(m: re.Match) -> str:
        k = int(m.group(2))
        return f"{MONTH_NAMES[k - 1]} {m.group(1)}" if 1 <= k <= 12 else m.group(0)
    t = re.sub(r"\b(\d{4})-(\d{1,2})\b(?!-\d)", month, str(time or "")).strip()
    if parse_date(t):
        return t
    return f"{t} (said {day})" if t else f"{day} (said)"


_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DAYS = (
    (re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r",?\s+(\d{4})", re.I),
     lambda m: (int(m[3]), MONTHS[m[2].lower()], int(m[1]))),
    (re.compile(r"\b" + _MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I),
     lambda m: (int(m[3]), MONTHS[m[1].lower()], int(m[2]))),
)
_MONTH_YEAR = re.compile(r"\b" + _MON + r",?\s+(\d{4})\b", re.I)


def shift_time(time: str, days: int) -> str:
    if not days or not time:
        return time

    def day(get):
        def sub(m: re.Match) -> str:
            try:
                return (date(*get(m)) + timedelta(days=days)).isoformat()
            except ValueError:
                return m.group(0)
        return sub

    def month(m: re.Match) -> str:
        d = date(int(m[2]), MONTHS[m[1].lower()], 15) + timedelta(days=days)
        return f"{MONTH_NAMES[d.month - 1]} {d.year}"
    for pat, get in _DAYS:
        time = pat.sub(day(get), time)
    return _MONTH_YEAR.sub(month, time)


def text_of(turn: dict) -> str:
    t = turn["content"]
    return t if turn["role"] == "user" else t[:CHARS]


def record(sid: str) -> Record:
    d, turns = sessions()[sid]
    return Record(f"Chat session dated {d} between the user and an AI assistant.",
                  [Turn(x["role"], text_of(x)) for x in turns])


def add_session(store: Store, facts: dict, sid: str, when: str = "") -> None:
    ref, turns = sessions()[sid]
    ids = [f"{sid}#start"] + [f"{sid}#{i}" for i in range(len(turns))]
    gap = (parse_date(when)[0] - parse_date(ref)[0]).days if when else 0
    events = []
    for x in facts.get("events", []):
        x = x | {"steps": [ids[i] for i in x["steps"] if 0 <= i < len(ids)]}
        x["time"] = shift_time(v3_time(x.get("time", ""), ref.split(" ")[0]), gap)
        events.append(x)
    add_extracted(store, dict(facts) | {"events": events}, sid)


def build_store(qid: str, facts_of) -> Store:
    q = by_id()[qid]
    store = Store()
    for sid, when in zip(q["haystack_session_ids"], q["haystack_dates"]):
        f = facts_of(sid)
        if f is not None:
            add_session(store, f, sid, when)
    return store


def raw_turns(qid: str) -> list[str]:
    q = by_id()[qid]
    out = []
    for sid, d, s in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        out += [f"[{sid}#{i}] ({d}) {x['role']}: {text_of(x)}" for i, x in enumerate(s)]
    return out


def answer_turns(qid: str) -> set[str]:
    q = by_id()[qid]
    return {f"{sid}#{i}" for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"])
            for i, x in enumerate(s) if x.get("has_answer")}
