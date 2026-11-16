from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_benchmarks import locomo as LOCOMO
from graph_benchmarks import longmemeval as LME
from graph_benchmarks.baselines import flat_brief, raw_brief
from graph_benchmarks.common import add_llm_args, clients, dump
from graph_memory.reader import Reader
from graph_memory.store import Store

ARMS = ("N", "R", "C", "F", "G")
NOTHING = "(nothing relevant recorded)"
ANSWER_LOCOMO = ("You answer questions about a long conversation between {a} and {b}. Use only the memory below. "
                 "Reply with a short answer (a few words; for 'when' give the date), no explanation. If the memory "
                 "does not contain the answer, reply exactly: not mentioned.")
ANSWER_LME = ("You answer questions about the user's past chat sessions with an AI assistant. The current date is "
              "{now}. Use only the memory below. Reply with a short answer (a few words or one sentence), no "
              "explanation. If the memory does not contain the answer, reply exactly: not mentioned.")
JUDGE_LOCOMO = ("You grade an answer to a question about a conversation. Given the question, the gold answer and the "
                "response, reply CORRECT if the response states the same thing as the gold answer (paraphrases, extra "
                "detail and other date formats are fine; for dates the same day, month or year as asked), otherwise "
                "WRONG. Reply with one word.")


class Locomo:
    budget = 4000

    def __init__(self, out: Path):
        self.dev, self.formal = LOCOMO.split()
        self.dir = out / "stores"
        self._stores: dict[str, Store] = {}

    def units(self, split: str) -> list[str]:
        return {"dev": self.dev, "formal": self.formal}[split]

    def missing(self, unit: str) -> bool:
        return not (self.dir / f"{unit}.json").exists()

    def items(self, unit: str) -> list[dict]:
        c = LOCOMO.conv(unit)
        a, b = c["conversation"]["speaker_a"], c["conversation"]["speaker_b"]
        return [q | {"unit": unit, "qtype": str(q["category"]), "now": "", "base": ANSWER_LOCOMO.format(a=a, b=b)}
                for q in LOCOMO.questions(c)]

    def store(self, unit: str) -> Store:
        if unit not in self._stores:
            p = self.dir / f"{unit}.json"
            self._stores[unit] = Store.from_json(json.loads(p.read_text(encoding="utf-8")))
        return self._stores[unit]

    def raw(self, unit: str) -> list[str]:
        return LOCOMO.raw_turns(LOCOMO.conv(unit))

    def full(self, unit: str) -> str | None:
        return LOCOMO.full_text(LOCOMO.conv(unit))

    def evidence(self, it: dict) -> set[str]:
        return set(it.get("evidence") or [])

    async def grade(self, it: dict, pred: str, judge) -> dict:
        row = {"score": LOCOMO.score(pred, it["answer"], it["category"])}
        if it["category"] == 5:
            row["judge"] = int(row["score"] > 0)
        else:
            j = await judge.chat([{"role": "system", "content": JUDGE_LOCOMO},
                                  {"role": "user", "content": f"Question: {it['question']}\nGold answer: "
                                                              f"{it['answer']}\nResponse: {pred}"}])
            row["judge"] = int("CORRECT" in (j.content or "").upper())
        return row


class LongMemEval:
    budget = 6000

    def __init__(self, out: Path):
        self.dev, self.formal = LME.sample()
        self.dir = out / "sessions"
        self._facts: dict[str, dict | None] = {}

    def units(self, split: str) -> list[str]:
        return {"dev": self.dev, "formal": self.formal}[split]

    def missing(self, unit: str) -> bool:
        return any(self.facts(s) is None for s in LME.by_id()[unit]["haystack_session_ids"])

    def items(self, unit: str) -> list[dict]:
        q = LME.by_id()[unit]
        return [{"qid": unit, "unit": unit, "qtype": q["question_type"], "question": q["question"],
                 "answer": str(q["answer"]), "abstention": "_abs" in unit, "now": q["question_date"],
                 "base": ANSWER_LME.format(now=q["question_date"])}]

    def facts(self, sid: str) -> dict | None:
        if sid not in self._facts:
            p = self.dir / f"{sid}.json"
            self._facts[sid] = json.loads(p.read_text(encoding="utf-8"))["facts"] if p.exists() else None
        return self._facts[sid]

    def store(self, unit: str) -> Store:
        return LME.build_store(unit, self.facts)

    def raw(self, unit: str) -> list[str]:
        return LME.raw_turns(unit)

    def full(self, unit: str) -> str | None:
        return None

    def evidence(self, it: dict) -> set[str]:
        return LME.answer_turns(it["qid"])

    async def grade(self, it: dict, pred: str, judge) -> dict:
        prompt = LME.judge_prompt(it["qtype"], it["question"], it["answer"], pred, it["abstention"])
        j = await judge.chat([{"role": "user", "content": prompt}])
        ok = int("yes" in (j.content or "").lower())
        return {"score": ok, "judge": ok}


def first_line(text: str | None) -> str:
    t = (text or "").strip()
    return t.splitlines()[0].strip() if t else ""


async def main(args: argparse.Namespace) -> None:
    if args.bench == "locomo":
        LOCOMO.DATA = Path(args.data or LOCOMO.DATA)
    else:
        LME.DATA = Path(args.data or LME.DATA)
    out = Path(args.out or f"runs/{args.bench}")
    bench = Locomo(out) if args.bench == "locomo" else LongMemEval(out)
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS or (a == "C" and args.bench != "locomo")]
    if bad:
        sys.exit(f"unknown arms {bad}; choose from {','.join(ARMS)} (C: locomo only)")
    units = bench.units(args.split)
    if args.only:
        units = [u for u in units if u in args.only.split(",")]
    if any(a in ("F", "G") for a in arms):
        lost = [u for u in units if bench.missing(u)]
        if lost:
            sys.exit(f"not extracted yet: {len(lost)} units, e.g. {lost[0]}")
    items = [it for u in units for it in bench.items(u)]
    if args.sample:
        seen: dict[str, int] = {}
        keep = []
        for it in items:
            seen[it["qtype"]] = seen.get(it["qtype"], 0) + 1
            if seen[it["qtype"]] <= args.sample:
                keep.append(it)
        items = keep
    readers: dict[str, Reader] = {}

    def reader(unit: str) -> Reader:
        if unit not in readers:
            if len(readers) > 64:
                readers.clear()
            readers[unit] = Reader(bench.store(unit))
        return readers[unit]

    t0 = time.monotonic()
    async with AsyncExitStack() as stack:
        cs = clients(args, {"answer": (False, {"max_tokens": 120}), "judge": (True, {"max_tokens": 8})}, out)
        for c in cs.values():
            await stack.enter_async_context(c)
        llm, judge = cs["answer"], cs["judge"]

        def material(arm: str, it: dict) -> tuple[str, dict]:
            if arm == "N":
                return "(no memory)", {}
            if arm == "C":
                return bench.full(it["unit"]) or "(no memory)", {}
            if arm == "R":
                return raw_brief(bench.raw(it["unit"]), it["question"], bench.budget) or NOTHING, {}
            if arm == "F":
                return flat_brief(bench.store(it["unit"]), it["question"], bench.budget) or NOTHING, {}
            text, meta = reader(it["unit"]).brief(it["question"], bench.budget, it["now"])
            return text or NOTHING, meta

        for arm in arms:
            path = out / arm / "answers.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(x)["qid"] for x in path.open(encoding="utf-8")} if path.exists() else set()
            todo = [it for it in items if it["qid"] not in done]
            print(f"[{arm}] {len(todo)} of {len(items)} to answer", flush=True)
            lock = asyncio.Lock()
            with path.open("a", encoding="utf-8") as f:

                async def one(it: dict) -> None:
                    try:
                        text, meta = material(arm, it)
                        res = await llm.chat([{"role": "system", "content": it["base"]},
                                              {"role": "user", "content": f"Memory:\n{text}\n\nQuestion: "
                                                                          f"{it['question']}\nAnswer:"}])
                        pred = first_line(res.content)
                        graded = await bench.grade(it, pred, judge)
                    except Exception as exc:
                        print(f"[{arm}] {it['qid']}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                        return
                    row = {k: it[k] for k in ("qid", "unit", "qtype", "question", "answer") if k in it}
                    row.update({"arm": arm, "prediction": pred, "memory_chars": len(text),
                                "prompt_tokens": int((res.usage or {}).get("prompt_tokens") or 0),
                                "memory": text if arm in ("R", "F", "G") else "", "meta": meta,
                                "evidence": sorted(bench.evidence(it))} | graded)
                    async with lock:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()

                await asyncio.gather(*(one(it) for it in todo))
        scores = {}
        for arm in arms:
            rows = [json.loads(x) for x in (out / arm / "answers.jsonl").open(encoding="utf-8")]
            rows = [r for r in rows if r["unit"] in set(units)]
            by: dict[str, list[float]] = {}
            for r in rows:
                by.setdefault(r["qtype"], []).append(r["score"])
            scores[arm] = {"n": len(rows), "overall": sum(r["score"] for r in rows) / max(1, len(rows)),
                           **{t: sum(v) / len(v) for t, v in sorted(by.items())}}
            print(f"[{arm}] " + json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                            for k, v in scores[arm].items()}), flush=True)
        dump(out / f"summary_{args.split}.json",
             {"bench": args.bench, "split": args.split, "arms": arms, "items": len(items), "scores": scores,
              "answer_llm": llm.totals, "judge_llm": judge.totals,
              "minutes": round((time.monotonic() - t0) / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Answer LoCoMo or LongMemEval questions once per memory arm.")
    ap.add_argument("--bench", choices=["locomo", "longmemeval"], required=True)
    ap.add_argument("--data", default="", help="default: the benchmark's data path")
    ap.add_argument("--out", default="", help="the extraction output folder (default runs/<bench>)")
    ap.add_argument("--split", choices=["dev", "formal"], default="formal")
    ap.add_argument("--arms", default="R,F,G", help=f"comma-separated, from {','.join(ARMS)} (C: locomo only)")
    ap.add_argument("--only", default="", help="comma-separated unit ids")
    ap.add_argument("--sample", type=int, default=0, help="only the first N questions of each type")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
