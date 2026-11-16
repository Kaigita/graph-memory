from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_benchmarks import meme as MEME
from graph_benchmarks.baselines import bm25, cap
from graph_benchmarks.common import add_llm_args, clients, dump
from graph_memory.chat import VERSIONS
from graph_memory.reader import Reader
from graph_memory.store import Store, fact_lines

BUDGET = 6000
R_TOP = 5
T_FIRST, T_ROUNDS, T_QUERIES, T_PER_QUERY = 15, 3, 3, 5
EMPTY = "(no relevant facts)"
ARMS = ("R", "C", "F", "Fa", "T", "Gr", "G")
PLAN = """You are looking up a user's memory to answer a question about them.

Question: {question}

Memory records found so far:
{records}

Could the correct answer depend on records not shown yet? For example, a record may say that the asked value depends on something else ("is determined by ...", "if ... changes ...") -- then look up whether that other thing changed later, and whether what it depends on changed; or the asked value may have been changed or removed later.
If more records are needed, write up to 3 short search queries, one per line, and nothing else. If the records shown are enough, write only: DONE"""


class Memory:

    def __init__(self, out: Path, version: str):
        self.dir = out / f"facts_{version}"
        self.version = version
        self._facts: dict[tuple[str, int], dict | None] = {}
        self._stores: dict[tuple[str, int], Store] = {}
        self._lines: dict[tuple[str, int], list[str]] = {}

    def facts(self, eid: str, k: int) -> dict | None:
        if (eid, k) not in self._facts:
            p = self.dir / eid / f"{k:02d}.json"
            self._facts[(eid, k)] = json.loads(p.read_text(encoding="utf-8"))["facts"] if p.exists() else None
        return self._facts[(eid, k)]

    def missing(self, eid: str, upto: int) -> list[int]:
        return [k for k in range(upto) if self.facts(eid, k) is None]

    def store(self, eid: str, upto: int) -> Store:
        if (eid, upto) not in self._stores:
            self._stores[(eid, upto)] = MEME.build_store(eid, upto, self.facts, self.version)
        return self._stores[(eid, upto)]

    def lines(self, eid: str, upto: int) -> list[str]:
        if (eid, upto) not in self._lines:
            self._lines[(eid, upto)] = fact_lines(self.store(eid, upto))
        return self._lines[(eid, upto)]


def bullets(lines: list[str]) -> str:
    return "\n".join(f"- {x}" for x in lines)


async def main(args: argparse.Namespace) -> None:
    MEME.DATA = Path(args.data)
    out = Path(args.out)
    dev, formal = MEME.split()
    units = {"dev": dev, "formal": formal}[args.split]
    if args.only:
        units = [u for u in units if u in args.only.split(",")]
    mem = Memory(out, args.version)
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        sys.exit(f"unknown arms {bad}; choose from {','.join(ARMS)}")
    if any(a not in ("R", "C") for a in arms):
        lost = {u: mem.missing(u, MEME.points(MEME.by_id()[u])[1]) for u in units}
        lost = {u: v for u, v in lost.items() if v}
        if lost:
            sys.exit(f"sessions not extracted yet: {lost}")
    t0 = time.monotonic()
    async with AsyncExitStack() as stack:
        cs = clients(args, {"answer": (False, {"max_tokens": 500}), "plan": (False, {"max_tokens": 200}),
                            "judge": (True, {"max_tokens": 300})}, out)
        for c in cs.values():
            await stack.enter_async_context(c)
        llm, planner, judge = cs["answer"], cs["plan"], cs["judge"]
        readers: dict[tuple[str, int, bool], Reader] = {}

        def reader(eid: str, upto: int, propagate: bool) -> Reader:
            key = (eid, upto, propagate)
            if key not in readers:
                readers[key] = Reader(mem.store(eid, upto), propagate=propagate)
            return readers[key]

        async def t_context(eid: str, upto: int, question: str) -> tuple[str, list[dict]]:
            lines = mem.lines(eid, upto)
            order = {x: i for i, x in reversed(list(enumerate(lines)))}
            picked = bm25(lines, question)[:T_FIRST]
            calls = []
            for _ in range(T_ROUNDS):
                recs = bullets(sorted(picked, key=order.get)) or EMPTY
                res = await planner.chat([{"role": "user", "content": PLAN.format(question=question, records=recs)}])
                text = (res.content or "").strip()
                calls.append({"reply": text, "prompt_tokens": (res.usage or {}).get("prompt_tokens", 0),
                              "completion_tokens": (res.usage or {}).get("completion_tokens", 0)})
                queries = [q.strip(" -*•\t\"'") for q in text.splitlines() if q.strip()]
                queries = [q for q in queries if q.upper().rstrip(".") != "DONE"][:T_QUERIES]
                if not queries:
                    break
                new = []
                for q in queries:
                    for x in bm25(lines, q)[:T_PER_QUERY]:
                        if x not in picked and x not in new:
                            new.append(x)
                if not new:
                    break
                picked += new
            kept, used = [], 0
            for x in picked:
                if used + len(x) + 3 > BUDGET:
                    break
                kept.append(x)
                used += len(x) + 3
            return bullets(sorted(kept, key=order.get)), calls

        async def context(arm: str, it: dict) -> tuple[str, dict]:
            ep = MEME.by_id()[it["unit"]]
            upto = MEME.points(ep)[0 if it["phase"] == "before" else 1]
            q = it["question"]
            if arm == "R":
                return "\n\n---\n\n".join(bm25(MEME.chunks(it["unit"], upto), q)[:R_TOP]), {}
            if arm == "C":
                return MEME.transcript(it["unit"], upto), {}
            if arm == "F":
                return cap(bm25(mem.lines(it["unit"], upto), q), BUDGET), {}
            if arm == "Fa":
                return bullets(mem.lines(it["unit"], upto)), {}
            if arm == "T":
                text, calls = await t_context(it["unit"], upto, q)
                return text, {"plan": calls}
            return reader(it["unit"], upto, arm == "G").brief(q, BUDGET)

        items = [it for u in units for it in MEME.items(u)]
        summary = {}
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
                        text, meta = await context(arm, it)
                        text = text.strip() or EMPTY
                        res = await llm.chat([{"role": "user", "content": MEME.ANSWER.format(
                            context=text, question=it["question"])}])
                        answer = (res.content or "").strip()
                        req = MEME.judge_request(it, answer)
                        if it["task"] == "ER":
                            ok, verdict = MEME.er_pass(it, answer), "substring"
                        else:
                            j = await judge.chat([{"role": "user", "content": req}])
                            verdict = (j.content or "").strip()
                            ok = MEME.parse_verdict(it, verdict)
                    except Exception as exc:
                        print(f"[{arm}] {it['qid']}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                        return
                    row = {k: it[k] for k in ("qid", "unit", "phase", "task", "hop", "entity", "question", "gold")}
                    row.update({"arm": arm, "answer": answer, "pass": int(ok), "verdict": verdict,
                                "context_chars": len(text),
                                "prompt_tokens": (res.usage or {}).get("prompt_tokens", 0),
                                "completion_tokens": (res.usage or {}).get("completion_tokens", 0),
                                "finish_reason": res.finish_reason, "meta": meta,
                                "context": text if arm not in ("R", "C") else ""})
                    async with lock:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()

                await asyncio.gather(*(one(it) for it in todo))
            rows = [json.loads(x) for x in path.open(encoding="utf-8")]
            rows = [r for r in rows if r["unit"] in set(units)]
            summary[arm] = MEME.score(rows)
            print(f"[{arm}] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                            for k, v in summary[arm].items()}), flush=True)
        dump(out / f"summary_{args.split}.json",
             {"split": args.split, "units": len(units), "arms": arms, "scores": summary,
              "answer_llm": llm.totals, "plan_llm": planner.totals, "judge_llm": judge.totals,
              "minutes": round((time.monotonic() - t0) / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Answer and judge every MEME question once per memory arm.")
    ap.add_argument("--data", default=str(MEME.DATA))
    ap.add_argument("--out", default="runs/meme", help="the extraction output folder; answers go to <out>/<arm>")
    ap.add_argument("--split", choices=["dev", "formal"], default="formal")
    ap.add_argument("--arms", default="G,Gr,Fa", help=f"comma-separated, from {','.join(ARMS)}")
    ap.add_argument("--only", default="", help="comma-separated episode ids")
    ap.add_argument("--version", choices=VERSIONS, default="m1", help="extractor version of the facts")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
