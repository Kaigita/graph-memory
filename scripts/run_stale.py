from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_benchmarks import stale as STALE
from graph_benchmarks.common import add_llm_args, clients, dump, totals
from graph_memory.reader import Reader
from graph_memory.store import Store, fact_lines

EMPTY = "(no relevant facts)"
ARMS = ("C", "Fa", "Gr", "G")
FIX = ("Your reply was not a valid JSON object with dim1_eval, dim2_eval and dim3_eval, each holding a boolean "
       "\"pass\". Return the whole JSON object only.")


class Memory:

    def __init__(self, out: Path):
        self.dir = out / f"facts_{STALE.VERSION}"
        self._stores: dict[str, Store] = {}

    def facts(self, uid: str, k: int) -> dict | None:
        p = self.dir / uid / f"{k:02d}.json"
        return json.loads(p.read_text(encoding="utf-8"))["facts"] if p.exists() else None

    def missing(self, uid: str) -> list[int]:
        return [k for k in range(len(STALE.by_uid()[uid]["haystack_session"])) if self.facts(uid, k) is None]

    def store(self, uid: str) -> Store:
        if uid not in self._stores:
            n = len(STALE.by_uid()[uid]["haystack_session"])
            self._stores[uid] = STALE.build_store(uid, n, self.facts)
        return self._stores[uid]


async def main(args: argparse.Namespace) -> None:
    STALE.DATA = Path(args.data)
    out = Path(args.out)
    dev, formal = STALE.split()
    uids = {"dev": dev, "formal": formal}[args.split]
    if args.subset:
        uids = STALE.subset(uids, args.subset)
    if args.only:
        uids = [u for u in uids if u in args.only.split(",")]
    if args.first:
        uids = uids[: args.first]
    mem = Memory(out)
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        sys.exit(f"unknown arms {bad}; choose from {','.join(ARMS)}")
    if any(a in ("Fa", "Gr", "G") for a in arms):
        lost = {u: mem.missing(u) for u in uids}
        lost = {u: v for u, v in lost.items() if v}
        if lost:
            sys.exit(f"sessions not extracted yet: {len(lost)} scenarios, e.g. {next(iter(lost.items()))}")
    t0 = time.monotonic()
    async with AsyncExitStack() as stack:
        cs = clients(args, {"answer": (False, {"max_tokens": 4096}), "judge": (True, {"max_tokens": 2000}),
                            "judge_retry": (True, {"max_tokens": 2000, "temperature": 0.7})}, out)
        for c in cs.values():
            await stack.enter_async_context(c)
        llm, judge, judge_retry = cs["answer"], cs["judge"], cs["judge_retry"]

        def context(arm: str, uid: str, q: str) -> tuple[str, dict]:
            sc = STALE.by_uid()[uid]
            if arm == "C":
                return STALE.format_haystack(sc["haystack_session"], sc.get("timestamps", [])), {}
            store = mem.store(uid)
            if arm == "Fa":
                return "\n".join(f"- {x}" for x in fact_lines(store)), {}
            rd = Reader(store, propagate=arm == "G", inferred=args.use_inferred, profile_chars=args.profile_chars)
            return rd.brief(q, args.budget)

        async def answer(arm: str, uid: str, d: str) -> dict:
            sc = STALE.by_uid()[uid]
            key = f"{d}_query"
            q = sc["probing_queries"][key]
            text, meta = context(arm, uid, q)
            text = text.strip() or EMPTY
            sys_p, user_p = (STALE.build_prompts(text, q, key) if arm == "C"
                             else STALE.memory_prompts(text, q, key))
            res = await llm.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
            return {"answer": (res.content or "").strip(), "context_chars": len(text),
                    "prompt_tokens": (res.usage or {}).get("prompt_tokens", 0),
                    "completion_tokens": (res.usage or {}).get("completion_tokens", 0),
                    "finish_reason": res.finish_reason, "meta": meta,
                    "context": text if arm in ("Gr", "G") else ""}

        async def verdict(uid: str, answers: dict) -> tuple[dict, str, int]:
            sc = STALE.by_uid()[uid]
            msgs = [{"role": "system", "content": STALE.SYSTEM_PROMPT_ALL_IN_ONE_JUDGE},
                    {"role": "user", "content": STALE.judge_user(sc, answers["dim1"], answers["dim2"],
                                                                 answers["dim3"])}]
            tries = 0
            for client in (judge, judge_retry):
                res = await client.chat(msgs)
                tries += 1
                got = STALE.parse_judge(res.content or "")
                if got is None:
                    fix = await client.chat(msgs + [{"role": "assistant", "content": res.content or ""},
                                                    {"role": "user", "content": FIX}])
                    tries += 1
                    got = STALE.parse_judge(fix.content or "")
                    res = fix if got is not None else res
                if got is not None:
                    return got, res.content or "", tries
            return {d: False for d in STALE.DIMS}, "unparsed: " + (res.content or "")[:2000], tries

        summary = {}
        for arm in arms:
            path = out / arm / "answers.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            done = {json.loads(x)["uid"] for x in path.open(encoding="utf-8")} if path.exists() else set()
            todo = [u for u in uids if u not in done]
            print(f"[{arm}] {len(todo)} of {len(uids)} scenarios to answer", flush=True)
            lock = asyncio.Lock()
            with path.open("a", encoding="utf-8") as f:

                async def one(uid: str) -> None:
                    try:
                        got = await asyncio.gather(*(answer(arm, uid, d) for d in STALE.DIMS))
                        res = dict(zip(STALE.DIMS, got))
                        passed, raw, tries = await verdict(uid, {d: res[d]["answer"] for d in STALE.DIMS})
                    except Exception as exc:
                        print(f"[{arm}] {uid}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                        return
                    sc = STALE.by_uid()[uid]
                    row = {"uid": uid, "type": sc["type"], "arm": arm, "pass": passed, "judge": raw,
                           "judge_tries": tries, "dims": res}
                    async with lock:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()

                await asyncio.gather(*(one(u) for u in todo))
            rows = [json.loads(x) for x in path.open(encoding="utf-8")]
            rows = [r for r in rows if r["uid"] in set(uids)]
            summary[arm] = STALE.score(rows)
            print(f"[{arm}] " + json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                                            for k, v in summary[arm].items()}), flush=True)
        tag = args.split + (f"_sub{args.subset}" if args.subset else "")
        dump(out / f"summary_{tag}.json",
             {"split": args.split, "scenarios": len(uids), "arms": arms, "scores": summary,
              "budget": args.budget, "profile_chars": args.profile_chars, "inferred": args.use_inferred,
              "answer_llm": llm.totals, "judge_llm": totals(judge, judge_retry),
              "minutes": round((time.monotonic() - t0) / 60, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Answer STALE's three probes per scenario once per arm, then judge.")
    ap.add_argument("--data", default=str(STALE.DATA))
    ap.add_argument("--out", default="runs/stale", help="the extraction output folder; answers go to <out>/<arm>")
    ap.add_argument("--split", choices=["dev", "formal"], default="formal")
    ap.add_argument("--subset", type=int, default=0, help="only the first N scenarios of each type")
    ap.add_argument("--arms", default="G,Gr,Fa", help=f"comma-separated, from {','.join(ARMS)}")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--first", type=int, default=0, help="only the first N scenarios")
    ap.add_argument("--budget", type=int, default=18000, help="G / Gr memory text budget in characters")
    ap.add_argument("--profile-chars", type=int, default=16000, help="G / Gr cap of the profile section")
    ap.add_argument("--use-inferred", action="store_true", help="G / Gr also use the rules marked said = no")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
