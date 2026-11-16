from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_benchmarks import longmemeval as LME
from graph_benchmarks.common import add_llm_args, clients, dump, totals
from graph_memory.extract import extract
from graph_memory.store import Store

RETRIES = 2


async def main(args: argparse.Namespace) -> None:
    LME.DATA = Path(args.data)
    out = Path(args.out)
    out_dir, cache = out / "sessions", out / "cache"
    dev, formal = LME.sample()
    of = lambda ids: list(dict.fromkeys(s for q in ids for s in LME.by_id()[q]["haystack_session_ids"]))
    d, f = of(dev), of(formal)
    sids = {"dev": d, "formal": [s for s in f if s not in set(d)], "both": d + [s for s in f if s not in set(d)]}[
        args.split]
    qs = {"dev": dev, "formal": formal, "both": dev + formal}[args.split]
    ans = {sid for q in qs for sid, s in zip(LME.by_id()[q]["haystack_session_ids"], LME.by_id()[q]["haystack_sessions"])
           if any(t.get("has_answer") for t in s)}
    sids = sorted(sids, key=lambda s: s not in ans)
    if args.limit:
        sids = sids[: args.limit]
    if args.redo_failed:
        for s in sids:
            p = out_dir / f"{s}.json"
            if p.exists() and not json.loads(p.read_text(encoding="utf-8"))["ok"]:
                p.unlink()
    todo = [s for s in sids if not (out_dir / f"{s}.json").exists()]
    print(f"{len(sids)} sessions, {len(todo)} to extract", flush=True)
    sem = asyncio.Semaphore(args.llm_concurrency * 2)
    fails, done, t0 = [], 0, time.monotonic()
    async with AsyncExitStack() as stack:
        cs = clients(args, {"extract": (False, {"temperature": 0.2, "max_tokens": 8192}),
                            "extract_retry": (False, {"temperature": 0.7, "max_tokens": 8192})}, out)
        for c in cs.values():
            await stack.enter_async_context(c)
        llm, llm_retry = cs["extract"], cs["extract_retry"]

        async def one(sid: str) -> None:
            nonlocal done
            async with sem:
                ok, tries = False, 0
                for client in [llm] + [llm_retry] * RETRIES:
                    log: list[dict] = []
                    tries += 1
                    try:
                        facts = await extract(LME.record(sid), client, cache, LME.WORLD, LME.KINDS, Store(),
                                              log=log.append)
                    except Exception as exc:
                        print(f"{sid}: {type(exc).__name__}: {exc}", flush=True)
                        fails.append(sid)
                        return
                    ok = not (log and not log[-1].get("ok"))
                    if ok:
                        break
                if not ok:
                    fails.append(sid)
                dump(out_dir / f"{sid}.json", {"sid": sid, "facts": facts, "ok": ok, "tries": tries})
                done += 1
                if done % 200 == 0:
                    print(f"{done}/{len(todo)} done, {len(fails)} failed, {time.monotonic() - t0:.0f}s", flush=True)

        await asyncio.gather(*(one(s) for s in todo))
        llm_totals = totals(llm, llm_retry)
    n_ev = bad = 0
    for s in sids:
        p = out_dir / f"{s}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            n_ev += len(data["facts"].get("events", []))
            bad += not data["ok"]
    summary = {"sessions": len(sids), "failed": bad, "events": n_ev, "llm": llm_totals,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.limit:
        dump(out / f"extract_summary_{args.split}.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract facts from every LongMemEval session the sampled questions use.")
    ap.add_argument("--data", default=str(LME.DATA))
    ap.add_argument("--out", default="runs/longmemeval")
    ap.add_argument("--split", choices=["dev", "formal", "both"], default="dev",
                    help="the sessions of the dev sample, of the formal sample (minus dev's), or both")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sessions")
    ap.add_argument("--redo-failed", action="store_true", help="extract again the sessions whose file has ok: false")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
