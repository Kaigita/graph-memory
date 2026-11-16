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
from graph_benchmarks.common import add_llm_args, clients, dump, totals
from graph_memory.chat import KINDS, VERSIONS, parts, world
from graph_memory.extract import extract
from graph_memory.store import Store

RETRIES = 2


def session_file(out: Path, version: str, eid: str, k: int) -> Path:
    return out / f"facts_{version}" / eid / f"{k:02d}.json"


async def main(args: argparse.Namespace) -> None:
    MEME.DATA = Path(args.data)
    out = Path(args.out)
    cache = out / "cache"
    dev, formal = MEME.split()
    eids = {"dev": dev, "formal": formal, "both": dev + formal}[args.split]
    if args.only:
        eids = [e for e in eids if e in args.only.split(",")]
    sem = asyncio.Semaphore(args.llm_concurrency)
    fails: list[str] = []
    done, t0 = 0, time.monotonic()
    async with AsyncExitStack() as stack:
        cs = clients(args, {"extract": (False, {"temperature": 0.2, "max_tokens": 8192}),
                            "extract_retry": (False, {"temperature": 0.7, "max_tokens": 8192})}, out)
        for c in cs.values():
            await stack.enter_async_context(c)
        llm, llm_retry = cs["extract"], cs["extract_retry"]

        async def episode(eid: str) -> None:
            nonlocal done
            async with sem:
                store = Store()
                for k in range(len(MEME.by_id()[eid]["sessions"])):
                    p = session_file(out, args.version, eid, k)
                    if p.exists():
                        facts = json.loads(p.read_text(encoding="utf-8"))["facts"]
                    else:
                        ok, tries = False, 0
                        for client in [llm] + [llm_retry] * RETRIES:
                            log: list[dict] = []
                            tries += 1
                            try:
                                facts = await extract(MEME.record(eid, k), client, cache, world(store, args.version),
                                                      KINDS, Store(), log=log.append, parts=parts(args.version))
                            except Exception as exc:
                                print(f"{eid}#{k}: {type(exc).__name__}: {exc}", flush=True)
                                fails.append(f"{eid}#{k}")
                                return
                            ok = not (log and not log[-1].get("ok"))
                            if ok:
                                break
                        if not ok:
                            fails.append(f"{eid}#{k}")
                        dump(p, {"episode": eid, "k": k, "facts": facts, "ok": ok, "tries": tries,
                                 "version": args.version})
                    MEME.add_session(store, facts, eid, k, args.version)
                done += 1
                print(f"{eid} done ({done}/{len(eids)}), {len(store.events)} facts, {len(store.rules)} rules, "
                      f"{time.monotonic() - t0:.0f}s", flush=True)

        await asyncio.gather(*(episode(e) for e in eids))
        llm_totals = totals(llm, llm_retry)
    n_sess = bad = 0
    for eid in eids:
        for k in range(len(MEME.by_id()[eid]["sessions"])):
            p = session_file(out, args.version, eid, k)
            if p.exists():
                n_sess += 1
                bad += not json.loads(p.read_text(encoding="utf-8"))["ok"]
    summary = {"version": args.version, "episodes": len(eids), "sessions": n_sess, "failed": bad, "llm": llm_totals,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.only:
        dump(out / f"extract_summary_{args.split}_{args.version}.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract facts and rules from every MEME session, in order.")
    ap.add_argument("--data", default=str(MEME.DATA))
    ap.add_argument("--out", default="runs/meme")
    ap.add_argument("--split", choices=["dev", "formal", "both"], default="dev")
    ap.add_argument("--only", default="", help="comma-separated episode ids")
    ap.add_argument("--version", choices=VERSIONS, default="m1")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
