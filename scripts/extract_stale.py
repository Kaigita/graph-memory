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
from graph_memory.chat import KINDS, parts, world
from graph_memory.extract import extract
from graph_memory.store import Store

RETRIES = 2


def session_file(out: Path, uid: str, k: int) -> Path:
    return out / f"facts_{STALE.VERSION}" / uid / f"{k:02d}.json"


async def main(args: argparse.Namespace) -> None:
    STALE.DATA = Path(args.data)
    out = Path(args.out)
    cache = out / "cache"
    dev, formal = STALE.split()
    uids = {"dev": dev, "formal": formal, "both": dev + formal}[args.split]
    if args.only:
        uids = [u for u in uids if u in args.only.split(",")]
    if args.first:
        uids = uids[: args.first]
    sem = asyncio.Semaphore(args.llm_concurrency)
    fails: list[str] = []
    done, t0 = 0, time.monotonic()
    async with AsyncExitStack() as stack:
        cs = clients(args, {"extract": (False, {"temperature": 0.2, "max_tokens": 8192}),
                            "extract_retry": (False, {"temperature": 0.7, "max_tokens": 8192})}, out)
        for c in cs.values():
            await stack.enter_async_context(c)
        llm, llm_retry = cs["extract"], cs["extract_retry"]

        async def scenario(uid: str) -> None:
            nonlocal done
            async with sem:
                store = Store()
                for k in range(len(STALE.by_uid()[uid]["haystack_session"])):
                    p = session_file(out, uid, k)
                    if p.exists():
                        facts = json.loads(p.read_text(encoding="utf-8"))["facts"]
                    else:
                        ok, tries = False, 0
                        for client in [llm] + [llm_retry] * RETRIES:
                            log: list[dict] = []
                            tries += 1
                            try:
                                facts = await extract(STALE.record(uid, k), client, cache,
                                                      world(store, STALE.VERSION), KINDS, Store(), log=log.append,
                                                      parts=parts(STALE.VERSION))
                            except Exception as exc:
                                print(f"{uid}#{k}: {type(exc).__name__}: {exc}", flush=True)
                                fails.append(f"{uid}#{k}")
                                return
                            ok = not (log and not log[-1].get("ok"))
                            if ok:
                                break
                        if not ok:
                            fails.append(f"{uid}#{k}")
                        dump(p, {"uid": uid, "k": k, "facts": facts, "ok": ok, "tries": tries,
                                 "version": STALE.VERSION})
                    STALE.add_session(store, facts, uid, k)
                done += 1
                print(f"{uid} done ({done}/{len(uids)}), {len(store.events)} facts, {len(store.rules)} rules, "
                      f"{time.monotonic() - t0:.0f}s", flush=True)

        await asyncio.gather(*(scenario(u) for u in uids))
        llm_totals = totals(llm, llm_retry)
    n_sess = bad = 0
    for uid in uids:
        for k in range(len(STALE.by_uid()[uid]["haystack_session"])):
            p = session_file(out, uid, k)
            if p.exists():
                n_sess += 1
                bad += not json.loads(p.read_text(encoding="utf-8"))["ok"]
    summary = {"version": STALE.VERSION, "scenarios": len(uids), "sessions": n_sess, "failed": bad,
               "llm": llm_totals, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.only and not args.first:
        dump(out / f"extract_summary_{args.split}_{STALE.VERSION}.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract facts and rules from every STALE session, in order.")
    ap.add_argument("--data", default=str(STALE.DATA))
    ap.add_argument("--out", default="runs/stale")
    ap.add_argument("--split", choices=["dev", "formal", "both"], default="dev")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--first", type=int, default=0, help="only the first N scenarios of the split")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
