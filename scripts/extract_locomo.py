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
from graph_benchmarks.common import add_llm_args, clients, dump
from graph_memory.extract import extract
from graph_memory.store import Store


async def extract_conv(sample: str, llm, out: Path) -> Store:
    path = out / "stores" / f"{sample}.json"
    if path.exists():
        return Store.from_json(json.loads(path.read_text(encoding="utf-8")))
    c = LOCOMO.conv(sample)
    store = Store()
    log: list[dict] = []
    for sess in LOCOMO.sessions(c):
        snap = Store.from_json(json.loads(json.dumps(store.to_json())))
        facts = await extract(LOCOMO.record(c, sess), llm, out / "cache", LOCOMO.WORLD, LOCOMO.KINDS, snap,
                              log=log.append)
        LOCOMO.add_session(store, facts, c, sess)
        print(f"{sample} session {sess['n']}: {len(facts.get('events', []))} events (store {len(store.events)})",
              flush=True)
    dump(path, store.to_json())
    dump(out / "stores" / f"{sample}.log.json", log)
    return store


async def main(args: argparse.Namespace) -> None:
    LOCOMO.DATA = Path(args.data)
    out = Path(args.out)
    dev, formal = LOCOMO.split()
    convs = {"dev": dev, "formal": formal, "both": dev + formal}[args.split]
    if args.only:
        convs = [x for x in convs if x in args.only.split(",")]
    t0 = time.monotonic()
    async with AsyncExitStack() as stack:
        cs = clients(args, {"extract": (False, {"temperature": 0.2, "max_tokens": 8192})}, out)
        llm = await stack.enter_async_context(cs["extract"])
        stores = await asyncio.gather(*(extract_conv(s, llm, out) for s in convs))
        summary = {"convs": convs, "events": {s: len(st.events) for s, st in zip(convs, stores)},
                   "llm": dict(llm.totals), "minutes": round((time.monotonic() - t0) / 60, 1),
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(out / f"extract_summary_{args.split}.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract one fact store per LoCoMo conversation, session by session.")
    ap.add_argument("--data", default=str(LOCOMO.DATA))
    ap.add_argument("--out", default="runs/locomo")
    ap.add_argument("--split", choices=["dev", "formal", "both"], default="formal")
    ap.add_argument("--only", default="", help="comma-separated sample ids")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
