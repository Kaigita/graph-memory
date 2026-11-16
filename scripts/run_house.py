from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph_benchmarks import house as H  # noqa: E402
from graph_benchmarks.common import add_llm_args, clients, dump, totals  # noqa: E402
from graph_benchmarks.house_memory import HouseMemory  # noqa: E402

ARMS = ("N", "F", "G")
MODES = {"F": "flat", "G": "graph"}
BUDGETS = {"h1": (40, 60, 1200), "b": (40, 60, 1200), "l": (50, 75, 1600)}


def ep_path(out: Path, task_id: str) -> Path:
    return out / (task_id.replace("/", "__") + ".json")


async def play(ids: list[str], tasks: dict, memory: Any, llm: Any, out: Path, budget: tuple[int, int],
               episodes: int, tag: str) -> list[dict]:
    sem = asyncio.Semaphore(episodes)
    t0, done = time.monotonic(), 0

    async def one(tid: str) -> dict:
        nonlocal done
        path = ep_path(out, tid)
        if path.exists():
            ep = json.loads(path.read_text(encoding="utf-8"))
        else:
            async with sem:
                ep = await H.run_episode(tid, tasks[tid], memory, llm, *budget)
            if ep["end_reason"] != "error":
                dump(path, ep)
        done += 1
        print(f"[{tag}] {done:3d}/{len(ids)} {'OK' if ep['success'] else '--'} {tid} steps={len(ep['steps'])} "
              f"t={time.monotonic() - t0:.0f}s", flush=True)
        return ep

    eps = list(await asyncio.gather(*(one(t) for t in ids)))
    for _ in range(2):
        bad = [e["task_id"] for e in eps if e["end_reason"] == "error"]
        if not bad:
            break
        again = {e["task_id"]: e for e in await asyncio.gather(*(one(t) for t in bad))}
        eps = [again.get(e["task_id"], e) for e in eps]
    return eps


def rate(eps: list[dict]) -> float | None:
    return round(100 * sum(e["success"] for e in eps) / len(eps), 1) if eps else None


def score(out: Path, arms: list[str], data: dict, ids: set[str]) -> dict:
    res: dict[str, Any] = {}
    for a in arms:
        eps = []
        for k in range(data["gather"], len(data["batches"])):
            for tid in data["batches"][k]:
                p = ep_path(out / a / f"b{k + 1}", tid)
                if tid in ids and p.exists():
                    eps.append(json.loads(p.read_text(encoding="utf-8")))
        by_type: dict[str, list[dict]] = {}
        for e in eps:
            by_type.setdefault(e["type"], []).append(e)
        res[a] = {"n": len(eps), "success": rate(eps), "by_type": {t: rate(v) for t, v in sorted(by_type.items())}}
    return res


async def main(args: argparse.Namespace) -> None:
    data = H.load_set(args.set, Path(args.data))
    arms = args.arms.split(",")
    if "N" not in arms or not set(arms) <= set(ARMS):
        raise SystemExit(f"--arms must include N and come from {','.join(ARMS)}: N's episodes feed every memory")
    seeds = data["seeds"][:args.houses] if args.houses else data["seeds"]
    ids = {t for t, s in data["tasks"].items() if s["seed"] in seeds}
    batches = [[t for t in b if t in ids] for b in data["batches"]]
    max_steps, max_calls, brief_chars = BUDGETS[args.set]
    budget = (args.max_steps or max_steps, args.max_calls or max_calls)
    out = Path(args.out) / args.set
    out.mkdir(parents=True, exist_ok=True)
    llms = clients(args, {a: (False, {"temperature": 0.6}) for a in arms}, out / "logs")
    mems: dict[str, Any] = {a: HouseMemory(MODES[a], args.brief_chars or brief_chars) for a in arms if a != "N"}
    for a, m in mems.items():
        if (out / a / "memory" / "latest.json").exists():
            m.load(out / a / "memory" / "latest.json")
    state_path = out / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"done": 0, "batches": []}
    try:
        for k in range(state["done"], len(batches)):
            b, t0 = k + 1, time.monotonic()
            playing = ["N"] if k < data["gather"] else arms
            for a, m in mems.items():
                m.save(out / a / "memory" / f"before_b{b}.json")
            print(f"== batch {b} ({len(batches[k])} tasks): {','.join(playing)}", flush=True)
            got = await asyncio.gather(*(play(batches[k], data["tasks"], mems.get(a), llms[a], out / a / f"b{b}",
                                              budget, args.episodes, a) for a in playing))
            rec: dict[str, Any] = {"batch": b, "arms": {a: {"n": len(r), "won": sum(e["success"] for e in r),
                                                            "errors": sum(e["end_reason"] == "error" for e in r)}
                                                        for a, r in zip(playing, got)}}
            scored = got[playing.index("N")]
            retry = [e["task_id"] for e in scored if not e["success"] and e["end_reason"] != "error"]
            retried = await play(retry, data["tasks"], None, llms["N"], out / "N" / f"b{b}_retry", budget,
                                 args.episodes, "N retry")
            ok = [e for e in scored if e["end_reason"] != "error"]
            okr = [e for e in retried if e["end_reason"] != "error"]
            rec["feed"] = {"scored": len(ok), "retried": len(okr), "retry_won": sum(e["success"] for e in okr)}
            for a, m in mems.items():
                m.add(ok, "s")
                m.add(okr, "r")
                m.save(out / a / "memory" / "latest.json")
                rec[f"{a}_store"] = m.stats()
            rec["wall"] = round(time.monotonic() - t0, 1)
            state["batches"].append(rec)
            state["done"] = k + 1
            dump(state_path, state)
            print(f"== batch {b} done: {json.dumps(rec['arms'])} feed {json.dumps(rec['feed'])}", flush=True)
    finally:
        for c in llms.values():
            await c.aclose()
    if state["done"] >= len(batches):
        res = score(out, arms, data, ids)
        dump(out / "summary.json", {"set": args.set, "houses": seeds, "budget": budget, "scores": res,
                                     "llm": totals(*llms.values())})
        for a in arms:
            print(f"{a}: {res[a]['success']} on {res[a]['n']} test tasks, by type {res[a]['by_type']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(BUDGETS), default="h1")
    ap.add_argument("--data", default=str(H.DATA))
    ap.add_argument("--out", default="runs/house")
    ap.add_argument("--arms", default="N,F,G", help=f"comma-separated, from {','.join(ARMS)}")
    ap.add_argument("--houses", type=int, default=0, help="only the first N houses")
    ap.add_argument("--episodes", type=int, default=8, help="episodes in flight per arm")
    ap.add_argument("--max-steps", type=int, default=0, help="default: 40, or 50 on the l set")
    ap.add_argument("--max-calls", type=int, default=0, help="default: 60, or 75 on the l set")
    ap.add_argument("--brief-chars", type=int, default=0, help="default: 1200, or 1600 on the l set")
    add_llm_args(ap)
    asyncio.run(main(ap.parse_args()))
