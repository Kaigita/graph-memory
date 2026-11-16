from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph_benchmarks.house import CONTAINER_KIND, layer, make_house, solve, state_at  # noqa: E402

GATHER = 3
H1 = {"seeds": [31001, 31002, 31003, 31004], "batches": 9, "max_plan": 30,
      "test": [("static", 2), ("moved", 2), ("locked", 3), ("new", 1)]}
H2 = {"b": {"first": 32001, "max_plan": 30}, "l": {"first": 33001, "max_plan": 50}}
H2_BATCHES, H2_HOUSES = 7, 4
H2_SLOTS = [("L0m", 1), ("L0s", 1), ("L1", 2), ("L2", 2), ("L3", 2)]


def make_spec(seed: int, b: int, target: str, rng: random.Random) -> dict:
    h, st = make_house(seed), state_at(seed, b)
    t_room = h.containers[st.placement[target]]
    if rng.random() < 0.5:
        room = rng.choice([r for r in h.rooms if r != t_room and not h.locks_to_room(r)])
        return {"kind": "bring", "target": target, "dest_room": room, "goal": f"Bring the {target} to the {room}."}
    c = rng.choice([c for c in h.open_access() if h.containers[c] != t_room])
    prep = "on" if CONTAINER_KIND[c] == "surface" else "in"
    return {"kind": "put", "target": target, "dest_container": c, "dest_room": h.containers[c],
            "goal": f"Put the {target} {prep} the {c} in the {h.containers[c]}."}


def h1_candidates(seed: int, b: int, kind: str) -> list[str]:
    h, st = make_house(seed), state_at(seed, b)
    objs = sorted(o for o in st.placement if o not in h.keys)
    if kind == "any":
        return objs
    if kind == "static":
        return [o for o in objs if st.last_moved.get(o, 1) <= b - 2 and h.accessible(st.placement[o])]
    if kind == "moved":
        return [o for o in st.moved if o in objs]
    if kind == "locked":
        return [o for o in objs if not h.accessible(st.placement[o])]
    return list(st.new)


def h1_set() -> dict:
    tasks: dict[str, dict] = {}
    batches: list[list[str]] = [[] for _ in range(H1["batches"])]
    for seed in H1["seeds"]:
        rng = random.Random(seed)
        used: set[str] = set()
        for b in range(1, H1["batches"] + 1):
            slots = [("any", 8)] if b <= GATHER else H1["test"]
            i = 0
            for kind, n in slots:
                for _ in range(n):
                    cands = h1_candidates(seed, b, kind)
                    fresh = [o for o in cands if o not in used] or cands
                    for _attempt in range(200):
                        target = rng.choice(fresh)
                        spec = make_spec(seed, b, target, rng)
                        p = solve(seed, b, spec)
                        if p is not None and len(p) <= H1["max_plan"]:
                            break
                    else:
                        raise RuntimeError(f"no task for {seed} b{b} {kind}")
                    used.add(target)
                    tid = f"house/{seed}/b{b}/{i:02d}"
                    tasks[tid] = spec | {"seed": seed, "batch": b, "type": kind if b > GATHER else "familiar",
                                         "plan_len": len(p)}
                    batches[b - 1].append(tid)
                    i += 1
    return {"seeds": H1["seeds"], "gather": GATHER, "max_plan": H1["max_plan"], "batches": batches, "tasks": tasks}


def h2_candidates(seed: int, b: int, kind: str) -> list[str]:
    h, st = make_house(seed), state_at(seed, b)
    objs = sorted(o for o in st.placement if o not in h.keys)
    lay = {o: layer(h, st.placement, o) for o in objs}
    if kind in ("L0m", "L0s") and b <= GATHER:
        return [o for o in objs if lay[o] == 0]
    if kind == "L0m":
        return [o for o in st.moved if o in lay and lay[o] == 0]
    if kind == "L0s":
        return [o for o in objs if lay[o] == 0 and st.last_moved.get(o, 1) <= b - 2]
    return [o for o in objs if lay[o] == int(kind[1])]


def h2_house(seed: int, max_plan: int) -> list[list[tuple[str, dict]]] | None:
    h = make_house(seed)
    rng = random.Random(seed)
    used: set[str] = set()
    out = []
    for b in range(1, H2_BATCHES + 1):
        row, i = [], 0
        for kind, n in H2_SLOTS:
            for _ in range(n):
                cands = h2_candidates(seed, b, kind)
                if not cands:
                    return None
                fresh = [o for o in cands if o not in used] or cands
                for attempt in range(200):
                    target = rng.choice(fresh if attempt < 100 else cands)
                    spec = make_spec(seed, b, target, rng)
                    p = solve(seed, b, spec)
                    if p is not None and len(p) <= max_plan:
                        break
                else:
                    return None
                used.add(target)
                L = layer(h, state_at(seed, b).placement, target)
                tid = f"house/{seed}/b{b}/{i:02d}"
                row.append((tid, spec | {"seed": seed, "batch": b, "type": f"L{L}" if b > GATHER else "familiar",
                                         "layer": L, "moved": kind == "L0m" and b > GATHER, "slot": kind,
                                         "plan_len": len(p),
                                         "plan_keys": sum(c.startswith("take ") and c[5:] in h.keys for c in p)}))
                i += 1
        out.append(row)
    return out


def h2_set(name: str) -> dict:
    cfg = H2[name]
    seeds, rejected, per_house = [], [], {}
    seed = cfg["first"]
    while len(seeds) < H2_HOUSES:
        got = h2_house(seed, cfg["max_plan"])
        if got is None:
            rejected.append(seed)
        else:
            seeds.append(seed)
            per_house[seed] = got
        seed += 1
    tasks: dict[str, dict] = {}
    batches: list[list[str]] = [[] for _ in range(H2_BATCHES)]
    for s in seeds:
        for b, row in enumerate(per_house[s]):
            for tid, spec in row:
                tasks[tid] = spec
                batches[b].append(tid)
    return {"seeds": seeds, "rejected_seeds": rejected, "gather": GATHER, "max_plan": cfg["max_plan"],
            "batches": batches, "tasks": tasks}


def main(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.sets.split(","):
        data = h1_set() if name == "h1" else h2_set(name)
        (out / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        by_type: dict[str, list[int]] = {}
        for t in data["tasks"].values():
            by_type.setdefault(t["type"], []).append(t["plan_len"])
        print(f"{name}: houses {data['seeds']}, {len(data['tasks'])} tasks; plan length by type:",
              {k: (len(v), round(sum(v) / len(v), 1), max(v)) for k, v in sorted(by_type.items())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="h1,b,l", help="comma-separated, from h1, b, l")
    ap.add_argument("--out", default="data/house")
    main(ap.parse_args())
