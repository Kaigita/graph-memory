import json
import random
from collections import deque

import pytest

from graph_benchmarks import house as H
from graph_benchmarks.house_memory import BRIEF_CHARS, HEADER, FactStore, HouseMemory, bm25, graph_brief
from graph_memory.llm import ToolCall


def reach(h: H.House, keys: set[str]) -> set[str]:
    seen, q = {H.START}, deque([H.START])
    while q:
        r = q.popleft()
        for r2 in h.exits[r].values():
            door = h.lock_of_door(r, r2)
            if r2 not in seen and (door is None or h.door_key[door] in keys):
                seen.add(r2)
                q.append(r2)
    return seen


@pytest.mark.parametrize("seed,rooms,locks", [(31001, 10, 5), (32001, 10, 7), (33001, 20, 7)])
def test_house_structure(seed, rooms, locks):
    h = H.make_house(seed)
    assert len(h.rooms) == rooms and h.rooms[0] == H.START and len(h.keys) == locks
    for r in h.rooms:
        for d, r2 in h.exits[r].items():
            assert h.exits[r2][H.OPP[d]] == r
    have: set[str] = set()
    for _ in range(locks + 1):
        have |= {k for k in h.keys if h.accessible(h.placement[k], have)}
    assert have == set(h.keys) and reach(h, set(h.keys)) == set(h.rooms)


def test_moves_keep_locked_places():
    seed = 31001
    h = H.make_house(seed)
    free = set(h.open_access())
    for b in range(2, 10):
        prev, cur = H.state_at(seed, b - 1), H.state_at(seed, b)
        assert len([m for m in cur.moved if m not in h.keys]) == H.MOVES_PER_BATCH
        for o in cur.moved:
            assert cur.placement[o] in free and prev.placement[o] != cur.placement[o]
        for o, c in prev.placement.items():
            if c not in free and o not in h.keys:
                assert cur.placement[o] == c


def test_episode_commands_and_room_text_round_trip():
    h = H.make_house(31001)
    ep = H.Episode(h, H.state_at(31001, 1).placement)
    door = next(iter(h.door_key))
    a, b = sorted(door)
    ep.room = a
    d = next(dd for dd, r2 in h.exits[a].items() if r2 == b)
    assert "locked" in ep.do(f"go {d}")[0]
    k = h.door_key[door]
    ep.loc[k] = "@carried"
    wrong = next(kk for kk in h.keys if kk != k)
    ep.loc[wrong] = "@carried"
    assert "does not fit" in ep.do(f"unlock {d} door with {wrong}")[0]
    assert ep.do(f"unlock {d} door with {k}")[1] and ep.do(f"go {d}")[1] and ep.room == b
    assert not ep.do("dance")[1]
    rng = random.Random(0)
    ep.unlocked = set(h.container_key) | set(h.door_key)
    for r in h.rooms:
        ep.room = r
        ep.opened |= {c for c in ep.here() if H.CONTAINER_KIND[c] == "closable" and rng.random() < 0.5}
        info = H.parse_room(ep.describe())
        assert info["room"] == r and set(info["containers"]) == set(ep.here())
        assert all(items == ep.contents(c) for c, (_, items) in info["containers"].items() if items is not None)
        assert {dd: r2 for dd, (r2, _) in info["exits"].items()} == h.exits[r]


def locked_task(seed: int, batch: int) -> dict:
    h, st = H.make_house(seed), H.state_at(seed, batch)
    target = next(o for o in sorted(st.placement) if o not in h.keys and H.layer(h, st.placement, o) >= 2)
    return {"kind": "bring", "target": target, "dest_room": H.START, "goal": f"Bring the {target} to the front hall.",
            "seed": seed, "batch": batch, "type": "L2"}


def scripted(spec: dict, tid: str) -> dict:
    world = H.episode(spec)
    ep = {"task_id": tid, "initial_obs": world.describe(), "steps": [], "end_reason": "won"}
    for cmd in H.solve(spec["seed"], spec["batch"], spec):
        text, ok = world.do(cmd)
        ep["steps"].append({"action": cmd, "valid": ok, "observation": text, "won": world.goal_met(spec)})
    assert ep["steps"][-1]["won"]
    return ep


def test_graph_briefing_composes_the_key_chain_and_flat_ranks_lines():
    spec = locked_task(32001, 2)
    g, f = HouseMemory("graph"), HouseMemory("flat")
    for m in (g, f):
        m.add([scripted(spec, "house/32001/b2/00")], "s")
    assert g.stats() == f.stats() and f.brief(32002, spec["goal"]) == ""
    assert g.brief(32002, spec["goal"]).endswith("not seen in this house yet.")
    later = dict(spec, batch=3)
    text = g.brief(32001, later["goal"])
    lines = "\n".join(graph_brief(g.store(32001), later["goal"]))
    h, st = H.make_house(32001), H.state_at(32001, 2)
    keys = H.key_closure(h, st.placement, st.placement[spec["target"]])
    assert text.startswith(HEADER) and len(text) <= BRIEF_CHARS
    assert all(f"Get the {k} first" in lines for k in keys) and f"Then get the {spec['target']}" in lines
    assert f.brief(32001, later["goal"]).split("\n")[1].startswith(f"- {spec['target']}: seen")
    assert spec["target"] in g.answer(32001, spec["target"], H.START)


def test_belief_marks_moved_objects():
    s = FactStore()
    s.containers = {"fridge": {"room": "kitchen", "kind": "closable"}, "desk": {"room": "study", "kind": "surface"}}
    s.sights = {"mug": [[2, "e1", "fridge"]]}
    s.snaps = {"fridge": [[2, "e1", ["mug"], []], [4, "e2", ["plate"], []]]}
    assert s.belief("mug")["gone"] == 4
    s.snaps["fridge"].append([5, "e3", ["plate"], ["mug"]])
    assert s.belief("mug")["gone"] == 4
    s.sights["mug"].append([5, "e4", "desk"])
    assert s.belief("mug")["place"] == "desk" and s.belief("mug")["gone"] is None
    assert bm25(["mug: seen in the fridge", "plate: seen in the sideboard"], "Bring the mug.")[0].startswith("mug")


class Res:
    def __init__(self, content: str, calls: list[tuple[str, str]]):
        self.content, self.reasoning, self.usage = content, "", {"prompt_tokens": 10, "completion_tokens": 2}
        self.tool_calls = [ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)]


class Script:
    def __init__(self, replies: list[Res]):
        self.replies, self.seen = replies, []

    async def chat(self, messages, tools=None):
        self.seen.append((json.loads(json.dumps(messages)), [t["function"]["name"] for t in tools]))
        return self.replies.pop(0) if self.replies else Res("", [])


async def test_agent_loop_runs_tools_nudges_and_wins():
    spec = locked_task(31002, 4)
    mem = HouseMemory("graph")
    mem.add([scripted(dict(spec, batch=3), "house/31002/b3/00")], "s")
    cmds = H.solve(spec["seed"], spec["batch"], spec)
    replies = [Res("done?", []), Res("", [("memory", json.dumps({"query": spec["target"]}))]),
               Res("", [("act", "{bad"), ("search", "{}")])]
    replies += [Res("", [("act", json.dumps({"action": c}))]) for c in cmds]
    llm = Script(replies)
    ep = await H.run_episode("house/31002/b4/00", spec, mem, llm, 40, 60)
    assert ep["success"] and ep["end_reason"] == "won" and len(ep["steps"]) == len(cmds)
    assert [c["tool"] for c in ep["memory_calls"]] == ["briefing", "memory"] and ep["no_tool_calls"] == 1
    first, names = llm.seen[0]
    assert names == ["act", "memory"] and first[0]["content"].endswith(mem.policy)
    assert first[1]["content"].startswith(f"Your task: {spec['goal']}") and HEADER in first[1]["content"]
    nudge, tools = llm.seen[1][0][-1]["content"], [m["content"] for m in llm.seen[3][0] if m["role"] == "tool"]
    assert nudge.startswith("You replied without calling a tool") and tools[-2:] == [
        "Tool arguments were not valid JSON. Call the tool again.", "Unknown tool `search`."]
    plain = await H.run_episode("house/31002/b4/00", spec, None, Script([Res("", [])] * 3), 40, 3)
    assert plain["end_reason"] == "max_llm_calls" and not plain["memory_calls"]
