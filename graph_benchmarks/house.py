from __future__ import annotations

import json
import random
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA = Path("data/house")
START = "front hall"
N_ROOMS = 10
N_OBJECTS = 40
MOVES_PER_BATCH = 4
KEY_MOVE_P = 0.25

DIRS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}

ROOMS: dict[str, list[str]] = {
    "front hall": ["coat rack", "shoe cabinet", "side table"],
    "hallway": ["console table", "linen cupboard", "umbrella stand"],
    "kitchen": ["fridge", "kitchen cupboard", "kitchen drawer", "counter"],
    "pantry": ["pantry shelf", "storage bin", "spice rack"],
    "dining room": ["sideboard", "dining table", "glass cabinet"],
    "living room": ["bookshelf", "coffee table", "tv cabinet"],
    "study": ["desk", "filing cabinet", "safe", "study shelf"],
    "bedroom": ["wardrobe", "nightstand", "chest of drawers"],
    "guest room": ["guest wardrobe", "trunk", "guest nightstand"],
    "bathroom": ["bathroom cabinet", "laundry basket", "towel rail"],
    "laundry room": ["washing machine", "utility shelf", "detergent cupboard"],
    "garage": ["toolbox", "workbench", "garage shelf"],
    "attic": ["old trunk", "attic box", "dusty shelf"],
    "nursery": ["toy chest", "changing table", "nursery shelf"],
}
CONTAINER_KIND: dict[str, str] = {
    "coat rack": "surface", "shoe cabinet": "closable", "side table": "surface",
    "console table": "surface", "linen cupboard": "closable", "umbrella stand": "open",
    "fridge": "closable", "kitchen cupboard": "closable", "kitchen drawer": "closable", "counter": "surface",
    "pantry shelf": "surface", "storage bin": "open", "spice rack": "surface",
    "sideboard": "closable", "dining table": "surface", "glass cabinet": "closable",
    "bookshelf": "surface", "coffee table": "surface", "tv cabinet": "closable",
    "desk": "surface", "filing cabinet": "closable", "safe": "closable", "study shelf": "surface",
    "wardrobe": "closable", "nightstand": "closable", "chest of drawers": "closable",
    "guest wardrobe": "closable", "trunk": "closable", "guest nightstand": "closable",
    "bathroom cabinet": "closable", "laundry basket": "open", "towel rail": "surface",
    "washing machine": "closable", "utility shelf": "surface", "detergent cupboard": "closable",
    "toolbox": "closable", "workbench": "surface", "garage shelf": "surface",
    "old trunk": "closable", "attic box": "closable", "dusty shelf": "surface",
    "toy chest": "closable", "changing table": "surface", "nursery shelf": "surface",
}
LOCKABLE = ["safe", "filing cabinet", "chest of drawers", "trunk", "old trunk", "toolbox", "tv cabinet",
            "guest wardrobe", "attic box", "glass cabinet", "toy chest", "sideboard"]
OBJECTS: dict[str, str] = {
    "mug": "kitchen", "plate": "kitchen", "kettle": "kitchen", "frying pan": "kitchen", "spatula": "kitchen",
    "salt shaker": "kitchen", "tea tin": "kitchen", "bread knife": "kitchen",
    "jar of honey": "pantry", "bag of rice": "pantry", "olive oil": "pantry", "flashlight": "pantry",
    "candle holder": "dining room", "napkin ring": "dining room", "wine glass": "dining room",
    "remote control": "living room", "photo album": "living room", "board game": "living room", "vase": "living room",
    "stapler": "study", "notebook": "study", "fountain pen": "study", "calculator": "study", "passport": "study",
    "envelope": "study",
    "alarm clock": "bedroom", "scarf": "bedroom", "wristwatch": "bedroom", "jewelry box": "bedroom",
    "slippers": "bedroom",
    "spare pillow": "guest room", "travel bag": "guest room",
    "toothbrush": "bathroom", "hairdryer": "bathroom", "bath towel": "bathroom", "soap bar": "bathroom",
    "iron": "laundry room", "clothes pegs": "laundry room", "sewing kit": "laundry room",
    "hammer": "garage", "screwdriver": "garage", "tape measure": "garage", "bicycle pump": "garage",
    "paint brush": "garage",
    "picture frame": "attic", "old diary": "attic", "christmas lights": "attic",
    "teddy bear": "nursery", "rattle": "nursery", "picture book": "nursery",
    "umbrella": "front hall", "dog leash": "front hall", "sunglasses": "front hall", "gloves": "hallway",
}
RESERVE = ["phone charger", "birthday card", "yoga mat", "cookbook", "chess set", "headphones", "water bottle",
           "sun hat", "gift bag", "tennis racket", "knitting needles", "harmonica"]
KEYS = ["brass key", "silver key", "iron key", "copper key", "bronze key", "steel key", "golden key"]

EXTRA_ROOMS: dict[str, list[str]] = {
    "office": ["office desk", "file drawer", "office cupboard", "printer table"],
    "library": ["reading table", "tall bookcase", "map chest", "library cabinet"],
    "cellar": ["wine rack", "cellar crate", "cellar cupboard"],
    "playroom": ["toy box", "craft table", "game cabinet"],
    "sunroom": ["plant stand", "wicker basket", "garden chest"],
    "workshop": ["tool cabinet", "work table", "parts bin"],
    "music room": ["piano bench", "music stand", "record cabinet"],
    "dressing room": ["dresser", "shoe rack", "hat box"],
    "mudroom": ["boot bench", "mudroom locker", "coat hooks"],
}
CONTAINER_KIND.update({
    "office desk": "surface", "file drawer": "closable", "office cupboard": "closable", "printer table": "surface",
    "reading table": "surface", "tall bookcase": "surface", "map chest": "closable", "library cabinet": "closable",
    "wine rack": "surface", "cellar crate": "open", "cellar cupboard": "closable",
    "toy box": "closable", "craft table": "surface", "game cabinet": "closable",
    "plant stand": "surface", "wicker basket": "open", "garden chest": "closable",
    "tool cabinet": "closable", "work table": "surface", "parts bin": "open",
    "piano bench": "closable", "music stand": "surface", "record cabinet": "closable",
    "dresser": "closable", "shoe rack": "surface", "hat box": "closable",
    "boot bench": "surface", "mudroom locker": "closable", "coat hooks": "surface",
})
EXTRA_LOCKABLE = ["file drawer", "map chest", "library cabinet", "cellar cupboard", "game cabinet", "garden chest",
                  "tool cabinet", "record cabinet", "hat box", "mudroom locker"]
EXTRA_OBJECTS: dict[str, str] = {
    "laptop": "office", "paper clips": "office", "desk lamp": "office", "sticky notes": "office",
    "printer paper": "office",
    "atlas": "library", "magnifying glass": "library", "bookmark": "library", "encyclopedia": "library",
    "reading glasses": "library",
    "wine bottle": "cellar", "jar of pickles": "cellar", "cider jug": "cellar", "potato sack": "cellar",
    "toy car": "playroom", "jigsaw puzzle": "playroom", "crayons": "playroom", "building blocks": "playroom",
    "kite": "playroom",
    "watering can": "sunroom", "flower pot": "sunroom", "garden gloves": "sunroom", "seed packet": "sunroom",
    "wrench": "workshop", "pliers": "workshop", "sandpaper": "workshop", "wood glue": "workshop", "drill": "workshop",
    "violin": "music room", "sheet music": "music room", "metronome": "music room", "tuning fork": "music room",
    "necktie": "dressing room", "belt": "dressing room", "perfume": "dressing room", "hand mirror": "dressing room",
    "hairbrush": "dressing room",
    "rain boots": "mudroom", "raincoat": "mudroom", "wool hat": "mudroom",
    "cutting board": "kitchen", "whisk": "kitchen", "teapot": "kitchen",
    "throw blanket": "living room", "cushion": "living room", "table lamp": "living room",
    "pajamas": "bedroom", "hot water bottle": "bedroom",
    "shampoo": "bathroom", "razor": "bathroom", "toothpaste": "bathroom",
    "car wax": "garage", "jump cables": "garage",
    "ruler": "study", "eraser": "study",
    "baby blanket": "nursery", "pacifier": "nursery",
    "rocking horse": "attic", "old lamp": "attic",
    "guest towel": "guest room", "reading lamp": "guest room",
    "lint roller": "laundry room", "spray starch": "laundry room",
    "salad bowl": "dining room", "tablecloth": "dining room",
    "box of cereal": "pantry", "tin of beans": "pantry",
    "doormat": "front hall", "shoe horn": "hallway",
}


@dataclass(frozen=True)
class HouseSpec:
    name: str
    n_rooms: int
    n_doors: int
    n_locked: int
    n_objects: int
    moves: int
    grid: int
    far_max: int
    extra_edges: int
    big_pool: bool
    n_batches: int = 7
    min_per_layer: int = 3


SPECS: dict[int, HouseSpec] = {
    32: HouseSpec("B", n_rooms=10, n_doors=3, n_locked=4, n_objects=40, moves=4, grid=2, far_max=3, extra_edges=2,
                  big_pool=False),
    33: HouseSpec("L", n_rooms=20, n_doors=3, n_locked=4, n_objects=100, moves=10, grid=3, far_max=5,
                  extra_edges=4, big_pool=True),
}


def spec_of(seed: int) -> HouseSpec | None:
    return SPECS.get(seed // 1000)


ENV_NAME = ("a text-based house. You work in the same house every day, one task at a time; other people may have "
            "moved things since your last task")
TOOL_INTRO = "Solve the task by interacting with the house through the `act` tool."
ACT_TOOL = {
    "type": "function",
    "function": {
        "name": "act",
        "description": "Execute one command in the house.",
        "parameters": {"type": "object",
                       "properties": {"action": {"type": "string", "description": "One command."}},
                       "required": ["action"]},
    },
}
RULE = ("Each `act` call executes exactly one command:\n"
        "  go <north|south|east|west>  (or: go to <adjacent room>)\n"
        "  open <container> / close <container>  -- things inside a closed container are not visible\n"
        "  take <object>  -- an object you can see in this room\n"
        "  put <object> in <container> / put <object> on <surface>\n"
        "  drop <object>  -- put it on the floor of this room\n"
        "  unlock <container> with <key> / unlock <direction> door with <key>\n"
        "  look / inventory")


def article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def item_list(names: list[str]) -> str:
    return ", ".join(article(n) for n in names) if names else "nothing"


@dataclass
class House:
    seed: int
    rooms: list[str]
    pos: dict[str, tuple[int, int]]
    exits: dict[str, dict[str, str]]
    tree: set[frozenset]
    door_key: dict[frozenset, str]
    containers: dict[str, str]
    container_key: dict[str, str]
    placement: dict[str, str]
    objects: list[str]
    keys: list[str]
    reserve: list[str]
    spec: str = "H1"

    def lock_of_door(self, a: str, b: str) -> frozenset | None:
        e = frozenset((a, b))
        return e if e in self.door_key else None

    def key_of(self, lock: str | frozenset) -> str:
        return self.door_key[lock] if isinstance(lock, frozenset) else self.container_key[lock]

    def locks_to_room(self, room: str) -> list[frozenset]:
        prev: dict[str, str | None] = {START: None}
        q = deque([START])
        while q:
            r = q.popleft()
            for r2 in self.exits[r].values():
                if r2 not in prev and frozenset((r, r2)) in self.tree:
                    prev[r2] = r
                    q.append(r2)
        out, r = [], room
        while prev.get(r) is not None:
            p = prev[r]
            if frozenset((p, r)) in self.door_key:
                out.append(frozenset((p, r)))
            r = p
        return out[::-1]

    def locks_to_container(self, c: str) -> list[str | frozenset]:
        locks: list[str | frozenset] = list(self.locks_to_room(self.containers[c]))
        if c in self.container_key:
            locks.append(c)
        return locks

    def accessible(self, c: str, keys: set[str] | frozenset = frozenset()) -> bool:
        return all(self.key_of(lock) in keys for lock in self.locks_to_container(c))

    def open_access(self) -> list[str]:
        return sorted(c for c in self.containers if self.accessible(c))


def _make_house_once(seed: int) -> House | None:
    rng = random.Random(seed)
    others = sorted(r for r in ROOMS if r != START)
    rooms = [START] + rng.sample(others, N_ROOMS - 1)
    pos = {START: (0, 0)}
    taken = {(0, 0): START}
    tree: set[frozenset] = set()
    for room in rooms[1:]:
        while True:
            base = rng.choice(sorted(pos, key=rooms.index))
            d = rng.choice(sorted(DIRS))
            x, y = pos[base][0] + DIRS[d][0], pos[base][1] + DIRS[d][1]
            if (x, y) not in taken and abs(x) <= 2 and abs(y) <= 2:
                pos[room], taken[(x, y)] = (x, y), room
                tree.add(frozenset((base, room)))
                break

    def far_side(edge: frozenset) -> set[str]:
        seen = {START}
        q = deque([START])
        while q:
            r = q.popleft()
            for e in tree:
                if r in e and e != edge:
                    (r2,) = e - {r}
                    if r2 not in seen:
                        seen.add(r2)
                        q.append(r2)
        return set(rooms) - seen
    cand = [e for e in sorted(tree, key=lambda e: sorted(e)) if 1 <= len(far_side(e)) <= 3]
    if len(cand) < 2:
        return None
    doors = rng.sample(cand, 2)
    region = {r: frozenset(e for e in doors if r in far_side(e)) for r in rooms}
    edges = set(tree)
    extra = [frozenset((a, b)) for a in rooms for b in rooms if a < b and frozenset((a, b)) not in tree
             and abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1 and region[a] == region[b]]
    rng.shuffle(extra)
    edges |= set(extra[:2])
    exits: dict[str, dict[str, str]] = {r: {} for r in rooms}
    for e in edges:
        a, b = sorted(e)
        dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
        d = next(k for k, v in DIRS.items() if v == (dx, dy))
        exits[a][d], exits[b][OPP[d]] = b, a
    containers: dict[str, str] = {}
    for r in rooms:
        for c in rng.sample(ROOMS[r], rng.choice([2, 3]) if len(ROOMS[r]) > 2 else 2):
            containers[c] = r
    lockable = [c for c in sorted(containers) if c in LOCKABLE]
    if len(lockable) < 3:
        return None
    locked_c = rng.sample(lockable, 3)
    keys = rng.sample(KEYS, 5)
    locks: list[str | frozenset] = list(doors) + locked_c
    rng.shuffle(locks)
    door_key, container_key = {}, {}
    for lock, key in zip(locks, keys):
        (door_key if isinstance(lock, frozenset) else container_key)[lock] = key
    house = House(seed=seed, rooms=rooms, pos=pos, exits=exits, tree=tree, door_key=door_key, containers=containers,
                  container_key=container_key, placement={}, objects=[], keys=keys, reserve=[])
    have: set[str] = set()
    for lock in locks:
        key = house.key_of(lock)
        spots = [c for c in sorted(containers) if house.accessible(c, have) and c != lock]
        deeper = [c for c in spots if not house.accessible(c)]
        house.placement[key] = rng.choice(deeper if deeper and rng.random() < 0.6 else spots)
        have.add(key)
    pool = sorted(OBJECTS)
    house.objects = rng.sample(pool, N_OBJECTS)
    load: dict[str, int] = {c: 0 for c in containers}
    for item in house.placement.values():
        load[item] += 1
    for obj in house.objects:
        home = [c for c in sorted(containers) if containers[c] == OBJECTS[obj] and load[c] < 5]
        anywhere = [c for c in sorted(containers) if load[c] < 5]
        c = rng.choice(home if home and rng.random() < 0.5 else anywhere)
        house.placement[obj] = c
        load[c] += 1
    house.reserve = rng.sample(RESERVE, len(RESERVE))
    return house


def key_closure(house: House, placement: dict[str, str], c: str, _seen: frozenset = frozenset()) -> set[str]:
    keys: set[str] = set()
    for lock in house.locks_to_container(c):
        k = house.key_of(lock)
        if k in keys or k in _seen:
            continue
        keys.add(k)
        place = placement.get(k)
        if place in house.containers:
            keys |= key_closure(house, placement, place, _seen | {k})
    return keys


def layer(house: House, placement: dict[str, str], item: str) -> int:
    return len(key_closure(house, placement, placement[item]))


def _make_house_v2(seed: int, spec: HouseSpec, outer: int) -> House | None:
    rng = random.Random(seed)
    pool_rooms = {**ROOMS, **EXTRA_ROOMS} if spec.big_pool else ROOMS
    pool_objects = {**OBJECTS, **EXTRA_OBJECTS} if spec.big_pool else OBJECTS
    others = sorted(r for r in pool_rooms if r != START)
    rooms = [START] + rng.sample(others, spec.n_rooms - 1)
    pos = {START: (0, 0)}
    taken = {(0, 0): START}
    tree: set[frozenset] = set()
    for room in rooms[1:]:
        for _ in range(2000):
            base = rng.choice(sorted(pos, key=rooms.index))
            d = rng.choice(sorted(DIRS))
            x, y = pos[base][0] + DIRS[d][0], pos[base][1] + DIRS[d][1]
            if (x, y) not in taken and abs(x) <= spec.grid and abs(y) <= spec.grid:
                pos[room], taken[(x, y)] = (x, y), room
                tree.add(frozenset((base, room)))
                break
        else:
            return None

    def far_side(edge: frozenset) -> set[str]:
        seen = {START}
        q = deque([START])
        while q:
            r = q.popleft()
            for e in tree:
                if r in e and e != edge:
                    (r2,) = e - {r}
                    if r2 not in seen:
                        seen.add(r2)
                        q.append(r2)
        return set(rooms) - seen
    cand = [e for e in sorted(tree, key=lambda e: sorted(e)) if 1 <= len(far_side(e)) <= spec.far_max]
    if len(cand) < spec.n_doors:
        return None
    doors = rng.sample(cand, spec.n_doors)
    region = {r: frozenset(e for e in doors if r in far_side(e)) for r in rooms}
    edges = set(tree)
    extra = [frozenset((a, b)) for a in rooms for b in rooms if a < b and frozenset((a, b)) not in tree
             and abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) == 1 and region[a] == region[b]]
    rng.shuffle(extra)
    edges |= set(extra[:spec.extra_edges])
    exits: dict[str, dict[str, str]] = {r: {} for r in rooms}
    for e in edges:
        a, b = sorted(e)
        dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
        d = next(k for k, v in DIRS.items() if v == (dx, dy))
        exits[a][d], exits[b][OPP[d]] = b, a
    containers: dict[str, str] = {}
    for r in rooms:
        for c in rng.sample(pool_rooms[r], rng.choice([2, 3]) if len(pool_rooms[r]) > 2 else 2):
            containers[c] = r
    lockable = [c for c in sorted(containers) if c in LOCKABLE or c in EXTRA_LOCKABLE]
    if len(lockable) < spec.n_locked:
        return None
    locked_c = rng.sample(lockable, spec.n_locked)
    keys = rng.sample(KEYS, spec.n_doors + spec.n_locked)
    locks: list[str | frozenset] = list(doors) + locked_c
    rng.shuffle(locks)
    door_key, container_key = {}, {}
    for lock, key in zip(locks, keys):
        (door_key if isinstance(lock, frozenset) else container_key)[lock] = key
    house = House(seed=seed, rooms=rooms, pos=pos, exits=exits, tree=tree, door_key=door_key, containers=containers,
                  container_key=container_key, placement={}, objects=[], keys=keys, reserve=[], spec=spec.name)
    have: set[str] = set()
    for lock in locks:
        key = house.key_of(lock)
        spots = [c for c in sorted(containers) if house.accessible(c, have) and c != lock]
        depth = {c: len(key_closure(house, house.placement, c)) for c in spots}
        ok = [c for c in spots if depth[c] <= 2]
        if not ok:
            return None
        top = max(depth[c] for c in ok)
        deepest = [c for c in ok if depth[c] == top]
        house.placement[key] = rng.choice(deepest if rng.random() < 0.7 else ok)
        have.add(key)
    house.objects = rng.sample(sorted(pool_objects), spec.n_objects)
    load: dict[str, int] = {c: 0 for c in containers}
    for item in house.placement.values():
        load[item] += 1
    for obj in house.objects:
        home = [c for c in sorted(containers) if containers[c] == pool_objects[obj] and load[c] < 5]
        anywhere = [c for c in sorted(containers) if load[c] < 5]
        c = rng.choice(home if home and rng.random() < 0.5 else anywhere)
        house.placement[obj] = c
        load[c] += 1
    house.reserve = rng.sample(RESERVE, len(RESERVE))
    st = Batch(dict(house.placement), last_moved={i: 1 for i in house.placement})
    for b in range(1, spec.n_batches + 1):
        if b > 1:
            st = _next_state(house, outer, b, st)
        count = Counter(layer(house, st.placement, o) for o in st.placement if o not in house.keys)
        if any(count[L] < spec.min_per_layer for L in (1, 2, 3)):
            return None
    return house


@lru_cache(maxsize=64)
def make_house(seed: int) -> House:
    spec = spec_of(seed)
    for attempt in range(1000):
        if spec is None:
            house = _make_house_once(seed * 1000 + attempt)
        else:
            house = _make_house_v2(seed * 1000 + attempt, spec, seed)
        if house is not None:
            house.seed = seed
            return house
    raise RuntimeError(f"no valid house for seed {seed}")


@dataclass
class Batch:
    placement: dict[str, str]
    moved: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    last_moved: dict[str, int] = field(default_factory=dict)


@lru_cache(maxsize=512)
def state_at(seed: int, batch: int) -> Batch:
    house = make_house(seed)
    if batch <= 1:
        return Batch(dict(house.placement), last_moved={i: 1 for i in house.placement})
    return _next_state(house, seed, batch, state_at(seed, batch - 1))


def _next_state(house: House, seed: int, batch: int, prev: Batch) -> Batch:
    rng = random.Random(seed * 7919 + batch)
    placement = dict(prev.placement)
    free = house.open_access()
    movable = sorted(o for o, c in placement.items() if o not in house.keys and c in free)
    spec = spec_of(seed)
    moved = rng.sample(movable, spec.moves if spec else MOVES_PER_BATCH)
    for o in moved:
        placement[o] = rng.choice([c for c in free if c != placement[o]])
    new = [house.reserve[(batch - 2) % len(house.reserve)]] if batch - 2 < len(house.reserve) else []
    for o in new:
        placement[o] = rng.choice(free)
    if rng.random() < KEY_MOVE_P:
        k = rng.choice(sorted(house.keys))
        placement[k] = rng.choice([c for c in free if c != placement[k]])
        moved.append(k)
    last = dict(prev.last_moved)
    for o in moved + new:
        last[o] = batch
    return Batch(placement, moved, new, last)


_ARTICLES = re.compile(r"^(the|a|an|some)\s+")


def _clean(text: str) -> str:
    text = " ".join(text.lower().strip().rstrip(".!").split())
    return _ARTICLES.sub("", text)


def _match(name: str, candidates: list[str]) -> str | None:
    name = _clean(name)
    if name in candidates:
        return name
    hits = [c for c in candidates if c.endswith(" " + name) or name in c.split(" ")]
    hits = hits or [c for c in candidates if name in c]
    return hits[0] if len(hits) == 1 else None


class Episode:

    def __init__(self, house: House, placement: dict[str, str]):
        self.h = house
        self.loc = dict(placement)
        self.room = START
        self.opened: set[str] = set()
        self.unlocked: set[str | frozenset] = set()

    def carried(self) -> list[str]:
        return sorted(i for i, l in self.loc.items() if l == "@carried")

    def room_of(self, item: str) -> str:
        loc = self.loc[item]
        if loc == "@carried":
            return self.room
        if loc.startswith("floor:"):
            return loc[6:]
        return self.h.containers[loc]

    def locked(self, lock: str | frozenset) -> bool:
        return (lock in self.h.door_key or lock in self.h.container_key) and lock not in self.unlocked

    def visible_in(self, c: str) -> bool:
        kind = CONTAINER_KIND[c]
        return kind != "closable" or c in self.opened

    def contents(self, c: str) -> list[str]:
        return sorted(i for i, l in self.loc.items() if l == c)

    def here(self) -> list[str]:
        return sorted(c for c, r in self.h.containers.items() if r == self.room)

    def visible_items(self) -> list[str]:
        items = [i for i, l in self.loc.items() if l == f"floor:{self.room}"]
        for c in self.here():
            if self.visible_in(c):
                items += self.contents(c)
        return sorted(items)

    def container_desc(self, c: str) -> str:
        kind = CONTAINER_KIND[c]
        if kind == "surface":
            return f"{article(c)} (on it: {item_list(self.contents(c))})"
        if kind == "open":
            return f"{article(c)} (in it: {item_list(self.contents(c))})"
        if self.locked(c):
            return f"{article(c)} (locked)"
        if c in self.opened:
            return f"{article(c)} (open; in it: {item_list(self.contents(c))})"
        return f"{article(c)} (closed)"

    def describe(self) -> str:
        floor = sorted(i for i, l in self.loc.items() if l == f"floor:{self.room}")
        exits = []
        for d in ("north", "south", "east", "west"):
            r2 = self.h.exits[self.room].get(d)
            if r2:
                door = self.h.lock_of_door(self.room, r2)
                exits.append(f"{d} to the {r2}" + (" (locked door)" if door and self.locked(door) else ""))
        return (f"You are in the {self.room}.\n"
                f"You see: {', '.join(self.container_desc(c) for c in self.here())}.\n"
                f"On the floor: {item_list(floor)}.\n"
                f"Exits: {', '.join(exits)}.")

    def do(self, action: str) -> tuple[str, bool]:
        a = " ".join(action.lower().strip().rstrip(".!").split())
        if a in ("look", "l", "look around"):
            return self.describe(), True
        if a in ("inventory", "i", "inv"):
            items = self.carried()
            return (f"You are carrying: {item_list(items)}." if items else "You are carrying nothing."), True
        m = re.match(r"^(?:go|move|walk|head)\s+(?:to\s+)?(?:the\s+)?(.+)$", a) or re.match(
            r"^(north|south|east|west|n|s|e|w)$", a)
        if m:
            return self._go(m.group(1))
        m = re.match(r"^open\s+(.+)$", a)
        if m:
            return self._open(m.group(1))
        m = re.match(r"^close\s+(.+)$", a)
        if m:
            return self._close(m.group(1))
        m = re.match(r"^(?:take|get|pick up|grab)\s+(.+?)(?:\s+(?:from|off|out of)\s+(.+))?$", a)
        if m:
            return self._take(m.group(1), m.group(2))
        m = re.match(r"^(?:put|place)\s+(.+?)\s+(?:in|into|inside|on|onto)\s+(.+)$", a)
        if m:
            return self._put(m.group(1), m.group(2))
        m = re.match(r"^drop\s+(.+)$", a)
        if m:
            return self._drop(m.group(1))
        m = re.match(r"^unlock\s+(.+?)\s+with\s+(.+)$", a)
        if m:
            return self._unlock(m.group(1), m.group(2))
        return ("I don't understand that. Commands: go <direction>, open / close <container>, take <object>, "
                "put <object> in/on <container>, drop <object>, unlock <container or direction door> with <key>, "
                "look, inventory."), False

    def _target_dir(self, text: str) -> str | None:
        t = _clean(text)
        short = {"n": "north", "s": "south", "e": "east", "w": "west"}
        t = short.get(t, t)
        for d in DIRS:
            if re.search(rf"\b{d}\b", t):
                return d if d in self.h.exits[self.room] else None
        for d, r2 in self.h.exits[self.room].items():
            if r2 in t or t in r2:
                return d
        return None

    def _go(self, text: str) -> tuple[str, bool]:
        d = self._target_dir(text)
        if d is None:
            return "You can't go that way from here.", False
        r2 = self.h.exits[self.room][d]
        door = self.h.lock_of_door(self.room, r2)
        if door and self.locked(door):
            return f"The door to the {r2} is locked.", False
        self.room = r2
        return f"You go {d} to the {r2}.\n" + self.describe(), True

    def _container(self, text: str) -> str | None:
        return _match(text, self.here())

    def _open(self, text: str) -> tuple[str, bool]:
        c = self._container(text)
        if c is None:
            o = _match(text, self.visible_items() + self.carried())
            if o is not None:
                return f"The {o} cannot be opened.", False
            return f"There is no {_clean(text)} here.", False
        if CONTAINER_KIND[c] != "closable":
            return f"The {c} cannot be opened.", False
        if self.locked(c):
            return f"The {c} is locked.", False
        if c in self.opened:
            return f"The {c} is already open. In it: {item_list(self.contents(c))}.", False
        self.opened.add(c)
        return f"You open the {c}. In it: {item_list(self.contents(c))}.", True

    def _close(self, text: str) -> tuple[str, bool]:
        c = self._container(text)
        if c is None or CONTAINER_KIND[c] != "closable" or c not in self.opened:
            return "You can't close that.", False
        self.opened.discard(c)
        return f"You close the {c}.", True

    def _take(self, text: str, src: str | None) -> tuple[str, bool]:
        o = _match(text, self.visible_items())
        if o is None:
            return f"There is no {_clean(text)} here that you can see.", False
        if src is not None:
            c = self._container(src)
            if c is None or self.loc[o] != c:
                return f"There is no {o} in that.", False
        self.loc[o] = "@carried"
        return f"You take the {o}.", True

    def _put(self, text: str, dest: str) -> tuple[str, bool]:
        o = _match(text, self.carried())
        if o is None:
            return f"You are not carrying {_clean(text)}.", False
        c = self._container(dest)
        if c is None:
            return f"There is no {_clean(dest)} here.", False
        if self.locked(c):
            return f"The {c} is locked.", False
        if CONTAINER_KIND[c] == "closable" and c not in self.opened:
            return f"The {c} is closed.", False
        self.loc[o] = c
        prep = "on" if CONTAINER_KIND[c] == "surface" else "in"
        return f"You put the {o} {prep} the {c}.", True

    def _drop(self, text: str) -> tuple[str, bool]:
        o = _match(text, self.carried())
        if o is None:
            return f"You are not carrying {_clean(text)}.", False
        self.loc[o] = f"floor:{self.room}"
        return f"You drop the {o}.", True

    def _unlock(self, text: str, key_text: str) -> tuple[str, bool]:
        k = _match(key_text, self.carried())
        if k is None or k not in self.h.keys:
            return f"You are not carrying {_clean(key_text)}.", False
        c = self._container(text)
        if c is not None and "door" not in _clean(text):
            if c not in self.h.container_key:
                return f"The {c} has no lock.", False
            if c in self.unlocked:
                return f"The {c} is already unlocked.", False
            if self.h.container_key[c] != k:
                return f"The {k} does not fit the {c}.", False
            self.unlocked.add(c)
            return f"You unlock the {c} with the {k}.", True
        d = self._target_dir(text)
        if d is None:
            return f"There is no {_clean(text)} here to unlock.", False
        r2 = self.h.exits[self.room][d]
        door = self.h.lock_of_door(self.room, r2)
        if door is None:
            return f"The way to the {r2} has no lock.", False
        if door in self.unlocked:
            return f"The door to the {r2} is already unlocked.", False
        if self.h.door_key[door] != k:
            return f"The {k} does not fit the door to the {r2}.", False
        self.unlocked.add(door)
        return f"You unlock the door to the {r2} with the {k}.", True

    def goal_met(self, spec: dict) -> bool:
        target = spec["target"]
        if target not in self.loc:
            return False
        if spec["kind"] == "bring":
            return self.room_of(target) == spec["dest_room"] and (
                self.loc[target] != "@carried" or self.room == spec["dest_room"])
        return self.loc[target] == spec["dest_container"]


def episode(spec: dict) -> Episode:
    return Episode(make_house(spec["seed"]), state_at(spec["seed"], spec["batch"]).placement)


def solve(seed: int, batch: int, spec: dict) -> list[str] | None:
    house = make_house(seed)
    ep = Episode(house, state_at(seed, batch).placement)
    out: list[str] = []

    def run(cmd: str) -> None:
        text, ok = ep.do(cmd)
        if not ok:
            raise RuntimeError(f"plan step failed: {cmd} -> {text}")
        out.append(cmd)

    def go_to(room: str) -> None:
        prev: dict[str, tuple[str, str] | None] = {ep.room: None}
        q = deque([ep.room])
        while q:
            r = q.popleft()
            for d in ("north", "south", "east", "west"):
                r2 = house.exits[r].get(d)
                if r2 is None or r2 in prev:
                    continue
                door = house.lock_of_door(r, r2)
                if door and ep.locked(door) and house.door_key[door] not in ep.carried():
                    continue
                prev[r2] = (r, d)
                q.append(r2)
        if room not in prev:
            raise RuntimeError(f"no route to {room}")
        steps, r = [], room
        while prev[r] is not None:
            p, d = prev[r]
            steps.append((p, d, r))
            r = p
        for p, d, r2 in reversed(steps):
            door = house.lock_of_door(p, r2)
            if door and ep.locked(door):
                run(f"unlock {d} door with {house.door_key[door]}")
            run(f"go {d}")

    def fetch(item: str) -> None:
        if ep.loc[item] == "@carried":
            return
        c = ep.loc[item]
        for lock in house.locks_to_container(c):
            if ep.locked(lock):
                fetch(house.key_of(lock))
        go_to(house.containers[c])
        if c in house.container_key and ep.locked(c):
            run(f"unlock {c} with {house.container_key[c]}")
        if CONTAINER_KIND[c] == "closable" and c not in ep.opened:
            run(f"open {c}")
        run(f"take {item}")

    try:
        fetch(spec["target"])
        if spec["kind"] == "bring":
            go_to(spec["dest_room"])
        else:
            dest = spec["dest_container"]
            go_to(house.containers[dest])
            if CONTAINER_KIND[dest] == "closable" and dest not in ep.opened:
                run(f"open {dest}")
            prep = "on" if CONTAINER_KIND[dest] == "surface" else "in"
            run(f"put {spec['target']} {prep} {dest}")
    except RuntimeError:
        return None
    return out if ep.goal_met(spec) else None


ROOM_RE = re.compile(r"You are in the ([a-z ]+)\.")
SEE_RE = re.compile(r"You see: (.*)\.$", re.M)
CONT_RE = re.compile(r"an? ([a-z ]+?) \(([^)]*)\)")
FLOOR_RE = re.compile(r"On the floor: (.*)\.$", re.M)
EXITS_RE = re.compile(r"Exits: (.*)\.$", re.M)
EXIT_RE = re.compile(r"(north|south|east|west) to the ([a-z ]+?)( \(locked door\))?$")
OPEN_RE = re.compile(r"(?:You open the|The) ([a-z ]+?)(?: is already open)?\. In it: (.*)\.$")
UNLOCK_OK_RE = re.compile(r"You unlock the (door to the )?([a-z ]+?) with the ([a-z ]+)\.$")
UNLOCK_BAD_RE = re.compile(r"The ([a-z ]+?) does not fit the (door to the )?([a-z ]+)\.$")
TAKE_RE = re.compile(r"You take the ([a-z ]+)\.$")


def parse_items(text: str) -> list[str]:
    text = text.strip()
    if text == "nothing" or not text:
        return []
    return [_ARTICLES.sub("", part.strip()) for part in text.split(", ")]


def parse_room(text: str) -> dict | None:
    m = ROOM_RE.search(text)
    see, floor, exits = SEE_RE.search(text), FLOOR_RE.search(text), EXITS_RE.search(text)
    if not (m and see and floor and exits):
        return None
    containers = {}
    for name, inner in CONT_RE.findall(see.group(1)):
        if inner.startswith("on it: "):
            containers[name] = ("surface", parse_items(inner[7:]))
        elif inner.startswith("in it: "):
            containers[name] = ("open", parse_items(inner[7:]))
        elif inner.startswith("open; in it: "):
            containers[name] = ("opened", parse_items(inner[13:]))
        else:
            containers[name] = (inner, None)
    ex = {}
    for part in exits.group(1).split(", "):
        em = EXIT_RE.match(part.strip())
        if em:
            ex[em.group(1)] = (em.group(2), bool(em.group(3)))
    return {"room": m.group(1), "containers": containers, "floor": parse_items(floor.group(1)), "exits": ex}


def load_set(name: str, data: Path = DATA) -> dict:
    return json.loads((data / f"{name}.json").read_text(encoding="utf-8"))


def task_seed(task_id: str) -> int:
    return int(task_id.split("/")[1])


def task_batch(task_id: str) -> int:
    return int(task_id.split("/")[2][1:])


SYSTEM = ("You are an expert agent operating in {env}.\n{intro}\n\nRules:\n- {rule}\n"
          "- Call one tool per turn and wait for its result before deciding the next action.")
NUDGE = ("You replied without calling a tool. The task is NOT complete yet: the environment ends the episode by "
         "itself as soon as the goal is achieved, and it has not ended. Re-read the goal: \"{goal}\". Work out which "
         "part of it is not satisfied yet, then call `act` with the next action.\nCurrent observation: {obs}")
OBS_WINDOW = 20
OLD_OBS_CHARS = 200


def system_prompt(memory: Any = None) -> str:
    base = SYSTEM.format(env=ENV_NAME, intro=TOOL_INTRO, rule=RULE)
    return f"{base}\n\n{memory.policy}" if memory is not None else base


def current_room(steps: list[dict]) -> str:
    for step in reversed(steps):
        m = ROOM_RE.search(step["observation"] or "")
        if m:
            return m.group(1)
    return START


def _assistant(res: Any) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": res.content or ""}
    if res.tool_calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                             for c in res.tool_calls]
    return msg


def _compact(messages: list[dict]) -> None:
    n = 0
    for msg in reversed(messages):
        if msg["role"] != "tool":
            continue
        n += 1
        if n > OBS_WINDOW and len(msg["content"]) > OLD_OBS_CHARS:
            msg["content"] = msg["content"][:OLD_OBS_CHARS] + " …[truncated]"


async def run_episode(task_id: str, spec: dict, memory: Any, llm: Any, max_steps: int, max_calls: int) -> dict:
    ep: dict[str, Any] = {"task_id": task_id, "type": spec["type"], "goal": spec["goal"], "initial_obs": "",
                          "steps": [], "memory_calls": [], "success": False, "end_reason": "", "error": None,
                          "llm_calls": 0, "no_tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    try:
        world = episode(spec)
        obs = world.describe()
        ep["initial_obs"] = obs
        first = f"Your task: {spec['goal']}\n\n{obs}"
        tools = [ACT_TOOL]
        if memory is not None:
            brief = memory.brief(spec["seed"], spec["goal"])
            ep["memory_calls"].append({"t": 0, "tool": "briefing", "args": {"mode": memory.mode, "chars": len(brief)},
                                       "result": brief})
            first = f"{first}\n\n{brief}" if brief else first
            tools = tools + memory.tool_specs
        messages = [{"role": "system", "content": system_prompt(memory)}, {"role": "user", "content": first}]
        won = False
        ep["end_reason"] = "max_env_steps"
        while len(ep["steps"]) < max_steps and not won:
            if ep["llm_calls"] >= max_calls:
                ep["end_reason"] = "max_llm_calls"
                break
            res = await llm.chat(messages, tools=tools)
            ep["llm_calls"] += 1
            ep["prompt_tokens"] += int(res.usage.get("prompt_tokens") or 0)
            ep["completion_tokens"] += int(res.usage.get("completion_tokens") or 0)
            messages.append(_assistant(res))
            if not res.tool_calls:
                ep["no_tool_calls"] += 1
                messages.append({"role": "user", "content": NUDGE.format(goal=spec["goal"], obs=obs.strip()[:300])})
                continue
            for call in res.tool_calls:
                args = call.parsed_args()
                if args is None:
                    result = "Tool arguments were not valid JSON. Call the tool again."
                elif memory is not None and call.name == "memory":
                    result = memory.answer(spec["seed"], str(args.get("query", "")), current_room(ep["steps"]))
                    ep["memory_calls"].append({"t": len(ep["steps"]), "tool": "memory", "args": args,
                                               "result": result})
                elif call.name == "act" and not won and len(ep["steps"]) < max_steps:
                    action = str(args.get("action", "")).strip()
                    obs, valid = world.do(action)
                    won = world.goal_met(spec)
                    ep["steps"].append({"t": len(ep["steps"]), "action": action, "valid": valid, "observation": obs,
                                        "won": won, "content": res.content, "reasoning": res.reasoning})
                    result = obs
                elif call.name == "act":
                    result = "Episode already finished."
                else:
                    result = f"Unknown tool `{call.name}`."
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            _compact(messages)
        if won:
            ep["success"], ep["end_reason"] = True, "won"
    except Exception as exc:
        ep["end_reason"], ep["error"] = "error", f"{type(exc).__name__}: {exc}"
    return ep
