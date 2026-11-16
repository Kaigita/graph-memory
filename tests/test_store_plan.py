import json

import pytest

from graph_memory.extract import Record, Turn, add_extracted, canon, prompt, transcript, validate
from graph_memory.plan import plan, plan_answer, plan_brief, track
from graph_memory.store import Store, flat_lines

GAME_KINDS = ("links", "things", "acts", "enter")


def toy() -> Store:
    s = Store()
    for e, k in [("hall", "place"), ("cellar", "place"), ("study", "place"), ("lamp", "thing"), ("key", "thing"),
                 ("gem", "thing")]:
        s.entity(e, e.title() if k == "place" else e, k)
    s.start, s.start_have = "hall", []
    s.add_at("lamp", "hall", "", "e1")
    s.add_get("lamp", "take lamp", "hall", "", [], [], "e1")
    s.add_link("hall", "north", "cellar", [], [], "e1")
    s.add_link("cellar", "south", "hall", [], [], "e1")
    s.add_enter("cellar", ["lamp"], ["light lamp"], "e1")
    s.add_act("light lamp", ["lamp"], "e1")
    s.add_get("key", "take key", "cellar", "", [], [], "e1")
    s.add_link("hall", "east", "study", [], ["open door", "look"], "e1")
    s.add_link("hall", "east", "study", [], ["open door"], "e2")
    s.add_get("gem", "take gem", "study", "chest", [], ["unlock chest", "open chest"], "e2")
    s.add_act("unlock chest", ["key"], "e2")
    return s


def test_conditions_are_intersected_and_store_round_trips():
    s = toy()
    assert s.link_do("hall|east") == ["open door"]
    z = Store.from_json(json.loads(json.dumps(s.to_json())))
    assert z.digest() == s.digest() and z.link_do("hall|east") == ["open door"]


def test_plan_chains_tools_entry_conditions_and_gates():
    s = toy()
    lines, actions = plan(s, "gem", "hall", set())
    assert actions == ["take lamp", "light lamp", "north", "take key", "south", "open door", "east",
                       "unlock chest", "open chest", "take gem"]
    assert len(lines) == 4 and "no record" not in " ".join(lines)
    assert plan_brief(s, "Get the gem. The task is done as soon as you are holding it.") == lines
    assert "you are holding the key" in plan_answer(s, "key", "study", {"key"})[0]
    s.links.pop("hall|east")
    assert "no known way" in " ".join(plan(s, "gem", "hall", set())[0])


def test_flat_lines():
    text = "\n".join(flat_lines(toy()))
    assert "entering the Cellar needed: holding the lamp, 'light lamp' done" in text
    assert "before 'east' from the Hall, this was done there first: open door" in text


def test_alignment_merges_modified_names_only_when_unique():
    s = Store()
    assert canon(s, "the brass lamp", "thing") == "brass lamp"
    assert canon(s, "Lamp", "thing") == "brass lamp"
    assert canon(s, "Kitchen", "place") == "kitchen" and canon(s, "the kitchen", "place") == "kitchen"
    canon(s, "red key", "thing")
    canon(s, "blue key", "thing")
    assert canon(s, "key", "thing") == "key"


def facts(**kw):
    d = {"start_place": "Hall", "start_have": [], "links": [], "things": [], "acts": [], "enter": []}
    d.update(kw)
    return d


def test_validate_and_add_extracted():
    good = facts(links=[{"from": "Hall", "action": "N", "to": "Cellar", "have": [], "do": [], "steps": [1]}],
                 things=[{"thing": "lamp", "place": "Hall", "in": "", "got_by": "take lamp", "have": [], "do": [],
                          "steps": [2]}])
    assert validate(json.loads(json.dumps(good)), GAME_KINDS, 2)
    with pytest.raises(ValueError):
        validate(facts(extra=[]), GAME_KINDS, 2)
    with pytest.raises(ValueError):
        validate(facts(links=[{"from": "Hall", "action": "n", "to": "Cellar", "steps": [9]}]), GAME_KINDS, 2)
    s = Store()
    add_extracted(s, validate(good, GAME_KINDS, 2), "e1")
    assert s.start == "hall" and "hall|n" in s.links and s.gets["lamp"]["action"] == "take lamp"
    assert "places: Cellar; Hall" in prompt("a text game", GAME_KINDS, s)


def test_extracted_needs_keep_only_named_things_and_entries_drop_moves():
    s = Store()
    f = facts(start_have=["A room key", "sword"],
              links=[{"from": "Hall", "action": "n", "to": "Wharf", "have": ["room key"], "do": [], "steps": [1]}],
              acts=[{"action": "move rug", "place": "Hall", "have": ["sword", "rug"], "steps": [1]},
                    {"action": "unlock door with key", "place": "Hall", "have": ["room key"], "steps": [1]}],
              enter=[{"place": "Wharf", "have": ["room key"], "do": ["n"], "steps": [1]},
                     {"place": "Attic", "have": ["lantern"], "do": ["turn on lantern"], "steps": [1]}])
    add_extracted(s, validate(f, GAME_KINDS, 1), "e1")
    assert s.names["room key"] == "room key" and s.link_have("hall|n") == []
    assert "move rug" not in s.acts and s.act_have("unlock door with key") == ["room key"]
    assert "wharf" not in s.enters and s.enter_do("attic") == ["turn on lantern"]


def test_tracker_follows_places_and_holdings():
    s = toy()
    turns = [Turn("take lamp", "Taken."), Turn("north", "Cellar\nIt is damp."), Turn("get key", "Taken.")]
    here, held = track(s, turns)
    assert here == "cellar" and held == {"lamp", "key"}


def test_transcript_format():
    rec = Record("Hall", [Turn("north", "Cellar"), Turn("take key", "Taken.")])
    assert transcript(rec) == {"observation_0": "Hall", "steps": [
        {"i": 1, "command": "north", "observation": "Cellar"},
        {"i": 2, "command": "take key", "observation": "Taken."}]}
